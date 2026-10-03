# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "torchvision", "pillow", "transformers>=5.0", "peft>=0.17", "datasets>=3.0", "accelerate",
#   "llm2jev", "flash-linear-attention", "huggingface_hub",
# ]
# ///
"""ezjev 训练任务（HF Jobs）。由 tools/build.py 从 parts/ 生成，不要直接改这个文件。

用法见 jobs/README.md。所有配置都从环境变量读。
"""
import os, json, gzip

HF_REPO    = os.environ["HF_REPO"]
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")
DATA_SCALE = float(os.environ.get("DATA_SCALE", "1.0"))
SEED       = int(os.environ.get("SEED", "0"))
MAX_LEN    = int(os.environ.get("MAX_LEN", "8192"))
LORA_R     = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LR         = float(os.environ.get("LR", "1e-4"))
BRIER_WEIGHT = float(os.environ.get("BRIER_WEIGHT", "1.0"))
TOKENS_PER_BATCH  = int(os.environ.get("TOKENS_PER_BATCH", "48000"))   # A100 80GB
EXAMPLES_PER_STEP = int(os.environ.get("EXAMPLES_PER_STEP", "32"))
LOG_EVERY, SAVE_EVERY = 20, 10**9
DEVICE     = "cuda"
WORK       = "/tmp/ezjev"
MERGED_DIR = "/tmp/ezjev-merged"
os.makedirs(WORK, exist_ok=True)
print(f"BASE_MODEL={BASE_MODEL} DATA_SCALE={DATA_SCALE} -> {HF_REPO}", flush=True)


# ===== 构造训练数据 =====
# 每条样本 = {"src", "state", "questions": {"q": {...}}, "expected": {"q": key}}，格式与榜单评测请求一致。
# 只用公开数据集的 train split；榜单评测集（各 benchmark 的 test/validation）一律不碰。
import json, random, re, os, gzip, itertools
import datasets

R = random.Random(SEED)

def choice(instr, criteria):
    return {"type": "choice", "instructions": instr, "criteria": criteria}

def noul(instr, yes=None, no=None):
    q = {"type": "noul", "instructions": instr}
    if yes or no:
        q["criteria"] = {"true": yes or "Yes", "false": no or "No"}
    return q

def sample(src, state, q, gold):
    return {"src": src, "state": state, "questions": {"q": q}, "expected": {"q": gold}}

def stream(name, cfg=None, split="train", n=1000, buffer=20000):
    """流式读取并随机抽 n 条（不必下载整个数据集）。"""
    ds = datasets.load_dataset(name, cfg, split=split, streaming=True)
    return list(itertools.islice(ds.shuffle(seed=SEED, buffer_size=buffer), n))

def mc(src, state, instr, options, gold_idx):
    """选择题：打乱选项，随机用 A/B/C、1/2/3 或选项原文做 key（模拟评测集的多种写法）。"""
    order = list(range(len(options)))
    R.shuffle(order)
    opts = [options[i] for i in order]
    gold = order.index(gold_idx)
    style = R.random()
    if style < 0.4 and len(opts) <= 26:
        keys = [chr(65 + i) for i in range(len(opts))]
        crit = dict(zip(keys, opts))
    elif style < 0.7:
        keys = [str(i + 1) for i in range(len(opts))]
        crit = dict(zip(keys, opts))
    else:
        keys = [o.strip()[:200] for o in opts]
        if len(set(keys)) != len(keys):
            keys = [str(i + 1) for i in range(len(opts))]
            crit = dict(zip(keys, opts))
        else:
            crit = {k: None for k in keys}
    return sample(src, state, choice(instr, crit), keys[gold])

def labelset(src, text, instr, names, gold_name, desc=None):
    """分类题：所有类别都给出（评测集不做选项裁剪，训练也一样）。"""
    names = list(names)
    if R.random() < 0.5:
        R.shuffle(names)
    if desc or R.random() < 0.6:
        crit = {n: (desc.get(n) if desc else None) for n in names}
    else:
        crit = {n: (n.replace("_", " ") if "_" in n else None) for n in names}
    return sample(src, text, choice(instr, crit), gold_name)

def pick(*xs):
    return R.choice(xs)

# ---------- 各数据源 ----------
def nli(name, cfg, split, n, src):
    out = []
    names = ["entailment", "neutral", "contradiction"]
    desc = {"entailment": "The hypothesis must be true given the premise.",
            "neutral": "The hypothesis may or may not be true given the premise.",
            "contradiction": "The hypothesis cannot be true given the premise."}
    for ex in stream(name, cfg, split, n):
        lab = int(ex["label"])
        if lab not in (0, 1, 2):
            continue
        state = {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
        if R.random() < 0.75:
            out.append(labelset(src, state, pick("What is the relation between the premise and the hypothesis?",
                                                 "Does the premise entail, contradict, or say nothing about the hypothesis?"),
                                names, names[lab], desc))
        else:
            out.append(sample(src, state, noul("Is the hypothesis supported by the premise?"), lab == 0))
    return out

def anli(n):
    out = []
    for split in ("train_r1", "train_r2", "train_r3"):
        out += nli("facebook/anli", None, split, n // 3, "anli_train")
    return out

def boolq(n):
    out = []
    for ex in stream("google/boolq", n=n):
        state = f"Passage:\n{ex['passage']}"
        q = ex["question"].strip().rstrip("?") + "?"
        if R.random() < 0.6:
            out.append(sample("boolq", state, noul(q[0].upper() + q[1:]), bool(ex["answer"])))
        else:
            out.append(mc("boolq", state, q[0].upper() + q[1:], ["Yes", "No"], 0 if ex["answer"] else 1))
    return out

def full_labels(name, cfg, split, text_key, label_key, n, src, instr):
    """小数据集整份加载，拿到完整类别表（label 是 ClassLabel 就用其 names，否则 label 本身就是类名）。"""
    ds = datasets.load_dataset(name, cfg, split=split)
    feat = ds.features[label_key]
    names = list(feat.names) if hasattr(feat, "names") else sorted(set(ds[label_key]))
    out = []
    for i in R.sample(range(len(ds)), min(n, len(ds))):
        ex = ds[i]
        gold = names[ex[label_key]] if hasattr(feat, "names") else ex[label_key]
        out.append(labelset(src, ex[text_key], instr(), names, gold))
    return out

def banking(n):
    return full_labels("mteb/banking77", None, "train", "text", "label_text", n, "banking77_train",
                       lambda: pick("Which banking intent does the customer message express?",
                                    "Classify the customer's request.", "What does the customer want?"))

def clinc(n):
    return full_labels("clinc/clinc_oos", "plus", "train", "text", "intent", n, "clinc_train",
                       lambda: pick("Which intent does the user utterance express? Use oos if it is out of scope.",
                                    "Route this assistant query to the right intent (oos = none of these).",
                                    "Classify the user's intent."))

def massive(n):
    return full_labels("mteb/amazon_massive_intent", "en", "train", "text", "label_text", n, "massive_train",
                       lambda: pick("Which virtual-assistant intent matches this command?", "Classify the user's intent."))

def topic(name, cfg, text_fn, label_key, n, src, instr):
    ds = datasets.load_dataset(name, cfg, split="train", streaming=True)
    names = ds.features[label_key].names
    out = []
    for ex in stream(name, cfg, "train", n):
        out.append(labelset(src, text_fn(ex), instr, names, names[ex[label_key]]))
    return out

def sst5(n):
    out = []
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    for ex in stream("SetFit/sst5", n=n):
        if R.random() < 0.5:
            out.append(labelset("sst5", ex["text"], "What is the sentiment of this review?", levels, levels[int(ex["label"])]))
        else:
            out.append(sample("sst5", ex["text"], {"type": "score", "instructions": "How positive is this review?",
                                                   "criteria": levels}, str(int(ex["label"]))))
    return out

def irony(n):
    out = []
    for ex in stream("cardiffnlp/tweet_eval", "irony", n=n):
        if R.random() < 0.5:
            out.append(sample("irony", ex["text"], noul(pick("Is this tweet ironic or sarcastic?", "Is the author being sarcastic?")),
                              int(ex["label"]) == 1))
        else:
            out.append(labelset("irony", ex["text"], "Is the tweet sarcastic?", ["sarcastic", "not_sarcastic"],
                                "sarcastic" if int(ex["label"]) == 1 else "not_sarcastic"))
    return out

def winogrande(n):
    out = []
    for ex in stream("allenai/winogrande", "winogrande_xl", n=n):
        out.append(mc("winogrande", ex["sentence"], "Which option correctly fills the blank (_)?",
                      [ex["option1"], ex["option2"]], int(ex["answer"]) - 1))
    return out

def hellaswag(n):
    out = []
    for ex in stream("Rowan/hellaswag", n=n):
        state = f"{ex['activity_label']}: {ex['ctx']}"
        out.append(mc("hellaswag", state, "Which ending is the most plausible continuation?", ex["endings"], int(ex["label"])))
    return out

def mmlu_aux(n):
    out = []
    for ex in stream("cais/mmlu", "auxiliary_train", n=n):
        ex = ex["train"]
        out.append(mc("mmlu_aux", "", ex["question"], ex["choices"], int(ex["answer"])))
    return out

def keyed_mc(name, cfg, n, src, qkey="question"):
    out = []
    for ex in stream(name, cfg, n=n):
        ch = ex["choices"]
        if ex["answerKey"] not in ch["label"]:
            continue
        state = ""
        if "fact1" in ex:  # QASC：带上事实作为 state
            state = f"Fact 1: {ex['fact1']}\nFact 2: {ex['fact2']}" if R.random() < 0.5 else ""
        out.append(mc(src, state, ex[qkey], ch["text"], ch["label"].index(ex["answerKey"])))
    return out

def arc(n):
    return keyed_mc("allenai/ai2_arc", "ARC-Challenge", n // 2, "arc_train") + \
           keyed_mc("allenai/ai2_arc", "ARC-Easy", n // 2, "arc_train")

def sciq(n):
    out = []
    for ex in stream("allenai/sciq", n=n):
        opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        state = ex["support"] if ex["support"] and R.random() < 0.5 else ""
        out.append(mc("sciq", state, ex["question"], opts, 0))
    return out

def aqua(n):
    out = []
    for ex in stream("deepmind/aqua_rat", "raw", n=n):
        opts = [re.sub(r"^[A-E]\)\s*", "", o) for o in ex["options"]]
        gold = "ABCDE".index(ex["correct"])
        out.append(mc("aqua", "", ex["question"], opts, gold))
    return out

def gsm8k(n):
    """GSM8K train → 4 选 1 / 10 选 1 的数值选择题（评测集也是这两种形式）。"""
    out = []
    for ex in stream("openai/gsm8k", "main", n=n):
        ans = ex["answer"].split("####")[-1].strip().replace(",", "")
        try:
            a = float(ans)
        except ValueError:
            continue
        k = R.choice([4, 10])
        nums = {float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ex["answer"].split("####")[0].replace(",", ""))}
        cands = {a + d for d in (-10, -5, -2, -1, 1, 2, 5, 10)} | {a * 2, a / 2, a * 10} | nums
        cands = [c for c in cands if c != a and c >= 0]
        R.shuffle(cands)
        fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
        opts, seen = [fmt(a)], {fmt(a)}
        for c in cands:
            if len(opts) == k:
                break
            if fmt(c) not in seen:
                opts.append(fmt(c)); seen.add(fmt(c))
        if len(opts) < k:
            continue
        out.append(mc("gsm8k_train", "", ex["question"], opts, 0))
    return out

def hotpot(n):
    """多段落检索：哪个段落包含答案 / 某段落是否相关（对应 BRIGHT、ToolRet、HoVer 这类任务）。"""
    out, pool = [], []
    for ex in stream("hotpotqa/hotpot_qa", "distractor", n=n * 2, buffer=5000):
        titles, sents = ex["context"]["title"], ex["context"]["sentences"]
        support = set(ex["supporting_facts"]["title"])
        paras = [" ".join(s) for s in sents]
        extra = [tp for tp in pool if tp[0] not in titles]
        pool = (pool + list(zip(titles, paras)))[-300:]
        if R.random() < 0.5:
            hits = [i for i, p in enumerate(paras) if ex["answer"] in p and titles[i] in support]
            if len(hits) != 1:
                continue
            gold = titles[hits[0]]
            docs = dict(zip(titles, paras))
            extra = [tp for tp in extra if ex["answer"] not in tp[1]]
            if R.random() < 0.3 and len(extra) >= 60:  # 长上下文版：再混入几十段无关段落（评测集里有很长的文档）
                for t, p in R.sample(extra, R.randint(20, 60)):
                    docs.setdefault(t, p)
                items = list(docs.items()); R.shuffle(items); docs = dict(items)
            state = {"question": ex["question"], "passages": docs}
            crit = {t: None for t in docs}
            out.append(sample("hotpot_train", state, choice("Which passage contains the answer to the question?", crit), gold))
        else:
            i = R.randrange(len(titles))
            state = f"Query: {ex['question']}\n\nDocument ({titles[i]}):\n{paras[i]}"
            rel = titles[i] in support
            if R.random() < 0.5:
                out.append(sample("hotpot_train", state, noul("Is this document relevant to answering the query?"), rel))
            else:
                out.append(labelset("hotpot_train", state, "How relevant is the document to the query?",
                                    ["relevant", "not_relevant"], "relevant" if rel else "not_relevant"))
        if len(out) >= n:
            break
    return out

FUNC_NAME = re.compile(r'^\s{0,4}"name":\s*"([^"]+)"', re.M)
FUNC_DESC = re.compile(r'"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"')

def glaive(n):
    """函数调用：该调用哪个工具，还是不调用（对应 BFCL / API-Bank / When2Call）。"""
    out = []
    for ex in stream("glaiveai/glaive-function-calling-v2", n=n * 2, buffer=10000):
        names = FUNC_NAME.findall(ex["system"])
        if not names:
            continue
        descs = dict(FUNC_DESC.findall(ex["system"]))
        m = re.search(r"USER:(.*?)ASSISTANT:(.*?)(?:<\|endoftext\|>|USER:|FUNCTION RESPONSE:|$)", ex["chat"], re.S)
        if not m:
            continue
        user, reply = m.group(1).strip(), m.group(2)
        call = re.search(r'<functioncall>\s*\{"name":\s*"([^"]+)"', reply)
        gold = call.group(1) if call else "no_tool_call"
        if gold != "no_tool_call" and gold not in names:
            continue
        tools = ex["system"].split("-", 1)[-1].strip()
        state = [{"role": "system", "content": "Available functions:\n" + tools}, {"role": "user", "content": user}]
        crit = {nm: descs.get(nm, None) for nm in names}
        crit["no_tool_call"] = "Do not call a function: answer directly, ask for missing information, or decline."
        if R.random() < 0.7:
            out.append(sample("glaive_tools", state, choice("Which function should the assistant call next?", crit), gold))
        else:
            out.append(sample("glaive_tools", state, noul("Should the assistant call a function now?"), gold != "no_tool_call"))
        if len(out) >= n:
            break
    return out

def shp(n):
    """人类偏好：哪个回复更受欢迎（对应 Arts & Human Judgment 里的偏好类任务）。"""
    out = []
    for ex in stream("stanfordnlp/SHP", n=n * 3, buffer=20000):
        if ex["score_ratio"] < 2 or len(ex["history"]) + len(ex["human_ref_A"]) + len(ex["human_ref_B"]) > 6000:
            continue
        state = {"post": ex["history"], "response_1": ex["human_ref_A"], "response_2": ex["human_ref_B"]}
        crit = {"response_1": None, "response_2": None}
        out.append(sample("shp", state, choice("Which response did readers prefer?", crit),
                          "response_1" if ex["labels"] == 1 else "response_2"))
        if len(out) >= n:
            break
    return out

def hh(n):
    out = []
    for ex in stream("Anthropic/hh-rlhf", n=n * 2):
        a, b = ex["chosen"], ex["rejected"]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        cut = a.rfind("Assistant:", 0, k + 1)
        if cut < 0:
            continue
        ctx, ra, rb = a[:cut].strip(), a[cut + 10:].strip(), b[cut + 10:].strip()
        if not ra or not rb or ra == rb or len(ctx) > 6000:
            continue
        out.append(mc("hh_rlhf", {"conversation": ctx}, "Which final assistant reply is more helpful and harmless?", [ra, rb], 0))
        if len(out) >= n:
            break
    return out

SOURCES = {
    "mnli":       lambda n: nli("nyu-mll/multi_nli", None, "train", n, "mnli"),
    "snli":       lambda n: nli("stanfordnlp/snli", None, "train", n, "snli"),
    "anli":       anli,
    "boolq":      boolq,
    "banking77":  banking,
    "clinc":      clinc,
    "massive":    massive,
    "ag_news":    lambda n: topic("fancyzhx/ag_news", None, lambda e: e["text"], "label", n, "ag_news", "What is the topic of this news article?"),
    "dbpedia":    lambda n: topic("fancyzhx/dbpedia_14", None, lambda e: f"{e['title']}: {e['content']}", "label", n, "dbpedia", "Which category does this entity belong to?"),
    "yahoo":      lambda n: topic("community-datasets/yahoo_answers_topics", None, lambda e: f"{e['question_title']}\n{e['question_content']}", "topic", n, "yahoo", "Which topic does this question belong to?"),
    "emotion":    lambda n: topic("dair-ai/emotion", None, lambda e: e["text"], "label", n, "emotion", "Which emotion does the text express?"),
    "sst5":       sst5,
    "irony":      irony,
    "winogrande": winogrande,
    "hellaswag":  hellaswag,
    "mmlu_aux":   mmlu_aux,
    "arc":        arc,
    "csqa":       lambda n: keyed_mc("tau/commonsense_qa", None, n, "csqa"),
    "obqa":       lambda n: keyed_mc("allenai/openbookqa", "main", n, "obqa", "question_stem"),
    "qasc":       lambda n: keyed_mc("allenai/qasc", None, n, "qasc"),
    "sciq":       sciq,
    "aqua":       aqua,
    "gsm8k":      gsm8k,
    "hotpot":     hotpot,
    "tools":      glaive,
    "shp":        shp,
    "hh":         hh,
}

def build(sizes, path):
    rows = []
    for name, n in sizes.items():
        n = max(8, int(n * DATA_SCALE))
        try:
            got = SOURCES[name](n)
        except Exception as e:  # 某个数据源挂了不影响整体
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        rows += got
        print(f"{name:12s} {len(got):6d}", flush=True)
    R.shuffle(rows)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total", len(rows), "->", path)
    return rows


# ===== 每个数据源抽多少题（再乘以 DATA_SCALE）=====
SIZES = {
    # 语言理解 / NLI（对应 ANLI、ContractNLI、NLI4CT、VAST、RAGTruth）
    "mnli": 3000, "snli": 1000, "anli": 4500, "boolq": 3000,
    # 检索与分类（BANKING77、CLINC150、Amazon ESCI、PhishNChips……）
    "banking77": 3500, "clinc": 4000, "massive": 2500,
    "ag_news": 1000, "dbpedia": 1000, "yahoo": 1500, "emotion": 1500, "sst5": 1500, "irony": 1500,
    # 知识与推理（MMLU-Pro、GPQA、BBH、GSM8K、WinoGrande、HellaSwag……）
    "winogrande": 3000, "hellaswag": 3000, "mmlu_aux": 5000, "arc": 2000, "csqa": 1500, "obqa": 1000,
    "qasc": 1000, "sciq": 1500, "aqua": 1500, "gsm8k": 2500,
    # 多段落检索（BRIGHT、ToolRet、HoVer）
    "hotpot": 3000,
    # 工具调用（BFCL、API-Bank、When2Call）
    "tools": 3500,
    # 人类偏好（Arts & Human Judgment）
    "shp": 2000, "hh": 1000,
}



rows = build(SIZES, f"{WORK}/all.jsonl.gz")
n_dev = min(1500, max(50, len(rows) // 30))
for name, part in (("dev", rows[:n_dev]), ("train", rows[n_dev:])):
    with gzip.open(f"{WORK}/{name}.jsonl.gz", "wt") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train", len(rows) - n_dev, "dev", n_dev, flush=True)


# ===== 训练：LoRA，损失 = 标签上的交叉熵 + Brier =====
# 和 llm2jev 推理完全同一套提示词：render() 生成文本，在 "Answer:" 之后那个位置读各选项字母的 logit。
import math, time, json, gzip, random
import torch, torch.nn as nn, torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from llm2jev.prompt import render, find_labels

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
_, probe, _ = render(tok, "x", {"q": {"type": "noul"}}, ["A", "B"])
LABELS, LABEL_IDS = find_labels(tok, probe["q"][0])

def encode(row):
    """一条样本 -> (token ids, 选项数, 正确选项下标)。"""
    _, out, _ = render(tok, row["state"], row["questions"], LABELS)
    text, keys = out["q"]
    gold = row["expected"]["q"]
    gold = {True: "true", False: "false"}.get(gold, gold) if isinstance(gold, bool) else gold
    return tok.encode(text, add_special_tokens=False), len(keys), keys.index(gold)

def load_encoded(path):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    enc, dropped = [], 0
    for r in rows:
        ids, k, g = encode(r)
        if len(ids) > MAX_LEN:
            dropped += 1
            continue
        enc.append((ids, k, g, r["src"]))
    print(f"{path}: {len(enc)} 条, 超过 {MAX_LEN} tokens 丢弃 {dropped} 条", flush=True)
    return enc

def batches(data, budget, shuffle=True):
    """按长度分桶，每个 micro-batch 的 padding 后 token 数不超过 budget。"""
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    out, cur = [], []
    for i in order:
        L = len(data[i][0])
        if cur and (len(cur) + 1) * max(L, len(data[cur[-1]][0])) > budget:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(SEED).shuffle(out)
    return out

def collate(data, idx, device):
    L = max(len(data[i][0]) for i in idx)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(idx), L), pad, dtype=torch.long)
    mask = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):  # 右侧 padding：因果模型里真实 token 不受后面 padding 影响
        x = data[i][0]
        ids[r, :len(x)] = torch.tensor(x)
        mask[r, :len(x)] = 1
    last = mask.sum(1) - 1
    return ids.to(device), mask.to(device), last.to(device)

def core_of(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m

def label_logits(model, ids, mask, last, kmax):
    """只取最后一个真实 token 的 hidden state，再乘选项字母那几行 lm_head（不算完整词表，省显存）。"""
    core = core_of(model)
    h = core.model(input_ids=ids, attention_mask=mask).last_hidden_state
    h = h[torch.arange(h.size(0), device=h.device), last]
    W = core.lm_head.weight[LABEL_IDS_T[:kmax]]
    return h.float() @ W.float().T

def loss_fn(logits, ks, golds):
    total = 0.0
    for r, (k, g) in enumerate(zip(ks, golds)):
        lg = logits[r, :k]
        p = lg.softmax(-1)
        onehot = F.one_hot(torch.tensor(g, device=lg.device), k).float()
        total = total + F.cross_entropy(lg[None], torch.tensor([g], device=lg.device)) + BRIER_WEIGHT * ((p - onehot) ** 2).sum()
    return total

@torch.no_grad()
def evaluate(model, data, T=1.0, budget=None):
    model.eval()
    correct, nll, brier, per = 0, 0.0, 0.0, {}
    all_logits = []
    for idx in batches(data, budget or TOKENS_PER_BATCH, shuffle=False):
        ids, mask, last = collate(data, idx, DEVICE)
        kmax = max(data[i][1] for i in idx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            lg = label_logits(model, ids, mask, last, kmax)
        for r, i in enumerate(idx):
            _, k, g, src = data[i]
            l = lg[r, :k].float().cpu()
            all_logits.append((l, g))
            p = (l / T).softmax(-1)
            ok = int(p.argmax()) == g
            correct += ok
            nll -= math.log(max(p[g].item(), 1e-12))
            brier += ((p - F.one_hot(torch.tensor(g), k).float()) ** 2).sum().item()
            a, n = per.get(src, (0, 0)); per[src] = (a + ok, n + 1)
    n = len(data)
    model.train()
    return {"acc": correct / n, "nll": nll / n, "brier": brier / n,
            "per_src": {s: round(a / m, 3) for s, (a, m) in sorted(per.items())}}, all_logits

def fit_temperature(all_logits):
    """在 dev 上找使 NLL 最小的温度 T（全局一个，llm2jev --temperature 用它；不改变 argmax）。"""
    best = (1.0, float("inf"))
    for T in [x / 100 for x in range(50, 301, 2)]:
        nll = sum(-(l / T).log_softmax(-1)[g].item() for l, g in all_logits) / len(all_logits)
        if nll < best[1]:
            best = (T, nll)
    return best


train_data = load_encoded(f"{WORK}/train.jsonl.gz")
dev_data = load_encoded(f"{WORK}/dev.jsonl.gz")


# ===== 加载模型 + LoRA + 训练循环（可断点续训）=====
import os, json, math, time
from peft import PeftModel

def load_base():
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    multimodal = hasattr(cfg, "vision_config")
    cls = getattr(transformers, "AutoModelForMultimodalLM", transformers.AutoModelForImageTextToText) if multimodal \
        else transformers.AutoModelForCausalLM
    m = cls.from_pretrained(BASE_MODEL, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                            device_map={"": 0} if DEVICE == "cuda" else None)
    return m

model = load_base()
LABEL_IDS_T = torch.tensor(LABEL_IDS, device=DEVICE)
# 只在语言模型的线性层上挂 LoRA，视觉塔保持不动
targets = sorted({n for n, mod in model.named_modules()
                  if isinstance(mod, nn.Linear) and not any(s in n for s in ("visual", "vision", "lm_head", "audio"))})
print("LoRA 层数:", len(targets), "例如", targets[:3])
ckpt_dir = f"{WORK}/lora"
state_file = f"{ckpt_dir}/train_state.json"
if os.path.exists(state_file):
    model = PeftModel.from_pretrained(model, ckpt_dir, is_trainable=True)
    done = json.load(open(state_file))["step"]
    print("从断点续训, 已完成 step", done)
else:
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, target_modules=targets))
    done = 0
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False




dev_before, _ = evaluate(model, dev_data)
print("训练前 dev:", json.dumps(dev_before), flush=True)


micro = batches(train_data, TOKENS_PER_BATCH)
steps, cur, n = [], [], 0
for b in micro:  # 每个优化 step 累积约 EXAMPLES_PER_STEP 条样本
    cur.append(b); n += len(b)
    if n >= EXAMPLES_PER_STEP:
        steps.append(cur); cur, n = [], 0
if cur:
    steps.append(cur)
total = len(steps)
print(f"{len(train_data)} 条训练样本, {len(micro)} 个 micro-batch, {total} 个优化 step", flush=True)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
for _ in range(done):
    sched.step()

model.train()
t0, run_loss, run_n = time.time(), 0.0, 0
for s in range(done, total):
    nex = sum(len(b) for b in steps[s])
    for idx in steps[s]:
        ids, mask, last = collate(train_data, idx, DEVICE)
        kmax = max(train_data[i][1] for i in idx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            lg = label_logits(model, ids, mask, last, kmax)
        loss = loss_fn(lg, [train_data[i][1] for i in idx], [train_data[i][2] for i in idx]) / nex
        loss.backward()
        run_loss += loss.item()  # 已按样本数归一化，累加后就是本 step 的平均每样本损失
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    run_n += 1
    if (s + 1) % LOG_EVERY == 0 or s == done:
        el = time.time() - t0
        eta = el / (s + 1 - done) * (total - s - 1)
        print(f"step {s+1}/{total}  loss/样本 {run_loss / run_n:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"已用 {el/60:.1f} 分钟, 预计还要 {eta/60:.1f} 分钟", flush=True)
        run_loss, run_n = 0.0, 0
    if (s + 1) % SAVE_EVERY == 0 or s + 1 == total:
        model.save_pretrained(ckpt_dir)
        json.dump({"step": s + 1, "total": total}, open(state_file, "w"))
print("训练完成", flush=True)



dev_after, dev_logits = evaluate(model, dev_data)
TEMPERATURE = round(fit_temperature(dev_logits)[0], 3)
print("训练后 dev:", json.dumps(dev_after), flush=True)
print("温度 T =", TEMPERATURE, flush=True)


# ===== 合并 LoRA → 完整权重，保存到 Drive 并上传 HF =====
import gc, json, os
from transformers import AutoProcessor
from huggingface_hub import HfApi

try:
    del model
except NameError:
    pass
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

merged = PeftModel.from_pretrained(load_base(), f"{WORK}/lora").merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
try:
    AutoProcessor.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)
except Exception:
    tok.save_pretrained(MERGED_DIR)
json.dump({"base_model": BASE_MODEL, "temperature": TEMPERATURE, "prompt": "llm2jev --prompt chat",
           "dev_before": dev_before, "dev_after": dev_after}, open(f"{MERGED_DIR}/ezjev.json", "w"), indent=1, ensure_ascii=False)
open(f"{MERGED_DIR}/README.md", "w").write(f"""---
base_model: {BASE_MODEL}
library_name: transformers
tags: [decision-model, jev, typed-decisions]
---

# {HF_REPO.split('/')[-1]}

A typed-decision model (Jev-style `/v1/systemone`: `choice`, `noul`, `score` questions answered with a probability
per option from one forward pass). LoRA fine-tune of `{BASE_MODEL}`, merged into full weights. Trained only on train
splits of public datasets; no Decision Index suite rows were used.

Serve with [llm2jev](https://github.com/tic-top/llm2jev) (chat prompt, temperature {TEMPERATURE}):

```bash
vllm serve {HF_REPO} --max-logprobs 256 --return-tokens-as-token-ids --port 8000
llm2jev --model {HF_REPO} --backend vllm --url http://127.0.0.1:8000 --port 8080 --temperature {TEMPERATURE}
```

Held-out dev accuracy (our own splits): base {dev_before['acc']:.3f} -> fine-tuned {dev_after['acc']:.3f}.

Some training sources (e.g. ANLI, CC BY-NC 4.0) are non-commercial; check their licences before commercial use.
""")
print("已保存到", MERGED_DIR, flush=True)
del merged; gc.collect()

api = HfApi()
api.create_repo(HF_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_REPO, commit_message="ezjev weights")
print(f"已上传: https://huggingface.co/{HF_REPO} （目前是私有仓库，提交榜单前改成公开）", flush=True)
