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
DATA_VERSION = int(os.environ.get("DATA_VERSION", "1"))
DATA_REPO  = os.environ.get("DATA_REPO")   # 设了就直接下载 jobs/data_job.py 做好的数据（{DATA_NAME}/train.jsonl.gz、dev.jsonl.gz）
DATA_NAME  = os.environ.get("DATA_NAME")
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

def yn(instr, yes="Yes", no="No"):
    """评测集里大量 yes/no 判断写成 choice（key 是 yes/no），不是 noul。"""
    return choice(instr, {"yes": yes, "no": no})

def sample(src, state, q, gold):
    return {"src": src, "state": state, "questions": {"q": q}, "expected": {"q": gold}}

def multi(src, state, qs, golds):
    """一个 state 配多个问题（ContractNLI、BFCL、ToolRet 这类评测题就是这样）。"""
    return {"src": src, "state": state, "questions": qs, "expected": golds}

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
    cut = (0.3, 0.45, 0.75) if DATA_VERSION >= 2 else (0.4, 0.7, 0.7)
    if style < cut[0] and len(opts) <= 26:
        keys = [chr(65 + i) for i in range(len(opts))]
        crit = dict(zip(keys, opts))
    elif style < cut[1]:
        keys = [str(i + 1) for i in range(len(opts))]
        crit = dict(zip(keys, opts))
    elif style < cut[2]:  # option_0、option_1……（GSM8K、CRUXEval、API-Bank、Habermas 等评测题的写法）
        keys = [f"option_{i}" for i in range(len(opts))]
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
    if DATA_VERSION >= 2 and R.random() < 0.35:  # BANKING77、CLINC 评测题：key 是 option_i，类名放在描述里
        keys = [f"option_{i}" for i in range(len(names))]
        crit = {k: (f"{n}: {desc[n]}" if desc and desc.get(n) else n) for k, n in zip(keys, names)}
        return sample(src, text, choice(instr, crit), keys[names.index(gold_name)])
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

def inline_state(row):
    """把短 state 并进题目说明、state 置空（ANLI、WinoGrande、HellaSwag、VAST、CLadder 等评测题就是这种写法）。"""
    st, (qid, q), = row["state"], *row["questions"].items()
    if not isinstance(q.get("instructions"), str):
        return row
    if isinstance(st, str) and 0 < len(st) < 3000:
        body = st
    elif isinstance(st, dict) and st and all(isinstance(v, str) for v in st.values()) and sum(map(len, st.values())) < 3000:
        body = "\n".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in st.items())
    else:
        return row
    q = {**q, "instructions": q["instructions"] + "\n" + body}
    return {**row, "state": R.choice(({}, "")), "questions": {qid: q}}

def build(sizes, path):
    rows = []
    for name, n in sizes.items():
        n = max(8, int(n * DATA_SCALE))
        try:
            got = SOURCES[name](n)
        except Exception as e:  # 某个数据源挂了不影响整体
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        if DATA_VERSION >= 2:
            got = [inline_state(r) if len(r["questions"]) == 1 and R.random() < 0.3 else r for r in got]
        rows += got
        print(f"{name:12s} {len(got):6d}", flush=True)
    R.shuffle(rows)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total", len(rows), "->", path)
    return rows


# ===== 第二版数据：针对 Decision Index 各 benchmark 的题型补充训练数据 =====
# 每个来源只用 train（或 dev）部分；评测用的是 test 部分（见 decision-index kit 的 suite/build/*.py）。
# 题目格式照抄评测请求的写法（state 结构、instructions、criteria 的 key），但内容全部来自训练数据。
# 构造完还会和评测集做一遍去重（parts/decontam.py），重合的样本删掉。
import ast, csv, io, urllib.request, zipfile

RAW = "/tmp/ezjev-raw"
os.makedirs(RAW, exist_ok=True)

def fetch(url, name):
    path = f"{RAW}/{name}"
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "ezjev-data"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path

GH = "https://raw.githubusercontent.com"
YES_NO_REL = {"no": "Not relevant/useful to the query.", "yes": "Relevant/useful to the query."}
RANK_TASK = "Rank candidate documents/tools by relevance to this query."
CAND_TASK = "Assess whether this candidate is relevant/useful to the query. Use the full text below; return the probability of relevance."

# ---------- 语言理解 ----------
NLI3 = {"Entailment": "The contract entails the hypothesis.", "Contradiction": "The contract contradicts the hypothesis.",
        "NotMentioned": "The hypothesis is neither entailed nor contradicted by the contract."}

def contractnli(n):
    """ContractNLI train+dev：整份保密协议 + 2–4 个假设（评测是 test 的 123 份合同，每份 17 个假设）。"""
    z = zipfile.ZipFile(fetch("https://github.com/stanfordnlp/contract-nli/raw/gh-pages/resources/contract-nli.zip", "contract-nli.zip"))
    docs, labels = [], None
    for part in ("train", "dev"):
        d = json.loads(z.read(f"contract-nli/{part}.json"))
        docs += d["documents"]; labels = d["labels"]
    out = []
    while len(out) < n:
        doc = R.choice(docs)
        ann = doc["annotation_sets"][0]["annotations"]
        keys = R.sample(sorted(ann), min(len(ann), R.randint(2, 4)))
        qs = {k: choice("Classify the relationship between the contract and this hypothesis:\n" + labels[k]["hypothesis"], NLI3) for k in keys}
        out.append(multi("contractnli_train", doc["text"], qs, {k: ann[k]["choice"] for k in keys}))
    return out

def vast(n):
    """VAST train（评测是 vast_test.csv）。label: 0 反对 / 1 支持 / 2 中立。"""
    rows = list(csv.DictReader(open(fetch(f"{GH}/emilyallaway/zero-shot-stance/master/data/VAST/vast_train.csv", "vast_train.csv"),
                                    encoding="utf-8")))
    names = ["against", "favor", "neutral"]
    out = []
    for r in R.sample(rows, min(n, len(rows))):
        instr = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
        lab = names[int(r["label"])]
        if R.random() < 0.6:  # 评测题的写法：A/B/C 固定顺序
            out.append(sample("vast_train", {}, choice(instr, dict(zip("ABC", names))), "ABC"[names.index(lab)]))
        else:
            out.append(labelset("vast_train", {}, instr, names, lab))
    return out

def nli4ct(n):
    """NLI4CT（SemEval-2024 Task 2）train+dev（评测是 test.json）。"""
    z = zipfile.ZipFile(fetch("https://github.com/ai-systems/Task-2-SemEval-2024/raw/main/training_data.zip", "nli4ct.zip"))
    names = {m.split("/")[-1][:-5]: m for m in z.namelist() if m.startswith("CT json/") and m.endswith(".json")}
    ct = lambda i: json.loads(z.read(names[i]))
    items = list(json.loads(z.read("train.json")).values()) + list(json.loads(z.read("dev.json")).values())
    crit = {"Entailment": "The clinical trial evidence entails the statement.",
            "Contradiction": "The clinical trial evidence contradicts the statement."}
    out = []
    for it in R.sample(items, min(n, len(items))):
        sec = it["Section_id"]
        state = {"primary_trial": {"id": it["Primary_id"], "section": sec, "text": ct(it["Primary_id"])[sec]}}
        if it.get("Secondary_id"):
            state["secondary_trial"] = {"id": it["Secondary_id"], "section": sec, "text": ct(it["Secondary_id"])[sec]}
        out.append(sample("nli4ct_train", state, choice("Classify the statement against the supplied clinical trial evidence:\n"
                                                        + it["Statement"], crit), it["Label"]))
    return out

def ragtruth(n):
    """RAGTruth train：回答里有没有原文不支持的内容（评测是 test split）。"""
    q = noul("The response contains content that is not supported by the context in the prompt.",
             "The response contains content that is not supported by the context in the prompt.",
             "All content of the response is supported by the context in the prompt.")
    out = []
    for ex in stream("wandb/RAGTruth-processed", n=n, buffer=20000):
        lab = ex["hallucination_labels_processed"]
        lab = ast.literal_eval(lab) if isinstance(lab, str) else lab
        bad = bool(lab.get("evident_conflict") or lab.get("baseless_info"))
        ctx = ex["context"] if isinstance(ex["context"], str) else json.dumps(ex["context"], ensure_ascii=False)
        out.append(sample("ragtruth_train", {"prompt": f"{ex['query']}\n{ctx}", "response": ex["output"]}, q, bad))
    return out

def isarcasm(n):
    """iSarcasmEval train（英文 + 阿拉伯文；评测只用 test/）。"""
    out = []
    for lang, col in (("En", "tweet"), ("Ar", "text")):
        rows = list(csv.DictReader(open(fetch(f"{GH}/iabufarha/iSarcasmEval/main/train/train.{lang}.csv", f"isarc.{lang}.csv"),
                                        encoding="utf-8")))
        for r in R.sample(rows, min(n // 2, len(rows))):
            if not r.get(col):
                continue
            sar = str(r["sarcastic"]).strip() == "1"
            out.append(sample("isarcasm_train", r[col], yn(pick("Is this text intended to be sarcastic?", "Is the author being sarcastic?")),
                              "yes" if sar else "no"))
    return out

ACOS_SENT = {"0": "negative", "1": "neutral", "2": "positive"}

def acos(n):
    """ACOS train（评测是 *_test.tsv）：评论里有没有某个 方面类别 + 情感 组合。"""
    out = []
    for dom, fn in (("Laptop-ACOS", "laptop_quad_train.tsv"), ("Restaurant-ACOS", "rest16_quad_train.tsv")):
        lines = open(fetch(f"{GH}/NUSTM/ACOS/main/data/{dom}/{fn}", fn), encoding="utf-8").read().splitlines()
        parsed = []
        for line in lines:
            parts = line.split("\t")
            pairs = {(q.split()[1], ACOS_SENT.get(q.split()[2], "neutral")) for q in parts[1:] if len(q.split()) >= 3}
            parsed.append((parts[0], pairs))
        cats = sorted({c for _, ps in parsed for c, _ in ps})
        for review, pairs in R.sample(parsed, min(n // 2, len(parsed))):
            cand = list(pairs) + [(R.choice(cats), R.choice(list(ACOS_SENT.values()))) for _ in range(4)]
            cand = list(dict.fromkeys(cand))[:6]
            qs, gold = {}, {}
            for c, s in cand:
                k = f"{c}__sentiment_{s}"
                qs[k] = choice(f"Given the review below, decide whether it expresses at least one aspect-category/sentiment pair ({c}, {s}) "
                               "anywhere in the review. Answer yes only when that exact category and sentiment pair is present; answer no "
                               "otherwise. Multiple opposite sentiments are separate valid questions.",
                               {"yes": "The exact category/sentiment pair is present.", "no": "The exact category/sentiment pair is absent."})
                gold[k] = "yes" if (c, s) in pairs else "no"
            out.append(multi("acos_train", {"review": review, "task": "ACOS fixed category/sentiment presence"}, qs, gold))
    return out

def legal_rules(n):
    """长文档规则推理：ShARC（条款 + 场景 → 是否满足）和 ConditionalQA（政策文档问答）。"""
    out = []
    ds = datasets.load_dataset("UCLNLP/sharc", revision="refs/convert/parquet", split="train", streaming=True)  # 原仓库是加载脚本
    for ex in itertools.islice(ds.shuffle(seed=SEED, buffer_size=10000), n * 3):
        ans = ex["answer"].strip().lower()
        if ans not in ("yes", "no"):
            continue
        h = ast.literal_eval(ex["history"]) if isinstance(ex["history"], str) else ex["history"]
        hist = "\n".join(f"Q: {x['follow_up_question']}\nA: {x['follow_up_answer']}" for x in h)
        state = {"rule": ex["snippet"], "scenario": ex["scenario"] or "(none)", "question": ex["question"], "follow_ups": hist or "(none)"}
        out.append(sample("sharc", state, yn("Based only on the rule text and the user's information, is the answer to the question yes?"), ans))
        if len(out) >= n:
            break
    return out

# ---------- 检索与分类 ----------
def esci(n):
    """Amazon ESCI train（评测是 split==test）。"""
    lab = {"Exact": "E", "Substitute": "S", "Complement": "C", "Irrelevant": "I"}
    crit = {"E": "Exact: the product satisfies the search query.", "S": "Substitute: a product that could substitute for the requested product.",
            "C": "Complement: a product that complements the requested product.", "I": "Irrelevant: the product does not address the requested product need."}
    out, per = [], {}
    for ex in stream("tasksource/esci", n=n * 4, buffer=50000):
        if ex["product_locale"] != "us" or ex["esci_label"] not in lab:
            continue
        g = lab[ex["esci_label"]]
        if per.get(g, 0) >= n * (0.4 if g == "E" else 0.25):  # E 占原始数据六成以上，压一压
            continue
        per[g] = per.get(g, 0) + 1
        prod = {k: ex[f"product_{k}"] for k in ("title", "description", "bullet_point", "brand", "color") if ex.get(f"product_{k}")}
        out.append(sample("esci_train", {"search_query": ex["query"].strip(), "product": prod},
                          choice("Classify the relevance of this product to the search query using the ESCI categories.", crit), g))
        if len(out) >= n:
            break
    return out

def rank_rows(src, query, cands, gold_set, k_max=8):
    """ToolRet / BRIGHT 的格式：state 只有 query，每个候选一个 yes/no 问题。"""
    qs, gold = {}, {}
    for i, c in enumerate(cands[:k_max]):
        qs[f"doc_{i}"] = {"type": "choice", "criteria": YES_NO_REL, "instructions": {"candidate": c, "task": CAND_TASK}}
        gold[f"doc_{i}"] = "yes" if i in gold_set else "no"
    return multi(src, {"query": query, "task": RANK_TASK}, qs, gold)

def qnli(n):
    """QNLI train → 段落是否能回答问题（ToolRet / BRIGHT 的相关性判断格式；这两个都没有 train split）。"""
    ex = stream("nyu-mll/glue", "qnli", n=n * 4, buffer=20000)
    pool = [e["sentence"] for e in ex]
    out = []
    for e in ex[:n]:
        cands, pos = [e["sentence"]], {0} if e["label"] == 0 else set()
        cands += R.sample(pool, R.randint(2, 5))
        order = list(range(len(cands))); R.shuffle(order)
        cands = [cands[i] for i in order]; pos = {order.index(p) for p in pos}
        out.append(rank_rows("qnli_rank", e["question"], cands, pos))
    return out

def sgd(n):
    """SGD（dstc8）train 对话：当前服务的 active intent（评测是 test/，服务和 train 不完全一样）。"""
    schema = {s["service_name"]: s for s in json.load(open(fetch(
        f"{GH}/google-research-datasets/dstc8-schema-guided-dialogue/master/train/schema.json", "sgd_schema.json")))}
    files = R.sample(range(1, 128), 12)
    dialogues = []
    for i in files:
        dialogues += json.load(open(fetch(f"{GH}/google-research-datasets/dstc8-schema-guided-dialogue/master/train/dialogues_{i:03d}.json",
                                          f"sgd_{i:03d}.json")))
    out = []
    instr = ("Using the dialogue history and service schema in state, choose the active intent for this service. "
             "Choose NONE when no service intent is active. Do not use future turns or hidden labels.")
    while len(out) < n:
        d = R.choice(dialogues)
        user_turns = [i for i, t in enumerate(d["turns"]) if t["speaker"] == "USER"]
        ti = R.choice(user_turns)
        fr = R.choice(d["turns"][ti]["frames"])
        sch = schema[fr["service"]]
        intents = [it["name"] for it in sch["intents"]]
        gold = fr["state"]["active_intent"] if fr["state"]["active_intent"] in intents else "NONE"
        hist = [{"speaker": t["speaker"], "utterance": t["utterance"]} for t in d["turns"][:ti + 1]]
        crit = {k: k for k in intents + ["NONE"]}
        out.append(sample("sgd_train", {"history": hist, "service": fr["service"], "schema": sch, "task": "SGD current-service intent"},
                          choice(instr, crit), gold))
    return out

# ---------- 工具调用 ----------
def toolace_items(n):
    """ToolACE：(可用工具列表, 对话前缀, 被调用的工具名列表 或 [] 表示不调用)。"""
    items = []
    for ex in stream("Team-ACE/ToolACE", n=n * 2, buffer=5000):
        m = re.search(r"\[\s*\{.*\}\s*\]", ex["system"], re.S)
        if not m:
            continue
        try:
            tools = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        conv = ex["conversations"]
        for j, turn in enumerate(conv):
            if turn["from"] != "assistant" or j == 0:
                continue
            prev = [{"role": {"user": "user", "assistant": "assistant", "tool": "tool"}.get(t["from"], "user"), "content": t["value"]}
                    for t in conv[:j]]
            names = re.findall(r"(?:^\[|,\s*)([A-Za-z_][\w .\-]*?)\(", turn["value"].strip()) if turn["value"].strip().startswith("[") else []
            names = [x.strip() for x in names if x.strip() in {t["name"] for t in tools}]
            items.append((tools, prev, names))
            break
        if len(items) >= n:
            break
    return items

def glaive_items(n):
    items = []
    for ex in stream("glaiveai/glaive-function-calling-v2", n=n * 2, buffer=10000):
        try:
            tools = [json.loads(b) for b in re.findall(r"\{.*?\}\s*(?=\n\s*\{|\s*$)", ex["system"].split("-", 1)[-1], re.S)]
        except json.JSONDecodeError:
            continue
        tools = [t for t in tools if isinstance(t, dict) and "name" in t]
        m = re.search(r"USER:(.*?)ASSISTANT:(.*?)(?:<\|endoftext\|>|USER:|FUNCTION RESPONSE:|$)", ex["chat"], re.S)
        if not tools or not m:
            continue
        call = re.search(r'<functioncall>\s*\{"name":\s*"([^"]+)"', m.group(2))
        names = [call.group(1)] if call and call.group(1) in {t["name"] for t in tools} else []
        items.append((tools, [{"role": "user", "content": m.group(1).strip()}], names))
        if len(items) >= n:
            break
    return items

BFCL_INSTR = ("Given the complete user conversation and the published function schemas in state, mark each candidate tool yes if it "
              "should be invoked to answer the request, or no otherwise. This evaluates tool-name selection only; do not produce "
              "arguments or call ordering.\n\n\n\nCandidate tool: ")

def tool_rows(n):
    """BFCL（每个工具 yes/no）、API-Bank（从目录里选一个 API）、ToolRet（工具相关性）三种格式。"""
    items = toolace_items(n // 2) + glaive_items(n // 2)
    pool = [t for tools, _, _ in items for t in tools]
    out = []
    for tools, prev, names in items:
        users = [m for m in prev if m["role"] == "user"]
        r = R.random()
        if r < 0.45:  # BFCL
            ts = tools + (R.sample(pool, R.randint(0, 3)) if len(tools) < 3 else [])
            ts = list({t["name"]: t for t in ts}.values()); R.shuffle(ts)
            qs = {f"tool_{t['name']}": choice(BFCL_INSTR + t["name"], {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}) for t in ts}
            gold = {f"tool_{t['name']}": "yes" if t["name"] in names else "no" for t in ts}
            out.append(multi("tools_bfcl", {"conversation": [prev], "functions": ts, "task": "BFCL tool-name selection"}, qs, gold))
        elif r < 0.75 and names:  # API-Bank：一个大目录里选当前该调用的 API
            ts = list({t["name"]: t for t in tools + R.sample(pool, R.randint(10, 40))}.values()); R.shuffle(ts)
            keys = [f"option_{i}" for i in range(len(ts))]
            dia = [{"role": "User" if m["role"] == "user" else "AI", "text": m["content"]} for m in prev if m["role"] in ("user", "assistant")]
            out.append(sample("tools_apibank", {"dialogue": dia, "available_tools": ts, "task": "API-Bank current API tool selection"},
                              choice("Select the single API tool to invoke for the dialogue prefix in state. Use the published catalog in "
                                     "state. Do not infer arguments or use the current/future API result.",
                                     dict(zip(keys, [t["name"] for t in ts]))),
                              keys[[t["name"] for t in ts].index(names[0])]))
        elif names and users:  # ToolRet：查询 + 候选工具
            pos = [t for t in tools if t["name"] in names][:1]
            neg = [t for t in R.sample(pool, 12) if t["name"] not in names][:R.randint(3, 7)]
            cands = pos + neg; R.shuffle(cands)
            out.append(rank_rows("tools_rank", users[-1]["content"], [json.dumps(c, ensure_ascii=False) for c in cands],
                                 {cands.index(pos[0])} if pos else set()))
    return out

REFUSE = "I'm sorry, but I can't help with that using the tools available to me."

def when2call(n):
    """When2Call train_pref（评测是 test 的 MCQ）：该直接调用工具、追问、还是拒绝。"""
    out = []
    for ex in stream("nvidia/When2Call", "train_pref", n=n, buffer=10000):
        tools = [json.loads(t) if isinstance(t, str) else t for t in ex["tools"]]
        user = [m["content"] for m in ex["messages"] if m["role"] == "user"]
        if not user:
            continue
        fmt = lambda c: re.sub(r"^<TOOLCALL>\[(.*)\]</TOOLCALL>$", r"\1", c.strip(), flags=re.S)
        good, bad = fmt(ex["chosen_response"]["content"]), fmt(ex["rejected_response"]["content"])
        opts = [good, bad]
        if not re.search(r"sorry|unable|can't|cannot", good, re.I):
            opts.append(REFUSE)
        out.append(mc("when2call_train", {"tools": tools, "question": user[-1]},
                      "Which response should the assistant give to the user's question, given the available tools?", opts, 0))
    return out

# ---------- 人文与判断 ----------
def humicroedit(n):
    """Humicroedit subtask-2 train+validation（评测是 subtask-2 test）：两个改了一个词的标题哪个更好笑。"""
    out = []
    for split in ("train", "validation"):
        for ex in stream("tasksource/humicroedit", "subtask-2", split=split, n=n):
            if ex["label"] not in (1, 2):
                continue
            h = [re.sub(r"<[^>]+/>", e, o) for o, e in ((ex["original1"], ex["edit1"]), (ex["original2"], ex["edit2"]))]
            out.append(sample("humicroedit_train", "", choice("Which edited news headline is funnier?", {"headline_1": h[0], "headline_2": h[1]}),
                              f"headline_{ex['label']}"))
    return out[:n]

def newyorker(n):
    """New Yorker 漫画配文匹配，fold 0 的 train（评测是 matching fold 0 的 test）。"""
    ds = datasets.load_dataset("jmhessel/newyorker_caption_contest", "matching", split="train", streaming=True).remove_columns(["image"])
    out = []
    for ex in itertools.islice(ds.shuffle(seed=SEED, buffer_size=5000), n):
        ch = ex["caption_choices"]
        ch = ast.literal_eval(ch) if isinstance(ch, str) else ch
        ent = ex["entities"]
        ent = ast.literal_eval(ent) if isinstance(ent, str) else ent
        keys = "ABCDE"[:len(ch)]
        out.append(sample("newyorker_train", {"scene": ex["image_location"], "description": ex["image_description"],
                                              "uncanny_description": ex["image_uncanny_description"], "entities": ent},
                          choice("Which caption was written for this cartoon?", dict(zip(keys, ch))), ex["label"]))
    return out

# ---------- 第二阶段补充：PhishNChips、HoVer 风格 ----------
PHISH_Q = {
    "verdict": choice("This email arrived in the user's inbox and contains a link (`link_url`). Is it a phishing attempt, or a legitimate "
                      "message whose link is safe to click?",
                      {"phishing": "The email or its link is a phishing attempt or otherwise malicious. The user should not click the link.",
                       "legitimate": "A normal, legitimate message. The link is safe to click."}),
    "is_phishing": noul("This email is a phishing attempt and the user should not click the link in `link_url`.",
                        "The email or its link is a phishing attempt or otherwise malicious.",
                        "The email is a normal, legitimate message and the link is safe to click."),
    "verdict_alt_click": choice("Should the user click the link in this email?",
                                {"do_not_click": "No. The link may be phishing or malicious.", "click": "Yes. The email and its link look legitimate."}),
    "verdict_alt_minimal": choice("Classify this email.", {"phishing": None, "legitimate": None}),
}
FIRST = ["Sarah", "Daniel", "Priya", "Tom", "Mei", "Carlos", "Anna", "James", "Fatima", "Lukas", "Grace", "Omar"]
LAST = ["Jenkins", "Morrison", "Patel", "Nguyen", "Garcia", "Schmidt", "Okafor", "Kim", "Rossi", "Silva", "Brown", "Cohen"]
SUBJ = [("Meeting notes - {p}", "I've compiled the notes from our {p} meeting. You can access them here: {u}. Let me know if anything is missing."),
        ("Updated schedule for {p}", "The updated schedule for {p} is posted. Please review it when you have a moment: {u}"),
        ("Invoice #{n} for {p}", "Please find the invoice for {p} at the link below.\n\n{u}\n\nThanks for your business."),
        ("Shared document: {p} plan", "I've shared the {p} plan with you. Open it here: {u}"),
        ("Quick question about {p}", "Following up on {p}. The reference material is here: {u}. Happy to discuss on our next call.")]
LURE = ["Your account will be suspended within 24 hours unless you verify your details.", "Unusual sign-in activity was detected. Confirm your identity now.",
        "Your mailbox is almost full. Sign in to keep receiving email.", "Action required: your password expires today."]
PROJ = ["Project Alpha", "the Q3 review", "the vendor onboarding", "the offsite", "the client renewal", "the audit", "the product launch"]

def _host(u):
    u = re.sub(r"^\w+://", "", u)
    return u.split("/")[0].split("?")[0].lower()

def phish(n):
    """PhishNChips 风格：真实的钓鱼 / 正常 URL（ealvaradob/phishing-dataset）放进模板生成的工作邮件。
    正常邮件里也会有第三方链接、"查看文档" 之类的话；钓鱼邮件有一半写得很平常，模型要看链接本身。"""
    urls = json.load(open(fetch("https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/urls.json", "phish_urls.json")))
    good = [u["text"] for u in urls if u["label"] == 0 and "." in _host(u["text"])]
    bad = [u["text"] for u in urls if u["label"] == 1 and "." in _host(u["text"])]
    out = []
    for i in range(n):
        is_ph = i % 2 == 0
        u = R.choice(bad if is_ph else good)
        u = u if re.match(r"^\w+://", u) else R.choice(["http://", "https://"]) + u
        first, last = R.choice(FIRST), R.choice(LAST)
        host = _host(u)
        org = host.split(".")[-2] if host.count(".") >= 1 else host
        r = R.random()
        if r < 0.5:
            frm = f"{first.lower()}.{last.lower()}@{host.replace('www.', '')}"
        elif r < 0.75:
            frm = f"{first.lower()}.{last.lower()}@{R.choice(['northwind', 'evergreenpartners', 'acmecorp', 'bluepeak', 'meridian'])}.com"
        else:
            frm = f"{R.choice(['support', 'it.helpdesk', 'billing', first.lower()])}{R.randint(1, 99)}@{R.choice(['gmail.com', 'outlook.com', 'yahoo.com'])}"
        subj, body = R.choice(SUBJ)
        p = R.choice(PROJ)
        text = body.format(p=p, u=u)
        if is_ph and R.random() < 0.5:
            text = R.choice(LURE) + "\n\n" + text
        state = {"sender": f"{first} {last}", "from": frm, "subject": subj.format(p=p, n=R.randint(1000, 9999)),
                 "body": f"Hi {R.choice(['Team', 'Alex', 'Drew', 'all'])},\n\n{text}\n\nBest,\n{first} {last}",
                 "link_display_text": R.choice(["access them here", "View document", "Open", u]), "link_url": u}
        gold = {"verdict": "phishing" if is_ph else "legitimate", "is_phishing": is_ph,
                "verdict_alt_click": "do_not_click" if is_ph else "click", "verdict_alt_minimal": "phishing" if is_ph else "legitimate"}
        keys = R.sample(list(PHISH_Q), R.randint(1, 3))
        out.append(multi("phish_gen", state, {k: PHISH_Q[k] for k in keys}, {k: gold[k] for k in keys}))
    return out

def hover_like(n):
    """HoVer 风格：HotpotQA train 的问题 + 答案写成一句说法，配上支撑段落；答案换成别的实体就是不被支持。"""
    crit = {"SUPPORTED": "The evidence supports the claim.", "NOT_SUPPORTED": "The evidence does not support the claim."}
    out = []
    for ex in stream("hotpotqa/hotpot_qa", "distractor", n=n * 2, buffer=5000):
        titles, sents = ex["context"]["title"], ex["context"]["sentences"]
        sup = [t for t in dict.fromkeys(ex["supporting_facts"]["title"]) if t in titles]
        if ex["answer"].lower() in ("yes", "no") or len(sup) < 2:
            continue
        ev = [{"title": t, "text": "".join(sents[titles.index(t)])} for t in sup]
        ok = R.random() < 0.5
        ans = ex["answer"]
        if not ok:
            others = [t for t in titles if t not in sup and t.lower() != ans.lower()]
            if not others:
                continue
            ans = R.choice(others)
        claim = f"The answer to \"{ex['question'].rstrip('?')}?\" is {ans}."
        out.append(sample("hover_like", {"claim": claim, "evidence": ev}, choice("Is the claim supported by the evidence?", crit),
                          "SUPPORTED" if ok else "NOT_SUPPORTED"))
        if len(out) >= n:
            break
    return out

SOURCES.update({"phish": phish, "hover_like": hover_like})

SOURCES.update({
    "contractnli": contractnli, "vast": vast, "nli4ct": nli4ct, "ragtruth": ragtruth, "isarcasm": isarcasm, "acos": acos,
    "sharc": legal_rules, "esci": esci, "qnli": qnli, "sgd": sgd, "tool_rows": tool_rows, "when2call": when2call,
    "humicroedit": humicroedit, "newyorker": newyorker,
})


# ===== 代码生成的题：答案由程序算出来（对应 CRUXEval、GSM8K、BBH、CLadder、SATA；这些 benchmark 没有可用的 train split）=====
# 题型格式照抄评测请求，内容全部随机生成，不含任何评测题文本。
import datetime as _dt, math

GR = random.Random(SEED + 1)
NAMES = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude", "Hana", "Ivan", "Jun", "Kofi", "Lena", "Mateo", "Nia",
         "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Uma", "Viktor", "Wen", "Yara", "Zane"]

def opt_keys(opts, style=None):
    style = style or GR.choice(["paren", "option", "letter"])
    if style == "paren":
        return [f"({chr(65 + i)})" for i in range(len(opts))]
    if style == "option":
        return [f"option_{i}" for i in range(len(opts))]
    return [chr(65 + i) for i in range(len(opts))]

def bbh_mc(src, text, opts, gold_idx):
    """BBH 风格：题干末尾带 Options 块，key 是 (A)/(B)…，描述是选项原文。"""
    order = list(range(len(opts))); GR.shuffle(order)
    opts = [opts[i] for i in order]; g = order.index(gold_idx)
    keys = [f"({chr(65 + i)})" for i in range(len(opts))]
    text = text + "\nOptions:\n" + "\n".join(f"{k} {o}" for k, o in zip(keys, opts))
    return sample(src, text, choice("Which option is the correct answer?", dict(zip(keys, opts))), keys[g])

def bbh_bin(src, text, yes, labels=("Yes", "No")):
    return sample(src, text, choice("Which option is the correct answer?", {labels[0]: labels[0], labels[1]: labels[1]}),
                  labels[0] if yes else labels[1])

# ---------- BBH 风格 ----------
def g_boolean():
    def expr(d):
        if d == 0 or GR.random() < 0.3:
            v = GR.choice([True, False]); return str(v), v
        r = GR.random()
        if r < 0.25:
            s, v = expr(d - 1); return f"not {s}", not v
        a, va = expr(d - 1); b, vb = expr(d - 1)
        if r < 0.5:
            return f"( {a} )", va
        op = GR.choice(["and", "or"])
        return f"{a} {op} {b}", (va and vb) if op == "and" else (va or vb)
    s, _ = expr(3)
    v = eval(s)
    return bbh_bin("gen_bbh", f"{s} is", v, ("True", "False"))

def g_web_of_lies():
    ppl = GR.sample(NAMES, GR.randint(4, 6))
    truth = GR.choice([True, False])
    parts = [f"{ppl[0]} {'tells the truth' if truth else 'lies'}."]
    for p, q in zip(ppl[1:], ppl):
        says = GR.choice([True, False])
        parts.append(f"{p} says {q} {'tells the truth' if says else 'lies'}.")
        truth = (says == truth)
    return bbh_bin("gen_bbh", "Question: " + " ".join(parts) + f" Does {ppl[-1]} tell the truth?", truth)

def g_navigate():
    x = y = 0; steps = []
    for _ in range(GR.randint(3, 8)):
        d, k = GR.choice(["forward", "backward", "left", "right"]), GR.randint(1, 10)
        steps.append(f"Take {k} step{'s' if k > 1 else ''} {d}.")
        x += {"left": -k, "right": k}.get(d, 0); y += {"forward": k, "backward": -k}.get(d, 0)
    if GR.random() < 0.4:  # 让一部分题真的回到原点
        if x: steps.append(f"Take {abs(x)} step{'s' if abs(x) > 1 else ''} {'left' if x > 0 else 'right'}.")
        if y: steps.append(f"Take {abs(y)} step{'s' if abs(y) > 1 else ''} {'backward' if y > 0 else 'forward'}.")
        x = y = 0
    text = ("If you follow these instructions, do you return to the starting point? Always face forward. " + " ".join(steps)
            + "\nOptions:\n- Yes\n- No")
    return bbh_bin("gen_bbh", text, x == 0 and y == 0)

ITEMS = {"books": ["Ulysses", "Frankenstein", "Lolita", "Moby Dick", "Emma", "Dracula", "Hamlet"],
         "balls": ["red ball", "blue ball", "green ball", "yellow ball", "pink ball", "black ball", "white ball"],
         "positions": ["striker", "goalkeeper", "left winger", "right winger", "benchwarmer", "center midfielder", "cheerleader"],
         "partners": ["Patrick", "Sam", "Jamie", "Lola", "Melissa", "Ophelia", "Karl"]}

def g_tracking():
    k = GR.choice([3, 5, 7]); ppl = NAMES[:k]
    kind = GR.choice(list(ITEMS)); things = GR.sample(ITEMS[kind], k)
    hold = dict(zip(ppl, things))
    swaps = []
    for _ in range(k):
        a, b = GR.sample(ppl, 2); hold[a], hold[b] = hold[b], hold[a]; swaps.append(f"{a} and {b} swap")
    who = GR.choice(ppl)
    text = (f"{', '.join(ppl[:-1])}, and {ppl[-1]} each start with one item: "
            + ", ".join(f"{p} has the {t}" for p, t in zip(ppl, things)) + ". Then " + ". Then ".join(swaps)
            + f". At the end, {who} has the")
    return bbh_mc("gen_bbh", text, things, things.index(hold[who]))

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

def g_logical_deduction():
    k = GR.choice([3, 5, 7]); objs = GR.sample(NAMES, k); order = objs[:]; GR.shuffle(order)  # order[0] = 最左/第一
    facts = set()
    while True:  # 加约束直到排列唯一
        r = GR.random()
        if r < 0.4:
            i = GR.randrange(k); facts.add(f"{order[i]} finished {ORD[i]}.")
        else:
            i, j = sorted(GR.sample(range(k), 2)); facts.add(f"{order[i]} finished above {order[j]}.")
        sols = 0
        for p in itertools.permutations(objs):
            ok = all((f"{p[int(ORD.index(f.split()[-1][:-1]))]}" == f.split()[0]) if " finished " in f and "above" not in f
                     else p.index(f.split()[0]) < p.index(f.split()[-1][:-1]) for f in facts)
            sols += ok
            if sols > 1:
                break
        if sols == 1:
            break
    pos = GR.randrange(k)
    text = (f"The following paragraphs each describe a set of {k} objects arranged in a fixed order. In a race, there were {k} "
            f"runners: {', '.join(objs)}. " + " ".join(sorted(facts, key=lambda _: GR.random())))
    opts = [f"{o} finished {ORD[pos]}" for o in objs]
    return bbh_mc("gen_bbh", text, opts, objs.index(order[pos]))

def g_date():
    d = _dt.date(GR.randint(1900, 2030), GR.randint(1, 12), GR.randint(1, 28))
    shift, phrase = GR.choice([(1, "tomorrow"), (-1, "yesterday"), (7, "one week from today"), (-7, "one week ago"),
                               (10, "10 days from today"), (-30, "30 days ago"), (365, "one year from today")])
    ans = d + _dt.timedelta(days=shift)
    fmt = lambda x: x.strftime("%m/%d/%Y")
    wrong = {fmt(ans + _dt.timedelta(days=GR.choice([-3, -2, -1, 1, 2, 3, 30, -30]))) for _ in range(8)} - {fmt(ans)}
    opts = [fmt(ans)] + list(wrong)[:GR.randint(3, 5)]
    return bbh_mc("gen_bbh", f"Today is {fmt(d)}. What is the date {phrase} in MM/DD/YYYY?", opts, 0)

COLORS = ["red", "blue", "green", "grey", "purple", "mauve", "teal", "orange", "black", "gold"]
THINGS = ["notebook", "pen", "cat toy", "mug", "keychain", "sheet of paper", "stress ball", "fidget spinner"]
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

def g_colored():
    items = [(GR.randint(1, 3), GR.choice(COLORS), GR.choice(THINGS)) for _ in range(GR.randint(3, 6))]
    c = GR.choice([it[1] for it in items]); rm = GR.choice([it[2] for it in items])
    left = sum(n for n, col, t in items if col == c and t != rm)
    desc = ", ".join(f"{NUMW[n]} {col} {t}{'s' if n > 1 else ''}" for n, col, t in items)
    text = f"On the desk, there are {desc}. If I remove all the {rm}s from the desk, how many {c} things remain on it?"
    return bbh_mc("gen_bbh", text, NUMW[:13], left) if left < 13 else g_colored()

ADJ = [("opinion", ["lovely", "ugly", "nice", "terrible", "wonderful"]), ("size", ["big", "small", "tiny", "enormous", "midsize"]),
       ("age", ["old", "new", "ancient", "brand-new"]), ("shape", ["square", "circular", "triangular", "rectangular"]),
       ("color", ["red", "grey", "blue", "green", "white"]), ("origin", ["Brazilian", "Japanese", "German", "Mexican", "Indian"]),
       ("material", ["wool", "plastic", "glass", "leather", "steel"]), ("purpose", ["hiking", "drinking", "smoking", "exercise"])]

def g_hyperbaton():
    cats = sorted(GR.sample(range(len(ADJ)), GR.randint(3, 4)))
    words = [GR.choice(ADJ[c][1]) for c in cats]
    noun = GR.choice(["sweater", "knife", "ship", "shoe", "car", "box"])
    bad = words[:]
    while bad == words:
        GR.shuffle(bad)
    return bbh_mc("gen_bbh", "Which sentence has the correct adjective order:", [" ".join(words + [noun]), " ".join(bad + [noun])], 0)

def g_shape():
    k = GR.choice([3, 4, 5, 6, 7, 8])
    name = {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}[k]
    cx, cy, r = GR.uniform(30, 70), GR.uniform(30, 70), GR.uniform(10, 25)
    pts = []
    for i in range(k):
        a = 2 * math.pi * i / k + GR.uniform(-0.2, 0.2)
        rr = r if k != 4 else r
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    if k == 4:
        w, h = GR.uniform(10, 30), GR.uniform(10, 30); pts = [(cx, cy), (cx + w, cy), (cx + w, cy + h), (cx, cy + h)]
    d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} " + " ".join(f"L {x:.2f},{y:.2f}" for x, y in pts[1:]) + f" L {pts[0][0]:.2f},{pts[0][1]:.2f}"
    opts = ["circle", "heptagon", "hexagon", "kite", "line", "octagon", "pentagon", "rectangle", "sector", "triangle"]
    text = f'This SVG path element <path d="{d}"/> draws a'
    keys = [f"({chr(65 + i)})" for i in range(len(opts))]
    text += "\nOptions:\n" + "\n".join(f"{kk} {o}" for kk, o in zip(keys, opts))
    return sample("gen_bbh", text, choice("Which option is the correct answer?", dict(zip(keys, opts))), keys[opts.index(name)])

def g_temporal():
    who = GR.choice(NAMES); place = GR.choice(["the bakery", "the gym", "the library", "the museum", "the park"])
    hrs = sorted(GR.sample(range(6, 21), 5))
    free = GR.randrange(4)
    lines, slots = [f"{who} woke up at {hrs[0]}:00."], []
    for i in range(4):
        slot = f"{hrs[i]}:00 to {hrs[i + 1]}:00"; slots.append(slot)
        if i != free:
            lines.append(f"{GR.choice(NAMES)} saw {who} {GR.choice(['reading', 'shopping', 'jogging', 'eating lunch', 'working'])} from {slot}.")
    lines.append(f"{place.capitalize()} was closed after {hrs[4]}:00.")
    text = f"Today, {who} went to {place}. Between what times could they have gone?\nWe know that:\n" + "\n".join(lines)
    return bbh_mc("gen_bbh", text, slots, free)

def bbh_like(n):
    fs = [g_boolean, g_web_of_lies, g_navigate, g_tracking, g_logical_deduction, g_date, g_colored, g_hyperbaton, g_shape, g_temporal]
    return [GR.choice(fs)() for _ in range(n)]

# ---------- CRUXEval 风格：给代码和输入，选正确输出 ----------
CRUX = [
    "def f(s):\n    return s[::{a}]",
    "def f(nums):\n    return [x * {a} for x in nums if x % {b} == 0]",
    "def f(s):\n    return s.replace('{c1}', '{c2}').upper()",
    "def f(nums):\n    out = []\n    for n in nums:\n        out.append((nums.count(n), n))\n    out.sort(reverse=True)\n    return out",
    "def f(text):\n    return ''.join(ch for ch in text if ch not in '{c1}{c2}')",
    "def f(nums):\n    nums = nums[:]\n    nums.insert({a}, {b})\n    return nums",
    "def f(d):\n    return sorted(d.items(), key=lambda kv: kv[1])[:{a}]",
    "def f(s):\n    return s.count('{c1}') + len(s.split('{c2}'))",
    "def f(nums):\n    total = 0\n    for i, n in enumerate(nums):\n        if i % {a} == 0:\n            total += n\n    return total",
    "def f(s):\n    words = s.split()\n    return ' '.join(w[::-1] if len(w) > {a} else w for w in words)",
    "def f(nums):\n    return sorted(set(nums), reverse={bool})[{a}:]",
    "def f(s):\n    return s.center(len(s) + {a}, '{c1}')",
    "def f(nums):\n    while len(nums) > {a}:\n        nums = nums[1:] + nums[:1]\n        nums.pop()\n    return nums",
    "def f(s):\n    d = {{}}\n    for ch in s:\n        d[ch] = d.get(ch, 0) + 1\n    return max(d, key=d.get)",
    "def f(a, b):\n    return a[{a}:] + b[:{b}]",
    "def f(nums):\n    return [n for n in nums[::-1] if n > {b}]",
]

def rand_input(code):
    if "(a, b)" in code:
        return (GR.choice(["abcdef", "hello", "xyzzy", "python"]), GR.choice(["12345", "world", "qwerty"]))
    if "(d)" in code:
        return ({k: GR.randint(0, 9) for k in GR.sample("abcdefg", GR.randint(2, 5))},)
    if "nums" in code:
        return ([GR.randint(-3, 12) for _ in range(GR.randint(3, 8))],)
    return (GR.choice(["banana split", "hello world", "abracadabra", "mississippi", "the quick brown fox", "aabbccdd"]),)

def crux_like(n):
    out = []
    while len(out) < n:
        t = GR.choice(CRUX)
        code = t.format(a=GR.randint(1, 3), b=GR.randint(1, 4), c1=GR.choice("abeilos"), c2=GR.choice("xyz-"), bool=GR.choice([True, False]))
        args = rand_input(code)
        ns = {}
        try:
            exec(code, ns)
            val = ns["f"](*[a.copy() if hasattr(a, "copy") else a for a in args])
            res = repr(val)
        except Exception:
            continue
        wrong = set()
        for _ in range(12):  # 干扰项：同一模板换参数后的输出、或对正确输出做小改动
            alt = t.format(a=GR.randint(1, 3), b=GR.randint(1, 4), c1=GR.choice("abeilos"), c2=GR.choice("xyz-"), bool=GR.choice([True, False]))
            try:
                ns2 = {}; exec(alt, ns2); w = repr(ns2["f"](*[a.copy() if hasattr(a, "copy") else a for a in args]))
            except Exception:
                continue
            if w != res:
                wrong.add(w)
        if isinstance(val, (str, list)) and len(val) > 1:  # 对正确输出本身做小改动（不是对它的 repr 字符串）
            wrong |= {repr(v) for v in (val[::-1], val[1:], val[:-1], val + val[-1:])}
        elif isinstance(val, int) and not isinstance(val, bool):
            wrong |= {repr(val + d) for d in (-2, -1, 1, 2)}
        wrong = list(wrong - {res})
        if len(wrong) < 1:
            continue
        opts = [res] + GR.sample(wrong, min(len(wrong), GR.randint(1, 3)))
        state = {"code": code, "input": ", ".join(repr(a) for a in args), "task": "CRUXEval output selection"}
        out.append(mc("gen_crux", state, "Choose the correct output of f called with the supplied input arguments. "
                                         "Candidates are Python literal values.", opts, 0))
    return out

# ---------- GSM8K 风格：数值选择题 ----------
def num_opts(ans, k):
    fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
    cands = [ans + d for d in (-10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10)] + [ans * 2, ans / 2, ans * 10, ans + ans / 2]
    GR.shuffle(cands)
    opts, seen = [fmt(ans)], {fmt(ans)}
    for c in cands:
        if c >= 0 and fmt(c) not in seen and len(opts) < k:
            opts.append(fmt(c)); seen.add(fmt(c))
    return opts

def gsm_row(src, question, ans):
    k = GR.choice([4, 10])
    opts = num_opts(ans, k)
    if len(opts) < k:
        return None
    return mc(src, {"question": question, "task": f"GSM8K deterministic {k}-choice numeric selection"},
              "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.",
              opts, 0)

WORD_PROBLEMS = [
    (lambda a, b, c: (f"{{n}} buys {a} boxes of pencils with {b} pencils in each box and gives away {c} pencils. How many pencils does {{n}} have left?", a * b - c)),
    (lambda a, b, c: (f"A shop sells apples for ${a} each and pears for ${b} each. {{n}} buys {c} apples and {c + 2} pears. How much does {{n}} spend in dollars?", a * c + b * (c + 2))),
    (lambda a, b, c: (f"{{n}} runs {a} km every weekday and {b} km on each weekend day. How many km does {{n}} run in {c} weeks?", (5 * a + 2 * b) * c)),
    (lambda a, b, c: (f"A tank holds {a * 10} liters. It is filled at {b} liters per minute and leaks {1 if b > 1 else 0} liter per minute. How many minutes until it is full, rounded down?", (a * 10) // max(1, b - (1 if b > 1 else 0)))),
    (lambda a, b, c: (f"{{n}} earns ${a * 5} per hour and works {b} hours a day for {c} days, then spends half of it. How many dollars are left?", a * 5 * b * c / 2)),
    (lambda a, b, c: (f"There are {a * 4} students. A quarter of them play chess and {b} of the rest play soccer. How many students play neither?", a * 4 - a - b) if a * 3 >= b else (f"There are {a + b} cats and {c} dogs. How many legs do they have in total?", 4 * (a + b + c))),
]

def gsm_like(n):
    out = []
    while len(out) < n:
        a, b, c = GR.randint(2, 12), GR.randint(2, 12), GR.randint(1, 9)
        q, ans = GR.choice(WORD_PROBLEMS)(a, b, c)
        r = gsm_row("gen_gsm", q.replace("{n}", GR.choice(NAMES)), float(ans))
        if r:
            out.append(r)
    return out

def gsm8k_fmt(n):
    """GSM8K train 按评测题的格式（state 里放题目、option_i 选项）。"""
    out = []
    for ex in stream("openai/gsm8k", "main", n=n):
        try:
            a = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        except ValueError:
            continue
        r = gsm_row("gsm8k_fmt", ex["question"], a)
        if r:
            out.append(r)
    return out

# ---------- CLadder 风格：因果概率 yes/no ----------
VARS = [("husband", "alarm set by husband", "ringing alarm"), ("smoking", "smoking", "lung cancer"), ("vaccination", "vaccination", "recovery"),
        ("tutoring", "tutoring", "passing the exam"), ("rain", "rain", "traffic jam"), ("xevo", "xevo", "gyzp"), ("zuph", "zuph", "rixq")]

def cladder_like(n):
    out = []
    for _ in range(n):
        xn, xdesc, y = GR.choice(VARS)
        p0, p1 = GR.randint(5, 95), GR.randint(5, 95)
        if p0 == p1:
            continue
        if GR.random() < 0.5:  # 无混杂：直接比较
            world = f"{xn.capitalize()} has a direct effect on {y}."
            info = (f"For those with no {xdesc}, the probability of {y} is {p0}%. For those with {xdesc}, the probability of {y} is {p1}%.")
            ate = p1 - p0
        else:  # 有混杂 Z：后门调整
            pz = GR.randint(10, 90); q = [GR.randint(5, 95) for _ in range(4)]  # P(Y|X=x,Z=z): x0z0, x0z1, x1z0, x1z1
            world = f"Confounder has a direct effect on {xn} and {y}. {xn.capitalize()} has a direct effect on {y}."
            info = (f"The overall probability of confounder is {pz}%. For those with no confounder and no {xdesc}, the probability of {y} is {q[0]}%. "
                    f"For those with confounder and no {xdesc}, {q[1]}%. For those with no confounder and {xdesc}, {q[2]}%. "
                    f"For those with confounder and {xdesc}, {q[3]}%.")
            ate = (q[2] * (100 - pz) + q[3] * pz) - (q[0] * (100 - pz) + q[1] * pz)
            if ate == 0:
                continue
        inc = GR.random() < 0.5
        ask = f"Will {xdesc} {'increase' if inc else 'decrease'} the chance of {y}?"
        text = ("Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors "
                f"or causal relationships: {world}\n\n{info}\n\n{ask}")
        yes = (ate > 0) == inc
        out.append(sample("gen_cladder", {}, choice(text, {"A": "yes", "B": "no"}), "A" if yes else "B"))
    return out

# ---------- SATA 风格：每个候选单独判断对不对 ----------
def sata_like(n):
    """SciQ train：带支撑段落的选择题拆成每个选项一个 yes/no；另加 "哪些数是质数" 这类多答案题。"""
    out = []
    for ex in stream("allenai/sciq", n=n):
        if not ex["support"]:
            continue
        cands = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]; GR.shuffle(cands)
        qs, gold = {}, {}
        for i, c in enumerate(cands):
            qs[f"option_{i}"] = choice(f"Question: {ex['question']}\nDoes this candidate correctly answer the question?\nCandidate: {c}",
                                       {"no": "No", "yes": "Yes"})
            gold[f"option_{i}"] = "yes" if c == ex["correct_answer"] else "no"
        out.append(multi("sata_sciq", {"paragraph": ex["support"]}, qs, gold))
    tests = [("prime", lambda v: v > 1 and all(v % d for d in range(2, int(v ** 0.5) + 1))), ("even", lambda v: v % 2 == 0),
             ("a perfect square", lambda v: int(v ** 0.5) ** 2 == v), ("divisible by 3", lambda v: v % 3 == 0)]
    for _ in range(n // 3):
        name, fn = GR.choice(tests); nums = GR.sample(range(2, 100), GR.randint(5, 9))
        q = f"Which of the following numbers are {name}?"
        qs = {f"option_{i}": choice(f"Question: {q}\nDoes this candidate correctly answer the question?\nCandidate: {v}", {"no": "No", "yes": "Yes"})
              for i, v in enumerate(nums)}
        out.append(multi("sata_numbers", {"paragraph": f"Consider the numbers {', '.join(map(str, nums))}."}, qs,
                         {f"option_{i}": "yes" if fn(v) else "no" for i, v in enumerate(nums)}))
    return out

SOURCES.update({"gen_bbh": bbh_like, "gen_crux": crux_like, "gen_gsm": gsm_like, "gsm8k_fmt": gsm8k_fmt,
                "gen_cladder": cladder_like, "sata": sata_like})


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

# ===== 第二版：补上 Decision Index 各 benchmark 的题型（parts/data_v2.py、parts/gen.py）=====
# 不在榜单里的分类数据（ag_news、yahoo、emotion、sst5 等）减量；GSM8K 换成评测题的格式。
SIZES_V2 = {
    **SIZES,
    "ag_news": 400, "dbpedia": 400, "yahoo": 500, "emotion": 500, "sst5": 500, "massive": 1500, "snli": 800, "mnli": 2000,
    "tools": 1000, "hotpot": 2000, "gsm8k": 0, "aqua": 800, "qasc": 600, "obqa": 600,
    # 语言理解：ContractNLI、VAST、NLI4CT、RAGTruth、iSarcasmEval、ACOS
    "contractnli": 450, "vast": 2500, "nli4ct": 1500, "ragtruth": 2500, "isarcasm": 2500, "acos": 1200, "sharc": 1200,
    # 检索与路由：ESCI、ToolRet/BRIGHT（相关性）、SGD
    "esci": 3000, "qnli": 1800, "sgd": 2500,
    # 工具：BFCL、API-Bank、ToolRet、When2Call
    "tool_rows": 4500, "when2call": 3000,
    # 人文：Humicroedit、New Yorker
    "humicroedit": 2000, "newyorker": 2000,
    # 代码生成的题：BBH、CRUXEval、GSM8K、CLadder、SATA
    "gen_bbh": 4000, "gen_crux": 2500, "gen_gsm": 1200, "gsm8k_fmt": 2500, "gen_cladder": 2000, "sata": 1200,
}
SIZES_V2 = {k: v for k, v in SIZES_V2.items() if v > 0}



if DATA_REPO:
    from huggingface_hub import hf_hub_download
    for name in ("train", "dev"):
        p = hf_hub_download(DATA_REPO, f"{DATA_NAME}/{name}.jsonl.gz", repo_type="dataset")
        open(f"{WORK}/{name}.jsonl.gz", "wb").write(open(p, "rb").read())
    print("训练数据来自", DATA_REPO, DATA_NAME, flush=True)
else:
    rows = build(SIZES_V2 if DATA_VERSION >= 2 else SIZES, f"{WORK}/all.jsonl.gz")
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
    """一条样本 -> [(token ids, 选项数, 正确选项下标), ...]，每个问题一项（和推理时一样，每个问题单独一个提示词）。
    expected 为 None 的问题跳过。"""
    _, out, _ = render(tok, row["state"], row["questions"], LABELS)
    items = []
    for qid, (text, keys) in out.items():
        gold = row["expected"].get(qid)
        if gold is None:
            continue
        gold = {True: "true", False: "false"}.get(gold, gold) if isinstance(gold, bool) else gold
        items.append((tok.encode(text, add_special_tokens=False), len(keys), keys.index(gold)))
    return items

def load_encoded(path):
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    enc, dropped = [], 0
    for r in rows:
        for ids, k, g in encode(r):
            if len(ids) > MAX_LEN:
                dropped += 1
                continue
            enc.append((ids, k, g, r["src"]))
    print(f"{path}: {len(rows)} 条样本 → {len(enc)} 个问题, 超过 {MAX_LEN} tokens 丢弃 {dropped} 个", flush=True)
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
