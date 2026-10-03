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
