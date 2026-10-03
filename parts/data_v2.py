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

SOURCES.update({
    "contractnli": contractnli, "vast": vast, "nli4ct": nli4ct, "ragtruth": ragtruth, "isarcasm": isarcasm, "acos": acos,
    "sharc": legal_rules, "esci": esci, "qnli": qnli, "sgd": sgd, "tool_rows": tool_rows, "when2call": when2call,
    "humicroedit": humicroedit, "newyorker": newyorker,
})
