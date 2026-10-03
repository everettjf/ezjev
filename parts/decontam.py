# ===== 和评测集去重：训练样本里只要有一段文字和评测题重合，就整条删掉 =====
# 两种检查：
#   1. 整句/整行完全相同（规范化后 ≥ 40 个字符；更短的多是 "The ticker symbol of the stock" 这类通用短语）；
#   2. 13 个词的 n-gram 重合（为了省内存只保留 1/8 的 n-gram）：命中 ≥ 3 个，且占这段文字 n-gram 的 20% 以上
#      （长合同里零星几句标准条款相同不算；整段相同的段落会被删）。
# 在评测集里出现于 50 行以上的文字（题目说明模板、固定的假设句、类别名）不算重合。
import collections, hashlib
from huggingface_hub import hf_hub_download

def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)

_WORD = re.compile(r"\w+")

def _words(s):
    return _WORD.findall(s.lower())

def _segments(s):
    yield s
    if "\n" in s:
        yield from s.split("\n")

def _key(ws):
    return int.from_bytes(hashlib.blake2b(" ".join(ws).encode(), digest_size=8).digest(), "little")

def _grams(ws, n=13):
    for i in range(len(ws) - n + 1):
        h = hash(tuple(ws[i:i + n]))
        if h % 8 == 0:
            yield h

def build_index(suite_repo, max_df=50):
    ex_df, ng_df = collections.Counter(), collections.Counter()
    nrows = 0
    for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        path = hf_hub_download(suite_repo, name, repo_type="dataset")
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            ex, ng = set(), set()
            for s in _strings([r.get("state"), r.get("questions")]):
                for seg in _segments(s):
                    ws = _words(seg)
                    if sum(map(len, ws)) >= 40:
                        ex.add(_key(ws))
                ng.update(_grams(_words(s)))
            ex_df.update(ex); ng_df.update(ng)
            nrows += 1
    EX = {h for h, c in ex_df.items() if c <= max_df}
    NG = {h for h, c in ng_df.items() if c <= max_df}
    print(f"评测集索引: {nrows} 行, 整句 {len(EX)} 个, n-gram {len(NG)} 个", flush=True)
    return EX, NG

def contaminated(row, EX, NG):
    for s in _strings([row["state"], row["questions"]]):
        for seg in _segments(s):
            ws = _words(seg)
            if sum(map(len, ws)) >= 40 and _key(ws) in EX:
                return "exact"
        gs = list(_grams(_words(s)))
        hits = sum(1 for g in gs if g in NG)
        if hits >= 3 and hits >= 0.2 * len(gs):
            return "ngram"
    return None

def decontaminate(rows, suite_repo):
    EX, NG = build_index(suite_repo)
    keep, dropped = [], collections.Counter()
    for r in rows:
        why = contaminated(r, EX, NG)
        if why:
            dropped[(r["src"], why)] += 1
        else:
            keep.append(r)
    print(f"去重: 保留 {len(keep)} 条, 删除 {len(rows) - len(keep)} 条", flush=True)
    for (src, why), c in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  删除 {src:20s} {why:6s} {c}", flush=True)
    return keep, {f"{s}:{w}": c for (s, w), c in dropped.items()}
