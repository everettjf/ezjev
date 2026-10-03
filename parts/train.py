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
