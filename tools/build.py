"""由 parts/ 下的代码生成 notebooks/ 里的两个 Colab notebook 和 jobs/train_job.py。

改训练代码时只改 parts/，然后运行：python3 tools/build.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "parts"
OUT = ROOT / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells, gpu="A100"):
    def cell(kind, text):
        text = text.strip("\n")
        lines = [l + "\n" for l in text.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if kind == "md":
            return {"cell_type": "markdown", "metadata": {}, "source": lines}
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}
    return {
        "cells": [cell(k, t) for k, t in cells],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": gpu, "machine_shape": "hm", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def src(name):
    return (HERE / name).read_text()


loop = src("loop.py")
setup, training = loop.split("micro = batches(train_data, TOKENS_PER_BATCH)\n", 1)
training = "micro = batches(train_data, TOKENS_PER_BATCH)\n" + training

# --------------------------------------------------------------------------------------------- 训练 notebook
train_cells = [
("md", """
# ezjev 训练（Colab）· 第 1 步：训练并上传模型

目标：在 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 0.2.1 上进 70 个参赛模型的
**前 25 名**，也就是 **38.5 分以上**（第 25 名 lev 是 38.54，第 15 名 JPT-4B 是 43.04）。

**做法**（照着榜上 4B 级最强的 JPT-4B（43.04）的公开配方来）：

- 底座 `Qwen/Qwen3.5-4B`，加 LoRA 微调后合并成完整权重；
- 提示词和推理时完全一样：[llm2jev](https://github.com/tic-top/llm2jev) 的 chat 格式，在 `Answer:` 后面读每个选项字母的 logit；
- 损失 = 选项上的交叉熵 + Brier（训的是概率，不只是对错）；
- 数据是约 5 万道题，全部来自公开数据集的 **train split**，覆盖分类、NLI、常识、数学、检索、工具调用、人类偏好；
  **榜单评测集一条都不用**（规则禁止，抽检也会发现）。

**需要准备**

1. Colab Pro，运行时选 **A100**（L4 也能跑，但大约慢 3 倍，要把 `TOKENS_PER_BATCH` 调小）；
2. Hugging Face 账号和一个 **write** 权限的 token，在左侧 🔑 Secrets 里添加名为 `HF_TOKEN` 的密钥；
3. Google Drive 留出约 20 GB（存数据、LoRA 断点和合并后的权重）。

**耗时**：A100 上构造数据约 15 分钟，训练约 2–3 小时，合并加上传约 10 分钟。
中途断线也不要紧：重新按顺序运行全部单元格，训练会从 Drive 上的断点接着跑。

跑完后打开第 2 个 notebook `ezjev_eval_colab.ipynb`，跑完整评测并提交。
"""),
("code", """
# 挂载 Drive，检查 GPU
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
("code", """
# 安装依赖（Colab 自带 torch）。flash-linear-attention 给 Qwen3.5 的线性注意力层提供快速 kernel，没有它会慢很多。
!pip -q install -U "transformers>=5.0" "peft>=0.17" "datasets>=3.0" accelerate llm2jev flash-linear-attention
# causal-conv1d 是可选加速，要现场编译，装不上也能训练
!pip -q install causal-conv1d --no-build-isolation || echo "causal-conv1d 没装上，可以忽略"
"""),
("code", """
# ===== 配置：一般只需要改 HF_REPO =====
import os, torch
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import whoami
HF_USER = whoami()["name"]

HF_REPO    = f"{HF_USER}/ezjev-4b"          # 训练好的模型上传到这里
BASE_MODEL = "Qwen/Qwen3.5-4B"               # 想先快速走通流程，可以换成 "Qwen/Qwen3.5-0.8B"（分数会低很多）
WORK       = "/content/drive/MyDrive/ezjev"  # 数据、断点都放在 Drive 上，断线不丢
MERGED_DIR = "/content/ezjev-merged"

SEED       = 0
DATA_SCALE = 1.0      # 数据量倍率：1.0 约 5.5 万题；先试跑可以用 0.05
MAX_LEN    = 8192     # 超过这个长度的训练样本直接丢掉
LORA_R, LORA_ALPHA = 16, 32
LR         = 1e-4
BRIER_WEIGHT = 1.0
TOKENS_PER_BATCH  = 24000   # 每个 micro-batch 的 token 数上限（A100 40GB 用 24000；L4 24GB 用 8000；OOM 就减半）
EXAMPLES_PER_STEP = 32      # 每个优化 step 的样本数（梯度累积）
LOG_EVERY, SAVE_EVERY = 20, 100
DEVICE = "cuda"
os.makedirs(WORK, exist_ok=True)
print("HF 用户:", HF_USER, "→", HF_REPO)
"""),
("md", """
## 1. 构造训练数据

每道题的格式和评测请求完全一样：`state` + `questions`（`choice` / `noul` / `score`）+ 正确答案。
选项顺序会打乱，key 的写法也会随机（A/B/C、1/2/3、选项原文），这样模型不会依赖选项位置或某一种写法。
"""),
("code", src("data.py")),
("code", src("sizes.py") + """
if os.path.exists(f"{WORK}/train.jsonl.gz"):
    print("训练数据已存在，跳过构造（想重新构造就先删掉", WORK, "里的 *.jsonl.gz）")
else:
    rows = build(SIZES, f"{WORK}/all.jsonl.gz")
    n_dev = min(1500, len(rows) // 30)          # 留 3% 做 dev，用来看效果和拟合温度
    for name, part in (("dev", rows[:n_dev]), ("train", rows[n_dev:])):
        with gzip.open(f"{WORK}/{name}.jsonl.gz", "wt") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print("train", len(rows) - n_dev, "dev", n_dev)
"""),
("md", """
## 2. 训练

先在 dev 上测一下底座模型零样本的准确率（作为对照），再开始训练。训练中每 `SAVE_EVERY` 步往 Drive 存一次 LoRA 断点。
"""),
("code", src("train.py")),
("code", """
train_data = load_encoded(f"{WORK}/train.jsonl.gz")
dev_data = load_encoded(f"{WORK}/dev.jsonl.gz")
lens = sorted(len(x[0]) for x in train_data)
print("训练样本 token 长度: 中位数", lens[len(lens) // 2], " 最长", lens[-1])
"""),
("code", setup),
("code", """
# 底座模型（LoRA 刚初始化时等价于底座）在 dev 上的表现
dev_before_file = f"{WORK}/dev_before.json"
if os.path.exists(dev_before_file):
    dev_before = json.load(open(dev_before_file))
else:
    dev_before, _ = evaluate(model, dev_data)
    json.dump(dev_before, open(dev_before_file, "w"))
print("训练前 dev:", {k: v for k, v in dev_before.items() if k != "per_src"})
"""),
("code", training),
("code", """
# 训练后 dev 表现 + 拟合温度
dev_after, dev_logits = evaluate(model, dev_data)
TEMPERATURE, nll = fit_temperature(dev_logits)
TEMPERATURE = round(TEMPERATURE, 3)
print("训练前:", {k: round(v, 4) for k, v in dev_before.items() if k != "per_src"})
print("训练后:", {k: round(v, 4) for k, v in dev_after.items() if k != "per_src"})
print("最优温度 T =", TEMPERATURE, " NLL", round(nll, 4))
print("各数据源准确率（训练前 → 训练后）:")
for s, a in dev_after["per_src"].items():
    print(f"  {s:16s} {dev_before['per_src'].get(s, float('nan')):.3f} → {a:.3f}")
json.dump({"temperature": TEMPERATURE, "dev_after": dev_after}, open(f"{WORK}/dev_after.json", "w"))
"""),
("md", """
## 3. 合并权重并上传到 Hugging Face

上传后是**私有**仓库。正式提交榜单前要改成公开，维护者需要下载权重来测延迟和抽检。
"""),
("code", src("export.py")),
("code", """
# 冒烟测试：用 llm2jev 进程内的 transformers 后端加载合并后的模型，问一个和 README 一样的问题
from llm2jev import LLM2Jev
from llm2jev.backends import HF
from transformers import AutoProcessor
jev = LLM2Jev(AutoProcessor.from_pretrained(MERGED_DIR), HF(MERGED_DIR), temperature=TEMPERATURE)
print(json.dumps(jev(
    "Refund policy: full refund within 30 days of purchase; 50% until day 60; none after.\\n"
    "Order 1182 was bought on 3 March and returned on 20 April.",
    {"refund": {"type": "choice", "instructions": "What refund does order 1182 get?",
                "criteria": {"full": "Full refund", "half": "50% refund", "none": "No refund"}},
     "late": {"type": "noul", "instructions": "Was the return made after day 30?"}}), indent=1))
"""),
("md", """
## 完成 ✅

记下上面打印的 `HF_REPO` 和温度 `TEMPERATURE`（也存在模型仓库的 `ezjev.json` 里），然后打开 **`ezjev_eval_colab.ipynb`** 跑完整评测。

**怎么判断有没有希望进第 15–25 名**：dev 准确率要比训练前明显提升，一般会从 0.6–0.7 升到 0.85 左右。
dev 只是我们自己从训练数据里切出来的，和榜单分数不能直接换算；真实分数要看第 2 个 notebook。
"""),
]

# --------------------------------------------------------------------------------------------- 评测 notebook
eval_cells = [
("md", """
# ezjev 评测（Colab）· 第 2 步：跑完整 Decision Index 并提交

流程：重建评测集 → 用 vLLM 加 llm2jev 起一个 `/v1/systemone` 服务 → 先跑 100 条冒烟 → 跑完整的 150,317 条请求 → 上传结果 → 提 PR。

**需要准备**

1. Colab Pro，运行时选 **A100**；
2. Secrets 里的 `HF_TOKEN`（和第 1 步同一个）；
3. 用这个 HF 账号**先接受 [HLE 数据集](https://huggingface.co/datasets/cais/hle) 的使用条款**（评测集里有 HLE，不接受的话重建会失败）；
4. Google Drive 留出约 10 GB。

**耗时**：重建评测集约 1 小时（只需要做一次，结果存在 Drive 上），完整评测预计 5–8 小时。
runner 是一条一条顺序发请求的，**支持断点续跑**：断线后重新按顺序运行全部单元格，已经完成的请求会跳过。

**规则提醒**（kit 里已经强制执行）：不截断、不删选项、不按 benchmark 调提示词、没回答的按错算。
上下文装不下的请求会记为 `unsupported`（算错），所以 `MAX_MODEL_LEN` 尽量开大。
"""),
("code", """
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
("code", """
# 安装：评测 kit（从 GitHub 拿源码，kit 里需要 hub/ 目录）、vLLM、llm2jev
!git clone -q https://github.com/apolinario/decision-index /content/decision-index 2>/dev/null || git -C /content/decision-index pull -q
!pip -q install -e "/content/decision-index[rebuild]"
!pip -q install -U "vllm>=0.30" llm2jev
"""),
("code", """
# ===== 配置 =====
import os, json, subprocess, time, requests
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import whoami, hf_hub_download
HF_USER = whoami()["name"]

HF_REPO   = f"{HF_USER}/ezjev-4b"            # 第 1 步上传的模型
RUN_NAME  = "ezjev-4b"
DI_DIR    = "/content/drive/MyDrive/decision-index"   # 评测集和结果放在 Drive 上
SUITE_DIR = f"{DI_DIR}/suite-0.2"
RUN_DIR   = f"{DI_DIR}/runs/{RUN_NAME}"
RESULTS_REPO = f"{HF_USER}/decision-index-results-{RUN_NAME}"   # 结果上传到这个 dataset
MAX_MODEL_LEN = 131072   # 越大越少 unsupported；A100 40GB 上如果 vLLM 启动 OOM，就降到 65536

TEMPERATURE = json.load(open(hf_hub_download(HF_REPO, "ezjev.json")))["temperature"]
os.makedirs(f"{DI_DIR}/runs", exist_ok=True)
print(HF_REPO, "T =", TEMPERATURE)
"""),
("md", """
## 1. 重建评测集（只做一次）

评测集不公开分发，要从各个公开来源重建。会下载约 7 GB，临时文件放在 Colab 本地磁盘，最后只把结果文件存到 Drive。
kit 会校验哈希，和官方逐字节一致才算成功。
"""),
("code", """
if os.path.exists(f"{SUITE_DIR}/manifest.json"):
    print("评测集已存在:", SUITE_DIR)
else:
    %cd /content/decision-index
    !python -m decision_index suite rebuild --work /content/di-work
    BUILT = "/content/di-work/artifacts/benchmark-suite/release-v2-rebuilt"
    !python -m decision_index suite import --dir {SUITE_DIR} --rows {BUILT}/selected-rows.jsonl.gz --added-rows {BUILT}/added-rows.jsonl.gz
!ls -la {SUITE_DIR}
"""),
("md", """
## 2. 启动服务：vLLM（算 logprob）+ llm2jev（`/v1/systemone`）

两个服务都在后台跑，日志写在 `/content/vllm.log` 和 `/content/llm2jev.log`。断线重连后要重新运行这个单元格。
"""),
("code", """
def wait(url, name, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(name, "就绪", flush=True); return
        except requests.RequestException:
            pass
        time.sleep(10)
    raise RuntimeError(f"{name} 没起来，看日志")

subprocess.run("pkill -f 'vllm serve' ; pkill -f llm2jev", shell=True)
vllm = subprocess.Popen(
    f"vllm serve {HF_REPO} --port 8000 --max-logprobs 256 --return-tokens-as-token-ids "
    f"--max-model-len {MAX_MODEL_LEN} --gpu-memory-utilization 0.90 > /content/vllm.log 2>&1", shell=True)
wait("http://127.0.0.1:8000/health", "vLLM")
jevsrv = subprocess.Popen(
    f"llm2jev --model {HF_REPO} --backend vllm --url http://127.0.0.1:8000 --port 8080 "
    f"--temperature {TEMPERATURE} > /content/llm2jev.log 2>&1", shell=True)
time.sleep(30)
!tail -n 3 /content/llm2jev.log
"""),
("code", """
# 手动问一个问题，确认服务正常
r = requests.post("http://127.0.0.1:8080/v1/systemone", json={
    "state": [{"role": "user", "content": "I was charged twice. Please refund the duplicate."}],
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}}}})
print(json.dumps(r.json(), indent=1))
"""),
("md", """
## 3. 冒烟测试：随机 100 条

确认没有报错，并看一下单条延迟。榜单要求在 RTX PRO 6000 上**中位延迟低于 1000 ms**；4B 模型通常只要几十到几百毫秒。
"""),
("code", """
%cd /content/decision-index
!python -m decision_index suite sample --dir {SUITE_DIR} --n 100 --out /content/sample-100.jsonl.gz
!python -m decision_index run --engine http --option base_url=http://127.0.0.1:8080 --option model={RUN_NAME} --suite-dir {SUITE_DIR} --rows /content/sample-100.jsonl.gz --out /content/runs/sample-100 --fresh
!python -m decision_index score --suite-dir {SUITE_DIR} --results /content/runs/sample-100/results.jsonl --out /content/runs/sample-100
import collections, statistics
rows = [json.loads(l) for l in open("/content/runs/sample-100/results.jsonl")]
print(collections.Counter(r["status"] for r in rows))
print("中位延迟 ms:", statistics.median(r["total_wall_ms"] for r in rows if r["status"] == "ok"))
for r in rows:
    if r["status"] != "ok":
        print(r["dataset"], r["status"], r.get("error", "")[:200])
"""),
("md", """
## 4. 完整评测（可断点续跑）

在后台运行，进度写进 Drive 上的 `results.jsonl`。`--compact` 不保存题目原文，这样结果可以公开上传，不违反评测集的授权条款。
跑完会自动打分，生成 `scores.json`。

**断线之后**：重新运行「配置」和「启动服务」两个单元格，再运行下面这个单元格，会从断点继续。
"""),
("code", """
%cd /content/decision-index
full = subprocess.Popen(
    f"python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 "
    f"--option model={RUN_NAME} --suite-dir {SUITE_DIR} --out {RUN_DIR} --compact "
    f">> {DI_DIR}/{RUN_NAME}.log 2>&1", shell=True)
print("已在后台启动，用下一个单元格看进度")
"""),
("code", """
# 看进度（随时可以重复运行）
import collections
TOTAL = 119898 + 30419
st = collections.Counter()
if os.path.exists(f"{RUN_DIR}/results.jsonl"):
    for l in open(f"{RUN_DIR}/results.jsonl"):
        st[json.loads(l)["status"]] += 1
done = sum(st.values())
print(f"{done}/{TOTAL} ({100 * done / TOTAL:.1f}%)", dict(st))
print("后台进程:", "运行中" if full.poll() is None else f"已结束 (exit {full.returncode})")
!tail -n 5 {DI_DIR}/{RUN_NAME}.log
"""),
("code", """
# 跑完后看分数（如果 pipeline 中途断过，也可以手动打一次分）
if not os.path.exists(f"{RUN_DIR}/scores.json"):
    !cd /content/decision-index && python -m decision_index score --suite-dir {SUITE_DIR} --results {RUN_DIR}/results.jsonl --out {RUN_DIR}
s = json.load(open(f"{RUN_DIR}/scores.json"))
idx = json.load(open(f"{RUN_DIR}/index.json"))
print("complete:", s.get("complete"))
print("Decision Index 0.2.1:", idx.get("index"), " raw:", idx.get("raw_index"))
for a, v in idx.get("areas", {}).items():
    print(f"  {a:10s} skill {v.get('skill')}  coverage {v.get('coverage')}")
"""),
("md", """
## 5. 上传结果并提交

1. 运行下面的单元格，把 `runs/ezjev-4b/` 上传到一个 HF dataset；
2. 在 Hugging Face 上把**模型仓库改成公开**；
3. fork [apolinario/decision-index](https://github.com/apolinario/decision-index)，在 `submissions/README.md`（没有就新建）加一行，然后提 PR：

```
| ezjev-4b | https://huggingface.co/<你>/ezjev-4b | https://huggingface.co/datasets/<你>/decision-index-results-ezjev-4b (runs/ezjev-4b/scores.json) | http engine → llm2jev 0.6.x over vLLM, temperature <T> | Colab A100 |
```

在 PR 描述里写清楚：底座是 Qwen3.5-4B，LoRA 合并；只用了公开数据集的 train split，没用任何评测集数据；
服务命令照抄模型卡；`MAX_MODEL_LEN` 设成多少（这算声明的容量上限，可以接受，只要没有截断）。
维护者会在 RTX PRO 6000 上自己测延迟、抽检答案，然后重新打分。
"""),
("code", """
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=RUN_DIR, path_in_repo=f"runs/{RUN_NAME}", repo_id=RESULTS_REPO, repo_type="dataset",
                  commit_message=f"Decision Index 0.2.1 run: {RUN_NAME}")
print(f"https://huggingface.co/datasets/{RESULTS_REPO}  （现在是私有的；提 PR 前改成公开）")
"""),
]

for name, cells in (("ezjev_train_colab.ipynb", train_cells), ("ezjev_eval_colab.ipynb", eval_cells)):
    (OUT / name).write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUT / name)

# --------------------------------------------------------------------------------------------- HF Jobs 训练脚本
JOB_HEADER = '''# /// script
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
'''

JOB_BUILD = '''
rows = build(SIZES, f"{WORK}/all.jsonl.gz")
n_dev = min(1500, max(50, len(rows) // 30))
for name, part in (("dev", rows[:n_dev]), ("train", rows[n_dev:])):
    with gzip.open(f"{WORK}/{name}.jsonl.gz", "wt") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
print("train", len(rows) - n_dev, "dev", n_dev, flush=True)
'''

JOB_EVAL = '''
dev_before, _ = evaluate(model, dev_data)
print("训练前 dev:", json.dumps(dev_before), flush=True)
'''

JOB_AFTER = '''
dev_after, dev_logits = evaluate(model, dev_data)
TEMPERATURE = round(fit_temperature(dev_logits)[0], 3)
print("训练后 dev:", json.dumps(dev_after), flush=True)
print("温度 T =", TEMPERATURE, flush=True)
'''

job = "\n\n".join([
    JOB_HEADER, src("data.py"), src("sizes.py"), JOB_BUILD, src("train.py"),
    'train_data = load_encoded(f"{WORK}/train.jsonl.gz")\ndev_data = load_encoded(f"{WORK}/dev.jsonl.gz")\n',
    setup, JOB_EVAL, training, JOB_AFTER, src("export.py"),
])
(ROOT / "jobs").mkdir(exist_ok=True)
(ROOT / "jobs" / "train_job.py").write_text(job)
print("wrote", ROOT / "jobs" / "train_job.py")
