# /// script
# requires-python = ">=3.10"
# dependencies = ["vllm>=0.30", "llm2jev", "huggingface_hub", "httpx>=0.27", "requests"]
# ///
"""ezjev 评测任务（HF Jobs）：vLLM + llm2jev 起 /v1/systemone 服务，用 decision-index 的 runner 和打分器跑。

环境变量：
  HF_REPO      要评测的模型（必填），比如 you/ezjev-4b
  SUITE_REPO   私有评测集 dataset（必填，由 suite_job.py 生成）
  RESULTS_REPO 结果上传到的 dataset（必填）
  RUN_NAME     结果目录名，默认取 HF_REPO 的名字
  PER_BENCH    >0 时是抽样估分：每个 benchmark 抽这么多个完整 case 组成 mini 评测集，用官方打分器打分
               （分数只是估计）；0 = 完整评测（可以提交榜单），默认 0
  MAX_MODEL_LEN vLLM 上下文长度，默认 131072；装不下的请求记为 unsupported（算错，但不违规）

完整评测会每 20 分钟把 results.jsonl 传到 RESULTS_REPO；任务中断后用同样参数重新提交，会从断点续跑。
"""
import collections, gzip, json, os, random, subprocess, sys, threading, time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = os.environ["HF_REPO"]
SUITE_REPO = os.environ["SUITE_REPO"]
RESULTS_REPO = os.environ["RESULTS_REPO"]
RUN_NAME = os.environ.get("RUN_NAME") or HF_REPO.split("/")[-1]
PER_BENCH = int(os.environ.get("PER_BENCH", "0"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "131072"))
os.environ["HF_HUB_DISABLE_XET"] = "1"

KIT, SUITE, OUT = Path("/tmp/decision-index"), Path("/tmp/suite-0.2"), Path(f"/tmp/runs/{RUN_NAME}")
PREFIX = f"runs/{RUN_NAME}" + ("" if PER_BENCH == 0 else f"-sample{PER_BENCH}")
BIN = Path(sys.executable).parent
api = HfApi()


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=KIT, env={**os.environ, "PYTHONPATH": str(KIT)}, **kw)


def di(args):
    sh(f"{sys.executable} -m decision_index {args}")


def wait(url, name, proc, timeout=2400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} 进程退出了 (exit {proc.returncode})")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"{name} 就绪 ({time.time() - t0:.0f}s)", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(10)
    raise RuntimeError(f"{name} 超时没起来")


# 1. 评测 kit + 评测集
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/apolinario/decision-index", str(KIT)], check=True)
di(f"suite download --dataset {SUITE_REPO} --dir {SUITE}")
suite_dir, verify = SUITE, ""

# 2. 抽样模式：每个 benchmark 取 PER_BENCH 个完整 case（同一 group 的请求一起取），写成 mini 评测集
if PER_BENCH > 0:
    mini = Path("/tmp/suite-mini")
    mini.mkdir(exist_ok=True)
    rng = random.Random(0)
    for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        with gzip.open(SUITE / name, "rt") as f:
            for line in f:
                e = json.loads(line)["_evaluation"]
                groups[e["catalog_id"]][e.get("group_id") or e["run_id"]].append(line)
        kept = 0
        with gzip.open(mini / name, "wt") as g:
            for cid, gs in sorted(groups.items()):
                for gid in rng.sample(sorted(gs), min(PER_BENCH, len(gs))):
                    g.writelines(gs[gid]); kept += len(gs[gid])
        print(f"{name}: 抽样 {kept} 条请求", flush=True)
    for name in ("excluded-questions.json", "manifest.json"):
        (mini / name).write_bytes((SUITE / name).read_bytes())
    suite_dir, verify = mini, "--no-verify"

# 3. 续跑：把之前传上去的 results.jsonl 拿回来
OUT.mkdir(parents=True, exist_ok=True)
try:
    p = hf_hub_download(RESULTS_REPO, f"{PREFIX}/results.jsonl", repo_type="dataset")
    (OUT / "results.jsonl").write_bytes(Path(p).read_bytes())
    print("从断点续跑，已有", sum(1 for _ in open(OUT / "results.jsonl")), "条", flush=True)
except Exception:
    print("全新开始", flush=True)

# 4. 起服务：vLLM 算 logprob，llm2jev 提供 /v1/systemone
cfg = json.load(open(hf_hub_download(HF_REPO, "ezjev.json")))
T = cfg["temperature"]
vllm = subprocess.Popen([str(BIN / "vllm"), "serve", HF_REPO, "--port", "8000", "--max-logprobs", "256",
                         "--return-tokens-as-token-ids", "--max-model-len", str(MAX_MODEL_LEN),
                         "--gpu-memory-utilization", "0.90"])
wait("http://127.0.0.1:8000/health", "vLLM", vllm)
jev = subprocess.Popen([sys.executable, "-m", "llm2jev", "--model", HF_REPO, "--backend", "vllm",
                        "--url", "http://127.0.0.1:8000", "--port", "8080", "--temperature", str(T)])
wait("http://127.0.0.1:8080/health", "llm2jev", jev, timeout=600)
r = requests.post("http://127.0.0.1:8080/v1/systemone", json={
    "state": "I was charged twice. Please refund the duplicate.",
    "questions": {"refund": {"type": "noul", "instructions": "Does the user request a refund?"},
                  "dept": {"type": "choice", "instructions": "Which department?",
                           "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}}}}, timeout=120)
print("服务自检:", r.status_code, r.text[:400], flush=True)
r.raise_for_status()

# 5. 跑评测（完整模式下后台定时上传进度）
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
stop = threading.Event()


def uploader():
    while not stop.wait(1200):
        try:
            api.upload_file(path_or_fileobj=str(OUT / "results.jsonl"), path_in_repo=f"{PREFIX}/results.jsonl",
                            repo_id=RESULTS_REPO, repo_type="dataset", commit_message="progress")
            print("[上传进度]", sum(1 for _ in open(OUT / "results.jsonl")), "条", flush=True)
        except Exception as ex:
            print("[上传进度失败]", ex, flush=True)


threading.Thread(target=uploader, daemon=True).start()
engine = f"--engine http --option base_url=http://127.0.0.1:8080 --option model={RUN_NAME}"
di(f"run {engine} --suite-dir {suite_dir} {verify} --out {OUT} --compact")
stop.set()
di(f"score --suite-dir {suite_dir} --results {OUT}/results.jsonl --out {OUT} --engine {RUN_NAME}")

# 6. 汇总并上传
s = json.load(open(OUT / "scores.json"))
rows = [json.loads(l) for l in open(OUT / "results.jsonl")]
ms = sorted(x["total_wall_ms"] for x in rows if x.get("status") == "ok")
print("=" * 60, flush=True)
print("模式:", "完整评测" if PER_BENCH == 0 else f"抽样估分（每个 benchmark {PER_BENCH} 个 case，分数仅供参考）")
print("状态:", dict(collections.Counter(x["status"] for x in rows)))
print("中位延迟 ms:", ms[len(ms) // 2] if ms else None)
print("Decision Index 0.2.1:", s["decision_index"], " complete:", s["complete"])
for a in s["areas"]:
    print(f"  {a['id']:10s} skill {a['skill']}  coverage {a['coverage']}")
print("=" * 60, flush=True)
api.upload_folder(folder_path=str(OUT), path_in_repo=PREFIX, repo_id=RESULTS_REPO, repo_type="dataset",
                  commit_message=f"{RUN_NAME}: index {s['decision_index']}")
print(f"结果: https://huggingface.co/datasets/{RESULTS_REPO}/tree/main/{PREFIX}", flush=True)
vllm.terminate(); jev.terminate()
