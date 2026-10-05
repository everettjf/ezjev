# /// script
# requires-python = ">=3.10"
# dependencies = ["vllm>=0.30", "llm2jev", "huggingface_hub", "requests"]
# ///
"""JevBench 公开题自测（HF Jobs）：vLLM + llm2jev 起 /v1/systemone，用 JevBench 的 typesafe adapter 串行跑 231 条公开题。

环境变量：
  HF_REPO       要评测的模型（必填），比如 you/ezjev-4b-s2
  RESULTS_REPO  结果上传到的 dataset（必填），放在 jevbench/<模型名>/ 下
  JEVBENCH_REV  JevBench 的 commit，默认 main
  PRICE_IN      成本估算用的输入价格（美元 / 百万 token），默认 0.04（Plumb-4B 用的 Qwen3.5-4B 公开价）

只是自测：正式分数还包含封存题，要在 JevBench 开 [bench request] issue 由维护者跑。
"""
import json, os, subprocess, sys, time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = os.environ["HF_REPO"]
RESULTS_REPO = os.environ["RESULTS_REPO"]
REV = os.environ.get("JEVBENCH_REV", "main")
PRICE_IN = os.environ.get("PRICE_IN", "0.04")
NAME = HF_REPO.split("/")[-1]
os.environ["HF_HUB_DISABLE_XET"] = "1"

KIT, OUT = Path("/tmp/jevbench"), Path(f"/tmp/jevbench-runs/{NAME}")
BIN = Path(sys.executable).parent
TIERS = ["easy", "original", "hard"]


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


# 1. JevBench 代码和公开题
subprocess.run(["git", "clone", "-q", "https://github.com/fstandhartinger/jevbench", str(KIT)], check=True)
subprocess.run(["git", "checkout", "-q", REV], cwd=KIT, check=True)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=KIT, text=True).strip()
print("JevBench", commit, flush=True)

# 2. 起服务（和 eval_job.py 一样）
cfg = json.load(open(hf_hub_download(HF_REPO, "ezjev.json")))
T = cfg["temperature"]
vllm = subprocess.Popen([str(BIN / "vllm"), "serve", HF_REPO, "--port", "8000", "--max-logprobs", "256",
                         "--return-tokens-as-token-ids", "--max-model-len", "32768",
                         "--gpu-memory-utilization", "0.90",
                         "--additional-config", json.dumps({"gdn_prefill_backend": "triton"})],
                        env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
wait("http://127.0.0.1:8000/health", "vLLM", vllm)
jev = subprocess.Popen([sys.executable, "-m", "llm2jev", "--model", HF_REPO, "--backend", "vllm",
                        "--url", "http://127.0.0.1:8000", "--port", "8080", "--temperature", str(T)])
wait("http://127.0.0.1:8080/health", "llm2jev", jev, timeout=600)
# 预热几次，别让第一次请求的编译时间算进延迟
for _ in range(3):
    requests.post("http://127.0.0.1:8080/v1/systemone", json={
        "state": "I was charged twice. Please refund the duplicate.",
        "questions": {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}}},
        timeout=120).raise_for_status()


# 3. 跑公开题：每个 tier 单独一个结果文件，串行请求（和官方测延迟的方式一样）
def jb(*args):
    return subprocess.run([sys.executable, "-m", "jevbench.cli", *args], cwd=KIT, check=False,
                          env={**os.environ, "PYTHONPATH": str(KIT)}, capture_output=True, text=True)


OUT.mkdir(parents=True, exist_ok=True)
summary = {"model": HF_REPO, "jevbench_commit": commit, "temperature": T, "price_in_per_m": float(PRICE_IN)}
for tier in TIERS:
    tasks = f"datasets/public/{tier}.jsonl"
    r = jb("run", "--tasks", tasks, "--adapter", "typesafe", "--endpoint", "http://127.0.0.1:8080",
           "--key-env", "", "--model", NAME, "--results", str(OUT / f"{tier}.results.jsonl"),
           "--raw-dir", str(OUT / f"raw-{tier}"), "--ledger", str(OUT / "ledger.jsonl"),
           "--price-in-per-m", PRICE_IN, "--price-out-per-m", "0", "--cap-usd", "100",
           "--cost-basis", f"estimate_qwen3.5-4b_public_input_price_{PRICE_IN}_per_m",
           "--manifest", str(OUT / f"{tier}.manifest.json"))
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    s = jb("summarize", "--tasks", tasks, "--results", str(OUT / f"{tier}.results.jsonl"),
           "--public-export", str(OUT / f"{tier}.summary.json"))
    if s.returncode != 0:
        print(s.stderr[-2000:], flush=True)
        continue
    summary[tier] = json.load(open(OUT / f"{tier}.summary.json"))

# 合在一起再算一次总的
allres = OUT / "all.results.jsonl"
allres.write_text("".join((OUT / f"{t}.results.jsonl").read_text() for t in TIERS))
s = jb("summarize", "--tasks", ",".join(f"datasets/public/{t}.jsonl" for t in TIERS),
       "--results", str(allres), "--public-export", str(OUT / "all.summary.json"))
if s.returncode == 0:
    summary["all"] = json.load(open(OUT / "all.summary.json"))
else:
    print(s.stderr[-2000:], flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("\n=== JevBench 公开题 ===", flush=True)
for k in TIERS + ["all"]:
    m = summary.get(k)
    if m:
        lat = m.get("latency") or {}
        print(f"{k:9s} acc={m['n_correct']}/{m['n_scorable']} ({m['accuracy']:.3f})  ece={m['ece']}  "
              f"brier={m['brier_mean']}  p50={lat.get('p50_s')}  p95={lat.get('p95_s')}  "
              f"$/1k={m['price_per_1000_decisions_usd']}  failed={m['n_attempted'] - m['n_valid']}", flush=True)

# 4. 上传（raw 目录里有题目原文，不传；结果文件只有 id、预测和概率）
api = HfApi()
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path=str(OUT),
                  path_in_repo=f"jevbench/{NAME}", ignore_patterns=["raw-*/**", "raw-*"])
print("已上传到", f"{RESULTS_REPO}/jevbench/{NAME}", flush=True)
vllm.terminate(); jev.terminate()
