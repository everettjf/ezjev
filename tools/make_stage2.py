"""第二阶段（弱项强化）数据：从 v2 训练数据里挑出指定的数据源，再混入一部分其他数据防止退化。

用法：python3 tools/make_stage2.py <输出名> <数据源,数据源,...> [每个弱项最多几条] [回放比例]
例：  python3 tools/make_stage2.py s2 tools_bfcl,tools_apibank,esci_train 3000 0.4
环境变量 EXTRA=v2b 可以把别的数据目录（比如只含新数据源的 v2b）整份加进弱项部分。
结果上传到 <DATA_REPO>/<输出名>/train.jsonl.gz、dev.jsonl.gz（dev 沿用 v2 的）。
"""
import gzip, json, os, random, sys, collections
from huggingface_hub import HfApi, hf_hub_download, whoami

name, srcs = sys.argv[1], set(sys.argv[2].split(","))
per_src = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
replay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
repo = os.environ.get("DATA_REPO") or f"{whoami()['name']}/ezjev-data"
R = random.Random(1)

rows = [json.loads(l) for l in gzip.open(hf_hub_download(repo, "v2/train.jsonl.gz", repo_type="dataset"), "rt")]
by = collections.defaultdict(list)
for r in rows:
    by[r["src"]].append(r)
weak = [r for s in sorted(srcs) for r in R.sample(by[s], min(per_src, len(by[s])))]
for extra in filter(None, os.environ.get("EXTRA", "").split(",")):
    weak += [json.loads(l) for l in gzip.open(hf_hub_download(repo, f"{extra}/train.jsonl.gz", repo_type="dataset"), "rt")]
others = [r for r in rows if r["src"] not in srcs]
mix = weak + R.sample(others, min(len(others), int(len(weak) * replay / (1 - replay))))
R.shuffle(mix)
print("弱项:", {s: min(per_src, len(by[s])) for s in sorted(srcs)}, " 回放:", len(mix) - len(weak), " 共", len(mix))

os.makedirs("/tmp/ezjev-s2", exist_ok=True)
with gzip.open("/tmp/ezjev-s2/train.jsonl.gz", "wt") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
api = HfApi()
api.upload_file(path_or_fileobj="/tmp/ezjev-s2/train.jsonl.gz", path_in_repo=f"{name}/train.jsonl.gz", repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj=hf_hub_download(repo, "v2/dev.jsonl.gz", repo_type="dataset"), path_in_repo=f"{name}/dev.jsonl.gz",
                repo_id=repo, repo_type="dataset")
print(f"已上传 {repo}/{name}")
