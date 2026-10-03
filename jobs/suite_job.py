# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub", "httpx>=0.27", "pyarrow>=20", "pandas>=2.0", "numpy>=1.26", "scipy>=1.11",
#   "mido>=1.3", "python-chess>=1.10", "tiktoken>=0.7",
# ]
# ///
"""重建 Decision Index 0.2.1 评测集并上传到你名下的【私有】dataset（只需要跑一次）。

评测集不公开分发，只能从各个来源重建；kit 会校验哈希，和官方逐字节一致才上传。
需要：HF 账号已接受 https://huggingface.co/datasets/cais/hle 的条款。
环境变量：SUITE_REPO，比如 you/decision-index-suite-0.2
"""
import os, subprocess, sys
from huggingface_hub import HfApi

SUITE_REPO = os.environ["SUITE_REPO"]
KIT = "/tmp/decision-index"
BUILT = "/tmp/work/artifacts/benchmark-suite/release-v2-rebuilt"
env = {**os.environ, "PYTHONPATH": KIT, "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1"}


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=KIT, env=env)


subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/apolinario/decision-index", KIT], check=True)
sh(f"{sys.executable} -m decision_index suite rebuild --work /tmp/work")
sh(f"{sys.executable} scripts/prepare_hub_upload.py --rows {BUILT}/selected-rows.jsonl.gz "
   f"--added-rows {BUILT}/added-rows.jsonl.gz --out /tmp/hub-upload")
api = HfApi()
api.create_repo(SUITE_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/tmp/hub-upload", repo_id=SUITE_REPO, repo_type="dataset",
                  commit_message="Decision Index 0.2 suite (rebuilt, hash-verified)")
print(f"完成: https://huggingface.co/datasets/{SUITE_REPO} （私有，不要公开）", flush=True)
