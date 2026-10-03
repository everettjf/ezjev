#!/usr/bin/env bash
# 提交 HF Jobs。先 `hf auth login`，然后：
#   jobs/launch.sh suite          重建评测集 → 私有 dataset（只需一次，CPU，约 1–2 小时，几毛钱）
#   jobs/launch.sh train-quick    快速试训：Qwen3.5-0.8B，约 2,700 题（A100，约 20–30 分钟）
#   jobs/launch.sh eval-quick     抽样估分试训出来的模型（RTX PRO 6000，每个 benchmark 20 个 case）
#   jobs/launch.sh data v1|v2     构造训练数据并和评测集去重 → 私有 dataset（CPU，几毛钱）
#   jobs/launch.sh ab v1|v2       对比测试：用 v1 / v2 数据训 Qwen3.5-0.8B（A100）
#   jobs/launch.sh eval <模型仓库> <每个 benchmark 抽几个 case>   抽样估分任意模型（0 = 完整评测）
#   jobs/launch.sh train          正式训练：Qwen3.5-4B，v2 数据（A100）
#   jobs/launch.sh eval-sample    正式模型抽样估分（每个 benchmark 50 个 case）
#   jobs/launch.sh eval-full      正式模型完整评测（可提交榜单；中断后重跑同一命令会续跑）
# 看任务：hf jobs ps / hf jobs logs <id> / hf jobs cancel <id>
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN="$(hf auth token 2>/dev/null)" || { echo "先运行 hf auth login"; exit 1; }
export HF_TOKEN
USER_NAME="$(hf auth whoami | head -n1 | sed 's/^user[:=] *//; s/ .*//')"
SUITE_REPO="${SUITE_REPO:-$USER_NAME/decision-index-suite-0.2}"
RESULTS_REPO="${RESULTS_REPO:-$USER_NAME/decision-index-results}"
QUICK_REPO="$USER_NAME/ezjev-0.8b-quick"
FULL_REPO="$USER_NAME/ezjev-4b"
DATA_REPO="${DATA_REPO:-$USER_NAME/ezjev-data}"
run() { hf jobs uv run --detach --secrets HF_TOKEN -e PYTHONUNBUFFERED=1 "$@"; }

case "${1:-}" in
  suite)       run --flavor cpu-upgrade --timeout 8h --name ezjev-suite -e SUITE_REPO="$SUITE_REPO" suite_job.py ;;
  train-quick) run --flavor a100-large --timeout 2h --name ezjev-train-quick \
                   -e HF_REPO="$QUICK_REPO" -e BASE_MODEL=Qwen/Qwen3.5-0.8B -e DATA_SCALE=0.05 train_job.py ;;
  eval-quick)  run --flavor rtx-pro-6000 --timeout 4h --name ezjev-eval-quick \
                   -e HF_REPO="$QUICK_REPO" -e SUITE_REPO="$SUITE_REPO" -e RESULTS_REPO="$RESULTS_REPO" -e PER_BENCH=20 eval_job.py ;;
  data)        v="${2:?v1 或 v2}"
               run --flavor cpu-upgrade --timeout 4h --name "ezjev-data-$v" -e DATA_REPO="$DATA_REPO" -e DATA_NAME="$v" \
                   -e DATA_VERSION="${v#v}" -e SUITE_REPO="$SUITE_REPO" data_job.py ;;
  ab)          v="${2:?v1 或 v2}"
               run --flavor a100-large --timeout 6h --name "ezjev-ab-$v" -e HF_REPO="$USER_NAME/ezjev-0.8b-$v" \
                   -e BASE_MODEL=Qwen/Qwen3.5-0.8B -e DATA_REPO="$DATA_REPO" -e DATA_NAME="$v" -e MAX_LEN=16384 train_job.py ;;
  eval)        m="${2:?模型仓库}"; k="${3:?每个 benchmark 的 case 数，0 = 完整评测}"
               run --flavor rtx-pro-6000 --timeout "$([ "$k" = 0 ] && echo 24h || echo 6h)" --name "ezjev-eval-${m##*/}-$k" \
                   -e HF_REPO="$m" -e SUITE_REPO="$SUITE_REPO" -e RESULTS_REPO="$RESULTS_REPO" -e PER_BENCH="$k" eval_job.py ;;
  train)       run --flavor a100-large --timeout 10h --name ezjev-train \
                   -e HF_REPO="${HF_REPO:-$FULL_REPO}" -e BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}" -e DATA_REPO="$DATA_REPO" \
                   -e DATA_NAME="${DATA_NAME:-v2}" -e MAX_LEN=16384 -e LR="${LR:-1e-4}" train_job.py ;;
  eval-sample) run --flavor rtx-pro-6000 --timeout 6h --name ezjev-eval-sample \
                   -e HF_REPO="$FULL_REPO" -e SUITE_REPO="$SUITE_REPO" -e RESULTS_REPO="$RESULTS_REPO" -e PER_BENCH=50 eval_job.py ;;
  eval-full)   run --flavor rtx-pro-6000 --timeout 24h --name ezjev-eval-full \
                   -e HF_REPO="$FULL_REPO" -e SUITE_REPO="$SUITE_REPO" -e RESULTS_REPO="$RESULTS_REPO" -e PER_BENCH=0 eval_job.py ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac
