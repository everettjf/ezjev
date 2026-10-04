# ezjev

训练一个 Jev 风格的决策模型（typed decisions：`choice` / `noul` / `score`，每个选项给一个概率），
参加 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 0.2.1 榜单。

## 结果

| 模型 | 数据 | Decision Index 0.2.1 |
|---|---|---|
| ezjev-0.8b-v1 | v1（27 个公开数据源） | 21.5（抽样估分） |
| ezjev-0.8b-v2 | v2（按榜单题型补数据） | 31.1（抽样估分） |
| ezjev-4b | v2 | 47.9（抽样估分） |
| **ezjev-4b-s2** | v2 + 弱项强化 | **51.15**（完整评测，150,759 条全部完成；抽样估分 49.3） |

抽样估分 = 每个 benchmark 随机抽 50 个 case（固定种子，各模型抽到的是同一批），用官方打分器算分，误差约几分。

## 方案

| | |
|---|---|
| 底座 | `Qwen/Qwen3.5-4B`，LoRA r=16（含 DeltaNet 线性注意力层），合并成完整权重 |
| 提示词 / 推理 | [llm2jev](https://github.com/tic-top/llm2jev) 的 chat 格式：一次 prefill，在 `Answer:` 后读选项字母的 logprob |
| 损失 | 选项上的交叉熵 + Brier，训练后在 dev 上拟合一个全局温度 |
| 数据 v2 | 约 8.5 万条样本 / 10.6 万个问题：v1 的公开数据源，加上按榜单题型补的 ContractNLI、VAST、NLI4CT、RAGTruth、iSarcasmEval、ACOS、ShARC、ESCI、QNLI、SGD、ToolACE/Glaive（BFCL / API-Bank / ToolRet 三种格式）、When2Call、Humicroedit、New Yorker，以及代码生成的 BBH / CRUXEval / GSM8K / CLadder / SATA 风格题 |
| 弱项强化（s2） | 在 ezjev-4b 上用低学习率（5e-5）再训一轮：RAGTruth、ESCI、CRUXEval、MMLU、SATA、NLI4CT，加上新的 PhishNChips 风格（真实钓鱼/正常 URL + 模板邮件）和 HoVer 风格数据，40% 回放 |
| 去重 | 所有训练数据和评测集做整句 + 13-gram 比对，重合的整条删掉（`parts/decontam.py`） |

所有来源只用 train（或 dev）部分；评测用的 split 见 decision-index kit 的 `suite/build/*.py`。
数据源清单、授权和与评测集的关系见 [`parts/data_v2.py`](parts/data_v2.py) 里每个函数的说明。

## 在 HF Jobs 上跑（推荐）

先 `hf auth login`（write token），接受 [HLE](https://huggingface.co/datasets/cais/hle) 的使用条款，然后：

```bash
jobs/launch.sh suite                         # 重建评测集 → 私有 dataset（只需一次）
jobs/launch.sh data v2                       # 构造训练数据并去重 → <你>/ezjev-data/v2
jobs/launch.sh train                         # Qwen3.5-4B + v2 数据（A100，约 3.5 小时）
jobs/launch.sh eval <你>/ezjev-4b 50         # 抽样估分（RTX PRO 6000，约 15 分钟）
jobs/launch.sh eval <你>/ezjev-4b 0          # 完整评测（约 6–7 小时，中断后重跑同一命令会续跑）
```

弱项强化：`DATA_ONLY="phish:3000,hover_like:2500" jobs/launch.sh data v2b`，
`EXTRA=v2b python3 tools/make_stage2.py s2 <数据源,...> 3000 0.4`，
然后 `HF_REPO=<你>/ezjev-4b-s2 BASE_MODEL=<你>/ezjev-4b DATA_NAME=s2 LR=5e-5 jobs/launch.sh train`。

这次全部流程（含 0.8B 对比测试）在 HF Jobs 上一共花了约 45 美元。

## 在 Colab 上跑

[`notebooks/ezjev_train_colab.ipynb`](notebooks/ezjev_train_colab.ipynb) 和 [`notebooks/ezjev_eval_colab.ipynb`](notebooks/ezjev_eval_colab.ipynb)
用的是 v1 数据（`DATA_VERSION = 1`），A100 上训练约 3 小时，完整评测约 5–8 小时。v2 数据要用 HF Jobs 构造（需要和私有评测集去重）。

改代码时只改 `parts/`，然后运行 `python3 tools/build.py` 重新生成 notebook 和 `jobs/train_job.py`、`jobs/data_job.py`。

## 注意

- 不能用评测集训练，也不能再发布评测集（规则和数据授权都禁止）。评测用 `--compact` 保存结果，不包含题目原文。
- 部分训练数据的授权是非商用的（例如 ANLI 是 CC BY-NC 4.0），VAST、NLI4CT、ACOS、Humicroedit 没有写明授权；训练出的权重如果要商用，请先核对授权。
- 本项目与 TypeSafe AI 无关。
