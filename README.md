# ezjev

用 Google Colab 训练一个 Jev 风格的决策模型（typed decisions：`choice` / `noul` / `score`，每个选项给一个概率），
目标是进入 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 0.2.1 榜单的 **第 15–25 名**，
也就是 **35–43 分**。

## 方案

| | |
|---|---|
| 底座 | `Qwen/Qwen3.5-4B`，LoRA r=16，合并成完整权重 |
| 提示词 / 推理 | [llm2jev](https://github.com/tic-top/llm2jev) 的 chat 格式：一次 prefill，在 `Answer:` 后读选项字母的 logprob |
| 损失 | 选项上的交叉熵 + Brier |
| 数据 | 约 5.5 万道题，全部来自 27 个公开数据源的 train split，不使用任何评测集数据 |
| 参照 | 同样配方的 JPT-4B 在 0.2.1 上得 43.04 分；4B 级其他模型在 35–41 分之间 |

## 你需要做的

1. **准备账号**
   - Colab Pro（要用 A100）；
   - Hugging Face 账号，建一个 **write** token，并接受 [HLE](https://huggingface.co/datasets/cais/hle) 的使用条款（评测要用）；
   - Google Drive 留出约 30 GB。
2. **训练**：在 Colab 打开 [`notebooks/ezjev_train_colab.ipynb`](notebooks/ezjev_train_colab.ipynb)，
   运行时选 A100，在 🔑 Secrets 里添加 `HF_TOKEN`，然后「全部运行」。大约 3 小时，模型会上传到 `<你>/ezjev-4b`（私有）。
3. **评测**：打开 [`notebooks/ezjev_eval_colab.ipynb`](notebooks/ezjev_eval_colab.ipynb)，同样选 A100，全部运行。
   第一次要重建评测集（约 1 小时），完整评测约 5–8 小时，断线后可以续跑。
4. **提交**：把模型仓库和结果 dataset 改成公开，然后给
   [apolinario/decision-index](https://github.com/apolinario/decision-index) 提 PR，在 `submissions/README.md` 里加一行
   （模板在评测 notebook 最后）。维护者会在 RTX PRO 6000 上测延迟（中位数要低于 1 秒）、抽检答案并重新打分，然后上榜。

打开 notebook 的方法：在 Colab 里选「文件 → 上传笔记本」；
或者把本仓库推到 GitHub，然后用 `https://colab.research.google.com/github/<用户>/<仓库>/blob/main/notebooks/ezjev_train_colab.ipynb` 打开。

## 如果分数不够

- 先看评测 notebook 打印的各领域分数，以及 `unsupported` 的数量。上下文装不下的请求会算错，可以调大 `MAX_MODEL_LEN`；
- 在训练 notebook 的 `SIZES` 里给弱的领域加数据，或者把 `DATA_SCALE` 调到 1.5–2；
- 换成 `Qwen/Qwen3.5-9B`（JPT-9B 用同类方法得了 46.89 分），这时 A100 40GB 要把 `TOKENS_PER_BATCH` 调小。

## 注意

- 不能用评测集训练，也不能再发布评测集（规则和数据授权都禁止）。评测 notebook 用 `--compact` 保存结果，不包含题目原文。
- 部分训练数据的授权是非商用的（例如 ANLI 是 CC BY-NC 4.0），训练出的权重如果要商用，请先核对授权。
- 本项目与 TypeSafe AI 无关。
