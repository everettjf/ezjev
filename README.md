# ezjev

Training a Jev-style decision model (typed decisions: `choice` / `noul` / `score`, with a probability for every option),
submitted to two leaderboards: [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
and [JevBench](https://benchmarkheaven.com/jev-models).

Project page: **https://xnu.app/ezjev/** (model overview, downloads and quickstart)

## Leaderboard submissions

| Leaderboard | Model | Submission | Status (2026-10-06) |
|---|---|---|---|
| Jev Decision Index | [ezjev-4b-s2](https://huggingface.co/everettjf/ezjev-4b-s2) | PR [apolinario/decision-index#55](https://github.com/apolinario/decision-index/pull/55) | **Listed in Decision Index 0.3** (2026-10-06): Full score **46.95**, #35 of 111 (tied group from #31), **best of all models ≤ 5B** |
| JevBench | [ezjev-4b-s3](https://huggingface.co/everettjf/ezjev-4b-s3) | Bench request [fstandhartinger/jevbench#193](https://github.com/fstandhartinger/jevbench/issues/193) | Acknowledged by the maintainer; in the free measurement queue. The official score (incl. sealed items) is measured by the maintainer |

### Decision Index 0.3 (official)

Decision Index 0.3 changed the scoring: the Full score is 20% public benchmarks, 50% private tests of the same skills and
30% private tasks from new domains. The private parts are never published, so nobody can train on them.

| Model | Full score | Public (37 benchmarks) | Private, same skills | Private, new domains | Rank (of 111) |
|---|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe) | 60.11 | 57.96 | 58.04 | 54.98 | #3 |
| **ezjev-4b-s2** | **46.95** | **50.82** | **46.22** | **40.73** | **#35** |
| vLLM-SR Decision 2.0 Nox 4B | 44.95 | 44.21 | 43.72 | 42.19 | #38 |
| jiwo 4B | 42.86 | 45.76 | 46.05 | 33.32 | #42 |
| Hopper (G) 1.2 | 42.61 | 41.15 | 42.37 | 39.61 | #43 |
| JPT-4B | 41.63 | 42.82 | 40.11 | 39.17 | #46 |

Every model ranked above ezjev-4b-s2 has at least 9B parameters. The score is lower than our 0.2.1 estimate (51.15): on the
public benchmarks we keep 50.82, but the private same-skill tests (46.22) and especially the new-domain tasks (40.73) show
that part of the gain from matching the public task formats does not transfer. Per area (public, skill score): knowledge 31.0,
language 60.5, retrieval 56.3, tools 69.9, arts 31.1. Source: the leaderboard's `data/index.json` (generated 2026-10-06).

## Development results (Decision Index 0.2.1, our own runs)

| Model | Data | Decision Index 0.2.1 |
|---|---|---|
| ezjev-0.8b-v1 | v1 (27 public data sources) | 21.5 (sampled estimate) |
| ezjev-0.8b-v2 | v2 (data added to match the leaderboard's task formats) | 31.1 (sampled estimate) |
| ezjev-4b | v2 | 47.9 (sampled estimate) |
| **ezjev-4b-s2** | v2 + weak-spot stage | **51.15** (full run, all 150,759 requests; sampled estimate 49.3) |

Sampled estimate = 50 random cases per benchmark (fixed seed, so every model gets the same cases), scored with the official
scorer; expect an error of a few points.

### JevBench

We also submitted to [JevBench](https://benchmarkheaven.com/jev-models) ([bench request #193](https://github.com/fstandhartinger/jevbench/issues/193))
with [ezjev-4b-s3](https://huggingface.co/everettjf/ezjev-4b-s3); the Decision Index entry stays on s2.
The JevBench score is the harmonic mean of Intelligence / Calibration / Speed / Cost, and the sealed items can only be run by
the maintainer. Self-test on the 231 public items:

| Model | easy (48) | original (72) | hard (111) | Overall | ECE | p50 latency |
|---|---|---|---|---|---|---|
| ezjev-4b-s2 | 48 | 68 | 65 | 0.784 | 0.059 | 0.034 s |
| **ezjev-4b-s3** | 48 | 71 | 68 | **0.810** | **0.044** | 0.034 s |

s3 = s2 + ~20k code-generated hard-tier style decisions ([`parts/gen_hard.py`](parts/gen_hard.py): multi-clause rules in long
policy documents, business-day / time-zone / leap-year deadlines, pro-rated refunds, multi-hop lookups, answer judging,
insufficient-information cases, traps), plus 40% replay, LR 5e-5.
No JevBench items were used for training; the public items are only used for the self-test (`jobs/launch.sh jevbench <model repo>`).

## Approach

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-4B`, LoRA r=16 (including the DeltaNet linear-attention layers), merged into full weights |
| Prompt / inference | The [llm2jev](https://github.com/tic-top/llm2jev) chat format: one prefill, then read the logprobs of the option letters after `Answer:` |
| Loss | Cross-entropy + Brier over the options; after training, one global temperature is fitted on dev |
| Data v2 | ~85k rows / ~106k questions: the v1 public sources, plus ContractNLI, VAST, NLI4CT, RAGTruth, iSarcasmEval, ACOS, ShARC, ESCI, QNLI, SGD, ToolACE/Glaive (in BFCL / API-Bank / ToolRet formats), When2Call, Humicroedit and New Yorker, added to match the leaderboard's task formats, and code-generated BBH / CRUXEval / GSM8K / CLadder / SATA-style items |
| Weak-spot stage (s2) | One more pass on ezjev-4b at a low learning rate (5e-5): RAGTruth, ESCI, CRUXEval, MMLU, SATA, NLI4CT, plus new PhishNChips-style data (real phishing / benign URLs in templated emails) and HoVer-style data, with 40% replay |
| Decontamination | All training data is checked against the evaluation suite (exact sentences + 13-grams); any overlapping row is dropped (`parts/decontam.py`) |

Only the train (or dev) split of each source is used; the splits used for evaluation are in the decision-index kit's `suite/build/*.py`.
The list of data sources, their licences and how they relate to the evaluation sets are documented on each function in
[`parts/data_v2.py`](parts/data_v2.py).

## Running on HF Jobs (recommended)

First `hf auth login` (write token) and accept the terms of use for [HLE](https://huggingface.co/datasets/cais/hle), then:

```bash
jobs/launch.sh suite                         # rebuild the evaluation suite → private dataset (once)
jobs/launch.sh data v2                       # build training data and decontaminate → <you>/ezjev-data/v2
jobs/launch.sh train                         # Qwen3.5-4B + v2 data (A100, ~3.5 hours)
jobs/launch.sh eval <you>/ezjev-4b 50        # sampled estimate (RTX PRO 6000, ~15 minutes)
jobs/launch.sh eval <you>/ezjev-4b 0         # full run (~6–7 hours; rerun the same command to resume after an interruption)
```

JevBench self-test: `jobs/launch.sh jevbench <model repo>` (RTX PRO 6000, ~9 minutes; results go to `<RESULTS_REPO>/jevbench/`).

s3: `DATA_ONLY="hard_temporal:7000,hard_policy:5000,hard_multihop:3500,hard_judge:3000,hard_trap:1500" jobs/launch.sh data v3h`,
`EXTRA=v3h python3 tools/make_stage2.py s3 contractnli_train,sharc,ragtruth_train,gen_crux,gen_bbh 1500 0.4`,
then `HF_REPO=<you>/ezjev-4b-s3 BASE_MODEL=<you>/ezjev-4b-s2 DATA_NAME=s3 LR=5e-5 jobs/launch.sh train` (A100, ~2.6 hours).

Weak-spot stage: `DATA_ONLY="phish:3000,hover_like:2500" jobs/launch.sh data v2b`,
`EXTRA=v2b python3 tools/make_stage2.py s2 <source,...> 3000 0.4`,
then `HF_REPO=<you>/ezjev-4b-s2 BASE_MODEL=<you>/ezjev-4b DATA_NAME=s2 LR=5e-5 jobs/launch.sh train`.

The whole Decision Index pipeline (including the 0.8B comparison runs) cost about $45 on HF Jobs.

## Running on Colab

[`notebooks/ezjev_train_colab.ipynb`](notebooks/ezjev_train_colab.ipynb) and [`notebooks/ezjev_eval_colab.ipynb`](notebooks/ezjev_eval_colab.ipynb)
use the v1 data (`DATA_VERSION = 1`): training takes ~3 hours on an A100, a full evaluation ~5–8 hours. The v2 data has to be built
on HF Jobs (it needs decontamination against the private evaluation suite).

When changing code, only edit `parts/`, then run `python3 tools/build.py` to regenerate the notebooks and `jobs/train_job.py` / `jobs/data_job.py`.

## Notes

- Do not train on the evaluation suite, and do not redistribute it (both the rules and the data licences forbid it). Evaluations
  save results with `--compact`, which leaves out the item text.
- Some training data is licensed for non-commercial use only (e.g. ANLI is CC BY-NC 4.0), and VAST, NLI4CT, ACOS and Humicroedit
  state no licence; check the licences before using the trained weights commercially.
- The code is under the [MIT License](LICENSE). MIT covers only the code in this repository, not the training data or the model
  weights: the weights remain subject to the data licences above.
- This project is not affiliated with TypeSafe AI.
