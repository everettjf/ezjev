# ===== 合并 LoRA → 完整权重，保存到 Drive 并上传 HF =====
import gc, json, os
from transformers import AutoProcessor
from huggingface_hub import HfApi

try:
    del model
except NameError:
    pass
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

merged = PeftModel.from_pretrained(load_base(), f"{WORK}/lora").merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
try:
    AutoProcessor.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)
except Exception:
    tok.save_pretrained(MERGED_DIR)
json.dump({"base_model": BASE_MODEL, "temperature": TEMPERATURE, "prompt": "llm2jev --prompt chat",
           "dev_before": dev_before, "dev_after": dev_after}, open(f"{MERGED_DIR}/ezjev.json", "w"), indent=1, ensure_ascii=False)
open(f"{MERGED_DIR}/README.md", "w").write(f"""---
base_model: {BASE_MODEL}
library_name: transformers
tags: [decision-model, jev, typed-decisions]
---

# {HF_REPO.split('/')[-1]}

A typed-decision model (Jev-style `/v1/systemone`: `choice`, `noul`, `score` questions answered with a probability
per option from one forward pass). LoRA fine-tune of `{BASE_MODEL}`, merged into full weights. Trained only on train
splits of public datasets; no Decision Index suite rows were used.

Serve with [llm2jev](https://github.com/tic-top/llm2jev) (chat prompt, temperature {TEMPERATURE}):

```bash
vllm serve {HF_REPO} --max-logprobs 256 --return-tokens-as-token-ids --port 8000
llm2jev --model {HF_REPO} --backend vllm --url http://127.0.0.1:8000 --port 8080 --temperature {TEMPERATURE}
```

Held-out dev accuracy (our own splits): base {dev_before['acc']:.3f} -> fine-tuned {dev_after['acc']:.3f}.

Some training sources (e.g. ANLI, CC BY-NC 4.0) are non-commercial; check their licences before commercial use.
""")
print("已保存到", MERGED_DIR, flush=True)
del merged; gc.collect()

api = HfApi()
api.create_repo(HF_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_REPO, commit_message="ezjev weights")
print(f"已上传: https://huggingface.co/{HF_REPO} （目前是私有仓库，提交榜单前改成公开）", flush=True)
