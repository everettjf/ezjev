# ===== 加载模型 + LoRA + 训练循环（可断点续训）=====
import os, json, math, time
from peft import PeftModel

def load_base():
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    multimodal = hasattr(cfg, "vision_config")
    cls = getattr(transformers, "AutoModelForMultimodalLM", transformers.AutoModelForImageTextToText) if multimodal \
        else transformers.AutoModelForCausalLM
    m = cls.from_pretrained(BASE_MODEL, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                            device_map={"": 0} if DEVICE == "cuda" else None)
    return m

model = load_base()
LABEL_IDS_T = torch.tensor(LABEL_IDS, device=DEVICE)
# 只在语言模型的线性层上挂 LoRA，视觉塔保持不动
targets = sorted({n for n, mod in model.named_modules()
                  if isinstance(mod, nn.Linear) and not any(s in n for s in ("visual", "vision", "lm_head", "audio"))})
print("LoRA 层数:", len(targets), "例如", targets[:3])
ckpt_dir = f"{WORK}/lora"
state_file = f"{ckpt_dir}/train_state.json"
if os.path.exists(state_file):
    model = PeftModel.from_pretrained(model, ckpt_dir, is_trainable=True)
    done = json.load(open(state_file))["step"]
    print("从断点续训, 已完成 step", done)
else:
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, target_modules=targets))
    done = 0
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False

micro = batches(train_data, TOKENS_PER_BATCH)
steps, cur, n = [], [], 0
for b in micro:  # 每个优化 step 累积约 EXAMPLES_PER_STEP 条样本
    cur.append(b); n += len(b)
    if n >= EXAMPLES_PER_STEP:
        steps.append(cur); cur, n = [], 0
if cur:
    steps.append(cur)
total = len(steps)
print(f"{len(train_data)} 条训练样本, {len(micro)} 个 micro-batch, {total} 个优化 step", flush=True)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
for _ in range(done):
    sched.step()

model.train()
t0, run_loss, run_n = time.time(), 0.0, 0
for s in range(done, total):
    nex = sum(len(b) for b in steps[s])
    for idx in steps[s]:
        ids, mask, last = collate(train_data, idx, DEVICE)
        kmax = max(train_data[i][1] for i in idx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            lg = label_logits(model, ids, mask, last, kmax)
        loss = loss_fn(lg, [train_data[i][1] for i in idx], [train_data[i][2] for i in idx]) / nex
        loss.backward()
        run_loss += loss.item()  # 已按样本数归一化，累加后就是本 step 的平均每样本损失
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    run_n += 1
    if (s + 1) % LOG_EVERY == 0 or s == done:
        el = time.time() - t0
        eta = el / (s + 1 - done) * (total - s - 1)
        print(f"step {s+1}/{total}  loss/样本 {run_loss / run_n:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"已用 {el/60:.1f} 分钟, 预计还要 {eta/60:.1f} 分钟", flush=True)
        run_loss, run_n = 0.0, 0
    if (s + 1) % SAVE_EVERY == 0 or s + 1 == total:
        model.save_pretrained(ckpt_dir)
        json.dump({"step": s + 1, "total": total}, open(state_file, "w"))
print("训练完成", flush=True)
