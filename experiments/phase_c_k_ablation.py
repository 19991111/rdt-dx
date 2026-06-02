#!/usr/bin/env python3
"""Phase C: K-Step Ablation — RDT-Fixed-K vs Baseline SFT.

Uses torch.multiprocessing for 4-GPU parallel training.
GPU 4: RDT-Fixed-16 (mlp bridge)
GPU 5: RDT-Fixed-8  (mlp bridge)
GPU 6: RDT-Fixed-4  (mlp bridge)
GPU 7: Baseline LoRA SFT (no loop, same trainable param budget)

Each RDT variant: progressive K bridge warmup (2→target_K, 20 steps/stage)
                   + short SFT at target K (30 steps CE+KL)
Baseline:           plain LoRA SFT on Qwen3.5-9B (50 steps CE)

Evaluation: CE loss per difficulty (easy/medium/hard)

Usage: source activate vllm_env && python phase_c_k_ablation.py
"""

import gc, json, logging, os, sys, time, math
from pathlib import Path
from typing import Dict, List, Optional
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase_c")

MODEL_PATH = "/data/model/Qwen/Qwen3___5-9B-Base"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "phase_c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LAYER_SPLIT = {"prefix_end": 12, "core_end": 28}
BRIDGE_TYPE = "mlp"  # Best performer from Phase B
WARMUP_STEPS = 20
SFT_STEPS = 30

VARIANTS = [
    {"name": "K16", "final_k": 16, "gpu": 4, "kind": "rdt",
     "desc": "RDT-Fixed-16 (mlp bridge, progressive warmup 2→4→8→16)"},
    {"name": "K8",  "final_k": 8,  "gpu": 5, "kind": "rdt",
     "desc": "RDT-Fixed-8 (mlp bridge, progressive warmup 2→4→8)"},
    {"name": "K4",  "final_k": 4,  "gpu": 6, "kind": "rdt",
     "desc": "RDT-Fixed-4 (mlp bridge, progressive warmup 2→4)"},
    {"name": "SFT", "final_k": 0,  "gpu": 7, "kind": "sft",
     "desc": "Baseline LoRA SFT (r=64, no loop)"},
]

TRAIN_TEXTS = [
    "患者: 我最近总是头疼，特别是下午，没有其他症状。\n医生: 建议注意休息，保持规律作息，如持续超过2周或加重建议门诊就诊。",
    "患者: 感冒了应该吃什么药？\n医生: 普通感冒为自限性疾病，建议多休息多饮水。如发热超过38.5℃可使用对乙酰氨基酚。不建议自行使用抗生素。",
    "患者: 最近两周胃痛，吃完饭更明显，有时反酸，晚上躺下加重，有胃炎病史。\n医生: 需考虑胃食管反流病或胃炎复发。建议消化内科门诊就诊。调整饮食，避免辛辣油腻。",
    "患者: 我父亲65岁，最近一个月间断性胸痛，活动后加重，休息可缓解，有高血压病史10年。\n医生: 高度疑似不稳定心绞痛，存在急性心肌梗死风险。建议立即前往急诊科，进行心电图和心肌酶谱检查。这是紧急情况。",
    "患者: 最近一个月瘦了8公斤，经常口渴，小便多，有时看东西模糊。\n医生: 需考虑糖尿病，建议尽快到内分泌科就诊，检查空腹血糖和糖化血红蛋白。",
]

EVAL_TEXTS = {
    "easy1":  "患者: 我最近总是头疼，特别是下午，没有其他症状。\n医生:",
    "easy2":  "患者: 感冒了嗓子疼怎么办？\n医生:",
    "medium": "患者: 最近两周胃痛，吃完饭更明显，反酸，有胃炎病史。\n医生:",
    "hard":   "患者: 我父亲65岁，最近一个月间断性胸痛，活动后加重，有高血压病史。\n医生:",
}


def worker(variant: dict, result_dir: str):
    name = variant["name"]
    gpu = variant["gpu"]
    device = torch.device(f"cuda:{gpu}")

    wlog = logging.getLogger(f"worker_{name}")
    wlog.info("[%s] Started on GPU %d (kind=%s)", name, gpu, variant["kind"])

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        if variant["kind"] == "rdt":
            result = _train_rdt(variant, device, tokenizer, wlog)
        else:
            result = _train_baseline(device, tokenizer, wlog)

        result_path = os.path.join(result_dir, f"result_{name}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        wlog.info("[%s] ✅ Saved.", name)

    except Exception as e:
        import traceback
        wlog.error("[%s] ❌ FAILED: %s", name, e)
        traceback.print_exc()
        result = {"name": name, "status": "failed", "error": str(e)}
        result_path = os.path.join(result_dir, f"result_{name}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)


def _train_rdt(variant: dict, device: torch.device, tokenizer, wlog) -> Dict:
    """Train RDT-Fixed-K: bridge warmup + short SFT."""
    from src.rdt_fixed import create_rdt_fixed_16
    from src.train_bridge_warmup import compute_kl_loss, compute_smoothness_loss, ReferenceModelWrapper

    name = variant["name"]
    target_k = variant["final_k"]
    bridge_type = BRIDGE_TYPE

    # Progressive K schedule: 2 → 4 → 8 → ... → target_k
    k_schedule = [2]
    while k_schedule[-1] * 2 <= target_k:
        k_schedule.append(k_schedule[-1] * 2)

    wlog.info("[%s] K schedule: %s, warmup=%d steps, sft=%d steps",
              name, k_schedule, WARMUP_STEPS, SFT_STEPS)

    # Build model (start at K=2)
    wlog.info("[%s] Building RDT model...", name)
    # Start with "last" aggregation since K=2 < 4; switch to last4_mean at K>=4
    model = create_rdt_fixed_16(
        base_model_path=MODEL_PATH, num_iters=2,
        prefix_layers=LAYER_SPLIT["prefix_end"],
        core_layers=LAYER_SPLIT["core_end"] - LAYER_SPLIT["prefix_end"],
        lora_r=None, bridge_type=bridge_type, aggregation="last",
        device_map=None,
    ).to(device)

    model.freeze_core(); model.freeze_suffix()
    for p in model.prefix_group.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW(list(model.bridge.parameters()), lr=2e-4, weight_decay=0.01)

    # Reference model (same GPU)
    wlog.info("[%s] Loading reference model...", name)
    ref_model = ReferenceModelWrapper(MODEL_PATH, device)

    # === Phase 1: Bridge Warmup ===
    wlog.info("[%s] === Bridge Warmup ===", name)
    warmup_log = {}
    model.train()

    for stage_k in k_schedule:
        model.num_iters = stage_k
        stage_label = f"K={stage_k}"
        total_kl = 0.0

        for step in range(WARMUP_STEPS):
            text = TRAIN_TEXTS[step % len(TRAIN_TEXTS)]
            inputs = tokenizer.encode(text, return_tensors="pt").to(device)
            inputs_ref = inputs.to(device)
            ref_logits = ref_model.get_logits(inputs_ref, torch.ones_like(inputs_ref, device=device)).to(device)

            optimizer.zero_grad()
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                          return_loop_outputs=True, core_no_grad=True)
            kl = compute_kl_loss(output.logits.float(), ref_logits.float(),
                                torch.ones_like(inputs, device=device))
            smooth = compute_smoothness_loss(output.loop_outputs)
            (kl + 2e-4 * smooth).backward()
            torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), 0.5)
            optimizer.step()
            total_kl += kl.item()

            if step == 0 or step == WARMUP_STEPS - 1:
                wlog.info("[%s]   warmup %s step %d: KL=%.4f Smooth=%.6f",
                          name, stage_label, step+1, kl.item(), smooth.item())

        avg_kl = total_kl / WARMUP_STEPS
        warmup_log[stage_label] = round(avg_kl, 4)
        wlog.info("[%s]   %s complete: avg_kl=%.4f", name, stage_label, avg_kl)

    # === Phase 2: Short SFT at target K ===
    wlog.info("[%s] === SFT Phase (K=%d) ===", name, target_k)
    model.num_iters = target_k
    # Switch to last4_mean aggregation when K >= 4
    if target_k >= 4:
        from src.aggregation import get_aggregation
        model.aggregation_fn = get_aggregation("last4_mean")
    model.unfreeze_core(); model.unfreeze_suffix()

    # Separate optimizers for bridge and lora
    bridge_p = list(model.bridge.parameters())
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "bridge" not in n]
    sft_optimizer = torch.optim.AdamW([
        {"params": bridge_p, "lr": 2e-4},
        {"params": lora_p, "lr": 5e-5},
    ], weight_decay=0.01)

    sft_log = []
    for step in range(SFT_STEPS):
        text = TRAIN_TEXTS[step % len(TRAIN_TEXTS)]
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        inputs_ref = inputs.to(device)

        with torch.no_grad():
            ref_logits = ref_model.get_logits(inputs_ref, torch.ones_like(inputs_ref, device=device)).to(device)

        sft_optimizer.zero_grad()
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                      labels=inputs, return_loop_outputs=True)

        # CE + KL + Smooth
        ce = output.loss
        kl = compute_kl_loss(output.logits.float(), ref_logits.float(),
                            torch.ones_like(inputs, device=device))
        smooth = compute_smoothness_loss(output.loop_outputs)
        loss = ce + 0.1 * kl + 2e-4 * smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_optimizer.step()

        sft_log.append({"step": step+1, "ce": round(ce.item(), 4), "kl": round(kl.item(), 4)})

        if step % 10 == 0 or step == SFT_STEPS - 1:
            wlog.info("[%s]   sft step %d: CE=%.4f KL=%.4f", name, step+1, ce.item(), kl.item())

    # === Evaluation ===
    wlog.info("[%s] === Evaluation (K=%d) ===", name, target_k)
    model.eval()
    eval_results = {}

    @torch.no_grad()
    def eval_one(text: str, label: str):
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        out = model(input_ids=inputs, labels=inputs)
        ce_val = out.loss.item()
        n_tok = max(inputs.shape[1] - 1, 1)
        ppl = math.exp(min(ce_val, 100))
        eval_results[label] = {"ce": round(ce_val, 4), "ppl": round(ppl, 2), "n_tokens": n_tok}
        return ce_val

    for diff, text in EVAL_TEXTS.items():
        eval_one(text, diff)

    del model, ref_model
    torch.cuda.empty_cache()

    return {"name": name, "kind": "rdt", "final_k": target_k, "status": "ok",
            "warmup": warmup_log, "sft_log": sft_log, "eval": eval_results}


def _train_baseline(device: torch.device, tokenizer, wlog) -> Dict:
    """Train baseline LoRA SFT on Qwen3.5-9B (no RDT loop)."""
    from transformers import AutoModelForCausalLM
    from src.rdt_fixed import LoRALinear, _inject_lora

    wlog.info("[SFT] Loading Qwen3.5-9B for LoRA SFT...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16, device_map=device,
    )

    # Apply LoRA with same rank as RDT variants (r=64)
    lang = base_model.model
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                      "in_proj_qkv", "in_proj_z", "gate_proj", "up_proj", "down_proj"]

    # Inject LoRA and immediately move to device (LoRA params default to CPU)
    for layer in lang.layers:
        _inject_lora(layer, target_modules, r=64, alpha=128, dropout=0.05)

    # Move any new LoRA params to GPU
    def _move_lora(m: nn.Module):
        for child in m.children():
            if isinstance(child, LoRALinear):
                child.lora_A = nn.Parameter(child.lora_A.data.to(device))
                child.lora_B = nn.Parameter(child.lora_B.data.to(device))
            _move_lora(child)
    _move_lora(base_model)

    # Count trainable params
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    wlog.info("[SFT] LoRA injected. Trainable: %s / %s (%.2f%%)",
              f"{trainable:,}", f"{total:,}", trainable/total*100)

    base_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=0.01,
    )

    # Train same total steps as RDT variants (SFT_STEPS)
    sft_log = []
    for step in range(SFT_STEPS):
        text = TRAIN_TEXTS[step % len(TRAIN_TEXTS)]
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)

        optimizer.zero_grad()
        out = base_model(input_ids=inputs, labels=inputs)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
        optimizer.step()

        sft_log.append({"step": step+1, "ce": round(out.loss.item(), 4)})

        if step % 10 == 0 or step == SFT_STEPS - 1:
            wlog.info("[SFT]   step %d: CE=%.4f", step+1, out.loss.item())

    # Evaluation
    wlog.info("[SFT] === Evaluation ===")
    base_model.eval()
    eval_results = {}

    @torch.no_grad()
    def eval_one(text: str, label: str):
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        out = base_model(input_ids=inputs, labels=inputs)
        ce_val = out.loss.item()
        n_tok = max(inputs.shape[1] - 1, 1)
        ppl = math.exp(min(ce_val, 100))
        eval_results[label] = {"ce": round(ce_val, 4), "ppl": round(ppl, 2), "n_tokens": n_tok}

    for diff, text in EVAL_TEXTS.items():
        eval_one(text, diff)

    del base_model
    torch.cuda.empty_cache()

    return {"name": "SFT", "kind": "sft", "final_k": 0, "status": "ok",
            "trainable_params": trainable, "trainable_pct": round(trainable/total*100, 2),
            "sft_log": sft_log, "eval": eval_results}


# === Report ===

def generate_report(results: List[Dict]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Phase C: K-Step Ablation Report")
    lines.append(f"Date: 2026-06-02  |  Bridge: {BRIDGE_TYPE}  |  Warmup: {WARMUP_STEPS}/stage  |  SFT: {SFT_STEPS} steps")
    lines.append("=" * 70)

    # 1. Training summary
    lines.append("\n## 1. Training Summary\n")
    lines.append("| Variant | Kind | K | Trainable Params | Final CE (train) |")
    lines.append("|---------|------|---|-----------------|-----------------|")
    for r in sorted(results, key=lambda x: x.get("final_k", 0), reverse=True):
        if r["status"] != "ok":
            lines.append(f"| {r['name']} | ❌ | - | - | - |")
            continue
        k = r.get("final_k", 0)
        kind = r.get("kind", "?")
        sft = r.get("sft_log", [])
        final_ce = sft[-1]["ce"] if sft else "N/A"
        tp = r.get("trainable_params", "N/A")
        if isinstance(tp, int): tp = f"{tp:,}"
        lines.append(f"| {r['name']} | {kind} | {k} | {tp} | {final_ce} |")
    lines.append("")

    # 2. Per-difficulty evaluation (THE KEY TABLE)
    lines.append("\n## 2. Per-Difficulty Evaluation (CE Loss / PPL)\n")
    lines.append("| Variant | K | Easy1 CE | Easy2 CE | Medium CE | Hard CE | Avg CE |")
    lines.append("|---------|---|----------|----------|-----------|---------|--------|")
    for r in sorted(results, key=lambda x: x.get("final_k", 0), reverse=True):
        if r["status"] != "ok":
            lines.append(f"| {r['name']} | - | - | - | - | - | - |")
            continue
        ev = r.get("eval", {})
        ces = [ev.get(d, {}).get("ce", float("nan")) for d in ["easy1", "easy2", "medium", "hard"]]
        avg = sum(c for c in ces if not math.isnan(c)) / max(1, sum(1 for c in ces if not math.isnan(c)))
        lines.append(f"| {r['name']} | {r.get('final_k', 0)} | {ces[0]} | {ces[1]} | {ces[2]} | {ces[3]} | {avg:.4f} |")
    lines.append("")

    # 3. Key analysis
    lines.append("\n## 3. Step Benefit Analysis\n")
    ok_rdt = [r for r in results if r["status"] == "ok" and r.get("kind") == "rdt"]
    ok_sft = [r for r in results if r["status"] == "ok" and r.get("kind") == "sft"]

    if len(ok_rdt) >= 2 and ok_sft:
        sft_ev = ok_sft[0].get("eval", {})
        sft_hard = sft_ev.get("hard", {}).get("ce", float("nan"))
        sft_easy = sft_ev.get("easy1", {}).get("ce", float("nan"))

        lines.append("| Variant | K | Easy Δ vs SFT | Hard Δ vs SFT | Hard/Easy Benefit Ratio |")
        lines.append("|---------|---|--------------|--------------|------------------------|")
        for r in sorted(ok_rdt, key=lambda x: x.get("final_k", 0)):
            ev = r.get("eval", {})
            k = r.get("final_k", 0)
            easy_ce = ev.get("easy1", {}).get("ce", float("nan"))
            hard_ce = ev.get("hard", {}).get("ce", float("nan"))

            easy_delta = easy_ce - sft_easy if not math.isnan(easy_ce) and not math.isnan(sft_easy) else float("nan")
            hard_delta = hard_ce - sft_hard if not math.isnan(hard_ce) and not math.isnan(sft_hard) else float("nan")

            # Positive delta = RDT is better (lower CE)
            ratio = abs(hard_delta / max(abs(easy_delta), 1e-8)) if easy_delta != 0 else float("inf")

            lines.append(f"| {r['name']} | {k} | {easy_delta:+.4f} | {hard_delta:+.4f} | {ratio:.1f}x |")

        lines.append("")
        lines.append("_Δ = SFT_CE - RDT_CE. Positive = RDT better (lower loss). Benefit Ratio > 1 = RDT helps hard cases more than easy._")

    # 4. Key findings
    lines.append("\n## 4. Key Findings\n")
    if ok_rdt:
        # Does more K help?
        ks = sorted([r["final_k"] for r in ok_rdt])
        hard_ces = []
        easy_ces = []
        for r in sorted(ok_rdt, key=lambda x: x["final_k"]):
            ev = r.get("eval", {})
            hard_ces.append(ev.get("hard", {}).get("ce", float("nan")))
            easy_ces.append(ev.get("easy1", {}).get("ce", float("nan")))

        hard_trend = "↓ improving" if all(hard_ces[i] >= hard_ces[i+1] for i in range(len(hard_ces)-1)) else "mixed"
        easy_trend = "→ stable" if max(easy_ces) - min(easy_ces) < 0.5 else "variable"

        lines.append(f"- **Hard CE trend with K**: {hard_trend} ({hard_ces})")
        lines.append(f"- **Easy CE trend with K**: {easy_trend} ({easy_ces})")

        if ok_sft:
            best_rdt = min(ok_rdt, key=lambda r: r.get("eval", {}).get("hard", {}).get("ce", 999))
            rdt_hard = best_rdt["eval"]["hard"]["ce"]
            sft_hard = ok_sft[0]["eval"]["hard"]["ce"]
            lines.append(f"- **Best RDT hard CE**: {rdt_hard:.4f} (K={best_rdt['final_k']}) vs SFT: {sft_hard:.4f} "
                         f"→ RDT is {'better' if rdt_hard < sft_hard else 'worse'} by {abs(rdt_hard-sft_hard):.4f}")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report at {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(lines)


# === Main ===

def main():
    import torch.multiprocessing as mp
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    logger.info("=" * 70)
    logger.info("PHASE C: K-Step Ablation — RDT-Fixed-K vs Baseline SFT")
    logger.info("Variants: %s", [v["name"] for v in VARIANTS])
    logger.info("=" * 70)

    result_dir = OUTPUT_DIR / "worker_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for f in result_dir.glob("result_*.json"): f.unlink()

    t0 = time.time()
    processes = []
    for v in VARIANTS:
        p = mp.Process(target=worker, args=(v, str(result_dir)))
        p.start(); processes.append(p)
        logger.info("Spawned [%s] on GPU %d (PID %d)", v["name"], v["gpu"], p.pid)

    for p in processes: p.join()
    elapsed = time.time() - t0
    logger.info("All done in %.1f s (%.1f min)", elapsed, elapsed/60)

    # Collect results
    results = []
    for v in VARIANTS:
        rp = result_dir / f"result_{v['name']}.json"
        if rp.exists():
            with open(rp) as f: results.append(json.load(f))
        else:
            results.append({"name": v["name"], "status": "failed", "error": "no result"})

    results.sort(key=lambda r: r.get("final_k", 0), reverse=True)

    report = generate_report(results)
    print("\n" + report)

    for pth in [OUTPUT_DIR / "phase_c_report.md", OUTPUT_DIR / "phase_c_data.json"]:
        with open(pth, "w") as f:
            f.write(report if pth.suffix == ".md" else json.dumps(results, indent=2, default=str))
    logger.info("Reports saved to %s", OUTPUT_DIR)

    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        logger.error("%d failed: %s", len(failures), [f["name"] for f in failures])
        sys.exit(1)


if __name__ == "__main__":
    main()
