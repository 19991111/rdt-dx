#!/usr/bin/env python3
"""Phase B: Bridge Ablation — 4-GPU parallel via torch.multiprocessing.

Each GPU process independently builds, trains, and evaluates one bridge variant.
Uses spawn-based multiprocessing for proper CUDA isolation.

Usage:
  source activate vllm_env && python phase_b_mp.py
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase_b_mp")

MODEL_PATH = "/data/model/Qwen/Qwen3___5-9B-Base"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "phase_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LAYER_SPLIT = {"prefix_end": 12, "core_end": 28}
K_SCHEDULE = [2, 4, 8, 16]
STEPS_PER_STAGE = 30

VARIANTS = [
    {"name": "mlp_step", "bridge_type": "mlp_step", "gpu": 4, "desc": "Bridge v2 (MLP+step)"},
    {"name": "none",     "bridge_type": "none",      "gpu": 5, "desc": "No Bridge"},
    {"name": "linear",   "bridge_type": "linear",    "gpu": 6, "desc": "Linear Bridge"},
    {"name": "mlp",      "bridge_type": "mlp",       "gpu": 7, "desc": "MLP (no step_emb)"},
]

TRAIN_TEXTS = [
    "患者: 我最近总是头疼，特别是下午，没有其他症状。\n医生: 建议注意休息，保持规律作息，如持续超过2周或加重建议门诊就诊。",
    "患者: 感冒了应该吃什么药？\n医生: 普通感冒为自限性疾病，建议多休息多饮水。如发热超过38.5℃可使用对乙酰氨基酚。",
    "患者: 最近两周胃痛，吃完饭更明显，有时反酸，晚上躺下加重。\n医生: 需考虑胃食管反流病或胃炎复发。建议消化内科门诊就诊。",
    "患者: 我父亲65岁，最近一个月间断性胸痛，活动后加重，休息可缓解，有高血压病史10年。\n医生: 高度疑似不稳定心绞痛，存在急性心肌梗死风险。建议立即前往急诊科。",
    "患者: 最近一个月瘦了8公斤，经常口渴，小便多，有时看东西模糊。\n医生: 需考虑糖尿病，建议尽快到内分泌科就诊，检查空腹血糖和糖化血红蛋白。",
]


# ---------------------------------------------------------------------------
# Worker function (runs in each spawned process)
# ---------------------------------------------------------------------------

def worker(variant: dict, result_dir: str):
    """Train and evaluate one bridge variant in a separate process."""
    name = variant["name"]
    bridge_type = variant["bridge_type"]
    gpu_id = variant["gpu"]
    device = torch.device(f"cuda:{gpu_id}")

    # Each process needs its own logger and imports
    import logging
    worker_logger = logging.getLogger(f"worker_{name}")
    worker_logger.info("[%s] Process started on GPU %d (bridge=%s)", name, gpu_id, bridge_type)

    try:
        # Lazy imports inside process
        from transformers import AutoTokenizer
        from src.rdt_fixed import create_rdt_fixed_16
        from src.train_bridge_warmup import compute_kl_loss, compute_smoothness_loss, ReferenceModelWrapper

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # Build RDT model
        worker_logger.info("[%s] Building model...", name)
        model = create_rdt_fixed_16(
            base_model_path=MODEL_PATH,
            num_iters=2,
            prefix_layers=LAYER_SPLIT["prefix_end"],
            core_layers=LAYER_SPLIT["core_end"] - LAYER_SPLIT["prefix_end"],
            lora_r=None,
            bridge_type=bridge_type,
            aggregation="last" if bridge_type == "none" else "last4_mean",
            device_map=None,
        ).to(device)

        # Freeze all but bridge
        model.freeze_core()
        model.freeze_suffix()
        for p in model.prefix_group.parameters():
            p.requires_grad = False

        bridge_params = list(model.bridge.parameters())
        is_trainable = len(bridge_params) > 0
        optimizer = torch.optim.AdamW(bridge_params, lr=2e-4, weight_decay=0.01) if is_trainable else None

        worker_logger.info("[%s] Model ready. Bridge params: %s, trainable: %s",
                          name, f"{sum(p.numel() for p in bridge_params):,}", is_trainable)

        # Reference model — use own GPU to avoid contention on GPU 3
        ref_gpu = gpu_id  # Each process uses its own GPU for reference
        ref_device = torch.device(f"cuda:{ref_gpu}")
        worker_logger.info("[%s] Loading reference model on cuda:%d...", name, ref_gpu)
        ref_model = ReferenceModelWrapper(MODEL_PATH, ref_device)

        # ---- Helper: compute diagnostics ----
        @torch.no_grad()
        def compute_diagnostics(k: int) -> Dict:
            model.eval()
            model.num_iters = k

            text = TRAIN_TEXTS[0]
            inputs = tokenizer.encode(text, return_tensors="pt").to(device)

            try:
                output = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    return_loop_outputs=True,
                )
            except Exception as e:
                # Reduce K if aggregation fails
                worker_logger.warning("[%s] K=%d forward failed: %s — retrying with K=%d",
                                     name, k, e, max(k, 4))
                model.num_iters = max(k, 4)
                output = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    return_loop_outputs=True,
                )
                k = max(k, 4)

            h_anchor = output.h_anchor
            loop_outputs = output.loop_outputs
            anchor_norm = h_anchor.norm(dim=-1).mean().item()

            drifts = []
            for h_t in loop_outputs:
                drift = (h_t - h_anchor).norm(dim=-1).mean().item() / (anchor_norm + 1e-8)
                drifts.append(round(drift, 6))

            last4_var = 0.0
            if len(loop_outputs) >= 4:
                last4 = torch.stack(loop_outputs[-4:], dim=0)
                last4_mean = last4.mean(dim=0)
                last4_var = ((last4 - last4_mean) ** 2).mean().item()

            # Top-5 overlap
            ref_inputs = inputs.to(ref_device)
            ref_out = ref_model.model(input_ids=ref_inputs)
            ref_logits = ref_out.logits.float()
            rdt_logits = output.logits.float()

            last_idx = -1
            ref_top5 = set(ref_logits[0, last_idx, :].topk(5).indices.tolist())
            rdt_top5 = set(rdt_logits[0, last_idx, :].topk(5).indices.tolist())
            top5 = len(ref_top5 & rdt_top5) / 5

            model.train()
            return {"k": k, "drifts": drifts, "last4_variance": round(last4_var, 8),
                    "top5_overlap": round(top5, 2), "drift_last": drifts[-1] if drifts else 0}

        # ---- Progressive K training ----
        all_diagnostics = {}
        model.train()

        for stage_k in K_SCHEDULE:
            stage_label = f"K={stage_k}"

            if not is_trainable:
                # No-bridge: diagnostics only
                diag = compute_diagnostics(stage_k)
                worker_logger.info("[%s] %s: top5=%d%% drift=%.4f",
                                  name, stage_label, int(diag["top5_overlap"]*100), diag["drift_last"])
                all_diagnostics[stage_label] = {"pre": diag, "post": diag, "avg_kl": 0.0}
                continue

            model.num_iters = stage_k

            # Pre-train diagnostic
            pre_diag = compute_diagnostics(stage_k)
            worker_logger.info("[%s] %s pre:  top5=%d%% drift=%.4f",
                              name, stage_label, int(pre_diag["top5_overlap"]*100), pre_diag["drift_last"])

            total_kl = 0.0
            for step in range(STEPS_PER_STAGE):
                text = TRAIN_TEXTS[step % len(TRAIN_TEXTS)]
                inputs = tokenizer.encode(text, return_tensors="pt").to(device)
                inputs_ref = inputs.to(ref_device)

                ref_logits = ref_model.get_logits(
                    inputs_ref, torch.ones_like(inputs_ref, device=ref_device)
                ).to(device)

                optimizer.zero_grad()
                output = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    return_loop_outputs=True,
                    core_no_grad=True,
                )

                kl_loss = compute_kl_loss(
                    output.logits.float(), ref_logits.float(),
                    torch.ones_like(inputs, device=device),
                )
                smooth_loss = compute_smoothness_loss(output.loop_outputs)
                loss = kl_loss + 2e-4 * smooth_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), 0.5)
                optimizer.step()

                total_kl += kl_loss.item()

                if step % 15 == 14 or step == 0:
                    worker_logger.info("[%s]   %s step %2d: KL=%.4f Smooth=%.6f",
                                      name, stage_label, step+1, kl_loss.item(), smooth_loss.item())

            avg_kl = total_kl / STEPS_PER_STAGE
            post_diag = compute_diagnostics(stage_k)
            worker_logger.info("[%s] %s post: top5=%d%% drift=%.4f avg_kl=%.4f",
                              name, stage_label, int(post_diag["top5_overlap"]*100),
                              post_diag["drift_last"], avg_kl)

            all_diagnostics[stage_label] = {"pre": pre_diag, "post": post_diag, "avg_kl": round(avg_kl, 4)}

        worker_logger.info("[%s] Training complete.", name)

        # ---- Final evaluation (K=16) ----
        model.num_iters = 16
        model.eval()

        eval_results = {}
        eval_texts = {
            "easy1": "患者: 我最近总是头疼，特别是下午。\n医生:",
            "easy2": "患者: 感冒了嗓子疼怎么办？\n医生:",
            "medium": "患者: 最近两周胃痛，吃完饭更明显，反酸，有胃炎病史。\n医生:",
            "hard": "患者: 我父亲65岁，最近一个月间断性胸痛，活动后加重，有高血压病史。\n医生:",
        }

        for diff, text in eval_texts.items():
            inputs = tokenizer.encode(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(input_ids=inputs, labels=inputs)
            eval_results[diff] = {"ce_loss": round(out.loss.item(), 4)}

        # Save result
        result = {
            "name": name, "bridge_type": bridge_type, "gpu": gpu_id, "status": "ok",
            "diagnostics": all_diagnostics, "evaluation": eval_results,
        }

        result_path = os.path.join(result_dir, f"result_{name}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        worker_logger.info("[%s] ✅ Complete. Result saved.", name)

    except Exception as e:
        import traceback
        worker_logger.error("[%s] ❌ FAILED: %s", name, e)
        traceback.print_exc()
        result = {"name": name, "bridge_type": bridge_type, "gpu": gpu_id,
                  "status": "failed", "error": str(e)}
        result_path = os.path.join(result_dir, f"result_{name}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[Dict]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Phase B: Bridge Ablation Report")
    lines.append(f"Date: 2026-06-02  |  Training: Progressive K {K_SCHEDULE}, {STEPS_PER_STAGE} steps/stage")
    lines.append("=" * 70)

    # 1. K=16 diagnostics
    lines.append("\n## 1. Final K=16 Diagnostics\n")
    lines.append("| Bridge | Top-5 Overlap | Drift (last) | Last4 Var | Avg KL |")
    lines.append("|--------|-------------|-------------|-----------|--------|")
    for r in sorted(results, key=lambda x: x["name"]):
        if r["status"] != "ok":
            lines.append(f"| {r['name']} | ❌ {r.get('error', 'FAILED')[:40]} | - | - | - |")
            continue
        k16 = r["diagnostics"].get("K=16", {}).get("post", {})
        lines.append(f"| {r['name']} | {k16.get('top5_overlap', 0)*100:.0f}% "
                     f"| {k16.get('drift_last', 0):.4f} "
                     f"| {k16.get('last4_variance', 0):.6f} "
                     f"| {r['diagnostics'].get('K=16', {}).get('avg_kl', 0):.4f} |")
    lines.append("")

    # 2. Drift evolution
    lines.append("\n## 2. Hidden Drift Evolution (post-train)\n")
    lines.append("| Bridge | K=2 | K=4 | K=8 | K=16 | Trend |")
    lines.append("|--------|-----|-----|-----|------|-------|")
    for r in sorted(results, key=lambda x: x["name"]):
        if r["status"] != "ok": continue
        drifts = []
        for k in K_SCHEDULE:
            post = r["diagnostics"].get(f"K={k}", {}).get("post", {})
            drifts.append(post.get("drift_last", 0))
        d = drifts
        if d[-1] < d[0] * 0.8: trend = "↓ converging"
        elif d[-1] > d[0] * 1.2: trend = "↑ diverging"
        else: trend = "→ stable"
        lines.append(f"| {r['name']} | {d[0]:.4f} | {d[1]:.4f} | {d[2]:.4f} | {d[3]:.4f} | {trend} |")
    lines.append("")

    # 3. Top-5 overlap evolution
    lines.append("\n## 3. Top-5 Overlap Evolution (post-train)\n")
    lines.append("| Bridge | K=2 | K=4 | K=8 | K=16 |")
    lines.append("|--------|-----|-----|-----|------|")
    for r in sorted(results, key=lambda x: x["name"]):
        if r["status"] != "ok": continue
        ov = []
        for k in K_SCHEDULE:
            post = r["diagnostics"].get(f"K={k}", {}).get("post", {})
            ov.append(f"{post.get('top5_overlap', 0)*100:.0f}%")
        lines.append(f"| {r['name']} | {ov[0]} | {ov[1]} | {ov[2]} | {ov[3]} |")
    lines.append("")

    # 4. Per-difficulty eval
    lines.append("\n## 4. Per-Difficulty Evaluation (K=16)\n")
    lines.append("| Bridge | Easy1 CE | Easy2 CE | Medium CE | Hard CE |")
    lines.append("|--------|----------|----------|-----------|---------|")
    for r in sorted(results, key=lambda x: x["name"]):
        if r["status"] != "ok": continue
        ev = r["evaluation"]
        lines.append(f"| {r['name']} "
                     f"| {ev.get('easy1', {}).get('ce_loss', 'N/A')} "
                     f"| {ev.get('easy2', {}).get('ce_loss', 'N/A')} "
                     f"| {ev.get('medium', {}).get('ce_loss', 'N/A')} "
                     f"| {ev.get('hard', {}).get('ce_loss', 'N/A')} |")
    lines.append("")

    # 5. Key findings
    lines.append("\n## 5. Key Findings\n")
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best_ov = max(ok, key=lambda r: r["diagnostics"].get("K=16", {}).get("post", {}).get("top5_overlap", 0))
        best_dr = min(ok, key=lambda r: r["diagnostics"].get("K=16", {}).get("post", {}).get("drift_last", 999))
        lines.append(f"- **Best Top-5 overlap (K=16)**: {best_ov['name']} ({best_ov['diagnostics']['K=16']['post']['top5_overlap']*100:.0f}%)")
        lines.append(f"- **Best drift control (K=16)**: {best_dr['name']} (drift={best_dr['diagnostics']['K=16']['post']['drift_last']:.4f})")

        nb = [r for r in ok if r["name"] == "none"]
        dv = [r for r in ok if r["name"] == "mlp_step"]
        if nb and dv:
            nbd = nb[0]["diagnostics"]["K=16"]["post"]["drift_last"]
            dvd = dv[0]["diagnostics"]["K=16"]["post"]["drift_last"]
            nbo = nb[0]["diagnostics"]["K=16"]["post"]["top5_overlap"]
            dvo = dv[0]["diagnostics"]["K=16"]["post"]["top5_overlap"]
            lines.append(f"- **Bridge v2 vs No-Bridge**: drift {nbd:.4f}→{dvd:.4f}, overlap {nbo*100:.0f}%→{dvo*100:.0f}%")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report at {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    logger.info("=" * 70)
    logger.info("PHASE B: Bridge Ablation — 4-GPU torch.multiprocessing")
    logger.info("Variants: %s", [v["name"] for v in VARIANTS])
    logger.info("=" * 70)

    # Temp dir for per-process results
    result_dir = OUTPUT_DIR / "worker_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous results
    for f in result_dir.glob("result_*.json"):
        f.unlink()

    t0 = time.time()

    # Spawn one process per variant
    processes = []
    for v in VARIANTS:
        p = mp.Process(target=worker, args=(v, str(result_dir)))
        p.start()
        processes.append(p)
        logger.info("Spawned process for [%s] on GPU %d (PID %d)", v["name"], v["gpu"], p.pid)

    # Wait for all
    for p in processes:
        p.join()

    elapsed = time.time() - t0
    logger.info("All processes completed in %.1f s (%.1f min)", elapsed, elapsed/60)

    # Collect results
    results = []
    for v in VARIANTS:
        rpath = result_dir / f"result_{v['name']}.json"
        if rpath.exists():
            with open(rpath) as f:
                results.append(json.load(f))
        else:
            results.append({"name": v["name"], "status": "failed", "error": "no result file"})

    results.sort(key=lambda r: r["name"])

    # Report
    report = generate_report(results)
    print("\n" + report)

    report_path = OUTPUT_DIR / "phase_b_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    data_path = OUTPUT_DIR / "phase_b_data.json"
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Reports saved to %s", OUTPUT_DIR)

    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        logger.error("%d failed: %s", len(failures), [f["name"] for f in failures])
        sys.exit(1)
    else:
        logger.info("✅ All %d variants passed!", len(results))


if __name__ == "__main__":
    main()
