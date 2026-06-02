"""Phase 2: RDT-Fixed-16 SFT training.

Trains the full RDT-Fixed-16 model (Bridge + Core LoRA + Suffix LoRA) on
medical SFT data. The goal is to make the 16-step recursive loop structure
serve medical task performance, improving complex case reasoning while
maintaining stability on easy cases.

Training objectives::

    L = ce_weight * L_CE + kl_ref_weight * L_KL + smooth_weight * L_smooth

Where:
    - ``L_CE``: Standard next-token cross-entropy on medical QA data.
    - ``L_KL``: KL divergence to reference model (prevents distribution drift).
    - ``L_smooth``: Step-to-step hidden state smoothness penalty.

Usage::

    python -m src.train_rdt_fixed \\
        --model_path /data/model/Qwen/Qwen3___5-9B-Base \\
        --data_path data/train.jsonl \\
        --output_dir output/rdt_sft \\
        --num_iters 16 --lora_r 64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rdt_fixed import RDTFixed16, create_rdt_fixed_16
from src.data_utils import (
    MedicalDataset,
    collate_fn,
    create_dataloader,
    generate_synthetic_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite loss function
# ---------------------------------------------------------------------------


def compute_rdt_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logits_ref: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    loop_outputs: Optional[List[torch.Tensor]] = None,
    ce_weight: float = 1.0,
    kl_ref_weight: float = 0.1,
    smooth_weight: float = 2e-4,
    kl_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the combined RDT SFT loss.

    Computes::

        L = ce_weight * L_CE + kl_ref_weight * L_KL + smooth_weight * L_smooth

    Args:
        logits: RDT model output logits of shape ``(B, S, V)``.
        labels: Token labels of shape ``(B, S)``. Positions with ``-100`` are
            ignored in the CE loss.
        logits_ref: Reference model logits for KL divergence. If ``None`` and
            ``kl_ref_weight > 0``, the KL term is skipped.
        attention_mask: Mask of shape ``(B, S)`` for KL loss masking.
        loop_outputs: Per-step hidden states ``[h_1, ..., h_K]`` for
            smoothness regularization. If ``None`` or fewer than 2 outputs,
            the smoothness term is skipped.
        ce_weight: Weight for cross-entropy loss. Defaults to 1.0.
        kl_ref_weight: Weight for KL divergence to reference. Defaults to 0.1.
        smooth_weight: Weight for smoothness regularization. Defaults to
            ``2e-4``.
        kl_temperature: Temperature for KL softmax. Defaults to 1.0.

    Returns:
        Tuple of ``(total_loss, metrics_dict)`` where ``metrics_dict`` contains
        ``"ce_loss"``, ``"kl_loss"`` (if applicable), ``"smooth_loss"``
        (if applicable), and ``"total_loss"``.
    """
    metrics: Dict[str, float] = {}

    # Cross-entropy loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    metrics["ce_loss"] = ce_loss.item()
    total = ce_weight * ce_loss

    # KL divergence to reference model
    if logits_ref is not None and kl_ref_weight > 0:
        log_probs_rdt = F.log_softmax(logits / kl_temperature, dim=-1)
        log_probs_ref = F.log_softmax(logits_ref / kl_temperature, dim=-1)
        probs_ref = F.softmax(logits_ref / kl_temperature, dim=-1)
        kl_per_token = (probs_ref * (log_probs_ref - log_probs_rdt)).sum(dim=-1)

        if attention_mask is not None:
            kl_per_token = kl_per_token * attention_mask.float()

        kl_loss = kl_per_token.sum() / (
            attention_mask.sum() if attention_mask is not None else kl_per_token.numel()
        )
        metrics["kl_loss"] = kl_loss.item()
        total = total + kl_ref_weight * kl_loss

    # Smoothness regularization
    if loop_outputs is not None and smooth_weight > 0 and len(loop_outputs) >= 2:
        smooth_loss = 0.0
        for t in range(1, len(loop_outputs)):
            delta = loop_outputs[t] - loop_outputs[t - 1]
            smooth_loss += (delta ** 2).mean()
        smooth_loss = smooth_loss / (len(loop_outputs) - 1)
        metrics["smooth_loss"] = smooth_loss.item()
        total = total + smooth_weight * smooth_loss

    metrics["total_loss"] = total.item()
    return total, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class RDTFixedTrainer:
    """Trainer for Phase 2: RDT-Fixed-16 SFT.

    Trains Bridge + Core LoRA + Suffix LoRA on medical SFT data. Supports
    loading a Bridge Warmup checkpoint for initialization and separate
    learning rates for Bridge and LoRA parameter groups.

    Attributes:
        model: ``RDTFixed16`` model (all trainable components unfrozen).
        ref_model: Optional reference model for KL regularization.
        optimizer: AdamW with separate LR groups for Bridge and LoRA.
        scheduler: Learning rate scheduler with warmup + cosine decay.
        device: Primary training device.
        config: Training configuration dict.
    """

    def __init__(
        self,
        model: RDTFixed16,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        config: dict,
        ref_model: Optional[nn.Module] = None,
    ):
        """Initialize the RDT SFT trainer.

        Args:
            model: RDT-Fixed-16 model with all trainable components unfrozen.
            optimizer: Optimizer (should have separate LR groups for Bridge
                and LoRA parameters).
            scheduler: Optional LR scheduler.
            device: Training device.
            config: Dict with keys ``ce_weight``, ``kl_ref_weight``,
                ``smooth_weight``, ``max_grad_norm``, ``use_amp``,
                ``log_dir``, and ``gradient_accumulation_steps``.
            ref_model: Optional reference model for KL loss. If provided,
                should be in eval mode with frozen parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.ref_model = ref_model

        self.writer = SummaryWriter(log_dir=config.get("log_dir", "./logs"))
        self.global_step = 0
        self.scaler = GradScaler(enabled=config.get("use_amp", True))
        self.use_amp = config.get("use_amp", True)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number (1-indexed).
            eval_dataloader: Optional evaluation data loader for per-epoch
                validation metrics.

        Returns:
            Dict of average metrics for the epoch, with optional ``"eval"``
            sub-dict containing validation metrics.
        """
        self.model.train()

        total_metrics: Dict[str, float] = {}
        total_batches = 0

        ce_weight = self.config.get("ce_weight", 1.0)
        kl_ref_weight = self.config.get("kl_ref_weight", 0.1)
        smooth_weight = self.config.get("smooth_weight", 2e-4)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            ref_logits = None
            if kl_ref_weight > 0 and self.ref_model is not None:
                with torch.no_grad():
                    ref_output = self.ref_model(
                        input_ids=input_ids, attention_mask=attention_mask,
                    )
                    ref_logits = ref_output.logits.float()

            self.optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=self.use_amp):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loop_outputs=(smooth_weight > 0),
                )

                loss, metrics = compute_rdt_loss(
                    logits=output.logits.float(),
                    labels=labels,
                    logits_ref=ref_logits,
                    attention_mask=attention_mask,
                    loop_outputs=output.loop_outputs if smooth_weight > 0 else None,
                    ce_weight=ce_weight,
                    kl_ref_weight=kl_ref_weight,
                    smooth_weight=smooth_weight,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            total_batches += 1

            for k, v in metrics.items():
                self.writer.add_scalar(f"rdt_sft/{k}", v, self.global_step)
            self.global_step += 1

            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                lr_bridge = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss: %.4f | CE: %.4f | LR: %.2e",
                    epoch, batch_idx, len(dataloader),
                    metrics.get("total_loss", 0), metrics.get("ce_loss", 0), lr_bridge,
                )

        avg_metrics = {k: v / total_batches for k, v in total_metrics.items()}
        logger.info("Epoch %d complete: %s", epoch, avg_metrics)

        if eval_dataloader is not None:
            eval_metrics = self.evaluate(eval_dataloader)
            avg_metrics["eval"] = eval_metrics
            logger.info("Epoch %d eval: %s", epoch, eval_metrics)

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a validation set.

        Computes token-averaged CE loss and perplexity over the entire
        validation loader.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict with keys ``"ce_loss"`` and ``"perplexity"``.
        """
        self.model.eval()

        total_ce_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
            )

            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

            n_tokens = (shift_labels != -100).sum().item()
            total_ce_loss += ce.item()
            total_tokens += n_tokens

        avg_ce = total_ce_loss / total_tokens if total_tokens > 0 else 0.0
        ppl = math.exp(avg_ce) if avg_ce < 100 else float("inf")

        self.model.train()
        return {"ce_loss": avg_ce, "perplexity": ppl}

    def save_checkpoint(
        self,
        output_dir: str | Path,
        epoch: Optional[int] = None,
        is_best: bool = False,
    ) -> None:
        """Save a training checkpoint.

        Args:
            output_dir: Directory to save to.
            epoch: Current epoch number (appended to filename).
            is_best: If ``True``, also save as ``"best_model.pt"``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_epoch{epoch}" if epoch is not None else ""
        filename = f"rdt_sft{suffix}.pt"
        path = output_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

        if is_best:
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("Best model saved to %s", best_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RDT SFT training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="RDT-Fixed-16 SFT Training (Phase 2)"
    )

    parser.add_argument("--model_path", type=str,
                        default="/data/model/Qwen/Qwen3___5-9B-Base")
    parser.add_argument("--bridge_warmup_ckpt", type=str, default=None)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--train_data", type=str, default="data/synthetic_train.jsonl")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--use_synthetic_data", action="store_true", default=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--bridge_lr", type=float, default=2e-4)
    parser.add_argument("--lora_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kl_ref_weight", type=float, default=0.1)
    parser.add_argument("--smooth_weight", type=float, default=2e-4)
    parser.add_argument("--num_iters", type=int, default=16)
    parser.add_argument("--bridge_type", type=str, default="mlp_step")
    parser.add_argument("--aggregation", type=str, default="last4_mean")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="output/rdt_sft")
    parser.add_argument("--log_dir", type=str, default="output/logs")

    return parser.parse_args()


def main() -> None:
    """Main entry point for Phase 2 RDT SFT training."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Configuration: %s", json.dumps(vars(args), indent=2, default=str))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare data
    train_path = Path(args.train_data)
    if not train_path.exists():
        if args.use_synthetic_data:
            generate_synthetic_data(train_path, num_easy=3, num_medium=3, num_hard=4)
        else:
            raise FileNotFoundError(f"Training data not found: {train_path}")

    train_loader = create_dataloader(
        data_path=train_path, tokenizer=tokenizer,
        batch_size=args.batch_size, max_length=args.max_length, shuffle=True,
    )

    eval_loader = None
    if args.eval_data:
        eval_path = Path(args.eval_data)
        if eval_path.exists():
            eval_loader = create_dataloader(
                data_path=eval_path, tokenizer=tokenizer,
                batch_size=args.batch_size, max_length=args.max_length, shuffle=False,
            )

    # Build RDT model
    model = create_rdt_fixed_16(
        base_model_path=args.model_path, num_iters=args.num_iters,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bridge_type=args.bridge_type,
        aggregation=args.aggregation, device_map=None,
    ).to(device)

    if args.bridge_warmup_ckpt:
        ckpt = torch.load(args.bridge_warmup_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("Loaded bridge warmup checkpoint from step %d", ckpt.get("global_step", 0))

    model.unfreeze_core()
    model.unfreeze_suffix()

    # Optimizer with separate LR groups
    param_groups = model.get_parameter_groups(
        bridge_lr=args.bridge_lr, lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # LR scheduler: warmup + cosine decay
    total_steps = args.num_epochs * len(train_loader) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Reference model for KL loss
    ref_model = None
    if args.kl_ref_weight > 0:
        from transformers import AutoModelForCausalLM
        ref_path = args.ref_model_path or args.model_path
        ref_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path, trust_remote_code=True, dtype=torch.bfloat16,
            device_map=ref_device,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    config = {
        "ce_weight": args.ce_weight,
        "kl_ref_weight": args.kl_ref_weight,
        "smooth_weight": args.smooth_weight,
        "max_grad_norm": args.max_grad_norm,
        "use_amp": args.use_amp and device.type == "cuda",
        "log_dir": args.log_dir,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    trainer = RDTFixedTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        device=device, config=config, ref_model=ref_model,
    )

    best_eval_ppl = float("inf")
    optimizer.zero_grad()

    for epoch in range(1, args.num_epochs + 1):
        logger.info("=" * 60)
        logger.info("Epoch %d / %d", epoch, args.num_epochs)
        logger.info("=" * 60)

        metrics = trainer.train_epoch(train_loader, epoch, eval_loader)
        trainer.save_checkpoint(output_dir, epoch=epoch)

        eval_ppl = metrics.get("eval", {}).get("perplexity", float("inf"))
        if eval_ppl < best_eval_ppl:
            best_eval_ppl = eval_ppl
            trainer.save_checkpoint(output_dir, epoch=epoch, is_best=True)

    logger.info("Training complete. Best eval PPL: %.4f", best_eval_ppl)


if __name__ == "__main__":
    main()
