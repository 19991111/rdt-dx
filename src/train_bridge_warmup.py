"""Phase 1: Bridge Warmup training for RDT-Fixed-16.

Trains the Bridge v2 module to stabilize the RDT loop structure before
task-specific SFT training. The goal is **not** to improve medical task
performance, but to ensure the looped Core + Bridge architecture produces
output distributions close to the reference (base) model.

Key design decisions:
    - Progressive K schedule: start with ``K=2``, ramp to ``K=16``.
    - KL divergence loss on the **full sequence** (not just the last token).
    - Only Bridge parameters are trained; Core and Suffix are frozen.
    - Core forward runs under ``torch.no_grad()`` for memory efficiency.
    - Smoothness regularization prevents step-to-step hidden state oscillation.

Usage::

    python -m src.train_bridge_warmup \\
        --model_path /data/model/Qwen/Qwen3___5-9B-Base \\
        --data_path data/train.jsonl \\
        --output_dir output/bridge_warmup \\
        --progressive_k 2,4,8,16
"""

from __future__ import annotations

import argparse
import json
import logging
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
# Loss functions
# ---------------------------------------------------------------------------


def compute_kl_loss(
    logits_rdt: torch.Tensor,
    logits_ref: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute token-wise KL divergence between RDT and reference logits.

    Computes ``KL(ref || rdt) = sum(ref_prob * (log(ref_prob) - log(rdt_prob)))``
    averaged over all non-masked tokens.

    Args:
        logits_rdt: RDT model logits of shape ``(B, S, V)``.
        logits_ref: Reference model logits of shape ``(B, S, V)``.
        attention_mask: Optional mask of shape ``(B, S)`` where non-padding
            tokens are 1 and padding tokens are 0.
        temperature: Softmax temperature. Defaults to 1.0 (no scaling).

    Returns:
        Scalar KL divergence averaged over all non-masked tokens.
    """
    B, S, V = logits_rdt.shape

    logits_rdt = logits_rdt / temperature
    logits_ref = logits_ref / temperature

    log_probs_rdt = F.log_softmax(logits_rdt, dim=-1)
    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
    probs_ref = F.softmax(logits_ref, dim=-1)

    kl_per_token = (probs_ref * (log_probs_ref - log_probs_rdt)).sum(dim=-1)

    if attention_mask is not None:
        kl_per_token = kl_per_token * attention_mask.float()

    return kl_per_token.sum() / (attention_mask.sum() if attention_mask is not None else B * S)


def compute_smoothness_loss(
    loop_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """Compute smoothness regularization across loop steps.

    Penalizes large changes in hidden states between consecutive iterations::

        L_smooth = mean(||h_t - h_{t-1}||^2)

    This encourages the loop to converge smoothly rather than oscillating.

    Args:
        loop_outputs: List of hidden states ``[h_1, ..., h_K]``, each of shape
            ``(B, S, D)``.

    Returns:
        Scalar smoothness loss. Returns 0 if fewer than 2 outputs are provided.
    """
    if len(loop_outputs) < 2:
        return torch.tensor(0.0, device=loop_outputs[0].device)

    total = 0.0
    for t in range(1, len(loop_outputs)):
        delta = loop_outputs[t] - loop_outputs[t - 1]
        total += (delta ** 2).mean()

    return total / (len(loop_outputs) - 1)


# ---------------------------------------------------------------------------
# Reference model wrapper
# ---------------------------------------------------------------------------


class ReferenceModelWrapper:
    """Wrapper that produces reference logits from a standard Qwen3.5 model.

    The reference model is the base Qwen3.5 **without** any RDT modifications.
    Its outputs serve as the target distribution for Bridge warmup KL loss.

    Attributes:
        model: The underlying ``Qwen3_5ForConditionalGeneration`` model.
        device: Device the model resides on.
    """

    def __init__(self, model_path: str, device: torch.device):
        """Load the reference model.

        Args:
            model_path: Path to the base Qwen3.5 model directory.
            device: Device to load the model onto.
        """
        from transformers import AutoModelForCausalLM

        logger.info("Loading reference model from %s ...", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device

    @torch.no_grad()
    def get_logits(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get reference logits for a batch.

        Args:
            input_ids: Token indices of shape ``(B, S)``.
            attention_mask: Attention mask of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, V)`` in float32.
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits.float()

    def to(self, device: torch.device) -> "ReferenceModelWrapper":
        """Move the reference model to a different device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        self.model = self.model.to(device)
        self.device = device
        return self


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class BridgeWarmupTrainer:
    """Trainer for Phase 1: Bridge Warmup.

    Uses a progressive K schedule to gradually increase loop depth while
    keeping the RDT output distribution close to the reference model via
    KL divergence loss.

    Attributes:
        model: ``RDTFixed16`` model (Bridge-only trainable).
        ref_model: Wrapped reference model for KL target.
        optimizer: AdamW optimizer (Bridge parameters only).
        scheduler: Optional LR scheduler.
        device: Primary training device.
        config: Training configuration dict.
    """

    def __init__(
        self,
        model: RDTFixed16,
        ref_model: ReferenceModelWrapper,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        config: dict,
    ):
        """Initialize the Bridge warmup trainer.

        Args:
            model: RDT-Fixed-16 model with Bridge parameters unfrozen.
            ref_model: Reference model wrapper.
            optimizer: Optimizer (should only include Bridge parameters).
            scheduler: Optional learning rate scheduler.
            device: Training device.
            config: Dict with keys ``lambda_smooth``, ``max_grad_norm``,
                ``use_amp``, and ``log_dir``.
        """
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.writer = SummaryWriter(log_dir=config.get("log_dir", "./logs"))
        self.global_step = 0
        self.scaler = GradScaler(enabled=config.get("use_amp", True))
        self.use_amp = config.get("use_amp", True)

    def train_stage(
        self,
        dataloader: DataLoader,
        num_iters: int,
        max_steps: int,
        stage_name: str = "",
    ) -> Dict[str, float]:
        """Train one stage of the progressive K schedule.

        Args:
            dataloader: Training data loader (cycles through data).
            num_iters: Number of core loop iterations for this stage (K).
            max_steps: Maximum training steps for this stage.
            stage_name: Label for logging (e.g., ``"K=4"``).

        Returns:
            Dict with keys ``kl_loss`` and ``smooth_loss`` (stage averages).
        """
        self.model.train()
        self.model.num_iters = num_iters
        self.model.freeze_core()
        self.model.freeze_suffix()

        for param in self.model.bridge.parameters():
            param.requires_grad = True

        total_kl_loss = 0.0
        total_smooth_loss = 0.0
        total_steps = 0

        lambda_smooth = self.config.get("lambda_smooth", 2e-4)

        logger.info(
            "Starting Bridge warmup stage %s (K=%d, max_steps=%d)",
            stage_name, num_iters, max_steps,
        )

        data_iter = iter(dataloader)

        for step in range(max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            ref_logits = self.ref_model.get_logits(input_ids, attention_mask)

            self.optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=self.use_amp):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loop_outputs=True,
                    core_no_grad=True,
                )

                kl_loss = compute_kl_loss(
                    output.logits.float(),
                    ref_logits.float(),
                    attention_mask=attention_mask,
                )
                smooth_loss = compute_smoothness_loss(output.loop_outputs)
                loss = kl_loss + lambda_smooth * smooth_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 0.5),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_kl_loss += kl_loss.item()
            total_smooth_loss += smooth_loss.item()
            total_steps += 1

            self.writer.add_scalar(
                f"bridge_warmup/{stage_name}/kl_loss", kl_loss.item(), self.global_step
            )
            self.writer.add_scalar(
                f"bridge_warmup/{stage_name}/smooth_loss", smooth_loss.item(), self.global_step
            )
            self.global_step += 1

            if step % 10 == 0 or step == max_steps - 1:
                logger.info(
                    "  %s step %d/%d | KL: %.6f | Smooth: %.6f | LR: %.2e",
                    stage_name, step, max_steps,
                    kl_loss.item(), smooth_loss.item(),
                    self.optimizer.param_groups[0]["lr"],
                )

        avg_kl = total_kl_loss / total_steps if total_steps > 0 else 0.0
        avg_smooth = total_smooth_loss / total_steps if total_steps > 0 else 0.0

        logger.info(
            "Stage %s complete. Avg KL: %.6f, Avg Smooth: %.6f",
            stage_name, avg_kl, avg_smooth,
        )

        return {"kl_loss": avg_kl, "smooth_loss": avg_smooth}

    def save_checkpoint(self, output_dir: str | Path, stage_name: str = "") -> None:
        """Save model checkpoint.

        Args:
            output_dir: Directory to save to.
            stage_name: Optional stage suffix for the filename (e.g.,
                ``"K=4"`` or ``"final"``).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"bridge_warmup_{stage_name}.pt" if stage_name else "bridge_warmup.pt"
        path = output_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Bridge warmup training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="RDT-Fixed-16 Bridge Warmup (Phase 1)"
    )

    parser.add_argument("--model_path", type=str,
                        default="/data/model/Qwen/Qwen3___5-9B-Base")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data/synthetic_train.jsonl")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_synthetic_data", action="store_true", default=True)
    parser.add_argument("--progressive_k", type=str, default="2,4,8,16")
    parser.add_argument("--steps_per_stage", type=int, default=50)
    parser.add_argument("--bridge_lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--lambda_smooth", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="output/bridge_warmup")
    parser.add_argument("--log_dir", type=str, default="output/logs")
    parser.add_argument("--bridge_type", type=str, default="mlp_step")
    parser.add_argument("--aggregation", type=str, default="last4_mean")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    """Main entry point for Phase 1 Bridge warmup training."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        if args.use_synthetic_data:
            generate_synthetic_data(data_path, num_easy=3, num_medium=3, num_hard=4)
        else:
            raise FileNotFoundError(f"Data not found at {data_path}")

    dataloader = create_dataloader(
        data_path=data_path, tokenizer=tokenizer,
        batch_size=args.batch_size, max_length=args.max_length, shuffle=True,
    )

    model = create_rdt_fixed_16(
        base_model_path=args.model_path, num_iters=2, lora_r=None,
        bridge_type=args.bridge_type, aggregation=args.aggregation,
        device_map=None,
    ).to(device)

    model.freeze_core()
    model.freeze_suffix()
    for param in model.prefix_group.parameters():
        param.requires_grad = False

    ref_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    ref_model = ReferenceModelWrapper(args.model_path, ref_device)

    bridge_params = list(model.bridge.parameters())
    optimizer = torch.optim.AdamW(bridge_params, lr=args.bridge_lr, weight_decay=args.weight_decay)

    total_steps = len(args.progressive_k.split(",")) * args.steps_per_stage
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    config = {
        "lambda_smooth": args.lambda_smooth,
        "max_grad_norm": args.max_grad_norm,
        "use_amp": args.use_amp and device.type == "cuda",
        "log_dir": args.log_dir,
    }

    trainer = BridgeWarmupTrainer(
        model=model, ref_model=ref_model, optimizer=optimizer,
        scheduler=scheduler, device=device, config=config,
    )

    k_values = [int(k.strip()) for k in args.progressive_k.split(",")]
    logger.info("Progressive K schedule: %s", k_values)

    for k in k_values:
        metrics = trainer.train_stage(
            dataloader=dataloader, num_iters=k,
            max_steps=args.steps_per_stage, stage_name=f"K={k}",
        )
        trainer.save_checkpoint(output_dir, stage_name=f"K={k}")

    # Final evaluation at K=16
    model.num_iters = 16
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ref_logits = ref_model.get_logits(input_ids, attention_mask)
            output = model(input_ids=input_ids, attention_mask=attention_mask,
                           return_loop_outputs=True)

            kl = compute_kl_loss(
                output.logits.float(), ref_logits.float(),
                attention_mask=attention_mask,
            )
            ref_top5 = ref_logits[0, -1, :].topk(5).indices
            rdt_top5 = output.logits[0, -1, :].topk(5).indices
            overlap = len(set(ref_top5.tolist()) & set(rdt_top5.tolist())) / 5

            logger.info("Eval batch %d: KL=%.6f, Top-5 overlap=%.1f%%",
                        batch_idx, kl.item(), overlap * 100)

    trainer.save_checkpoint(output_dir, stage_name="final")
    logger.info("Bridge warmup complete.")


if __name__ == "__main__":
    main()
