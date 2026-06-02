"""Evaluation utilities for RDT-Dx medical LLM.

Provides metric computation for both standard LM evaluation (CE loss,
perplexity) and RDT-specific diagnostics (step-wise KL divergence, top-k
token overlap, hidden state drift, last4 variance, etc.).

These metrics correspond to the evaluation system described in the
technical prototype document (§10).

Usage::

    from src.eval_utils import RDTevaluator
    evaluator = RDTevaluator(model, tokenizer, ref_model=ref_model)
    metrics = evaluator.evaluate(dataloader)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class RDTevaluator:
    """Evaluator for RDT-Fixed-16 models.

    Computes standard LM metrics and RDT-specific structure diagnostics that
    help assess whether the recursive loop is functioning correctly.

    Attributes:
        model: The ``RDTFixed16`` model under evaluation.
        ref_model: Optional reference model for KL- and overlap-based metrics.
        tokenizer: Tokenizer for decoding (reserved for future use).
        device: Device used for all computations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        ref_model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: RDT-Fixed-16 model to evaluate.
            tokenizer: HuggingFace tokenizer (for potential text decoding).
            ref_model: Reference model for distribution comparison metrics
                (KL divergence, top-k overlap). If ``None``, RDT-specific
                comparison metrics are skipped.
            device: Device for computation. Defaults to the device of the
                first model parameter.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_rdt_metrics: bool = True,
    ) -> Dict[str, float]:
        """Run full evaluation over a data loader.

        Iterates over batches, accumulating base metrics (CE loss, perplexity)
        and optionally RDT-specific structure diagnostics.

        Args:
            dataloader: DataLoader yielding dicts with keys ``input_ids``,
                ``attention_mask``, and ``labels``.
            compute_rdt_metrics: If ``True``, also compute per-step hidden
                drift, last4 variance, and top-k overlap (requires
                ``return_loop_outputs=True`` on the model forward).

        Returns:
            Dict mapping metric name to scalar average over evaluated batches.
            Batches are capped at 20 to bound evaluation time.
        """
        self.model.eval()

        metrics_accum: Dict[str, list] = {}

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_loop_outputs=compute_rdt_metrics,
            )

            base = self._compute_base_metrics(output, labels)
            for k, v in base.items():
                metrics_accum.setdefault(k, []).append(v)

            if compute_rdt_metrics and output.loop_outputs is not None:
                rdt = self._compute_rdt_metrics(output, input_ids, attention_mask)
                for k, v in rdt.items():
                    metrics_accum.setdefault(k, []).append(v)

            if batch_idx >= 20:
                break

        return {k: sum(v) / len(v) for k, v in metrics_accum.items()}

    def _compute_base_metrics(
        self, output, labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute standard language model evaluation metrics.

        Args:
            output: ``RDTOutput`` from a model forward pass.
            labels: Token labels with ``-100`` for ignored positions.

        Returns:
            Dict with keys ``ce_loss``, ``perplexity``, and ``n_tokens``.
        """
        ce_loss = output.loss
        if ce_loss is None:
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

        n_tokens = (labels[..., 1:] != -100).sum().item()
        ppl = torch.exp(ce_loss).item() if ce_loss.item() < 100 else float("inf")

        return {
            "ce_loss": ce_loss.item(),
            "perplexity": ppl,
            "n_tokens": n_tokens,
        }

    def _compute_rdt_metrics(
        self,
        output,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute RDT-specific structure diagnostics.

        Computes per-step hidden state drift (normalized distance from anchor),
        last4 hidden state variance, and (if a reference model is available)
        top-k token overlap at the last sequence position.

        Args:
            output: ``RDTOutput`` with ``loop_outputs`` and ``h_anchor``
                populated.
            input_ids: Input token indices of shape ``(B, S)``.
            attention_mask: Attention mask of shape ``(B, S)``.

        Returns:
            Dict of RDT diagnostic metrics (``hidden_drift_mean``,
            ``hidden_drift_max``, ``hidden_drift_last``, ``last4_variance``,
            and optionally ``top1_overlap`` / ``top5_overlap``).
        """
        loop_outputs = output.loop_outputs
        h_anchor = output.h_anchor

        metrics: Dict[str, float] = {}

        # Hidden drift: ||h_t - h_anchor|| / ||h_anchor|| per step
        if h_anchor is not None:
            anchor_norm = h_anchor.norm(dim=-1).mean().item()
            drifts = []
            for h_t in loop_outputs:
                drift = (h_t - h_anchor).norm(dim=-1).mean().item()
                normalized_drift = drift / (anchor_norm + 1e-8)
                drifts.append(normalized_drift)
            metrics["hidden_drift_mean"] = sum(drifts) / len(drifts)
            metrics["hidden_drift_max"] = max(drifts)
            metrics["hidden_drift_last"] = drifts[-1]

        # Last4 hidden state variance
        if len(loop_outputs) >= 4:
            last4 = torch.stack(loop_outputs[-4:], dim=0)  # (4, B, S, D)
            last4_mean = last4.mean(dim=0)
            variance = ((last4 - last4_mean) ** 2).mean()
            metrics["last4_variance"] = variance.item()

        # Top-k overlap with reference model at the last token position
        if self.ref_model is not None and attention_mask is not None:
            ref_logits = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float()

            last_idx = attention_mask.sum(dim=-1).long() - 1
            for k in [1, 5]:
                ref_topk: set[int] = set()
                rdt_topk: set[int] = set()
                for b in range(input_ids.shape[0]):
                    idx = last_idx[b].item()
                    ref_topk.update(ref_logits[b, idx, :].topk(k).indices.tolist())
                    rdt_topk.update(output.logits[b, idx, :].topk(k).indices.tolist())
                overlap = len(ref_topk & rdt_topk) / max(len(ref_topk | rdt_topk), 1)
                metrics[f"top{k}_overlap"] = overlap

        return metrics

    @torch.no_grad()
    def compute_step_benefit(
        self,
        dataloader: DataLoader,
        k_values: List[int] = [1, 2, 4, 8, 16],
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate at multiple loop depths for K-step ablation.

        Temporarily changes ``model.num_iters`` to each value in ``k_values``
        and runs evaluation, restoring the original value afterward.

        Args:
            dataloader: Evaluation data loader.
            k_values: List of K (loop iterations) to evaluate. Defaults to
                ``[1, 2, 4, 8, 16]``.

        Returns:
            Dict mapping each K to its metrics dict.
        """
        results: Dict[int, Dict[str, float]] = {}
        original_iters = self.model.num_iters

        for k in k_values:
            logger.info("Evaluating K=%d ...", k)
            self.model.num_iters = k
            metrics = self.evaluate(dataloader, compute_rdt_metrics=(k >= 4))
            results[k] = metrics

        self.model.num_iters = original_iters
        return results

    @torch.no_grad()
    def evaluate_by_difficulty(
        self,
        easy_loader: DataLoader,
        medium_loader: DataLoader,
        hard_loader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate separately on easy, medium, and hard cases.

        Essential for the "easy stability" check: verifying that multi-step
        computation does not degrade performance on simple cases.

        Args:
            easy_loader: DataLoader for easy-difficulty cases.
            medium_loader: DataLoader for medium-difficulty cases.
            hard_loader: DataLoader for hard-difficulty cases.

        Returns:
            Dict mapping difficulty label (``"easy"``, ``"medium"``,
            ``"hard"``) to its metrics dict.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, loader in [
            ("easy", easy_loader),
            ("medium", medium_loader),
            ("hard", hard_loader),
        ]:
            logger.info("Evaluating %s cases ...", name)
            results[name] = self.evaluate(loader)
        return results
