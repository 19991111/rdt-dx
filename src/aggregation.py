"""Aggregation strategies for RDT loop outputs.

After the Core loop completes (e.g., 16 iterations), per-step hidden states
must be fused into a single representation for the Suffix layers. This module
provides several strategies designed for ablation experiments (§11.3 of the
technical prototype).

Default strategy: ``last4_mean`` — averages the final 4 loop outputs,
balancing stability against single-step noise while preserving late-stage
reasoning signals.

Strategy summary:
    - ``last``: Use only the final step (vulnerable to single-step oscillation).
    - ``mean_all``: Average all steps equally (dilutes late-stage signal).
    - ``last4_mean``: Average steps 13-16 (default, near-convergence states).
    - ``gated_mean``: Learnable softmax-weighted fusion across all steps.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def aggregate_last4_mean(loop_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Average the last 4 loop outputs.

    Uses the final four steps (near-convergence states) rather than all steps
    (which dilutes late-stage reasoning) or the last step alone (which is
    vulnerable to single-step oscillation).

    Args:
        loop_outputs: List of per-step hidden states ``[h_1, ..., h_K]``, each
            of shape ``(batch, seq_len, hidden_dim)``. Must contain at least 4
            tensors.

    Returns:
        Fused tensor of shape ``(batch, seq_len, hidden_dim)``.

    Raises:
        ValueError: If fewer than 4 loop outputs are provided.
    """
    if len(loop_outputs) < 4:
        raise ValueError(
            f"last4_mean requires at least 4 loop outputs, "
            f"got {len(loop_outputs)}."
        )
    fused = torch.stack(loop_outputs[-4:], dim=0).mean(dim=0)
    return fused


def aggregate_last(loop_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Use only the final loop output.

    Args:
        loop_outputs: List of per-step hidden states ``[h_1, ..., h_K]``.

    Returns:
        The last tensor ``h_K`` of shape ``(batch, seq_len, hidden_dim)``.
    """
    return loop_outputs[-1]


def aggregate_mean_all(loop_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Average all loop outputs equally.

    Args:
        loop_outputs: List of per-step hidden states ``[h_1, ..., h_K]``.

    Returns:
        Mean tensor across all steps, shape ``(batch, seq_len, hidden_dim)``.
    """
    return torch.stack(loop_outputs, dim=0).mean(dim=0)


class GatedMeanAggregation(nn.Module):
    """Learnable gated fusion of loop outputs.

    Each step ``i`` is assigned a learned scalar weight ``w_i``, which is
    softmax-normalized across steps to produce a weighted average. This allows
    the model to learn which loop iterations are most informative for the
    downstream task.

    Attributes:
        num_iters: Number of loop iterations (K).
        gate: Learnable scalar parameters of shape ``(num_iters,)``. Initialized
            with a small bias favoring later steps.
    """

    def __init__(self, num_iters: int = 16, bias_init: float = 0.0) -> None:
        """Initialize gated mean aggregation.

        Args:
            num_iters: Number of loop iterations (K). Defaults to 16.
            bias_init: Initial bias toward later steps. Positive values
                increase the initial weight of later iterations.
        """
        super().__init__()
        self.num_iters = num_iters
        raw_weights = torch.zeros(num_iters)
        for i in range(num_iters):
            raw_weights[i] = bias_init * (i + 1) / num_iters
        self.gate = nn.Parameter(raw_weights)

    def forward(self, loop_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute gated weighted average of loop outputs.

        Args:
            loop_outputs: List of per-step tensors ``[h_1, ..., h_K]``, each of
                shape ``(K, batch, seq_len, hidden_dim)``.

        Returns:
            Weighted average tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        stacked = torch.stack(loop_outputs, dim=0)  # (K, B, S, D)
        weights = F.softmax(self.gate, dim=0)  # (K,)
        weights = weights.view(-1, 1, 1, 1).to(stacked.dtype)
        fused = (stacked * weights).sum(dim=0)
        return fused

    def get_step_weights(self) -> torch.Tensor:
        """Return the current softmax-normalized step weights.

        Returns:
            Tensor of shape ``(num_iters,)`` with probabilities summing to 1.
        """
        return F.softmax(self.gate, dim=0)


#: Registry mapping strategy names to callables for easy ablation switching.
_AGGREGATION_REGISTRY: dict[str, Callable] = {
    "last4_mean": aggregate_last4_mean,
    "last": aggregate_last,
    "mean_all": aggregate_mean_all,
}


def get_aggregation(name: str = "last4_mean", **kwargs):
    """Get an aggregation function or module by name.

    Args:
        name: Strategy identifier. One of ``"last4_mean"``, ``"last"``,
            ``"mean_all"``, or ``"gated_mean"``.
        **kwargs: Forwarded to ``GatedMeanAggregation`` if ``name`` is
            ``"gated_mean"``.

    Returns:
        A callable that maps ``List[Tensor] -> Tensor``, or a
        ``GatedMeanAggregation`` module for ``"gated_mean"``.

    Raises:
        ValueError: If ``name`` is not a recognized aggregation strategy.
    """
    if name in _AGGREGATION_REGISTRY:
        return _AGGREGATION_REGISTRY[name]
    elif name == "gated_mean":
        return GatedMeanAggregation(**kwargs)
    else:
        raise ValueError(
            f"Unknown aggregation: {name}. "
            f"Valid options: {list(_AGGREGATION_REGISTRY.keys())} + gated_mean."
        )
