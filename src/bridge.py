"""Bridge v2 state calibrator for RDT (Recursive Depth Transformer).

Bridge v2 prevents hidden-state distribution drift caused by repeatedly
applying the same Core Transformer layers. It uses a zero-initialized
residual MLP with step embedding to calibrate states after each iteration.

Canonical formulation::

    h_t = h_anchor + alpha * MLP(concat(h_anchor, z_t, z_t - h_anchor, step_emb(t)))

Key design decisions (from historical experiments, see prototype doc §16):
    - ``alpha`` is fixed at 0.2; adaptive alpha causes weight explosion and
      output collapse.
    - The final Linear layer is zero-initialized so that at initialization
      ``h_t ≈ h_anchor``, guaranteeing a stable training start.
    - Step embedding lets the bridge learn per-iteration calibration patterns,
      since drift behavior differs across loop iterations.
    - The difference term ``(z_t - h_anchor)`` explicitly encodes how far the
      current state has drifted from the anchor.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A lightweight alternative to LayerNorm that omits mean-centering. Used
    inside the Bridge MLP to stabilize intermediate representations without
    the overhead of computing means.

    Attributes:
        weight: Learnable scale parameter of shape ``(dim,)``.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Epsilon for numerical stability. Defaults to ``1e-6``.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalized tensor of the same shape and dtype as the input.
        """
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class BridgeV2(nn.Module):
    """Bridge v2: zero-init residual MLP with step embedding.

    Architecture::

        input  = concat(h_anchor, z_t, z_t - h_anchor, step_emb(t))
        delta  = Linear(input_dim -> bottleneck) -> RMSNorm -> GELU
               -> Linear(bottleneck -> hidden_dim)  [zero-init]
        h_t    = h_anchor + alpha * delta

    The difference term ``(z_t - h_anchor)`` provides an explicit drift signal,
    and the step embedding allows per-iteration calibration behavior.

    Attributes:
        hidden_dim: Dimensionality of the model's hidden states (e.g., 4096).
        step_embedding_dim: Dimensionality of the step embedding.
        num_iters: Maximum number of core iterations.
        alpha: Fixed residual scaling factor (0.2).
        step_embedding: Embedding lookup for step indices ``1..num_iters``.
        mlp: The MLP sub-network ``(input -> bottleneck -> output)``.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        step_embedding_dim: int = 32,
        num_iters: int = 16,
        alpha: float = 0.2,
        bottleneck_ratio: float = 0.5,
    ) -> None:
        """Initialize Bridge v2.

        Args:
            hidden_dim: Model hidden state dimensionality. Defaults to 4096.
            step_embedding_dim: Dimension of step embeddings. Set to 0 to
                disable step embedding. Defaults to 32.
            num_iters: Maximum number of core iterations. Defaults to 16.
            alpha: Fixed residual scaling factor. Must not be trainable;
                adaptive alpha empirically causes weight explosion.
                Defaults to 0.2.
            bottleneck_ratio: Ratio of bottleneck dim to ``hidden_dim``.
                Defaults to 0.5.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.step_embedding_dim = step_embedding_dim
        self.num_iters = num_iters
        self.alpha = alpha

        # Input: [h_anchor (hidden_dim), z_t (hidden_dim),
        #         z_t - h_anchor (hidden_dim), step_emb (step_embedding_dim)]
        self.has_step_embedding = step_embedding_dim > 0
        input_dim = 3 * hidden_dim + (step_embedding_dim if self.has_step_embedding else 0)
        bottleneck_dim = int(hidden_dim * bottleneck_ratio)

        if self.has_step_embedding:
            self.step_embedding = nn.Embedding(
                num_iters + 1, step_embedding_dim, padding_idx=0
            )
        else:
            self.step_embedding = None

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim, bias=False),
            RMSNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with recommended schemes.

        - Step embedding: ``normal_(0, 1/sqrt(dim))``.
        - First Linear: Xavier uniform.
        - Last Linear: **zeros** (critical for training stability — ensures
          ``h_t ≈ h_anchor`` at initialization).
        """
        if self.has_step_embedding:
            nn.init.normal_(
                self.step_embedding.weight,
                mean=0.0,
                std=1.0 / math.sqrt(self.step_embedding_dim),
            )
            with torch.no_grad():
                self.step_embedding.weight[0] = 0.0

        first_linear = self.mlp[0]
        nn.init.xavier_uniform_(first_linear.weight)

        last_linear = self.mlp[3]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        h_anchor: torch.Tensor,
        z_t: torch.Tensor,
        step_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate hidden states after one Core iteration.

        Args:
            h_anchor: Anchor states from prefix output, shape
                ``(batch, seq_len, hidden_dim)``. Should be detached from the
                autograd graph.
            z_t: Raw Core output at iteration ``t``, same shape as
                ``h_anchor``.
            step_idx: Current iteration index (1-indexed, ``1..num_iters``).

        Returns:
            Calibrated hidden states ``h_t`` of shape
            ``(batch, seq_len, hidden_dim)``.
        """
        if self.has_step_embedding:
            if isinstance(step_idx, int):
                step_idx = torch.tensor(
                    [step_idx], device=h_anchor.device, dtype=torch.long,
                )
            step_idx = step_idx.to(h_anchor.device)
            step_emb = self.step_embedding(step_idx)
            step_emb = step_emb.unsqueeze(1).expand(-1, h_anchor.shape[1], -1)
            if step_emb.shape[0] == 1 and h_anchor.shape[0] > 1:
                step_emb = step_emb.expand(h_anchor.shape[0], -1, -1)
            mlp_input = torch.cat(
                [h_anchor, z_t, z_t - h_anchor, step_emb], dim=-1
            )
        else:
            mlp_input = torch.cat([h_anchor, z_t, z_t - h_anchor], dim=-1)

        delta = self.mlp(mlp_input)
        h_t = h_anchor + self.alpha * delta
        return h_t

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"step_embedding_dim={self.step_embedding_dim}, "
            f"num_iters={self.num_iters}, "
            f"alpha={self.alpha}"
        )


class BridgeRegistry:
    """Registry of bridge variants for ablation experiments.

    Provides a single entry point for creating bridge modules by name::

        bridge = BridgeRegistry.create("mlp_step", hidden_dim=4096)

    Supported types:
        - ``"mlp_step"``: MLP Bridge with step embedding (default, BridgeV2).
        - ``"mlp"``: MLP Bridge **without** step embedding.
        - ``"linear"``: Simple Linear Bridge (ablation baseline).
        - ``"none"``: Identity bridge (no calibration, for ablation).
    """

    @staticmethod
    def create(
        bridge_type: str = "mlp_step",
        hidden_dim: int = 4096,
        **kwargs,
    ) -> nn.Module:
        """Create a bridge instance by type name.

        Args:
            bridge_type: Bridge variant identifier. One of ``"mlp_step"``,
                ``"mlp"``, ``"linear"``, or ``"none"``.
            hidden_dim: Hidden state dimensionality.
            **kwargs: Additional arguments forwarded to the bridge constructor
                (e.g., ``step_embedding_dim``, ``num_iters``, ``alpha``).

        Returns:
            A bridge ``nn.Module`` instance.

        Raises:
            ValueError: If ``bridge_type`` is not recognized.
        """
        if bridge_type == "mlp_step":
            return BridgeV2(hidden_dim=hidden_dim, **kwargs)
        elif bridge_type == "mlp":
            kwargs.pop("step_embedding_dim", None)
            return BridgeV2(hidden_dim=hidden_dim, step_embedding_dim=0, **kwargs)
        elif bridge_type == "linear":
            return _LinearBridge(hidden_dim=hidden_dim)
        elif bridge_type == "none":
            return _NoBridge()
        else:
            raise ValueError(
                f"Unknown bridge_type: {bridge_type}. "
                f"Valid options: mlp_step, mlp, linear, none."
            )


class _LinearBridge(nn.Module):
    """Simple linear bridge for ablation experiments.

    Uses a single zero-initialized ``nn.Linear`` layer without step embedding
    or non-linearity. Serves as a baseline to isolate the benefit of the MLP
    and step embedding components.

    Attributes:
        linear: The single Linear layer (zero-initialized).
        alpha: Fixed residual scaling factor (0.2).
    """

    def __init__(self, hidden_dim: int = 4096) -> None:
        """Initialize linear bridge.

        Args:
            hidden_dim: Model hidden state dimensionality.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.alpha = 0.2

    def forward(
        self,
        h_anchor: torch.Tensor,
        z_t: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """Apply linear calibration (step index is ignored).

        Args:
            h_anchor: Anchor states from prefix, shape ``(B, S, D)``.
            z_t: Raw Core output, shape ``(B, S, D)``.
            step_idx: Unused (no step embedding in linear bridge).

        Returns:
            Calibrated hidden states, shape ``(B, S, D)``.
        """
        del step_idx
        mlp_input = torch.cat([h_anchor, z_t, z_t - h_anchor], dim=-1)
        delta = self.linear(mlp_input)
        return h_anchor + self.alpha * delta


class _NoBridge(nn.Module):
    """Identity bridge for ablation experiments.

    Passes the Core output through unchanged, simulating the absence of any
    calibration mechanism. Used to demonstrate that Bridge is **essential**
    for loop stability.
    """

    def forward(
        self,
        h_anchor: torch.Tensor,
        z_t: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """Return ``z_t`` unchanged (identity).

        Args:
            h_anchor: Unused anchor states.
            z_t: Raw Core output.
            step_idx: Unused iteration index.

        Returns:
            ``z_t`` without modification.
        """
        del h_anchor, step_idx
        return z_t
