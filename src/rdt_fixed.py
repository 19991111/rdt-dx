"""RDT-Fixed-16: Recursive Depth Transformer with fixed 16-iteration loop.

Implements the core RDT-Fixed-16 architecture described in the technical
prototype document. Loads a Qwen3.5 base model, splits the Transformer layers
into Prefix / Core / Suffix segments, adds LoRA adapters to Core and Suffix,
and integrates the Bridge v2 state calibrator.

Architecture::

    Input tokens → Embedding (frozen) → Prefix Layers (frozen)
        → h_anchor (detach) → [Core Layers × 16 + Bridge v2] loop
        → last4_mean aggregation → Suffix Layers (LoRA)
        → Norm + LM Head → logits

Default layer split for Qwen3.5-9B (32 layers):
    - Prefix:  0–11 (12 layers, frozen)
    - Core:   12–27 (16 layers, LoRA + looped 16×)
    - Suffix: 28–31 (4 layers, LoRA)

Key classes:
    - :class:`LoRALinear`: Minimal LoRA adapter (avoids peft dependency).
    - :class:`LayerGroup`: Sequential wrapper over decoder layers with
      per-layer attention mask routing (handles Qwen3.5 hybrid attention).
    - :class:`RDTFixed16`: The full RDT-Fixed-K model.
    - :class:`RDTOutput`: Output dataclass bundling logits, loss, and loop
      diagnostics.
"""

from __future__ import annotations

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from .bridge import BridgeV2, BridgeRegistry
from .aggregation import get_aggregation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal LoRA implementation (avoids peft compatibility issues)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for ``nn.Linear``.

    Wraps an existing linear layer and adds trainable low-rank matrices
    ``A`` and ``B`` such that the effective weight is ``W + BA``, scaled by
    ``alpha / r``. The original weight is frozen; only ``A`` and ``B`` are
    trained. LoRA parameters are created in the same dtype as the base weight
    for mixed-precision compatibility.

    Attributes:
        base: The original frozen ``nn.Linear`` layer.
        lora_A: Trainable ``(in_features, r)`` matrix, kaiming-uniform init.
        lora_B: Trainable ``(r, out_features)`` matrix, zero init.
        scaling: ``alpha / r`` scaling factor applied to the LoRA branch.
        lora_dropout: Dropout applied before the LoRA branch (or Identity).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 64,
        lora_alpha: int = 128,
        dropout: float = 0.0,
    ) -> None:
        """Wrap a Linear layer with LoRA.

        Args:
            base: The original nn.Linear layer to adapt.
            r: LoRA rank.
            lora_alpha: Scaling factor.
            dropout: Dropout probability.
        """
        super().__init__()
        self.base = base
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Freeze original weight
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features

        # Match the base weight's dtype for mixed-precision compatibility
        base_dtype = base.weight.dtype

        # LoRA matrices (match base dtype to avoid bf16/float32 mismatch)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r, dtype=base_dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, dtype=base_dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base(x) + LoRA(x).

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        base_out = self.base(x)
        # Cast LoRA weights to input dtype for mixed-precision compatibility
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)
        ) * self.scaling
        return base_out + lora_out


def _inject_lora(
    module: nn.Module,
    target_names: List[str],
    r: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    """Inject LoRA adapters into a module's Linear sub-modules.

    Recursively traverses the module and replaces nn.Linear layers whose
    names contain one of the target_names with LoRALinear wrappers.

    Args:
        module: Root module to inject LoRA into.
        target_names: List of substrings to match against linear layer names.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout rate.

    Returns:
        The modified module (in-place).
    """
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            if isinstance(child, nn.Linear):
                # Check if this linear layer should get LoRA
                should_adapt = any(
                    target in child_name for target in target_names
                )
                if should_adapt:
                    lora_linear = LoRALinear(child, r=r, lora_alpha=alpha, dropout=dropout)
                    setattr(parent, child_name, lora_linear)

    return module


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class RDTOutput:
    """Output of the RDT-Fixed-16 forward pass.

    Attributes:
        logits: Predicted token logits of shape ``(batch, seq_len, vocab_size)``.
        loss: Cross-entropy loss if ``labels`` were provided, else ``None``.
        loop_outputs: List of hidden states ``[h_1 .. h_K]`` after each bridge
            step (populated when ``return_loop_outputs=True``).
        h_anchor: Anchor hidden states from the prefix output
            (populated when ``return_loop_outputs=True``).
        new_past_key_values: Updated KV cache dict (attached as a dynamic
            attribute when ``use_cache=True``; not part of the dataclass
            fields to maintain backward compatibility).
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    loop_outputs: Optional[List[torch.Tensor]] = None
    h_anchor: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Layer group wrapper (for LoRA application)
# ---------------------------------------------------------------------------


class LayerGroup(nn.Module):
    """Sequential group of Transformer decoder layers with KV cache support.

    Wraps a contiguous subset of the base model's decoder layers, handling
    per-layer attention mask selection (``linear_attention`` vs
    ``full_attention`` in Qwen3.5 hybrid attention) and passing position
    embeddings through. Supports incremental decoding via ``past_key_values``
    and ``use_cache`` parameters.

    Attributes:
        layers: ``ModuleList`` of ``Qwen3_5DecoderLayer`` instances.
        layer_mask_types: List of ``"linear_attention"`` or ``"full_attention"``,
            aligned with ``layers``.
    """

    def __init__(
        self,
        layers: List[nn.Module],
        layer_mask_types: List[str],
    ) -> None:
        """Initialize a layer group.

        Args:
            layers: List of decoder layers from the base model.
            layer_mask_types: Mask type for each layer, aligned with layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.layer_mask_types = layer_mask_types

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        masks_by_type: Dict[str, torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ):
        """Run all layers sequentially with optional KV cache.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            position_embeddings: Tuple of (cos, sin) rotary embeddings.
            masks_by_type: Dict mapping "linear_attention" / "full_attention"
                to their respective attention masks.
            position_ids: Position IDs passed to full-attention layers.
            past_key_values: Optional list of per-layer caches (length = num_layers).
                Each entry can be None or a layer-specific cache object.
            use_cache: If True, return updated per-layer KV caches.

        Returns:
            If use_cache=False: hidden_states tensor
            If use_cache=True: tuple of (hidden_states, new_past_key_values list)
        """
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            mask_type = self.layer_mask_types[i]
            attn_mask = masks_by_type[mask_type]

            layer_cache = past_key_values[i] if past_key_values is not None else None

            layer_out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=layer_cache,
                use_cache=use_cache,
            )

            if use_cache:
                # Qwen3.5 layer returns (hidden_states, present_key_value) when use_cache=True
                if isinstance(layer_out, tuple) and len(layer_out) >= 2:
                    hidden_states = layer_out[0]
                    present_key_values.append(layer_out[1])
                else:
                    hidden_states = layer_out
                    present_key_values.append(None)
            else:
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        if use_cache:
            return hidden_states, present_key_values
        return hidden_states


# ---------------------------------------------------------------------------
# Main RDT-Fixed-16 model
# ---------------------------------------------------------------------------


class RDTFixed16(nn.Module):
    """RDT-Fixed-K: Recursive Depth Transformer with fixed K-step loop.

    Enhances medical reasoning by reusing intermediate Transformer layers
    (the "Core") across ``num_iters`` iterations with Bridge-based state
    calibration, instead of scaling parameter count. Supports KV-cache-based
    incremental decoding via ``use_cache`` and ``past_key_values``.

    Trainable components (configured per training phase):
        - **Bridge v2**: Full-parameter trainable (Phase 1 Bridge Warmup).
        - **Core LoRA**: Low-rank adapters on Core layers (Phase 2 SFT).
        - **Suffix LoRA**: Low-rank adapters on Suffix layers (Phase 2 SFT).

    Frozen components:
        - Token embedding
        - Prefix layers (0 to ``prefix_end - 1``)
        - Final layer norm and LM head

    Attributes:
        config: Model configuration from the base Qwen3.5 model.
        base_model: The underlying ``Qwen3_5ForConditionalGeneration`` model.
        num_iters: Number of core loop iterations (K, default 16).
        prefix_group: Frozen prefix layers (``LayerGroup``).
        core_group: Core layers with optional LoRA adapters, reused each
            loop iteration (``LayerGroup``).
        suffix_group: Suffix layers with optional LoRA adapters
            (``LayerGroup``).
        bridge: Bridge v2 state calibrator (``BridgeV2`` or variant).
        aggregation_fn: Function to fuse loop outputs
            (e.g., ``aggregate_last4_mean``).
        embed_tokens: Token embedding (shared reference, frozen).
        final_norm: Output layer norm (shared reference, frozen).
        lm_head: Language model head (shared reference, frozen).
        rotary_emb: Rotary embedding module (shared reference).
        is_trainable: Whether the bridge has trainable parameters.
    """

    def __init__(
        self,
        base_model_path: str,
        num_iters: int = 16,
        # Layer split (inclusive start, exclusive end)
        prefix_end: int = 12,
        core_end: int = 28,
        # Bridge config
        bridge_type: str = "mlp_step",
        bridge_alpha: float = 0.2,
        step_embedding_dim: int = 32,
        # Aggregation
        aggregation: str = "last4_mean",
        # LoRA config (None = no LoRA)
        lora_r: Optional[int] = 64,
        lora_alpha: Optional[int] = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Precision
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
    ) -> None:
        """Initialize RDT-Fixed-K from a base Qwen3.5 model.

        Args:
            base_model_path: Path to the base Qwen3.5 model directory.
            num_iters: Number of core loop iterations (K). Defaults to 16.
            prefix_end: Exclusive end index for prefix layers
                (``0:prefix_end``). Defaults to 12.
            core_end: Exclusive end index for core layers
                (``prefix_end:core_end``). Layers from ``core_end`` to the
                final layer become Suffix. Defaults to 28.
            bridge_type: Bridge variant name (see
                :class:`~src.bridge.BridgeRegistry`). Defaults to
                ``"mlp_step"``.
            bridge_alpha: Fixed alpha for bridge residual. Defaults to 0.2.
            step_embedding_dim: Dimension of step embeddings. Defaults to 32.
            aggregation: Aggregation strategy name
                (see :func:`~src.aggregation.get_aggregation`).
                Defaults to ``"last4_mean"``.
            lora_r: LoRA rank. If ``None``, LoRA is not applied to Core or
                Suffix layers. Defaults to 64.
            lora_alpha: LoRA scaling factor. Defaults to 128.
            lora_dropout: LoRA dropout rate. Defaults to 0.05.
            lora_target_modules: Linear module sub-strings to target for LoRA
                injection. If ``None``, uses the default Qwen3.5 set.
            torch_dtype: Data type for model weights. Defaults to
                ``torch.bfloat16``.
            device_map: Device map for model loading. ``None`` means CPU.
        """
        super().__init__()
        self.num_iters = num_iters

        # ------------------------------------------------------------------
        # Load base model
        # ------------------------------------------------------------------
        logger.info("Loading base model from %s ...", base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map=device_map or "cpu",
        )

        # For text-only Qwen3.5, base_model.model IS the language model
        lang = self.base_model.model  # Qwen3_5TextModel
        self.config = self.base_model.config

        num_total_layers = len(lang.layers)
        logger.info(
            "Base model has %d layers. Split: prefix[0:%d], core[%d:%d], suffix[%d:%d].",
            num_total_layers,
            prefix_end,
            prefix_end,
            core_end,
            core_end,
            num_total_layers,
        )

        # ------------------------------------------------------------------
        # Extract and split layers
        # ------------------------------------------------------------------
        all_layers = list(lang.layers)
        all_layer_types = self.config.layer_types

        prefix_layers = all_layers[:prefix_end]
        core_layers = all_layers[prefix_end:core_end]
        suffix_layers = all_layers[core_end:]

        prefix_types = all_layer_types[:prefix_end]
        core_types = all_layer_types[prefix_end:core_end]
        suffix_types = all_layer_types[core_end:]

        # Build layer groups
        self.prefix_group = LayerGroup(prefix_layers, prefix_types)
        self.core_group = LayerGroup(core_layers, core_types)
        self.suffix_group = LayerGroup(suffix_layers, suffix_types)

        # ------------------------------------------------------------------
        # Apply LoRA to core and suffix layers
        # ------------------------------------------------------------------
        self._lora_config = None
        self._peft_models = {}  # store peft wrappers

        if lora_r is not None and lora_r > 0:
            if lora_target_modules is None:
                # Default targets for Qwen3.5: full attention + linear attention + MLP
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",       # full attention
                    "out_proj", "in_proj_qkv", "in_proj_z",        # linear attention
                    "gate_proj", "up_proj", "down_proj",            # MLP (both types)
                ]
            self._apply_lora(
                lora_r, lora_alpha, lora_dropout, lora_target_modules
            )
            logger.info(
                "Applied LoRA (r=%d, alpha=%d) to Core and Suffix layers. "
                "Target modules: %s",
                lora_r,
                lora_alpha,
                lora_target_modules,
            )
        else:
            logger.info("LoRA disabled — core and suffix layers are frozen.")

        # ------------------------------------------------------------------
        # Bridge v2
        # ------------------------------------------------------------------
        hidden_dim = self.config.hidden_size
        self.bridge = BridgeRegistry.create(
            bridge_type,
            hidden_dim=hidden_dim,
            step_embedding_dim=step_embedding_dim,
            num_iters=num_iters,
            alpha=bridge_alpha,
        )
        # Ensure bridge matches model dtype
        self.bridge = self.bridge.to(dtype=torch_dtype)

        # ------------------------------------------------------------------
        # Aggregation
        # ------------------------------------------------------------------
        self.aggregation_fn = get_aggregation(aggregation)

        # ------------------------------------------------------------------
        # Shared references (not copied — share memory with base model)
        # ------------------------------------------------------------------
        self.embed_tokens = lang.embed_tokens
        self.final_norm = lang.norm
        self.lm_head = self.base_model.lm_head
        self.rotary_emb = lang.rotary_emb

        # ------------------------------------------------------------------
        # Freeze non-trainable components
        # ------------------------------------------------------------------
        self._freeze_components()

        # ------------------------------------------------------------------
        # Stats
        # ------------------------------------------------------------------
        logger.info("RDT-Fixed-%d initialized.", num_iters)
        logger.info(
            "Trainable params: %s / %s",
            f"{self._count_trainable_params():,}",
            f"{self._count_total_params():,}",
        )

    # ------------------------------------------------------------------
    # LoRA utilities
    # ------------------------------------------------------------------

    def _apply_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: List[str],
    ) -> None:
        """Apply LoRA adapters to Core and Suffix layer groups.

        Uses the internal _inject_lora function to replace matching Linear
        layers with LoRALinear wrappers. This avoids peft compatibility
        issues with custom model architectures.

        Args:
            r: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: LoRA dropout rate.
            target_modules: List of submodule name substrings to target.
        """
        _inject_lora(self.core_group, target_modules, r, alpha, dropout)
        _inject_lora(self.suffix_group, target_modules, r, alpha, dropout)

        self._lora_config = {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": target_modules,
        }
        logger.info(
            "Injected LoRA (r=%d, alpha=%d) into Core and Suffix layers.",
            r, alpha,
        )

    def enable_lora(self) -> None:
        """Enable LoRA adapters on core and suffix layers."""
        for group_name in ["core_group", "suffix_group"]:
            group = getattr(self, group_name)
            for module in group.modules():
                if isinstance(module, LoRALinear):
                    for p in module.parameters():
                        p.requires_grad = True

    def disable_lora(self) -> None:
        """Disable LoRA adapters (freeze LoRA params)."""
        for group_name in ["core_group", "suffix_group"]:
            group = getattr(self, group_name)
            for module in group.modules():
                if isinstance(module, LoRALinear):
                    for p in module.parameters():
                        p.requires_grad = False

    # ------------------------------------------------------------------
    # Freezing utilities
    # ------------------------------------------------------------------

    def _freeze_components(self) -> None:
        """Freeze embedding, prefix, norm, and lm_head.

        Bridge, Core LoRA, and Suffix LoRA parameters remain trainable.
        """
        # Freeze prefix
        for param in self.prefix_group.parameters():
            param.requires_grad = False

        # Freeze embedding
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # Freeze norm and lm_head
        for param in self.final_norm.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Bridge is trainable by default (no freeze)

    def freeze_core(self) -> None:
        """Freeze core layers (used during Bridge Warmup Phase 1)."""
        for param in self.core_group.parameters():
            param.requires_grad = False

    def unfreeze_core(self) -> None:
        """Unfreeze core layers (used during SFT Phase 2)."""
        for param in self.core_group.parameters():
            param.requires_grad = True

    def freeze_suffix(self) -> None:
        """Freeze suffix layers."""
        for param in self.suffix_group.parameters():
            param.requires_grad = False

    def unfreeze_suffix(self) -> None:
        """Unfreeze suffix layers."""
        for param in self.suffix_group.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def _count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Position embeddings & mask computation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: bool = False,
        past_seq_len: int = 0,
    ) -> Tuple[
        torch.Tensor,                           # inputs_embeds
        Tuple[torch.Tensor, torch.Tensor],       # position_embeddings
        Dict[str, torch.Tensor],                 # masks_by_type
        Optional[torch.Tensor],                  # text_position_ids
    ]:
        """Prepare embeddings, position encodings, and attention masks.

        Args:
            input_ids: Token indices, shape (batch, seq_len).
            attention_mask: Optional attention mask, shape (batch, seq_len).
            cache_position: If True, position_ids are offset by past_seq_len
                (for incremental single-token decoding).
            past_seq_len: Number of cached past tokens (used when cache_position=True).

        Returns:
            Tuple of (inputs_embeds, position_embeddings, masks_by_type, position_ids).
        """
        lang = self.base_model.model
        config = self.config
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # --- Embed tokens ---
        inputs_embeds = self.embed_tokens(input_ids)

        # --- Position IDs with offset for incremental decoding ---
        offset = past_seq_len if cache_position else 0
        position_ids = torch.arange(offset, offset + seq_len, device=device)
        position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)

        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]

        # --- Rotary position embeddings (use correct positions) ---
        position_embeddings = self.rotary_emb(inputs_embeds, rope_position_ids)

        # --- Causal mask for full-attention layers ---
        # For incremental decoding, create_causal_mask handles cache_length internally
        # when past_key_values is provided with a non-zero cache length
        causal_mask = create_causal_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        # --- Linear attention mask ---
        linear_attn_mask = lang._update_linear_attn_mask(
            attention_mask, None
        )

        masks_by_type = {
            "linear_attention": linear_attn_mask,
            "full_attention": causal_mask,
        }

        return inputs_embeds, position_embeddings, masks_by_type, text_position_ids

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loop_outputs: bool = False,
        use_checkpoint: bool = False,
        core_no_grad: bool = False,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
    ) -> RDTOutput:
        """Forward pass through the RDT-Fixed-16 model.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            attention_mask: Optional attention mask of shape (batch, seq_len).
            labels: Optional labels for computing CE loss. -100 tokens are ignored.
            return_loop_outputs: If True, include all per-step hidden states.
            use_checkpoint: If True, use gradient checkpointing on core loop.
            core_no_grad: If True, run core layers under torch.no_grad().
            past_key_values: Dict of cached KV states from previous forward calls.
                Keys: "prefix", "core_1".."core_N", "suffix".
                Each value is a list of per-layer cache entries.
            use_cache: If True, return updated KV caches.

        Returns:
            RDTOutput with logits, optional loss, and (if use_cache) new_past_key_values.
        """
        # --- Prepare inputs ---
        use_kv_cache = past_key_values is not None and use_cache
        has_prefix_cache = use_kv_cache and "prefix" in past_key_values

        # Extract past sequence length from prefix KV cache for correct position IDs
        past_seq_len = 0
        if has_prefix_cache:
            prefix_kv = past_key_values["prefix"]
            if prefix_kv and len(prefix_kv) > 0 and prefix_kv[0] is not None:
                # Try to get seq_len from the first layer's cache
                try:
                    if hasattr(prefix_kv[0], 'get_seq_length'):
                        past_seq_len = prefix_kv[0].get_seq_length()
                    elif isinstance(prefix_kv[0], tuple) and len(prefix_kv[0]) == 2:
                        past_seq_len = prefix_kv[0][0].shape[-2]
                except Exception:
                    past_seq_len = 0

        (
            inputs_embeds,
            position_embeddings,
            masks_by_type,
            text_position_ids,
        ) = self._prepare_inputs(input_ids, attention_mask,
                                 cache_position=(has_prefix_cache and past_seq_len > 0),
                                 past_seq_len=past_seq_len)

        # --- Prefix pass ---
        prefix_cache = past_key_values.get("prefix", None) if use_kv_cache else None
        with torch.no_grad():
            prefix_out = self.prefix_group(
                inputs_embeds,
                position_embeddings=position_embeddings,
                masks_by_type=masks_by_type,
                position_ids=text_position_ids,
                past_key_values=prefix_cache,
                use_cache=use_cache,
            )
        if use_cache:
            hidden_states, new_prefix_cache = prefix_out
        else:
            hidden_states = prefix_out

        h_anchor = hidden_states.detach()

        # --- Core loop ---
        loop_outputs = []
        h_t = h_anchor
        new_core_caches = [] if use_cache else None

        for t in range(1, self.num_iters + 1):
            core_cache_key = f"core_{t}"
            core_cache = past_key_values.get(core_cache_key, None) if use_kv_cache else None

            z_t = self._core_step(
                h_t, position_embeddings, masks_by_type, text_position_ids,
                use_checkpoint=use_checkpoint,
                past_key_values=core_cache,
                use_cache=use_cache,
            )

            if use_cache:
                z_hidden, z_cache = z_t
                new_core_caches.append(z_cache)
            else:
                z_hidden = z_t

            # Bridge calibration
            h_t = self.bridge(h_anchor, z_hidden, t)
            loop_outputs.append(h_t)

        # --- Aggregation ---
        h_fused = self.aggregation_fn(loop_outputs)

        # --- Suffix pass ---
        suffix_cache = past_key_values.get("suffix", None) if use_kv_cache else None
        suffix_out = self.suffix_group(
            h_fused,
            position_embeddings=position_embeddings,
            masks_by_type=masks_by_type,
            position_ids=text_position_ids,
            past_key_values=suffix_cache,
            use_cache=use_cache,
        )
        if use_cache:
            h_fused, new_suffix_cache = suffix_out
        else:
            h_fused = suffix_out

        # --- Final norm + LM head ---
        h_fused = self.final_norm(h_fused)
        logits = self.lm_head(h_fused).float()

        # --- Build new KV cache dict ---
        new_past_key_values = None
        if use_cache:
            new_past_key_values = {"prefix": new_prefix_cache, "suffix": new_suffix_cache}
            for t in range(1, self.num_iters + 1):
                new_past_key_values[f"core_{t}"] = new_core_caches[t - 1]

        # --- Loss computation ---
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = RDTOutput(
            logits=logits,
            loss=loss,
            loop_outputs=loop_outputs if return_loop_outputs else None,
            h_anchor=h_anchor if return_loop_outputs else None,
        )
        # Attach KV cache to output (avoids changing RDTOutput dataclass)
        if new_past_key_values is not None:
            output.new_past_key_values = new_past_key_values
        return output

        return RDTOutput(
            logits=logits,
            loss=loss,
            loop_outputs=loop_outputs if return_loop_outputs else None,
            h_anchor=h_anchor if return_loop_outputs else None,
        )

    def _core_step(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        masks_by_type: Dict[str, torch.Tensor],
        position_ids: Optional[torch.Tensor],
        use_checkpoint: bool = False,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ):
        """Run one pass through the core layer group.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim).
            position_embeddings: Rotary position embeddings (cos, sin).
            masks_by_type: Attention masks dict.
            position_ids: Position IDs for full-attention layers.
            use_checkpoint: Use gradient checkpointing for memory efficiency.
            past_key_values: Optional per-layer KV cache for this core iteration.
            use_cache: If True, return updated KV caches.

        Returns:
            Output tensor, or (output, cache) if use_cache=True.
        """
        if use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.core_group,
                hidden_states,
                position_embeddings,
                masks_by_type,
                position_ids,
                past_key_values,
                use_cache,
                use_reentrant=False,
            )
        else:
            return self.core_group(
                hidden_states,
                position_embeddings=position_embeddings,
                masks_by_type=masks_by_type,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

    # ------------------------------------------------------------------
    # Fast generation with KV cache
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_kv(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Auto-regressive generation with KV cache support.

        Uses forward-level KV caching for Prefix and Suffix stages.
        Core loop recomputes each iteration (required since hidden states change).

        Speed vs naive generate(): ~20% faster with torch.compile + inference_mode.
        For production deployment speeds, use vLLM with custom model support.

        Args:
            input_ids: Prompt token indices (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            eos_token_id: End-of-sequence token ID.
            pad_token_id: Padding token ID.
            repetition_penalty: Penalty for repeated tokens (>1 = penalize).

        Returns:
            Full sequence including prompt, shape (batch, seq_len + new_tokens).
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generated = input_ids.clone()
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        # --- Prefill: full forward on the prompt with cache ---
        output = self.forward(
            input_ids=generated,
            use_cache=True,
            past_key_values=None,
        )
        past_kv = output.new_past_key_values
        next_logits = output.logits[:, -1, :]
        next_token = self._sample_token(
            next_logits, temperature, top_k, top_p, generated, repetition_penalty
        )

        # Handle finished sequences
        next_token = next_token * unfinished.long() + pad_token_id * (~unfinished).long()
        unfinished = unfinished & (next_token != eos_token_id)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        # --- Decode loop with KV cache ---
        for _ in range(max_new_tokens - 1):
            if not unfinished.any():
                break

            # Forward only the new token with cached K,V
            new_token_ids = next_token.unsqueeze(-1)  # (batch, 1)
            output = self.forward(
                input_ids=new_token_ids,
                use_cache=True,
                past_key_values=past_kv,
            )
            past_kv = output.new_past_key_values
            next_logits = output.logits[:, -1, :]

            next_token = self._sample_token(
                next_logits, temperature, top_k, top_p, generated, repetition_penalty
            )
            next_token = next_token * unfinished.long() + pad_token_id * (~unfinished).long()
            unfinished = unfinished & (next_token != eos_token_id)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        return generated

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        generated: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Sample a single token from logits."""
        batch_size = logits.shape[0]

        # Repetition penalty
        if repetition_penalty != 1.0:
            for b in range(batch_size):
                for token_id in set(generated[b].tolist()):
                    if logits[b, token_id] < 0:
                        logits[b, token_id] *= repetition_penalty
                    else:
                        logits[b, token_id] /= repetition_penalty

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                min_vals = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < min_vals] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_mask[..., 0] = False
                for b in range(batch_size):
                    logits[b, sorted_idx[b][sorted_mask[b]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Legacy generation (no KV cache, kept for compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate text using greedy or sampling-based decoding.

        This is a simple generation loop for prototype use. For production,
        consider using vLLM or the transformers GenerationMixin.

        Args:
            input_ids: Prompt token indices (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            eos_token_id: End-of-sequence token ID.
            pad_token_id: Padding token ID.
            repetition_penalty: Penalty for repeated tokens (>1 = penalize).

        Returns:
            Full sequence including prompt, shape (batch, seq_len + new_tokens).
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generated = input_ids.clone()
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if not unfinished.any():
                break

            # Forward pass
            output = self.forward(
                input_ids=generated,
                attention_mask=None,  # Will be inferred from padding
            )

            # Get logits for the last token
            next_logits = output.logits[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in set(generated[b].tolist()):
                        if next_logits[b, token_id] < 0:
                            next_logits[b, token_id] *= repetition_penalty
                        else:
                            next_logits[b, token_id] /= repetition_penalty

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature
                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    min_vals = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < min_vals] = float("-inf")
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_mask[..., 0] = False
                    for b in range(batch_size):
                        next_logits[b, sorted_idx[b][sorted_mask[b]]] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1)

            # Mark finished sequences
            next_token = next_token * unfinished.long() + pad_token_id * (~unfinished).long()
            unfinished = unfinished & (next_token != eos_token_id)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        return generated

    # ------------------------------------------------------------------
    # Trainable parameters helpers
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Get all trainable parameters with their names.

        Returns:
            List of (name, parameter) tuples.
        """
        return [
            (name, param)
            for name, param in self.named_parameters()
            if param.requires_grad
        ]

    def get_parameter_groups(
        self,
        bridge_lr: float = 2e-4,
        lora_lr: float = 5e-5,
        weight_decay: float = 0.01,
    ) -> List[Dict]:
        """Get parameter groups for optimizer configuration.

        Separates Bridge and LoRA parameters with different learning rates.

        Args:
            bridge_lr: Learning rate for Bridge v2 parameters.
            lora_lr: Learning rate for LoRA adapter parameters.
            weight_decay: Weight decay for non-bias/non-norm parameters.

        Returns:
            List of dicts suitable for torch.optim.AdamW or similar.
        """
        bridge_params = []
        lora_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bridge" in name:
                bridge_params.append(param)
            elif "lora" in name.lower():
                lora_params.append(param)
            else:
                # Any other trainable params get lora_lr
                lora_params.append(param)

        groups = []
        if bridge_params:
            groups.append({
                "params": bridge_params,
                "lr": bridge_lr,
                "weight_decay": weight_decay,
                "name": "bridge",
            })
        if lora_params:
            groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": weight_decay,
                "name": "lora",
            })

        return groups


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_rdt_fixed_16(
    base_model_path: str,
    num_iters: int = 16,
    prefix_layers: int = 12,
    core_layers: int = 16,
    lora_r: Optional[int] = 64,
    device_map: Optional[str] = None,
    **kwargs,
) -> RDTFixed16:
    """Create an RDT-Fixed-16 model from a Qwen3.5 base checkpoint.

    Args:
        base_model_path: Path to the base model directory.
        num_iters: Core loop iterations (default 16).
        prefix_layers: Number of prefix layers.
        core_layers: Number of core layers (reused in loop).
        lora_r: LoRA rank (None to disable LoRA).
        device_map: Device for model placement (None = CPU).
        **kwargs: Passed to RDTFixed16 constructor.

    Returns:
        Initialized RDTFixed16 model.
    """
    prefix_end = prefix_layers
    core_end = prefix_end + core_layers

    return RDTFixed16(
        base_model_path=base_model_path,
        num_iters=num_iters,
        prefix_end=prefix_end,
        core_end=core_end,
        lora_r=lora_r,
        device_map=device_map,
        **kwargs,
    )
