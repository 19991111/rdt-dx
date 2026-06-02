# RDT-Dx: Recursive Depth Transformer for Medical LLM

A medical diagnostic LLM prototype that enhances complex case reasoning by reusing intermediate Transformer layers with Bridge-based state calibration.

## Architecture

```
Input → Embedding → Prefix[0:11] → h_anchor
  → [ Core[12:27] × 16 + Bridge v2 ] loop
  → last4_mean → Suffix[28:31] → LM Head → Output
```

- **Base model**: Qwen3.5-9B (32 layers)
- **Trainable**: 138M params (1.52%) via LoRA + Bridge
- **Key innovation**: Computational depth replaces parameter scale

## Project Structure

```
├── src/                    # Core package
│   ├── rdt_fixed.py        # RDT-Fixed-16 model
│   ├── bridge.py           # Bridge v2 state calibrator
│   ├── aggregation.py      # Loop output aggregation
│   ├── data_utils.py       # Data loading utilities
│   ├── eval_utils.py       # Evaluation metrics
│   ├── train_bridge_warmup.py  # Phase 1 trainer
│   └── train_rdt_fixed.py      # Phase 2 trainer
├── configs/                # Training configs
├── experiments/            # Verification experiments
│   ├── phase_a_verify.py   # Structure feasibility
│   ├── phase_b_bridge_ablation.py  # Bridge comparison
│   └── phase_c_k_ablation.py       # K-step ablation
├── tests/                  # Unit tests
├── docs/                   # Documentation & reports
├── data/                   # Training data
└── output/                 # Checkpoints & logs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test
python tests/test_smoke.py --quick

# Build model
python -c "
from src import create_rdt_fixed_16
model = create_rdt_fixed_16(
    base_model_path='/data/model/Qwen/Qwen3___5-9B-Base',
    num_iters=16, prefix_layers=12, core_layers=16,
    bridge_type='mlp', lora_r=64
)
print(f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"
```

## Training Pipeline

| Phase | Script | Objective |
|-------|--------|-----------|
| 1: Bridge Warmup | `src/train_bridge_warmup.py` | Stabilize loop structure (KL loss) |
| 2: RDT SFT | `src/train_rdt_fixed.py` | Medical task training (CE + KL) |
| 3: GRPO RL | (future) | Medical quality optimization |

## Key Results

- **Phase B**: MLP Bridge controls drift to 0.37 (vs 54.6 without Bridge)
- **Phase C**: RDT-16 Hard CE = 0.43 vs SFT baseline 1.39 (3.2× better)
- **Parameter efficiency**: 138M RDT > 2.2B SFT on hard medical cases

See `docs/PRELIMINARY_REPORT.md` for full verification results.

## License

Research prototype — not for clinical use.
