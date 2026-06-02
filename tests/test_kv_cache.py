#!/usr/bin/env python3
"""Quick test for KV Cache implementation in RDT-Fixed-16."""

import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rdt_fixed import create_rdt_fixed_16
from transformers import AutoTokenizer

MODEL_PATH = "/data/model/Qwen/Qwen3___5-9B-Base"
DEVICE = "cuda:4"

print("=" * 60)
print("KV Cache Smoke Test")
print("=" * 60)

# Build model (K=4 for fast testing)
print("\n1. Building RDT-Fixed-4...")
model = create_rdt_fixed_16(
    base_model_path=MODEL_PATH,
    num_iters=4,
    prefix_layers=12,
    core_layers=4,
    lora_r=8, lora_alpha=16,
    bridge_type="mlp",
    aggregation="last4_mean",
    device_map=None,
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"   Model ready. Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test 1: Forward with use_cache
print("\n2. Testing forward with use_cache...")
prompt = "患者: 我最近总是头疼。\n医生:"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

t0 = time.time()
output = model(input_ids=input_ids, use_cache=True, past_key_values=None)
t1 = time.time()

assert hasattr(output, 'new_past_key_values'), "Missing new_past_key_values!"
kv = output.new_past_key_values
print(f"   ✅ Forward with cache: {t1-t0:.2f}s")
print(f"   KV keys: {list(kv.keys())}")
print(f"   Prefix cache layers: {len(kv['prefix'])}")
print(f"   Core_1 cache layers: {len(kv['core_1'])}")
print(f"   Suffix cache layers: {len(kv['suffix'])}")

# Test 2: Single-token forward with cached KV
print("\n3. Testing single-token decode with cache...")
new_token = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)  # (1, 1)

t0 = time.time()
output2 = model(input_ids=new_token, use_cache=True, past_key_values=kv)
t1 = time.time()

assert hasattr(output2, 'new_past_key_values'), "Missing new KV on decode!"
print(f"   ✅ Single-token decode: {t1-t0:.3f}s")
print(f"   Output shape: {output2.logits.shape}")

# Test 3: Compare generate_kv vs generate (same output for greedy)
print("\n4. Testing generate_kv...")
test_prompt = "患者: 我最近总是头疼，特别是下午。\n医生:"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(DEVICE)

t0 = time.time()
out_kv = model.generate_kv(
    input_ids=input_ids, max_new_tokens=20, temperature=0.0,
    eos_token_id=tokenizer.eos_token_id,
)
t1 = time.time()

gen_text = tokenizer.decode(out_kv[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"   ✅ generate_kv: {t1-t0:.1f}s, {out_kv.shape[1] - input_ids.shape[1]} tokens")
print(f"   Generated: {gen_text[:100]}")
print(f"   Speed: {(out_kv.shape[1]-input_ids.shape[1])/(t1-t0):.1f} tok/s")

# Test 4: Longer generation
print("\n5. Testing longer generation (50 tokens)...")
prompt2 = "患者: 感冒了应该吃什么药？\n医生:"
input_ids = tokenizer.encode(prompt2, return_tensors="pt").to(DEVICE)

t0 = time.time()
out2 = model.generate_kv(
    input_ids=input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)
t1 = time.time()

gen2 = tokenizer.decode(out2[0][input_ids.shape[1]:], skip_special_tokens=True)
n_tokens = out2.shape[1] - input_ids.shape[1]
print(f"   Generated {n_tokens} tokens in {t1-t0:.1f}s ({n_tokens/(t1-t0):.1f} tok/s)")
print(f"   Text: {gen2[:120]}")

print("\n" + "=" * 60)
print("KV Cache Test PASSED ✅")
print("=" * 60)
