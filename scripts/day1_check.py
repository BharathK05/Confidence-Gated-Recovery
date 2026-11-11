"""Day 1 check: environment, model load, generation, KV cache shape.

Run from the tamc/ folder:
    python scripts/day1_check.py

Passes on CPU (slow), MPS, or CUDA. Nothing here is project-specific yet -
it only proves the plumbing works before anything is built on top.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tamc.env import report
from tamc.loader import DEV_MODEL, kv_config, load

PROMPT = "Explain what a KV cache is in a transformer, in two sentences."


def main() -> int:
    report()

    print("\nLoading model (first run downloads ~1 GB)...")
    t0 = time.time()
    tokenizer, model = load(DEV_MODEL)
    print(f"  loaded {DEV_MODEL} in {time.time() - t0:.1f}s")

    cfg = kv_config(model)
    print("\nMODEL SHAPE")
    for key, value in cfg.items():
        print(f"  {key:<22}: {value}")
    if cfg["uses_gqa"]:
        group = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        print(f"  -> GQA: {group} query heads share each key/value head")

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]

    print(f"\nGenerating (prompt = {n_prompt} tokens)...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,          # greedy: required for reproducibility
            return_dict_in_generate=True,
            use_cache=True,
        )
    elapsed = time.time() - t0

    answer = tokenizer.decode(
        out.sequences[0][n_prompt:], skip_special_tokens=True
    ).strip()
    n_new = out.sequences.shape[1] - n_prompt

    print("\nOUTPUT")
    print("-" * 52)
    print(answer)
    print("-" * 52)
    print(f"  {n_new} tokens in {elapsed:.1f}s ({n_new / elapsed:.1f} tok/s)")

    # The cache is what the whole project compresses. Confirm it is reachable
    # and that its layer count matches the config.
    cache = out.past_key_values
    n_layers = len(cache) if not hasattr(cache, "layers") else len(cache.layers)
    key_0 = cache[0][0]
    print("\nKV CACHE")
    print(f"  layers in cache        : {n_layers} (config says {cfg['num_layers']})")
    print(f"  layer 0 key shape      : {tuple(key_0.shape)}")
    print(f"  -> (batch, kv_heads, tokens, head_dim)")

    assert n_layers == cfg["num_layers"], "cache layer count != config layer count"
    assert key_0.shape[1] == cfg["num_key_value_heads"], "unexpected kv head count"
    assert key_0.shape[2] == n_prompt + n_new - 1, "unexpected cached token count"

    print("\nDAY 1 PASSED - environment, model, generation, and cache all work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
