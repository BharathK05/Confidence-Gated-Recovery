"""Model loading, kept in one place so every script loads models identically."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import pick_device, pick_dtype

# Small enough to iterate on quickly. Weeks 2+ swap to the 1.5B / 3B models.
DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAIN_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def load(model_id: str = DEV_MODEL, device: str | None = None, eager: bool = True):
    """Load a tokenizer and model ready for KV-cache work.

    eager=True forces the plain attention implementation. This matters:
    `sdpa` and `flash_attention_2` are faster but do NOT return attention
    weights, and later phases need those to score tokens. Keep it eager
    until the scoring code computes its own scores.
    """
    device = device or pick_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must pad on the left or generation breaks.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager" if eager else "sdpa",
    ).to(device)
    model.eval()

    return tokenizer, model


def kv_config(model) -> dict:
    """Pull out the numbers needed to compute KV-cache size by hand.

    num_key_value_heads < num_attention_heads means the model uses GQA
    (grouped-query attention), which every model in this project does.
    """
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "hidden_size": cfg.hidden_size,
        "uses_gqa": n_kv_heads < cfg.num_attention_heads,
        "dtype_bytes": torch.finfo(model.dtype).bits // 8,
    }
