"""A tiny randomly-initialised model for offline development and tests.

Real weights are a ~1 GB download. This is 39 M random parameters built from a
config, so it needs no network and runs in milliseconds on CPU.

It is not a useful language model - its outputs are gibberish. What it *does*
reproduce is the structure that the compression code has to handle correctly:
GQA (4 query heads per key/value head), multiple layers, and a real KV cache.
That is enough to catch shape bugs, off-by-one budget errors, and padding bugs,
which is where nearly all the real bugs live.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # tokenizer only, a few MB


def toy_model(num_layers: int = 2, num_heads: int = 8, num_kv_heads: int = 2):
    """Return (tokenizer, model) with the same structural quirks as the real thing."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=2048,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return tokenizer, model.to(torch.float32).eval()


if __name__ == "__main__":
    tokenizer, model = toy_model()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=4, do_sample=False, return_dict_in_generate=True
        )
    print("cache layers:", len(out.past_key_values))
    print("layer 0 key :", tuple(out.past_key_values[0][0].shape))
