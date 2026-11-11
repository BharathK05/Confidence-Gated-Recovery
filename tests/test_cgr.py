"""Correctness checks for tamc/cgr.py on the real Qwen2.5-0.5B weights (CPU, ~1-2 min).

Run:  python -m pytest tests/test_cgr.py -q -s
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tamc import cgr  # noqa: E402
from tamc.loader import load  # noqa: E402

TEXT = (
    "The harbour town of Velmora was founded in 1820 by salt traders. "
    "Its lighthouse keeper, Ilse Brandt, kept a ledger of every ship. "
    "The special magic number for Lisbon is 4815162. "
    "In winter the fishing fleet stayed in port and the market sold dried cod. "
    "The town council met on the first Monday of each month in the old customs house. "
) * 6


@pytest.fixture(scope="module")
def env():
    tok, model = load(device="cpu")
    tap = cgr.QueryTap(model)
    msgs = [{"role": "user", "content": TEXT + "\nWhat is the special magic number for Lisbon? Answer with the number only."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    pf = cgr.prefill(model, tap, ids, window=32, chunk=64)
    eos = set(model.generation_config.eos_token_id) | {tok.eos_token_id}
    return tok, model, tap, ids, pf, eos


def test_chunked_prefill_matches_single_pass(env):
    tok, model, tap, ids, pf, eos = env
    with torch.no_grad():
        ref = model(input_ids=ids[None]).logits[0, -1].float()
    assert torch.allclose(pf.first_logits, ref, atol=1e-3), (pf.first_logits - ref).abs().max()
    assert pf.kvs[0][0].shape[2] == ids.numel()


def test_prefill_scores_match_model_attention(env):
    """Prefill recomputes attention layer by layer; it must equal the model's own weights."""
    tok, model, tap, ids, pf, eos = env
    with torch.no_grad():
        attns = model(input_ids=ids[None], output_attentions=True).attentions
    for l in (0, len(attns) // 2, len(attns) - 1):
        att = attns[l][0].float().view(tap.n_kv, tap.group, pf.n, pf.n).mean(1)   # (n_kv, q, k)
        win = att[:, pf.n_ctx:, : pf.n_ctx].mean(1)
        acc = att.sum(1)[:, : pf.n_ctx]
        assert torch.allclose(pf.win_score[l], win, atol=2e-3), (l, (pf.win_score[l] - win).abs().max())
        assert torch.allclose(pf.acc_score[l], acc, atol=2e-2, rtol=1e-2), (l, (pf.acc_score[l] - acc).abs().max())


def test_identity_full_matches_hf_generate(env):
    tok, model, tap, ids, pf, eos = env
    with torch.no_grad():
        ref = model.generate(ids[None], max_new_tokens=16, do_sample=False)[0, ids.numel():].tolist()
    r = cgr.generate(model, tap, pf, cgr.GenConfig(method="full", max_new_tokens=16), eos)
    assert r.tokens == ref, (tok.decode(r.tokens), tok.decode(ref))


def test_ratio_one_equals_full(env):
    tok, model, tap, ids, pf, eos = env
    full = cgr.generate(model, tap, pf, cgr.GenConfig(method="full", max_new_tokens=16), eos)
    backup = cgr.make_backup(pf)
    for trig in ("never", "always"):
        r = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=1.0, trigger=trig, cooldown=0,
                                                       max_new_tokens=16), eos, backup)
        assert r.tokens == full.tokens, trig
        assert r.gpu_kv_bytes == cgr.kv_bytes(pf.kvs)


def test_recovery_scores_match_model_attention(env):
    """The hand-computed attention (hooked query + our RoPE) must equal the model's own."""
    tok, model, tap, ids, pf, eos = env
    dev = ids.device
    nxt = torch.tensor([int(pf.first_logits.argmax())], device=dev)
    pos = torch.tensor([pf.n], device=dev)
    with torch.no_grad():
        _, _, attns = cgr.forward(model, nxt, pos, pf.kvs, output_attentions=True)
    qs = tap.rotated_queries(pf.n)
    for l in (0, len(qs) // 2, len(qs) - 1):
        keys = pf.kvs[l][0][0].float()                       # cache before this token
        q = qs[l].view(tap.n_kv, tap.group, tap.head_dim)
        logits = torch.einsum("kgd,ktd->kgt", q, keys) / math.sqrt(tap.head_dim)
        ref = attns[l][0, :, 0, :-1].float().view(tap.n_kv, tap.group, -1)   # drop self-attention column
        # renormalise the reference over the same key set
        ref = ref / ref.sum(-1, keepdim=True)
        mine = torch.softmax(logits, -1)
        assert torch.allclose(mine, ref, atol=2e-3), (l, (mine - ref).abs().max())


def test_recovery_keeps_cache_size_and_selection_valid(env):
    tok, model, tap, ids, pf, eos = env
    backup = cgr.make_backup(pf)
    base = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.1, max_new_tokens=16), eos)
    r = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.1, trigger="always", cooldown=0,
                                                   max_new_tokens=16), eos, backup)
    assert r.gpu_kv_bytes == base.gpu_kv_bytes            # same number of slots as plain SnapKV
    assert r.kept_tokens == base.kept_tokens
    assert len(r.trig_steps) == len(r.tokens) - 1          # fired on every compressed step
    assert r.backup_bytes > 0


def test_selection_shapes_and_sinks(env):
    tok, model, tap, ids, pf, eos = env
    B = cgr.budget_tokens(pf.n_ctx, 0.1)
    for m in ("random", "streaming", "h2o", "snapkv"):
        idx = cgr.select(m, pf, B)
        assert idx.shape == (len(pf.kvs), tap.n_kv, B)
        assert (idx[..., : cgr.N_SINK] == torch.arange(cgr.N_SINK)).all(), m      # sinks kept
        assert (idx[..., 1:] > idx[..., :-1]).all(), m                            # sorted, no duplicates
        assert idx.max() < pf.n_ctx


def test_conf_gate_fires_only_below_threshold(env):
    tok, model, tap, ids, pf, eos = env
    backup = cgr.make_backup(pf)
    r = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.05, trigger="conf", tau=0.6,
                                                   cooldown=0, max_new_tokens=16), eos, backup)
    for s in range(1, len(r.conf)):
        assert (s in r.trig_steps) == (r.conf[s] < 0.6)
    never = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.05, trigger="conf", tau=0.0,
                                                       max_new_tokens=16), eos, backup)
    plain = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.05, max_new_tokens=16), eos)
    assert never.tokens == plain.tokens and not never.trig_steps
