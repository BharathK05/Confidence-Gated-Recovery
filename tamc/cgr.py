"""Confidence-Gated Recovery (CGR) for KV-cache compression.

Base method: SnapKV (Li et al., NeurIPS 2024). SnapKV scores context tokens by
the attention they receive from the last `window` prompt tokens, keeps the top
ones and deletes the rest for good.

This module adds one thing on top: the deleted tokens are kept in a backup, and
while the model generates we watch its next-token confidence. When confidence
drops below a threshold, we re-score the backup against the query the model is
using right now and swap the most relevant tokens back in. The number of
context slots on the GPU never changes.

Everything (baselines and our method) runs through the same code path, so the
comparison is like for like: same prompt, same window, same budget.

Works with Hugging Face Llama/Qwen2-style models (RoPE, grouped-query
attention) loaded with attn_implementation="eager".
"""

from __future__ import annotations

import gc
import io
import json
import math
import os
import random
import re
import string
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import DynamicCache

N_SINK = 4  # first tokens are never evicted (attention sinks)


# --------------------------------------------------------------------------
# Cache helpers. We keep the cache as a plain list of (K, V) per layer and only
# wrap it in a DynamicCache for the duration of one forward call. This avoids
# depending on DynamicCache internals, which changed between versions.
# --------------------------------------------------------------------------

def cache_to_list(cache):
    if hasattr(cache, "layers"):  # transformers >= 4.54
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):  # transformers <= 4.53
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer[0], layer[1]) for layer in cache]


def list_to_cache(kvs):
    cache = DynamicCache()
    for i, (k, v) in enumerate(kvs):
        cache.update(k, v, i)
    return cache


def forward(model, ids, pos, kvs, output_attentions=False):
    """One forward call. `ids` and `pos` are 1-D tensors of the new tokens and
    their ORIGINAL positions. Returns (logits of last token, new kv list, attentions)."""
    out = model(
        input_ids=ids[None],
        position_ids=pos[None],
        past_key_values=list_to_cache(kvs) if kvs else DynamicCache(),
        use_cache=True,
        output_attentions=output_attentions,
    )
    return out.logits[0, -1].float(), cache_to_list(out.past_key_values), out.attentions


def kv_bytes(kvs):
    """Exact KV-cache size in bytes, counted from the tensors themselves."""
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kvs)


# --------------------------------------------------------------------------
# Query capture. A forward hook on every q_proj stores the query vectors of the
# most recent forward call; recovery uses the last token's query.
# --------------------------------------------------------------------------

class QueryTap:
    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.group = self.n_heads // self.n_kv
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.layers = model.model.layers
        self.q = [None] * len(self.layers)
        self.handles = [
            layer.self_attn.q_proj.register_forward_hook(self._hook(i))
            for i, layer in enumerate(self.layers)
        ]
        # RoPE frequencies: read them from the model itself. Newer transformers versions
        # moved `rope_theta` inside `config.rope_parameters`, so reading the config
        # attribute silently fell back to a wrong default there.
        rot = getattr(model.model, "rotary_emb", None)
        if rot is not None and getattr(rot, "inv_freq", None) is not None:
            self.inv_freq = rot.inv_freq.detach().float().cpu().reshape(-1)
        else:
            theta = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
            assert theta, "cannot find the RoPE base of this model"
            d = self.head_dim
            self.inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        assert self.inv_freq.numel() == self.head_dim // 2

    def _hook(self, i):
        def fn(module, inputs, output):
            self.q[i] = output[0].detach()  # every token of the call, (T, n_heads * head_dim)
        return fn

    def rotated_queries(self, position: int):
        """Last-token queries of every layer with RoPE applied -> list of (n_heads, d) fp32."""
        freqs = position * self.inv_freq
        emb = torch.cat([freqs, freqs])
        cos, sin = emb.cos(), emb.sin()
        out = []
        for q in self.q:
            q = q[-1].float().view(self.n_heads, self.head_dim)
            cos_d, sin_d = cos.to(q.device), sin.to(q.device)
            half = self.head_dim // 2
            rot = torch.cat([-q[:, half:], q[:, :half]], dim=-1)
            out.append(q * cos_d + rot * sin_d)
        return out

    def rotated_chunk(self, layer: int, positions):
        """All queries of the last forward call for one layer, RoPE applied -> (T, n_heads, d) fp32."""
        q = self.q[layer].float().view(-1, self.n_heads, self.head_dim)
        freqs = positions.float()[:, None] * self.inv_freq.to(q.device)[None, :]
        emb = torch.cat([freqs, freqs], -1)[:, None, :]            # (T, 1, d)
        half = self.head_dim // 2
        rot = torch.cat([-q[..., half:], q[..., :half]], dim=-1)
        return q * emb.cos() + rot * emb.sin()

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
# Prefill: run the prompt once, uncompressed, and collect everything every
# method needs. Done once per example and shared by all methods and budgets.
# --------------------------------------------------------------------------

@dataclass
class Prefill:
    kvs: list            # full cache, per layer (K, V), shape (1, n_kv, n, d)
    n: int               # prompt length in tokens
    n_ctx: int           # compressible prefix length (= n - window)
    first_logits: torch.Tensor
    win_score: torch.Tensor   # (L, n_kv, n_ctx) mean attention from the window queries
    acc_score: torch.Tensor   # (L, n_kv, n_ctx) attention accumulated over ALL queries (H2O)
    seconds: float


@torch.no_grad()
def prefill(model, tap: QueryTap, input_ids, window=32, chunk=256):
    t0 = time.time()
    n = input_ids.numel()
    assert n > window + N_SINK + 8, "prompt too short to compress"
    n_ctx = n - window
    dev = input_ids.device
    L, n_kv, G = len(tap.layers), tap.n_kv, tap.group
    acc = torch.zeros(L, n_kv, n, device=dev)
    win = torch.zeros(L, n_kv, n_ctx, device=dev)
    kvs, logits = [], None
    bounds = list(range(0, n_ctx, chunk)) + [n_ctx, n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        pos = torch.arange(a, b, device=dev)
        logits, kvs, _ = forward(model, input_ids[a:b], pos, kvs)
        # The attention weights are recomputed here from the hooked queries and the
        # cached keys, one layer at a time. Asking the model for output_attentions
        # keeps every layer's (heads x chunk x prompt) matrix alive at once, which
        # does not fit on a 16 GB GPU for long prompts.
        future = torch.arange(b, device=dev)[None, :] > pos[:, None]          # (T, b) causal mask
        for l in range(L):
            q = tap.rotated_chunk(l, pos).view(b - a, n_kv, G, tap.head_dim)
            keys = kvs[l][0][0, :, :b].float()                                # (n_kv, b, d)
            att = torch.einsum("qkgd,ktd->kgqt", q, keys) / math.sqrt(tap.head_dim)
            att = torch.softmax(att.masked_fill_(future, float("-inf")), -1).mean(1)   # (n_kv, T, b)
            acc[l, :, :b] += att.sum(1)
            if a == n_ctx:  # this chunk is exactly the observation window
                win[l] = att[:, :, :n_ctx].mean(1)
            del q, keys, att
    # fp16 can overflow on long prompts (seen with Qwen2.5-1.5B on a T4: every logit NaN,
    # every answer "!!!!"). Stop here rather than write garbage results.
    if not torch.isfinite(logits).all():
        raise FloatingPointError("the model produced NaN/inf logits for this prompt; load it in float32")
    return Prefill(kvs, n, n_ctx, logits, win, acc[:, :, :n_ctx].contiguous(), time.time() - t0)


# --------------------------------------------------------------------------
# Token selection. Every method returns sorted indices (L, n_kv, B) into the
# compressible prefix. Sink tokens are always kept.
# --------------------------------------------------------------------------

def pool(score, kernel=7):
    """SnapKV's local pooling: keeps the neighbours of a high-scoring token."""
    L, H, T = score.shape
    return F.avg_pool1d(score.view(L * H, 1, T), kernel, stride=1, padding=kernel // 2).view(L, H, T)


def top_b(score, B):
    score = score.clone()
    score[..., :N_SINK] = float("inf")
    return score.topk(B, dim=-1).indices.sort(dim=-1).values


def budget_tokens(n_ctx, ratio):
    return max(N_SINK + 4, min(n_ctx, round(ratio * n_ctx)))


def select(method, pf: Prefill, B, seed=0, kernel=7):
    L, H, T = pf.win_score.shape
    dev = pf.win_score.device
    position = torch.arange(T, device=dev, dtype=torch.float32).expand(L, H, T)
    if method == "full":
        return torch.arange(T, device=dev).expand(L, H, T)
    if method == "random":
        g = torch.Generator().manual_seed(seed)
        return top_b(torch.rand(L, H, T, generator=g).to(dev), B)
    if method == "streaming":  # StreamingLLM: sinks + most recent tokens
        return top_b(position, B)
    if method == "h2o":  # H2O: half heavy hitters (accumulated attention), half recent
        score = pf.acc_score.clone()
        n_recent = B // 2
        score[..., T - n_recent:] = float("inf")
        return top_b(score, B)
    if method == "snapkv":
        return top_b(pool(pf.win_score, kernel), B)
    raise ValueError(method)


def gather_kv(kvs, idx, n_ctx, device=None):
    """Build the compressed cache: selected prefix tokens + everything after the prefix."""
    out = []
    for (k, v), ix in zip(kvs, idx):
        ix = ix.to(k.device)[None, :, :, None].expand(1, -1, -1, k.shape[-1])
        ck = torch.cat([k[:, :, :n_ctx].gather(2, ix).to(device or k.device), k[:, :, n_ctx:].to(device or k.device)], 2)
        cv = torch.cat([v[:, :, :n_ctx].gather(2, ix).to(device or v.device), v[:, :, n_ctx:].to(device or v.device)], 2)
        out.append((ck, cv))
    return out


# --------------------------------------------------------------------------
# Generation with optional confidence-gated recovery.
# --------------------------------------------------------------------------

@dataclass
class GenConfig:
    method: str = "snapkv"        # full | random | streaming | h2o | snapkv
    ratio: float = 0.1            # fraction of the compressible prefix that is kept
    trigger: str = "never"        # never | conf | always | random
    tau: float = 0.5              # conf: recover when max next-token prob < tau
    p_random: float = 0.0         # random: per-step trigger probability
    cooldown: int = 2             # steps to wait after a recovery
    kernel: int = 7
    max_new_tokens: int = 32
    seed: int = 0

    def name(self):
        if self.method == "full":
            return "full"
        tag = {"never": "", "conf": f"+cgr(tau={self.tau})", "always": "+always", "random": "+randtrig"}[self.trigger]
        return f"{self.method}{tag}"


@dataclass
class GenResult:
    tokens: list
    conf: list                    # confidence at each step, before any recovery
    trig_steps: list = field(default_factory=list)
    conf_after: list = field(default_factory=list)   # confidence after recovery, per trigger
    flipped: int = 0              # triggers where recovery changed the chosen token
    swapped: float = 0.0          # mean fraction of slots replaced per recovery
    kept_tokens: int = 0
    gpu_kv_bytes: int = 0
    backup_bytes: int = 0
    gen_seconds: float = 0.0
    rec_seconds: float = 0.0


@torch.no_grad()
def recover(tap: QueryTap, backup, tail_kvs, keep_score, position, B, kernel=7):
    """Re-score every backed-up prefix token against the current query and
    return (new indices, updated scores).

    backup:     per layer (K, V) of the full prefix, held off the GPU
    tail_kvs:   per layer (K, V) of the tokens after the prefix (window + generated)
    keep_score: (L, n_kv, n_ctx) the scores the current selection was made from

    For each layer we compute the attention the current query WOULD put on every
    prefix token if the whole prefix were still in the cache. Only one layer's
    keys are on the GPU at a time. The new score is max(old, new), so a token
    that mattered for the prompt is not thrown out by one unrelated query.
    """
    qs = tap.rotated_queries(position)
    n_kv, G, d = tap.n_kv, tap.group, tap.head_dim
    new = torch.empty_like(keep_score)
    for l, q in enumerate(qs):
        dev = q.device
        k_ctx = backup[l][0][0].to(dev).float()            # (n_kv, n_ctx, d)
        k_tail = tail_kvs[l][0][0].float()                 # (n_kv, t, d)
        keys = torch.cat([k_ctx, k_tail], 1)               # (n_kv, n_ctx + t, d)
        q = q.view(n_kv, G, d)
        att = torch.softmax(torch.einsum("kgd,ktd->kgt", q, keys) / math.sqrt(d), -1)
        new[l] = att[:, :, : k_ctx.shape[1]].mean(1).to(new.device)
        del k_ctx, keys
    score = torch.maximum(keep_score, pool(new, kernel))
    return top_b(score, B), score


@torch.no_grad()
def generate(model, tap: QueryTap, pf: Prefill, cfg: GenConfig, eos_ids, backup=None):
    """Greedy decoding from a compressed cache.

    As in SnapKV, the first token comes from the uncompressed prefill pass and
    compression applies from the second token on. This is identical for all methods.
    `backup` is the prefix KV on the backup device (CPU on a GPU run); it is
    only needed when cfg.trigger != "never".
    """
    dev = pf.first_logits.device
    n_ctx = pf.n_ctx
    B = n_ctx if cfg.method == "full" else budget_tokens(n_ctx, cfg.ratio)
    idx = select(cfg.method, pf, B, cfg.seed, cfg.kernel)
    kvs = gather_kv(pf.kvs, idx, n_ctx)
    res = GenResult(tokens=[], conf=[], kept_tokens=B + (pf.n - n_ctx))
    res.gpu_kv_bytes = kv_bytes(kvs)
    use_rec = cfg.trigger != "never" and cfg.method != "full"
    if use_rec:
        assert backup is not None
        res.backup_bytes = kv_bytes(backup)
        keep_score = pool(pf.win_score, cfg.kernel) if cfg.method == "snapkv" else torch.zeros_like(pf.win_score)
    rng = random.Random(cfg.seed)
    t0 = time.time()

    probs = torch.softmax(pf.first_logits, -1)
    res.conf.append(probs.max().item())
    tok = int(probs.argmax())
    res.tokens.append(tok)
    wait = 0
    for step in range(1, cfg.max_new_tokens):
        if tok in eos_ids:
            break
        pos = pf.n + step - 1
        ids_t = torch.tensor([tok], device=dev)
        pos_t = torch.tensor([pos], device=dev)
        logits, new_kvs, _ = forward(model, ids_t, pos_t, kvs)
        probs = torch.softmax(logits, -1)
        conf = probs.max().item()
        res.conf.append(conf)

        fire = False
        if use_rec and wait == 0:
            if cfg.trigger == "always":
                fire = True
            elif cfg.trigger == "conf":
                fire = conf < cfg.tau
            elif cfg.trigger == "random":
                fire = rng.random() < cfg.p_random
        wait = max(0, wait - 1)

        if fire:
            t1 = time.time()
            tail = [(k[:, :, B:], v[:, :, B:]) for k, v in kvs]   # before this step's token
            new_idx, keep_score = recover(tap, backup, tail, keep_score, pos, B, cfg.kernel)
            changed = 1.0 - (new_idx[..., None] == idx[..., None, :]).any(-1).float().mean().item()
            idx = new_idx
            ctx = gather_kv(backup, idx, n_ctx, device=dev)        # prefix slots only (backup has no tail)
            kvs = [(torch.cat([ck, tk], 2), torch.cat([cv, tv], 2)) for (ck, cv), (tk, tv) in zip(ctx, tail)]
            logits, new_kvs, _ = forward(model, ids_t, pos_t, kvs)  # decode the same step again
            probs2 = torch.softmax(logits, -1)
            res.trig_steps.append(step)
            res.conf_after.append(probs2.max().item())
            res.flipped += int(probs2.argmax() != probs.argmax())
            res.swapped += changed
            probs = probs2
            wait = cfg.cooldown
            res.rec_seconds += time.time() - t1
        kvs = new_kvs
        tok = int(probs.argmax())
        res.tokens.append(tok)

    assert all(k.shape[2] == B + (pf.n - n_ctx) + len(res.tokens) - 1 for k, _ in kvs), "cache size drifted"
    if res.trig_steps:
        res.swapped /= len(res.trig_steps)
    res.gen_seconds = time.time() - t0
    return res


def make_backup(pf: Prefill, device="cpu"):
    """Copy of the compressible prefix, kept off the GPU."""
    return [(k[:, :, : pf.n_ctx].to(device), v[:, :, : pf.n_ctx].to(device)) for k, v in pf.kvs]


# --------------------------------------------------------------------------
# Data: LongBench QA sets and a synthetic two-needle test.
# --------------------------------------------------------------------------

QA_PROMPT = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    "\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
)
PROMPTS = {
    "hotpotqa": QA_PROMPT,
    "2wikimqa": QA_PROMPT,
    "musique": QA_PROMPT,
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, using a "
        "single phrase or sentence if possible. If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on "
        "the above article as concisely as you can, using a single phrase or sentence if possible. If the question "
        "cannot be answered based on the information in the article, write \"unanswerable\". If the question is a "
        "yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "needle": (
        "Some special magic numbers are hidden in the following text. Read it carefully.\n\n{context}\n\n"
        "Answer with the two numbers only. What are the special magic numbers for {input}?\nAnswer:"
    ),
}
MAX_NEW = {"hotpotqa": 32, "2wikimqa": 32, "musique": 32, "qasper": 128, "needle": 24}


def load_longbench(name):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        with z.open(f"data/{name}.jsonl") as f:
            return [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8")]


CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tallinn", "Hanoi", "Bogota", "Tunis", "Perth", "Oslo",
          "Lagos", "Minsk", "Havana", "Dhaka", "Zagreb", "Accra", "Kyoto", "Lima", "Riga", "Doha"]


def make_needles(n_examples, source_texts, tok, ctx_tokens, seed=0):
    """Two 'magic number' sentences hidden at random depths in real text.
    The text is cut to `ctx_tokens` BEFORE the needles go in, so truncation can never remove them."""
    rng = random.Random(seed)
    out = []
    for i in range(n_examples):
        text = source_texts[i % len(source_texts)]
        text = tok.decode(tok(text, add_special_tokens=False).input_ids[:ctx_tokens], skip_special_tokens=True)
        paras = [p for p in text.split("\n") if p.strip()]
        a, b = rng.sample(CITIES, 2)
        na, nb = rng.randint(1000000, 9999999), rng.randint(1000000, 9999999)
        for city, num in ((a, na), (b, nb)):
            paras.insert(rng.randint(1, max(1, len(paras) - 1)), f"The special magic number for {city} is {num}.")
        out.append({"context": "\n".join(paras), "input": f"{a} and {b}", "answers": [str(na), str(nb)]})
    return out


def build_prompt(tok, name, ex, max_len):
    """LongBench-style: fill the template, cut the middle if too long, wrap in the chat template."""
    prompt = PROMPTS[name].format(context=ex["context"], input=ex["input"])
    ids = tok(prompt, add_special_tokens=False).input_ids
    if len(ids) > max_len:
        half = max_len // 2
        prompt = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


# --------------------------------------------------------------------------
# Metrics (LongBench's QA F1).
# --------------------------------------------------------------------------

def normalize_answer(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, answers):
    best = 0.0
    p = normalize_answer(pred).split()
    for a in answers:
        g = normalize_answer(a).split()
        same = sum((Counter(p) & Counter(g)).values())
        if same == 0:
            continue
        prec, rec = same / len(p), same / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_output(name, pred, answers):
    if name == "needle":
        return sum(a in pred for a in answers) / len(answers)
    return qa_f1(pred, answers)


# --------------------------------------------------------------------------
# Experiment runner. Appends one JSON line per (example, config) and skips
# anything already in the file, so a dead Colab session loses nothing.
# --------------------------------------------------------------------------

def config_grid(ratios, taus):
    grid = [GenConfig(method="full")]
    for r in ratios:
        for m in ("random", "streaming", "h2o", "snapkv"):
            grid.append(GenConfig(method=m, ratio=r))
        for tau in taus:
            grid.append(GenConfig(method="snapkv", ratio=r, trigger="conf", tau=tau))
        grid.append(GenConfig(method="snapkv", ratio=r, trigger="always", cooldown=0))
        grid.append(GenConfig(method="snapkv", ratio=r, trigger="random"))  # p set per example
    return grid


def run_key(dataset, i, cfg):
    return f"{dataset}|{i}|{cfg.name()}|{cfg.ratio if cfg.method != 'full' else 1.0}"


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass  # half-written last line from a killed session
    return done


def run_dataset(model, tok, tap, dataset, examples, out_path, ratios, taus, max_len=6000,
                window=32, ref_tau=0.7, backup_device="cpu", log_every=5):
    """Run every config on every example. `ref_tau` is the threshold whose
    trigger rate the random-trigger ablation copies, example by example."""
    done = load_done(out_path)
    grid = config_grid(ratios, taus)
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
    dev = next(model.parameters()).device
    t_start = time.time()
    for i, ex in enumerate(examples):
        todo = [c for c in grid if run_key(dataset, i, c) not in done]
        if not todo:
            continue
        ids = build_prompt(tok, dataset, ex, max_len).to(dev)
        pf = backup = None
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        pf = prefill(model, tap, ids, window=window)
        backup = make_backup(pf, backup_device)
        ref_rate = {}  # ratio -> trigger rate of the reference CGR run on this example
        # reference CGR runs first so the random-trigger ablation can copy their rate
        todo.sort(key=lambda c: (c.trigger == "random", not (c.trigger == "conf" and c.tau == ref_tau)))
        with open(out_path, "a", encoding="utf-8") as f:
            for cfg in todo:
                cfg.max_new_tokens = MAX_NEW[dataset]
                cfg.seed = i
                if cfg.trigger == "random":
                    if cfg.ratio not in ref_rate:
                        continue  # reference run was done in an earlier session; rate unknown -> skip
                    cfg.p_random = ref_rate[cfg.ratio]
                r = generate(model, tap, pf, cfg, eos, backup if cfg.trigger != "never" else None)
                steps = max(1, len(r.tokens) - 1)
                if cfg.trigger == "conf" and cfg.tau == ref_tau:
                    ref_rate[cfg.ratio] = len(r.trig_steps) / steps
                text = tok.decode(r.tokens, skip_special_tokens=True).strip()
                row = {
                    "key": run_key(dataset, i, cfg), "dataset": dataset, "idx": i, "name": cfg.name(),
                    "method": cfg.method, "trigger": cfg.trigger, "tau": cfg.tau if cfg.trigger == "conf" else None,
                    "ratio": cfg.ratio if cfg.method != "full" else 1.0,
                    "score": score_output(dataset, text, ex["answers"]), "pred": text,
                    "n_prompt": pf.n, "kept_tokens": r.kept_tokens, "steps": steps,
                    "n_trig": len(r.trig_steps), "trig_steps": r.trig_steps, "flipped": r.flipped,
                    "swapped": r.swapped, "conf": [round(c, 4) for c in r.conf],
                    "conf_after": [round(c, 4) for c in r.conf_after],
                    "gpu_kv_bytes": r.gpu_kv_bytes, "full_kv_bytes": kv_bytes(pf.kvs),
                    "backup_bytes": r.backup_bytes, "gen_seconds": r.gen_seconds,
                    "rec_seconds": r.rec_seconds, "prefill_seconds": pf.seconds,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
        del pf, backup
        if (i + 1) % log_every == 0 or i + 1 == len(examples):
            mem = f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev.type == "cuda" else ""
            print(f"[{dataset}] {i + 1}/{len(examples)} examples, {time.time() - t_start:.0f}s{mem}", flush=True)
