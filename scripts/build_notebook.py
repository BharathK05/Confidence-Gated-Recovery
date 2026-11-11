"""Build notebooks/CGR_KV_Cache_Recovery.ipynb from tamc/cgr.py.

The notebook embeds the module verbatim (a %%writefile cell), so the code that
runs on Colab is byte-for-byte the code the tests in tests/test_cgr.py cover.

Run:  python scripts/build_notebook.py
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = open(os.path.join(ROOT, "tamc", "cgr.py"), encoding="utf-8").read()

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# Confidence-Gated Recovery for KV-Cache Compression

**Base paper:** SnapKV (Li et al., NeurIPS 2024, arXiv:2404.14469).
SnapKV keeps the context tokens that the last 32 prompt tokens attend to and deletes the rest for good.

**What this notebook adds:** the deleted tokens are kept in a backup in CPU RAM. During generation we watch
the model's next-token confidence. When it drops below a threshold `tau`, we re-score the backup against the
query the model is using at that moment and swap the most relevant tokens back in. The number of context
tokens on the GPU never changes.

**How to run**
1. `Runtime -> Change runtime type -> T4 GPU`.
2. `Runtime -> Run all`. Allow the Google Drive prompt (results are saved there after every example).
3. If Colab disconnects, just `Run all` again. Finished work is skipped.

The last section builds every table and figure for the paper and zips them.
""")

md("## 1. Settings")
code(r"""
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS   = ["needle", "hotpotqa", "2wikimqa", "qasper"]
N_EXAMPLES = 30                     # per dataset. Lower this first if you are short on time.
MAX_LEN    = 6000                   # prompt is cut in the middle to this many tokens (LongBench style)
RATIOS     = [0.02, 0.05, 0.10]     # fraction of the context kept on the GPU
TAUS       = [0.5, 0.7, 0.9]        # confidence thresholds for recovery
HEAD_TAU   = 0.7                    # the threshold reported as "ours" in the main table (fixed before running)
WINDOW     = 32                     # SnapKV observation window
USE_DRIVE  = True                   # save results to Google Drive so a disconnect loses nothing
RUN_NAME   = "cgr_qwen1.5b_fp32"    # new name: the earlier fp16 run produced NaN and must not be resumed
""")

md("## 2. Environment")
code(r"""
import os, sys, gc, json, time, random, platform
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, transformers
print("python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__)
assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> T4 GPU"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

BASE = next(d for d in ("/content", "/kaggle/working", ".") if os.path.isdir(d))   # Colab, Kaggle, anywhere else
OUT_DIR = f"{BASE}/{RUN_NAME}"
if USE_DRIVE and BASE == "/content":
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = f"/content/drive/MyDrive/{RUN_NAME}"
    except Exception as e:
        print("Drive not mounted, saving to /content instead:", e)
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS = f"{OUT_DIR}/results.jsonl"
print("results ->", RESULTS)
""")

md("""
## 3. The method

One file holds everything: the shared prefill, the four baselines, our recovery step, the data loaders,
the metric and the crash-safe runner. It is written to disk and imported so the same code can be unit-tested
outside Colab.
""")
code("%%writefile cgr.py\n" + SRC)

md("## 4. Load the model")
code(r"""
import importlib, cgr
importlib.reload(cgr)
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
# float32, not float16: in fp16 this model returned NaN for every long prompt on the T4 (all answers "!!!!").
# The 1.5B model in fp32 needs about 6 GB, which fits. eager attention: sanity check 4 reads the attention weights.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
tap = cgr.QueryTap(model)
EOS = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
          else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
print("layers", len(tap.layers), "| query heads", tap.n_heads, "| KV heads", tap.n_kv, "| head dim", tap.head_dim)
""")

md("""
## 5. Sanity checks

These must pass before any number is trusted.
1. Our decoding loop with the full cache gives exactly the same tokens as `model.generate`.
2. A 100% budget gives exactly the full-cache output, with and without recovery.
3. Recovery never changes the size of the GPU cache.
4. The attention we compute by hand for recovery matches the model's own attention.
""")
code(r"""
import math
story = ("The harbour town of Velmora was founded in 1820 by salt traders. Its lighthouse keeper, Ilse Brandt, "
         "kept a ledger of every ship. The special magic number for Lisbon is 4815162. In winter the fishing fleet "
         "stayed in port and the market sold dried cod. The council met on the first Monday of each month. ") * 8
msgs = [{"role": "user", "content": story + "\nWhat is the special magic number for Lisbon? Answer with the number only."}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          add_special_tokens=False, return_tensors="pt").input_ids[0].cuda()
pf = cgr.prefill(model, tap, ids, window=WINDOW)
backup = cgr.make_backup(pf)

with torch.no_grad():
    ref = model.generate(ids[None], max_new_tokens=16, do_sample=False)[0, ids.numel():].tolist()
full = cgr.generate(model, tap, pf, cgr.GenConfig(method="full", max_new_tokens=16), EOS)
assert full.tokens == ref, "decoding loop differs from model.generate"
print("1. identity with model.generate: OK ->", repr(tok.decode(full.tokens, skip_special_tokens=True)))

for trig in ("never", "always"):
    r = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=1.0, trigger=trig, cooldown=0, max_new_tokens=16), EOS, backup)
    assert r.tokens == full.tokens and r.gpu_kv_bytes == cgr.kv_bytes(pf.kvs)
print("2. 100% budget equals full cache: OK")

base = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.1, max_new_tokens=16), EOS)
rec = cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.1, trigger="always", cooldown=0, max_new_tokens=16), EOS, backup)
assert rec.gpu_kv_bytes == base.gpu_kv_bytes and len(rec.trig_steps) == len(rec.tokens) - 1
print(f"3. GPU cache size unchanged by recovery: OK ({base.gpu_kv_bytes / 2**20:.2f} MB of {cgr.kv_bytes(pf.kvs) / 2**20:.2f} MB)")

nxt, pos = torch.tensor([full.tokens[0]]).cuda(), torch.tensor([pf.n]).cuda()
with torch.no_grad():
    _, _, attns = cgr.forward(model, nxt, pos, pf.kvs, output_attentions=True)
qs, worst = tap.rotated_queries(pf.n), 0.0
for l in range(len(qs)):
    q = qs[l].view(tap.n_kv, tap.group, tap.head_dim)
    mine = torch.softmax(torch.einsum("kgd,ktd->kgt", q, pf.kvs[l][0][0].float()) / math.sqrt(tap.head_dim), -1)
    theirs = attns[l][0, :, 0, :-1].float().view(tap.n_kv, tap.group, -1)
    worst = max(worst, (mine - theirs / theirs.sum(-1, keepdim=True)).abs().max().item())
assert worst < 2e-2, worst
print(f"4. hand-computed attention matches the model: OK (max abs diff {worst:.1e})")
del pf, backup, attns; gc.collect(); torch.cuda.empty_cache()

# 5. Memory: one worst-case prompt (MAX_LEN tokens) through the heaviest configuration, before the long run.
torch.cuda.reset_peak_memory_stats()
long_ids = torch.randint(1000, 20000, (MAX_LEN,), generator=torch.Generator().manual_seed(0)).cuda()
pf = cgr.prefill(model, tap, long_ids, window=WINDOW)
backup = cgr.make_backup(pf)
cgr.generate(model, tap, pf, cgr.GenConfig(method="full", max_new_tokens=8), EOS)
cgr.generate(model, tap, pf, cgr.GenConfig(method="snapkv", ratio=0.2, trigger="always", cooldown=0, max_new_tokens=8), EOS, backup)
peak = torch.cuda.max_memory_allocated() / 2**30
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"5. memory for a {MAX_LEN}-token prompt: peak {peak:.1f} GB of {total:.1f} GB, prefill {pf.seconds:.0f}s", "OK" if peak < 0.8 * total else "TOO HIGH")
assert peak < 0.8 * total, "lower MAX_LEN before running the experiments"
del pf, backup, long_ids; gc.collect(); torch.cuda.empty_cache()
""")

md("""
## 6. Data

Three LongBench question-answering sets and a synthetic two-needle test (two "magic number" sentences hidden in
real text; the score is the fraction of the two numbers the model returns). Examples are picked with a fixed seed.
""")
code(r"""
data = {}
hp = cgr.load_longbench("hotpotqa")
random.Random(0).shuffle(hp)
for name in DATASETS:
    if name == "needle":   # haystacks come from hotpotqa passages that are NOT used in the hotpotqa evaluation
        data[name] = cgr.make_needles(N_EXAMPLES, [e["context"] for e in hp[N_EXAMPLES:2 * N_EXAMPLES]], tok, MAX_LEN - 300)
    elif name == "hotpotqa":
        data[name] = hp[:N_EXAMPLES]
    else:
        rows = cgr.load_longbench(name)
        random.Random(0).shuffle(rows)
        data[name] = rows[:N_EXAMPLES]
for name, rows in data.items():
    n_tok = [cgr.build_prompt(tok, name, ex, MAX_LEN).numel() for ex in rows[:10]]
    print(f"{name:10s} {len(rows)} examples, ~{sum(n_tok) // len(n_tok)} prompt tokens")
""")

md("""
## 7. Run the experiments

Per example the prompt is processed once, uncompressed. Every method and budget then starts from that same
cache, so all methods see exactly the same prompt. Configurations per example:

- `full`: no compression
- `random`, `streaming` (StreamingLLM), `h2o` (H2O), `snapkv` (SnapKV): baselines
- `snapkv+cgr(tau=...)`: ours, recovery when confidence < tau
- `snapkv+always`: recovery on every step (upper bound, the cost existing retrieval methods pay)
- `snapkv+randtrig`: recovery at random steps, as often as ours fired on that example (is confidence the right signal?)
""")
code(r"""
# Pilot: two real examples, printed, before the long run. The full-cache answers must be sensible text.
# If they are empty or look like "!!!!", STOP and do not run the next cell.
for name in DATASETS[:2]:
    ex = data[name][0]
    pf = cgr.prefill(model, tap, cgr.build_prompt(tok, name, ex, MAX_LEN).cuda(), window=WINDOW)
    backup = cgr.make_backup(pf)
    print(f"[{name}] {pf.n} prompt tokens, prefill {pf.seconds:.1f}s | gold: {ex['answers'][:2]}")
    for cfg in (cgr.GenConfig(method="full"), cgr.GenConfig(method="snapkv", ratio=0.05),
                cgr.GenConfig(method="snapkv", ratio=0.05, trigger="conf", tau=HEAD_TAU)):
        cfg.max_new_tokens = cgr.MAX_NEW[name]
        r = cgr.generate(model, tap, pf, cfg, EOS, backup)
        text = tok.decode(r.tokens, skip_special_tokens=True).strip()
        print(f"   {cfg.name():22s} score {cgr.score_output(name, text, ex['answers']):.2f} | {len(r.trig_steps)} recoveries | {text[:80]!r}")
        if cfg.method == "full":
            assert any(ch.isalnum() for ch in text), "the full-cache answer is not text; do not run the experiments"
    del pf, backup; gc.collect(); torch.cuda.empty_cache()
print("pilot OK")
""")

code(r"""
grid = cgr.config_grid(RATIOS, TAUS)
print(len(grid), "configurations per example,", sum(len(v) for v in data.values()), "examples")
t0 = time.time()
for name in DATASETS:
    cgr.run_dataset(model, tok, tap, name, data[name], RESULTS, RATIOS, TAUS,
                    max_len=MAX_LEN, window=WINDOW, ref_tau=HEAD_TAU)
print(f"done in {(time.time() - t0) / 60:.1f} min")
""")

md("""
## 8. Results

Everything below is computed from `results.jsonl` only. Only examples for which every configuration finished
are used, so all comparisons are paired.
""")
code(r"""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)

df = pd.read_json(open(RESULTS, encoding="utf-8"), lines=True)
per_ex = df.groupby(["dataset", "idx"]).name.count()
complete = per_ex[per_ex == per_ex.max()].index
df = df.set_index(["dataset", "idx"]).loc[complete].reset_index()
print("complete examples per dataset:", df[df.name == "full"].groupby("dataset").idx.count().to_dict())

HEAD = f"snapkv+cgr(tau={HEAD_TAU})"
df["rate"] = df.n_trig / df.steps
full_pred = df[df.name == "full"].set_index(["dataset", "idx"]).pred
# fidelity: token-F1 between a method's answer and the uncompressed model's answer on the same example
df["fidelity"] = [cgr.qa_f1(p, [full_pred[(d, i)]]) if full_pred[(d, i)].strip() else float(not p.strip())
                  for p, d, i in zip(df.pred, df.dataset, df.idx)]
df["ms_per_token"] = 1000 * df.gen_seconds / df.steps

LABEL = {"full": "Full cache", "random": "Random", "streaming": "StreamingLLM", "h2o": "H2O", "snapkv": "SnapKV",
         HEAD: "SnapKV + CGR (ours)", "snapkv+randtrig": "SnapKV + random trigger", "snapkv+always": "SnapKV + always recover"}
main = df[df.name.isin(LABEL)].copy()
main["Method"] = main.name.map(LABEL)
order = list(LABEL.values())

def table(value):
    t = main[main.name != "full"].pivot_table(index="Method", columns=["dataset", "ratio"], values=value, aggfunc="mean") * 100
    avg = main[main.name != "full"].pivot_table(index="Method", columns="ratio", values=value, aggfunc="mean") * 100
    avg.columns = pd.MultiIndex.from_product([["AVERAGE"], avg.columns])
    return pd.concat([t, avg], axis=1).reindex([m for m in order if m in t.index]).round(1)

t1 = table("score")
print("\nTABLE 1. Task score x100 (F1 for QA, recall for needle). Columns: dataset / fraction of context kept.")
print(t1.to_string())
print("\nFull cache:", (main[main.name == "full"].groupby("dataset").score.mean() * 100).round(1).to_dict(),
      "| average", round(main[main.name == "full"].score.mean() * 100, 1))
t1b = table("fidelity")
print("\nTABLE 1b. Agreement with the uncompressed model's answer x100 (token F1).")
print(t1b.to_string())
t1.to_csv(f"{OUT_DIR}/table1_score.csv"); t1b.to_csv(f"{OUT_DIR}/table1b_fidelity.csv")
""")

code(r"""
# TABLE 2. What recovery costs and what it does, pooled over datasets.
rec = df[df.trigger != "never"].copy()
rec["Variant"] = rec.name.map(lambda n: LABEL.get(n, n))
conf_gain = rec.apply(lambda r: np.mean(r.conf_after) - np.mean([r.conf[s] for s in r.trig_steps]) if r.n_trig else np.nan, axis=1)
rec["conf_gain"] = conf_gain
t2 = rec.groupby(["ratio", "name"]).agg(
    score=("score", lambda s: 100 * s.mean()), fidelity=("fidelity", lambda s: 100 * s.mean()),
    trigger_rate_pct=("rate", lambda s: 100 * s.mean()),
    examples_triggered_pct=("n_trig", lambda s: 100 * (s > 0).mean()),
    token_changed_pct=("flipped", "sum"), triggers=("n_trig", "sum"),
    slots_swapped_pct=("swapped", lambda s: 100 * s[s > 0].mean()),
    conf_gain=("conf_gain", "mean"), recover_ms=("rec_seconds", "sum")).reset_index()
t2["token_changed_pct"] = 100 * t2.token_changed_pct / t2.triggers.clip(lower=1)
t2["recover_ms"] = 1000 * t2.recover_ms / t2.triggers.clip(lower=1)
snap = df[df.name == "snapkv"].groupby("ratio").agg(score=("score", lambda s: 100 * s.mean()),
                                                     fidelity=("fidelity", lambda s: 100 * s.mean())).reset_index()
snap["name"] = "snapkv"
t2 = pd.concat([snap, t2]).sort_values(["ratio", "trigger_rate_pct"], na_position="first").round(2)
print("TABLE 2. Recovery behaviour (all datasets pooled). recover_ms = wall time of one recovery incl. the re-decoded step.")
print(t2.to_string(index=False))
t2.to_csv(f"{OUT_DIR}/table2_recovery.csv", index=False)

# Paired bootstrap: is the gain over SnapKV, and over random triggering, more than noise?
def paired_ci(a, b, n_boot=10000, seed=0):
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), len(d)

wide = df.pivot_table(index=["dataset", "idx", "ratio"], columns="name", values="score")
rows = []
for ratio in RATIOS:
    w = wide.xs(ratio, level="ratio")
    for other in ("snapkv", "snapkv+randtrig", "snapkv+always"):
        if other in w and HEAD in w:
            m, lo, hi, n = paired_ci(w[HEAD], w[other])
            rows.append({"ratio": ratio, "comparison": f"ours - {LABEL[other]}", "mean_diff": m, "ci_low": lo, "ci_high": hi, "n": n})
t3 = pd.DataFrame(rows).round(2)
print("\nTABLE 3. Paired differences in score x100 with 95% bootstrap intervals (an interval that excludes 0 is significant).")
print(t3.to_string(index=False))
t3.to_csv(f"{OUT_DIR}/table3_significance.csv", index=False)

# TABLE 4. Memory and speed. KV sizes are counted from the tensors, not from torch.cuda.max_memory_allocated.
mem = df[df.name.isin(["full", "snapkv", HEAD])].groupby(["name", "ratio"]).agg(
    gpu_kv_mb=("gpu_kv_bytes", lambda s: s.mean() / 2**20), backup_cpu_mb=("backup_bytes", lambda s: s.mean() / 2**20),
    kept_tokens=("kept_tokens", "mean"), prompt_tokens=("n_prompt", "mean"),
    ms_per_token=("ms_per_token", "mean"), prefill_s=("prefill_seconds", "mean")).reset_index()
mem["gpu_kv_saving_pct"] = 100 * (1 - mem.gpu_kv_mb / mem[mem.name == "full"].gpu_kv_mb.iloc[0])
print("\nTABLE 4. Memory and speed (means over all examples). backup_cpu_mb is the CPU RAM our method adds.")
print(mem.round(2).to_string(index=False))
mem.round(3).to_csv(f"{OUT_DIR}/table4_memory_speed.csv", index=False)
""")

code(r"""
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42})
# One fixed colour, marker and line style per method, so the figures also read in greyscale print.
STYLE = {"StreamingLLM": ("#eda100", "v", ":"), "H2O": ("#1baf7a", "s", "-."), "SnapKV": ("#2a78d6", "o", "--"),
         "SnapKV + CGR (ours)": ("#eb6834", "D", "-"), "SnapKV + random trigger": ("#6250d6", "^", ":"),
         "SnapKV + always recover": ("#52514e", "P", "-.")}
pct = [100 * r for r in RATIOS]

def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.pdf"); fig.savefig(f"{OUT_DIR}/{name}.png"); plt.show()

# FIGURE 1. Score against budget, one panel per dataset.
panels = DATASETS + ["average"]
fig, axes = plt.subplots(1, len(panels), figsize=(2.05 * len(panels), 2.3), sharex=True)
for ax, name in zip(axes, panels):
    sub = main if name == "average" else main[main.dataset == name]
    ax.axhline(100 * sub[sub.name == "full"].score.mean(), color="#0b0b0b", lw=1, ls=(0, (1, 1)), label="Full cache")
    for m in ("StreamingLLM", "H2O", "SnapKV", "SnapKV + CGR (ours)"):
        y = 100 * sub[sub.Method == m].groupby("ratio").score.mean().reindex(RATIOS)
        c, mk, ls = STYLE[m]
        ax.plot(pct, y, color=c, marker=mk, ls=ls, lw=1.6, ms=4.5, label=m)
    ax.set_xscale("log"); ax.set_xticks(pct); ax.set_xticklabels([f"{p:g}" for p in pct]); ax.minorticks_off()
    ax.set_title(name, fontsize=8.5); ax.set_xlabel("context kept (%)")
axes[0].set_ylabel("score (x100)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
fig.tight_layout(); save(fig, "fig1_score_vs_budget")

# FIGURE 2. Score against how often recovery fires, one panel per budget, all datasets pooled.
fig, axes = plt.subplots(1, len(RATIOS), figsize=(2.05 * len(RATIOS), 2.3), sharey=True)
for ax, ratio in zip(np.atleast_1d(axes), RATIOS):
    sub = df[df.ratio == ratio]
    ax.axhline(100 * df[df.name == "full"].score.mean(), color="#0b0b0b", lw=1, ls=(0, (1, 1)), label="Full cache")
    pts = sub[sub.trigger == "conf"].groupby("tau").agg(x=("rate", "mean"), y=("score", "mean")).sort_index()
    s0 = sub[sub.name == "snapkv"].score.mean()
    c, mk, ls = STYLE["SnapKV + CGR (ours)"]
    ax.plot([0] + list(100 * pts.x), [100 * s0] + list(100 * pts.y), color=c, ls=ls, lw=1.6, marker=mk, ms=4.5, label="SnapKV + CGR (ours)")
    for k, (tau, r) in enumerate(pts.iterrows()):   # labels staggered downwards so close points stay readable
        ax.annotate(f"τ={tau:g}", (100 * r.x, 100 * r.y), textcoords="offset points", xytext=(5, -10 - 8 * k), fontsize=6.5,
                    color="#52514e", arrowprops={"arrowstyle": "-", "color": "#b5b4ae", "lw": 0.5})
    for nm in ("snapkv", "snapkv+randtrig", "snapkv+always"):
        g = sub[sub.name == nm]
        if len(g):
            c, mk, _ = STYLE[LABEL[nm]]
            ax.plot([100 * g.rate.mean()], [100 * g.score.mean()], color=c, marker=mk, ms=6, ls="none",
                    markeredgecolor="white", markeredgewidth=0.8, label=LABEL[nm], zorder=3)
    ax.set_xscale("symlog", linthresh=5); ax.set_xlim(-0.5, 130); ax.set_xticks([0, 5, 20, 100]); ax.set_xticklabels(["0", "5", "20", "100"])
    ax.set_title(f"{100 * ratio:g}% of context kept", fontsize=8.5); ax.set_xlabel("steps with recovery (%)")
np.atleast_1d(axes)[0].set_ylabel("score (x100)")
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16), fontsize=7.5)
fig.tight_layout(); save(fig, "fig2_score_vs_trigger_rate")

# FIGURE 3. Confidence at the steps where recovery fired, before and after (ours, all budgets).
ours = df[(df.name == HEAD) & (df.n_trig > 0)]
before = np.concatenate([[r.conf[s] for s in r.trig_steps] for r in ours.itertuples()]) if len(ours) else np.array([])
after = np.concatenate([r.conf_after for r in ours.itertuples()]) if len(ours) else np.array([])
fig, ax = plt.subplots(figsize=(3.4, 2.2))
bins = np.linspace(0, 1, 21)
ax.hist(before, bins, color="#2a78d6", alpha=0.85, label=f"before recovery (mean {before.mean():.2f})" if len(before) else "before")
ax.hist(after, bins, histtype="step", color="#eb6834", lw=1.8, label=f"after recovery (mean {after.mean():.2f})" if len(after) else "after")
ax.axvline(HEAD_TAU, color="#52514e", lw=0.8, ls="--"); ax.text(HEAD_TAU + 0.01, ax.get_ylim()[1] * 0.92, "τ", fontsize=8, color="#52514e")
ax.set_xlabel("next-token confidence at trigger steps"); ax.set_ylabel("number of steps"); ax.legend(fontsize=7)
fig.tight_layout(); save(fig, "fig3_confidence_before_after")
print(f"{len(before)} recoveries in total")
""")

md("## 9. Examples and download")
code(r"""
# A few examples where recovery changed the answer, at the smallest budget. Useful for the paper's qualitative section.
r0 = min(RATIOS)
a = df[(df.name == "snapkv") & (df.ratio == r0)].set_index(["dataset", "idx"])
b = df[(df.name == HEAD) & (df.ratio == r0)].set_index(["dataset", "idx"])
changed = [k for k in a.index if a.pred[k] != b.pred[k]]
print(f"budget {100 * r0:g}%: recovery changed the answer on {len(changed)} of {len(a)} examples; "
      f"better on {sum(b.score[k] > a.score[k] for k in changed)}, worse on {sum(b.score[k] < a.score[k] for k in changed)}")
for k in sorted(changed, key=lambda k: a.score[k] - b.score[k])[:6]:
    print(f"\n[{k[0]} #{k[1]}]\n  full   : {full_pred[k][:90]!r}\n  snapkv : {a.pred[k][:90]!r}  (score {a.score[k]:.2f})"
          f"\n  ours   : {b.pred[k][:90]!r}  (score {b.score[k]:.2f}, {b.n_trig[k]} recoveries)")

import shutil
shutil.make_archive(f"{BASE}/cgr_outputs", "zip", OUT_DIR)
try:
    from google.colab import files
    files.download(f"{BASE}/cgr_outputs.zip")
except Exception as e:
    print(f"zip is at {BASE}/cgr_outputs.zip", e)
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
out = os.path.join(ROOT, "notebooks", "CGR_KV_Cache_Recovery.ipynb")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, f"({len(cells)} cells)")
