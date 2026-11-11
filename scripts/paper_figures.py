"""Figures for the paper, from results/final/results.jsonl -> paper/figures/*.pdf (+ .png).

    python scripts/paper_figures.py

Same data and encodings as the notebook figures; sizes are set for IEEE two-column pages
and Fig. 2 drops the per-point tau labels (they collided), naming the order in the legend instead.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
HEAD_TAU = 0.7
HEAD = f"snapkv+cgr(tau={HEAD_TAU})"

df = pd.read_json(open(os.path.join(ROOT, "results", "final", "results.jsonl"), encoding="utf-8"), lines=True)
df["rate"] = df.n_trig / df.steps
RATIOS = sorted(df[df.method != "full"].ratio.unique())
DATASETS = ["needle", "hotpotqa", "2wikimqa", "qasper"]
TITLE = {"needle": "Needle", "hotpotqa": "HotpotQA", "2wikimqa": "2WikiMQA", "qasper": "Qasper", "average": "Average"}
LABEL = {"streaming": "StreamingLLM", "h2o": "H2O", "snapkv": "SnapKV", HEAD: "SnapKV + CGR (ours)",
         "snapkv+randtrig": "SnapKV + random trigger", "snapkv+always": "SnapKV + always recover"}
STYLE = {"StreamingLLM": ("#eda100", "v", ":"), "H2O": ("#1baf7a", "s", "-."), "SnapKV": ("#2a78d6", "o", "--"),
         "SnapKV + CGR (ours)": ("#eb6834", "D", "-"), "SnapKV + random trigger": ("#6250d6", "^", ":"),
         "SnapKV + always recover": ("#52514e", "P", "-.")}
mpl.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42})
pct = [100 * r for r in RATIOS]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


# Fig. 1: score against budget, one panel per dataset (full page width).
panels = DATASETS + ["average"]
fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 1.9))
for ax, name in zip(axes, panels):
    sub = df if name == "average" else df[df.dataset == name]
    ax.axhline(100 * sub[sub.name == "full"].score.mean(), color="#0b0b0b", lw=1, ls=(0, (1, 1)), label="Full cache")
    for key in ("streaming", "h2o", "snapkv", HEAD):
        c, mk, ls = STYLE[LABEL[key]]
        y = 100 * sub[sub.name == key].groupby("ratio").score.mean().reindex(RATIOS)
        ax.plot(pct, y, color=c, marker=mk, ls=ls, lw=1.5, ms=4, label=LABEL[key])
    ax.set_xscale("log"); ax.set_xticks(pct); ax.set_xticklabels([f"{p:g}" for p in pct]); ax.minorticks_off()
    ax.set_title(TITLE[name], fontsize=8.5); ax.set_xlabel("context kept (%)")
axes[0].set_ylabel("score (×100)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=7.5)
fig.tight_layout(w_pad=0.6); save(fig, "fig1_score_vs_budget")

# Fig. 2: score against how often recovery fires, one panel per budget (full page width).
fig, axes = plt.subplots(1, len(RATIOS), figsize=(7.16, 2.0), sharey=True)
full = 100 * df[df.name == "full"].score.mean()
for ax, ratio in zip(axes, RATIOS):
    sub = df[df.ratio == ratio]
    ax.axhline(full, color="#0b0b0b", lw=1, ls=(0, (1, 1)), label="Full cache")
    pts = sub[sub.trigger == "conf"].groupby("tau").agg(x=("rate", "mean"), y=("score", "mean")).sort_index()
    s0 = sub[sub.name == "snapkv"].score.mean()
    c, mk, ls = STYLE["SnapKV + CGR (ours)"]
    ax.plot([0] + list(100 * pts.x), [100 * s0] + list(100 * pts.y), color=c, ls=ls, lw=1.5, marker=mk, ms=4,
            label="SnapKV + CGR (ours), τ = 0.5, 0.7, 0.9 from left")
    for key in ("snapkv", "snapkv+randtrig", "snapkv+always"):
        g = sub[sub.name == key]
        c, mk, _ = STYLE[LABEL[key]]
        ax.plot([100 * g.rate.mean()], [100 * g.score.mean()], color=c, marker=mk, ms=6.5, ls="none",
                markeredgecolor="white", markeredgewidth=0.8, label=LABEL[key], zorder=3)
    ax.set_xscale("symlog", linthresh=5); ax.set_xlim(-0.5, 130)
    ax.set_xticks([0, 5, 10, 20, 50, 100]); ax.set_xticklabels(["0", "5", "10", "20", "50", "100"]); ax.minorticks_off()
    ax.set_title(f"{100 * ratio:g}% of context kept", fontsize=8.5); ax.set_xlabel("decoding steps with a recovery (%)")
axes[0].set_ylabel("score (×100)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17), fontsize=7.5)
fig.tight_layout(w_pad=0.8); save(fig, "fig2_score_vs_trigger_rate")

# Fig. 3: confidence at the trigger steps, before and after recovery (one column).
ours = df[(df.name == HEAD) & (df.n_trig > 0)]
before = np.concatenate([[r.conf[s] for s in r.trig_steps] for r in ours.itertuples()])
after = np.concatenate([r.conf_after for r in ours.itertuples()])
fig, ax = plt.subplots(figsize=(3.4, 2.1))
bins = np.linspace(0, 1, 21)
ax.hist(before, bins, color="#2a78d6", alpha=0.85, label=f"before recovery (mean {before.mean():.2f})")
ax.hist(after, bins, histtype="step", color="#eb6834", lw=1.8, label=f"after recovery (mean {after.mean():.2f})")
ax.axvline(HEAD_TAU, color="#52514e", lw=0.8, ls="--")
ax.text(HEAD_TAU + 0.012, ax.get_ylim()[1] * 0.93, "τ", fontsize=8, color="#52514e")
ax.set_xlabel("next-token confidence at trigger steps"); ax.set_ylabel("number of steps"); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); save(fig, "fig3_confidence_before_after")
print(len(before), "recoveries;", "after >= tau:", round(100 * (after >= HEAD_TAU).mean(), 1), "%;",
      "after >= 0.95:", round(100 * (after >= 0.95).mean(), 1), "%")
# Fig. S: simplified score-vs-budget for the guide-format paper: QA average, needle, overall.
df["group"] = df.dataset.map(lambda d: "needle" if d == "needle" else "qa")
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
for ax, (title, sub) in zip(axes, (("Question answering (3 sets)", df[df.group == "qa"]),
                                   ("Needle retrieval", df[df.group == "needle"]), ("Average of all four tasks", df))):
    ax.axhline(100 * sub[sub.name == "full"].score.mean(), color="#0b0b0b", lw=1, ls=(0, (1, 1)), label="Full cache")
    for key in ("streaming", "h2o", "snapkv", HEAD):
        c, mk, ls = STYLE[LABEL[key]]
        y = 100 * sub[sub.name == key].groupby("ratio").score.mean().reindex(RATIOS)
        ax.plot(pct, y, color=c, marker=mk, ls=ls, lw=1.5, ms=4.5, label=LABEL[key].replace("SnapKV + CGR (ours)", "Proposed (SnapKV + CGR)"))
    ax.set_xscale("log"); ax.set_xticks(pct); ax.set_xticklabels([f"{p:g}" for p in pct]); ax.minorticks_off()
    ax.set_title(title, fontsize=8.5); ax.set_xlabel("context kept on the GPU (%)")
axes[0].set_ylabel("score (×100)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=7.5)
fig.tight_layout(w_pad=1.0); save(fig, "fig_score_simple")

# Fig. W: workflow of the proposed model (boxes and arrows, no data).
from matplotlib.patches import FancyBboxPatch, Polygon

fig, ax = plt.subplots(figsize=(7.16, 2.75))
ax.set_xlim(0, 100); ax.set_ylim(-2.5, 40); ax.axis("off"); ax.grid(False)
INK, BLUE, ORANGE, GREEN, GREY = "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def box(x, y, w, h, text, edge, face, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", lw=1.1, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3, color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1, color=GREY, text=None, dx=0, dy=0, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1, "ls": ls, "shrinkA": 0, "shrinkB": 0})
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va="center", fontsize=6.5, color=color)


# top row: prompt stage
box(1, 28, 14, 9, "Long prompt\n(~5,500 tokens)", GREY, "#f4f4f2")
box(21, 28, 16, 9, "Read prompt once\n(full KV cache)", GREY, "#f4f4f2")
box(43, 28, 18, 9, "SnapKV selection:\nkeep top 2–10%", BLUE, "#e6f0fb")
box(68, 30.5, 30, 6.5, "GPU cache:\nkept tokens only (small)", BLUE, "#e6f0fb", bold=True)
box(68, 21.5, 30, 6.5, "CPU backup:\nall evicted tokens", ORANGE, "#fdeee6", bold=True)
arrow(15.6, 32.5, 20.4, 32.5); arrow(37.6, 32.5, 42.4, 32.5)
arrow(61.6, 33.5, 67.4, 33.8, BLUE, "kept", dy=1.5); arrow(61.6, 30.5, 67.4, 25.5, ORANGE, "evicted", dx=-3.2, dy=-1.8)
# bottom row: decoding loop
box(1, 4, 15, 9, "Decode next\ntoken", GREY, "#f4f4f2")
ax.add_patch(Polygon([[22, 8.5], [31, 14.5], [40, 8.5], [31, 2.5]], closed=True, lw=1.1, edgecolor=INK, facecolor="#ffffff"))
ax.text(31, 8.5, "confidence\n< τ ?", ha="center", va="center", fontsize=6.3, color=INK)
box(46, 4, 17, 9, "Re-score backup\nwith current query", ORANGE, "#fdeee6")
box(69, 4, 14, 9, "Swap best\ntokens in", ORANGE, "#fdeee6")
box(88, 4, 11, 9, "Decode same\nstep again", ORANGE, "#fdeee6")
arrow(16.6, 8.5, 21.6, 8.5); arrow(40.4, 8.5, 45.4, 8.5, ORANGE, "yes", dy=1.6)
arrow(63.6, 8.5, 68.4, 8.5, ORANGE); arrow(83.6, 8.5, 87.4, 8.5, ORANGE)
arrow(76, 21, 76, 13.8, ORANGE, "recalled tokens", dx=9.5)                     # backup -> swap
ax.plot([31, 31], [14.9, 18], color=GREEN, lw=1.1); ax.plot([31, 8.5], [18, 18], color=GREEN, lw=1.1)
arrow(8.5, 18, 8.5, 13.8, GREEN); ax.text(20, 19.4, "no: emit token, continue", ha="center", fontsize=6.5, color=GREEN)
ax.plot([93.5, 93.5], [3.4, 0.8], color=GREY, lw=1.1); ax.plot([93.5, 8.5], [0.8, 0.8], color=GREY, lw=1.1)
arrow(8.5, 0.8, 8.5, 3.4, GREY); ax.text(51, -1.6, "emit token, continue (GPU cache size unchanged)", ha="center", fontsize=6.5, color=GREY)
ax.text(1, 23.5, "Step 1: prompt stage (once)", fontsize=7, color=GREY, style="italic")
ax.text(41, 16.2, "Step 2: generation loop", fontsize=7, color=GREY, style="italic")
save(fig, "fig_workflow")
print("wrote", OUT)
