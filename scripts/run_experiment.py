"""Run the CGR experiment outside Colab (Apple Silicon, a local CUDA GPU, or CPU).

    python scripts/run_experiment.py                 # full run, resumable
    python scripts/run_experiment.py --pilot         # two examples, printed, then stop (about 2 minutes)
    python scripts/run_experiment.py --n 20          # fewer examples per dataset

Same code and settings as notebooks/CGR_KV_Cache_Recovery.ipynb. Results go to
results/<run>/results.jsonl, one line per finished configuration, so the run can
be stopped (Ctrl+C) and started again at any time. When it finishes, the tables
and figures are built in the same folder.
"""
import argparse
import gc
import os
import random
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tamc import cgr  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--datasets", nargs="+", default=["needle", "hotpotqa", "2wikimqa", "qasper"])
p.add_argument("--n", type=int, default=30, help="examples per dataset")
p.add_argument("--max-len", type=int, default=6000)
p.add_argument("--ratios", nargs="+", type=float, default=[0.02, 0.05, 0.10])
p.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.7, 0.9])
p.add_argument("--head-tau", type=float, default=0.7)
p.add_argument("--window", type=int, default=32)
p.add_argument("--run", default="cgr_qwen1.5b_fp32")
p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
p.add_argument("--pilot", action="store_true")
a = p.parse_args()

dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
out_dir = os.path.join(ROOT, "results", a.run)
os.makedirs(out_dir, exist_ok=True)
results = os.path.join(out_dir, "results.jsonl")
print(f"device {dev} | torch {torch.__version__} | results -> {results}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
# float32 everywhere: fp16 gave NaN logits on long prompts with this model.
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
tap = cgr.QueryTap(model)
eos_id = model.generation_config.eos_token_id
EOS = set(eos_id if isinstance(eos_id, list) else [eos_id]) | {tok.eos_token_id}

data = {}
hp = cgr.load_longbench("hotpotqa")
random.Random(0).shuffle(hp)
for name in a.datasets:
    if name == "needle":
        data[name] = cgr.make_needles(a.n, [e["context"] for e in hp[a.n:2 * a.n]], tok, a.max_len - 300)
    elif name == "hotpotqa":
        data[name] = hp[:a.n]
    else:
        rows = cgr.load_longbench(name)
        random.Random(0).shuffle(rows)
        data[name] = rows[:a.n]

# Pilot: real answers, printed. The full-cache answer must be sensible text.
for name in a.datasets[:2]:
    ex = data[name][0]
    t0 = time.time()
    pf = cgr.prefill(model, tap, cgr.build_prompt(tok, name, ex, a.max_len).to(dev), window=a.window)
    backup = cgr.make_backup(pf)
    print(f"[{name}] {pf.n} prompt tokens, prefill {pf.seconds:.1f}s | gold: {ex['answers'][:2]}", flush=True)
    for cfg in (cgr.GenConfig(method="full"), cgr.GenConfig(method="snapkv", ratio=0.05),
                cgr.GenConfig(method="snapkv", ratio=0.05, trigger="conf", tau=a.head_tau)):
        cfg.max_new_tokens = cgr.MAX_NEW[name]
        r = cgr.generate(model, tap, pf, cfg, EOS, backup)
        text = tok.decode(r.tokens, skip_special_tokens=True).strip()
        print(f"   {cfg.name():22s} score {cgr.score_output(name, text, ex['answers']):.2f} | "
              f"{len(r.trig_steps)} recoveries | {r.gen_seconds:.1f}s | {text[:80]!r}", flush=True)
        if cfg.method == "full":
            assert any(ch.isalnum() for ch in text), "the full-cache answer is not text; stopping"
    per_cfg = (time.time() - t0) / 3
    del pf, backup
    gc.collect()
n_cfg = len(cgr.config_grid(a.ratios, a.taus))
n_ex = sum(len(v) for v in data.values())
print(f"pilot OK. {n_cfg} configurations x {n_ex} examples; rough estimate {per_cfg * n_cfg * n_ex / 3600:.1f} hours "
      f"(qasper is about 4x slower per example than the others)", flush=True)
if a.pilot:
    sys.exit(0)

t0 = time.time()
for name in a.datasets:
    cgr.run_dataset(model, tok, tap, name, data[name], results, a.ratios, a.taus,
                    max_len=a.max_len, window=a.window, ref_tau=a.head_tau, log_every=1)
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_local.py"), results], check=False)
