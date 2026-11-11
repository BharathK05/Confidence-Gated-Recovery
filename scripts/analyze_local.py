"""Run the notebook's analysis (section 8 and 9) on a results.jsonl, without a GPU.

    python scripts/analyze_local.py results/results.jsonl

Tables (CSV) and figures (PDF + PNG) are written next to the results file.
The analysis code is taken from the notebook itself, so the numbers are the
ones the notebook would print.
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from tamc import cgr  # noqa: E402,F401

RESULTS = os.path.abspath(sys.argv[1])
OUT_DIR = os.path.dirname(RESULTS)
nb = json.load(open(os.path.join(ROOT, "notebooks", "CGR_KV_Cache_Recovery.ipynb"), encoding="utf-8"))
cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
start = next(i for i, (s, t) in enumerate(cells) if t == "markdown" and s.lstrip().startswith("## 8."))
settings = next(s for s, t in cells if t == "code" and "N_EXAMPLES" in s and "MODEL_ID" in s)

g = {"__name__": "__main__", "cgr": cgr}
exec(settings, g)
g.update(RESULTS=RESULTS, OUT_DIR=OUT_DIR, BASE=OUT_DIR)
# budgets and datasets come from the file, in case the run used fewer than the notebook's defaults
rows = [json.loads(line) for line in open(RESULTS, encoding="utf-8") if line.strip()]
g["RATIOS"] = sorted({r["ratio"] for r in rows if r["method"] != "full"})
g["DATASETS"] = [d for d in g["DATASETS"] if any(r["dataset"] == d for r in rows)]
zip_base = os.path.join(OUT_DIR, "cgr_outputs").replace("\\", "/")
for src, kind in cells[start:]:
    if kind != "code":
        continue
    src = src.replace('shutil.make_archive("/content/cgr_outputs", "zip", OUT_DIR)', "pass")
    src = src.replace('files.download("/content/cgr_outputs.zip")', "pass").replace("from google.colab import files", "pass")
    exec(compile(src, "analysis", "exec"), g)
print("\nwrote tables and figures to", OUT_DIR)
