"""挑 DOTA 上适合展示的图：大图、选片比例低、置信检测几乎不丢。"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import select_slices  # noqa: E402

ds, prior, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
R.set_dataset(ds)
cache = pickle.loads(R.CACHE.read_bytes())["images"]
cfg = GlanceConfig()
rows = []
for r in cache:
    n = len(r["slices"])
    sel = select_slices(R.slice_scores(r, cfg, *R.prior_kinds(prior))[0], "threshold", th, 0)
    a, b = R.merge(r, np.arange(n), cfg)[0], R.merge(r, sel, cfg)[0]
    rows.append((r["id"], r["file_name"], n, len(sel), int((a[:, 4] >= 0.3).sum()), int((b[:, 4] >= 0.3).sum())))
g = pd.DataFrame(rows, columns=["id", "file", "n", "k", "na", "nb"])
g["frac"] = g.k / g.n
print(g[(g.n >= 30) & (g.na >= 8)].assign(keep=lambda d: d.nb / d.na).sort_values(["frac"]).head(15).to_string())
