"""报告用的汇总统计（读 sweep.csv / per_image.csv / cache.pkl）。

  python scripts/stats.py [op] [dataset]
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
op = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
ds = sys.argv[2] if len(sys.argv) > 2 else "visdrone"
RES = ROOT / "results" if ds == "visdrone" else ROOT / "results" / ds

df = pd.read_csv(RES / "sweep.csv")
cols = ["AP", "AP50", "APs", "slices_per_img", "slice_frac", "ms_per_img", "small_cov"]
print("== random (mean ± std over 3 seeds)")
r = df[df.family == "random"].groupby("threshold")[cols].agg(["mean", "std"])
print(r.round(4).to_string())

c = pickle.loads((RES / "cache.pkl").read_bytes())
ims = c["images"]
ms = lambda k: 1000 * np.mean([x[k] for x in ims])  # noqa: E731
print(f"== timing ms: glance={ms('t_glance'):.1f} edge_prior={ms('t_prior_edge'):.1f} "
      f"spectral_prior={ms('t_prior_spectral'):.1f} per_slice={1000 * np.mean(np.concatenate([x['t_slice'] for x in ims])):.1f} "
      f"slices/img={np.mean([len(x['slices']) for x in ims]):.2f}")

pi = pd.read_csv(RES / "per_image.csv")
g = pi[pi.method == f"glance[det+edge]@{op}"].copy()
g["f"] = g.n_run / g.n_slices
print("== per-image slice fraction quantiles (0,10,25,50,75,90,100%):",
      g.f.quantile([0, .1, .25, .5, .75, .9, 1]).round(2).tolist())
print(f"zero-slice images={int((g.n_run == 0).sum())}  all-slice images={int((g.f == 1).sum())}  total={len(g)}")
for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 10 ** 9)]:
    s = g[(g.n_gt >= lo) & (g.n_gt < hi)]
    print(f"  n_gt in [{lo},{hi}): {len(s)} imgs, mean slice frac={s.f.mean():.3f}")
