"""DOTA 上“仅边缘显著性”选片 vs 同数量随机选片（AP 层面的对照），输出 results/dota/edge_vs_random.csv。"""

import json
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
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

ds = sys.argv[1] if len(sys.argv) > 1 else "dota"
R.set_dataset(ds)
cache = pickle.loads(R.CACHE.read_bytes())
centers = R.gt_centers(json.loads(R.GT.read_text()))
cfg = GlanceConfig()
rows = []
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    def edge(r, t=th):
        return select_slices(R.slice_scores(r, cfg, *R.prior_kinds("edge"))[0], "threshold", t, 0)
    m, _ = R.run_method(cache, R.GT, centers, f"edge@{th}", edge, cfg, lambda r: r["t_prior_edge"])
    m.update(family="edge", threshold=th)
    rows.append(m)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        mr, _ = R.run_method(cache, R.GT, centers, f"random@{th}#s{seed}",
                             lambda r, g=rng: random_slices(len(r["slices"]), len(edge(r)), g), cfg)
        mr.update(family="random", threshold=th, seed=seed)
        rows.append(mr)
    rr = pd.DataFrame(rows[-3:])
    print(f"θ={th}: slices={m['slice_frac']:.1%} {m['ms_per_img']:.0f}ms | edge AP={m['AP']:.4f} AP50={m['AP50']:.4f} "
          f"cov={m['small_cov']:.3f} | random AP={rr.AP.mean():.4f}±{rr.AP.std():.4f} AP50={rr.AP50.mean():.4f} "
          f"cov={rr.small_cov.mean():.3f}")
pd.DataFrame(rows).to_csv(R.RES / "edge_vs_random.csv", index=False)
