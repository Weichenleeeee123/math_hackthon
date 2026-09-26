"""图像先验权重 λ 的敏感性：det+edge 在不同 λ 下的 AP / 切片比例（两个数据集，离线精确复现）。

  python scripts/lambda_check.py visdrone
  python scripts/lambda_check.py dota
输出 results/<ds>/lambda.csv
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import select_slices  # noqa: E402

ds = sys.argv[1] if len(sys.argv) > 1 else "dota"
R.set_dataset(ds)
cache = pickle.loads(R.CACHE.read_bytes())
centers = R.gt_centers(json.loads(R.GT.read_text()))
rows = []
for lam in [0.3, 0.6, 1.0]:
    cfg = GlanceConfig(img_weight=lam)
    for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        m, _ = R.run_method(cache, R.GT, centers, f"det+edge λ={lam}@{th}",
                            lambda r, c=cfg, t=th: select_slices(
                                R.slice_scores(r, c, *R.prior_kinds("det+edge"))[0], "threshold", t, 0),
                            cfg, lambda r: r["t_prior_edge"])
        m.update(lam=lam, threshold=th)
        rows.append(m)
        print(f"{m['method']:24s} AP={m['AP']:.4f} AP50={m['AP50']:.4f} APs={m['APs']:.4f} "
              f"slices={m['slice_frac']:.1%} {m['ms_per_img']:.0f}ms small_cov={m['small_cov']:.3f}")
pd.DataFrame(rows).to_csv(R.RES / "lambda.csv", index=False)
