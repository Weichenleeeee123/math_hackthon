"""覆盖感知去冗余（REPORT 3.19）：在 SAHI 全切 / Glance 选片之上再删"独占面积 < min_new"的低分片。

离线、无需 GPU（读 run_eval.py cache 的缓存）。对照组：从同一选中集里随机删同样多片。
用法：python scripts/prune_eval.py [--datasets visdrone visdrone_ft dota] → results[/<ds>]/prune.csv
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import prune_redundant, select_slices  # noqa: E402

# 每个数据集的主工作点：DOTA 上 COCO 检测器不认识俯视目标，用边缘先验 λ=1.0、θ=0.5（REPORT 3.7）
SETTINGS = {"visdrone": (0.3, "noisyor", (0.9, 0.99)),
            "visdrone_ft": (0.3, "noisyor", (0.9, 0.99)),
            "dota": (1.0, None, (0.5,))}


def evaluate(ds, min_news, seed=0):
    R.set_dataset(ds)
    cache = pickle.loads(R.CACHE.read_bytes())
    lam, det_kind, ths = SETTINGS[ds]
    cfg = GlanceConfig(img_weight=lam)
    centers = R.gt_centers(json.loads(R.GT.read_text()))
    scores = {r["id"]: R.slice_scores(r, cfg, det_kind, "edge")[0] for r in cache["images"]}
    t_prune = {}

    def pruned(base_fn, mn):
        def fn(r):
            sel = base_fn(r)
            t0 = time.perf_counter()
            out = prune_redundant(r["slices"], sel, scores[r["id"]], mn)
            t_prune[r["id"]] = time.perf_counter() - t0
            return out
        return fn

    rows = []

    def run(name, fn, base, extra=lambda r: 0.0):
        m, _ = R.run_method(cache, R.GT, centers, name, fn, cfg, extra)
        m.update(dataset=ds, base=base)
        rows.append(m)
        print(f"{ds:12s} {name:34s} AP={m['AP']*100:.2f} APs={m['APs']*100:.2f} "
              f"slices={m['slices_per_img']:.2f} ({m['slice_frac']:.1%}) ms={m['ms_per_img']:.1f} "
              f"cov_s={m['small_cov']:.3f}", flush=True)

    bases = [("sahi", lambda r: np.arange(len(r["slices"])))]
    bases += [(f"glance@{th}", lambda r, th=th: select_slices(scores[r["id"]], "threshold", th, 0)) for th in ths]
    for bname, bfn in bases:
        run(bname, bfn, bname)
        for mn in min_news:
            # 去冗余本身的耗时计入（t_extra），run_method 先调 select_fn 再调 t_extra_fn
            run(f"{bname}+prune{mn}", pruned(bfn, mn), bname, lambda r: t_prune[r["id"]])
        rng = np.random.default_rng(seed)

        def rnd(r, bfn=bfn, mn=min_news[0], rng=rng):
            sel = bfn(r)
            k = len(prune_redundant(r["slices"], sel, scores[r["id"]], mn))
            return np.sort(rng.choice(sel, size=k, replace=False)) if k else np.zeros(0, int)
        run(f"{bname}+random_drop(=prune{min_news[0]})", rnd, bname)
    out = R.RES / "prune.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("→", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SETTINGS))
    ap.add_argument("--min-new", nargs="+", type=float, default=[0.1, 0.2])
    a = ap.parse_args()
    for ds in a.datasets:
        evaluate(ds, a.min_new)


if __name__ == "__main__":
    main()
