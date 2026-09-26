"""与检测器无关的选片质量评估：被选切片覆盖了多少真值目标（中心落在切片内）。

AP 受检测器本身好坏影响（COCO 预训练模型在 DOTA 俯视图上几乎认不出车船），
而“目标覆盖率 vs 切片比例”只衡量选片本身：覆盖不到的目标，后面检测器再强也找不回来。

  python scripts/coverage.py dota
输出 results/<ds>/coverage.csv 与 figures/fig5_coverage.png
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures as MF  # noqa: E402
import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

plt = MF.plt


def main(ds):
    R.set_dataset(ds)
    cache = pickle.loads(R.CACHE.read_bytes())["images"]
    centers = R.gt_centers(json.loads(R.GT.read_text()))
    cfg = GlanceConfig()
    ths = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    def evaluate(name, fn, **tags):
        n_run = n_tot = hit_all = n_all = hit_s = n_s = 0
        for r in cache:
            sel = fn(r)
            c = centers.get(r["id"], np.zeros((0, 3), np.float32))
            h = R.covered(c, r["slices"], sel)
            n_run += len(sel)
            n_tot += len(r["slices"])
            hit_all += int(h.sum())
            n_all += len(c)
            if len(c):
                hit_s += int((h & (c[:, 2] > 0)).sum())
                n_s += int(c[:, 2].sum())
        return dict(method=name, slice_frac=n_run / n_tot, cov_all=hit_all / max(n_all, 1),
                    cov_small=hit_s / max(n_s, 1), **tags)

    rows = []
    for prior in ["det", "edge", "spectral", "det+edge"]:
        for th in ths:
            rows.append(evaluate(f"glance[{prior}]@{th}",
                                 lambda r, p=prior, t=th: select_slices(
                                     R.slice_scores(r, cfg, *R.prior_kinds(p))[0], "threshold", t, 0),
                                 family=prior, threshold=th))
    # 随机对照：与“仅边缘显著性”逐图同数量
    for th in ths:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            rows.append(evaluate(f"random@{th}#s{seed}", lambda r, t=th, g=rng: random_slices(
                len(r["slices"]),
                len(select_slices(R.slice_scores(r, cfg, *R.prior_kinds("edge"))[0], "threshold", t, 0)), g),
                family="random", threshold=th, seed=seed))
    # Oracle：只跑含任意真值目标的切片
    rows.append(evaluate("oracle_gt_any", lambda r: np.array(
        [k for k in range(len(r["slices"]))
         if R.covered(centers.get(r["id"], np.zeros((0, 3), np.float32)), r["slices"], [k]).any()], int),
        family="oracle"))
    df = pd.DataFrame(rows)
    df.to_csv(R.RES / "coverage.csv", index=False)
    print(df[df.family != "random"].round(3).to_string())
    print(df[df.family == "random"].groupby("threshold")[["slice_frac", "cov_all", "cov_small"]].mean().round(3))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = {"det+edge": "检测先验 + 边缘（λ=0.3）", "det": "仅检测先验", "edge": "仅边缘显著性",
              "spectral": "仅谱残差显著性"}
    for (key, lab), color in zip(labels.items(), MF.SERIES):
        d = df[df.family == key].sort_values("slice_frac")
        ax.plot(d.slice_frac * 100, d.cov_all * 100, "-o", color=color, label=lab,
                markeredgecolor=MF.SURFACE, markeredgewidth=1.2)
    rr = df[df.family == "random"].groupby("threshold", as_index=False)[["slice_frac", "cov_all"]].mean()
    rr = rr.sort_values("slice_frac")
    ax.plot(rr.slice_frac * 100, rr.cov_all * 100, "--", color=MF.MUTED, label="随机选片（对照）")
    o = df[df.family == "oracle"].iloc[0]
    ax.plot(o.slice_frac * 100, o.cov_all * 100, "*", color=MF.SERIES[6], markersize=14,
            markeredgecolor=MF.SURFACE, label="Oracle（只跑有目标的切片）")
    ax.set_xlabel("实际推理的切片占 SAHI 全部切片的比例（%）")
    ax.set_ylabel("真值目标覆盖率（%）")
    name = {"dota": "DOTA-v1.0 val", "visdrone": "VisDrone2019-DET-val"}.get(ds, ds)
    ax.set_title(f"选片质量（与检测器无关）：切得越少，还能覆盖多少目标（{name}）", loc="left",
                 color=MF.INK, fontsize=10)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    (R.RES / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(R.RES / "figures" / "fig5_coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dota")
