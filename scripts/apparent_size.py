"""扫视看得见吗：按"目标在扫视图里的表观边长"分箱，看选片覆盖率怎样随之变化。

SaccadeNet 的最优性命题有一个前提：目标在廉价的粗分辨率下看得见。这个前提被打破时，
"先扫一眼再放大"就会漏。本脚本把它变成可测的量（预注册 B3）：
  表观边长 = √(w·h) × 扫视输入尺寸 / 图像长边（像素，扫视图坐标）
  扫视命中 = 有低阈值粗检测（≥ glance_conf）的中心落在该目标框内
  选片覆盖 = 目标中心落在被选切片内
全程离线读 cache.pkl。输出 results/<ds>/apparent_size.csv 与 figures/fig_apparent_size.png。

  python scripts/apparent_size.py visdrone
  python scripts/apparent_size.py dota15 --theta 0.9
"""

import argparse
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
EDGES = [0, 4, 8, 16, 32, np.inf]


def gt_boxes(gt):
    by_img = {}
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            by_img.setdefault(a["image_id"], []).append(a["bbox"])
    return {k: np.array(v, dtype=np.float32) for k, v in by_img.items()}


def main(args):
    R.set_dataset(args.dataset, args.res_tag)
    cache = pickle.loads(R.CACHE.read_bytes())
    imgsz = cache.get("imgsz", 640)  # 旧缓存没有这个键，当时统一是 640
    boxes = gt_boxes(json.loads(R.GT.read_text()))
    cfg = GlanceConfig(img_weight=args.img_weight)
    rng = np.random.default_rng(0)

    rows = []
    for rec in cache["images"]:
        b = boxes.get(rec["id"])
        if b is None or not len(b):
            continue
        h, w = rec["hw"]
        side = np.sqrt(b[:, 2] * b[:, 3]) * imgsz / max(h, w)
        cx, cy = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
        centers = np.stack([cx, cy, np.zeros_like(cx)], 1)
        g = rec["glance"]
        gx, gy = (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2
        seen = ((gx[None] >= b[:, :1]) & (gx[None] <= b[:, :1] + b[:, 2:3])
                & (gy[None] >= b[:, 1:2]) & (gy[None] <= b[:, 1:2] + b[:, 3:4])).any(1)
        sel = {p: select_slices(R.slice_scores(rec, cfg, *R.prior_kinds(p))[0], "threshold", args.theta, 0)
               for p in ("det", "det+edge")}
        sel["random"] = random_slices(len(rec["slices"]), len(sel["det+edge"]), rng)
        cov = {p: R.covered(centers, rec["slices"], s) for p, s in sel.items()}
        for i in range(len(b)):
            rows.append({"image_id": rec["id"], "side": side[i], "seen": bool(seen[i]),
                         **{f"cov_{p}": bool(c[i]) for p, c in cov.items()}})
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.side, EDGES, right=False)
    agg = df.groupby("bin", observed=True).agg(n=("side", "size"), seen=("seen", "mean"), cov_det=("cov_det", "mean"),
                                                cov_det_edge=("cov_det+edge", "mean"), cov_random=("cov_random", "mean"))
    agg = agg.reset_index()
    agg["bin"] = agg["bin"].astype(str)
    out = R.RES / "apparent_size.csv"
    agg.to_csv(out, index=False)
    print(f"{args.dataset}：扫视输入 {imgsz}，θ={args.theta}，目标 {len(df)} 个")
    print(agg.round(3).to_string(index=False))

    small, large = df[df.side < 8], df[df.side >= 16]
    if len(small) and len(large):
        d = 100 * (large.cov_det.mean() - small.cov_det.mean())
        print(f"\n预注册 B3：仅检测先验覆盖率，表观 ≥16px 比 <8px 高 {d:.1f} 个点"
              f"（预测 ≥ 10；<8px {len(small)} 个，≥16px {len(large)} 个）")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(agg))
    for col, lab, color in (("seen", "扫视中被粗检测命中", MF.SERIES[0]), ("cov_det", "选片覆盖：仅检测先验", MF.SERIES[1]),
                            ("cov_det_edge", "选片覆盖：检测先验 + 边缘", MF.SERIES[2])):
        ax.plot(x, agg[col] * 100, "-o", color=color, label=lab)
    ax.plot(x, agg.cov_random * 100, "--", color=MF.MUTED, label="同数量随机选片")
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(agg["bin"], agg.n)], fontsize=8)
    ax.set_xlabel("目标在扫视图中的表观边长（像素）")
    ax.set_ylabel("比例（%）")
    ax.set_ylim(-2, 102)
    ax.set_title(f"扫视看得见，才选得到（{args.dataset}，θ={args.theta}）", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    (R.RES / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(R.RES / "figures" / "fig_apparent_size.png", dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="visdrone")
    ap.add_argument("--res-tag", default="")
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--img-weight", type=float, default=0.3)
    main(ap.parse_args())
