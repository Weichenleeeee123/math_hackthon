"""单图对比可视化（读缓存，不需要 GPU）：SAHI 均匀切片 vs Glance-SAHI 只切可疑区域。

用法：
  python scripts/visualize.py              # 自动挑 3 张：省得多 / 中位 / 覆盖最差（失败案例）
  python scripts/visualize.py --ids 12 40  # 指定 image_id
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures  # noqa: E402,F401  （复用字体与配色设置）
from make_figures import INK, SERIES, SURFACE  # noqa: E402
import run_eval as R  # noqa: E402
from run_eval import gt_centers, load_rgb, merge, slice_scores  # noqa: E402

from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.saliency import heatmap_prior  # noqa: E402
from glance_sahi.selector import select_slices  # noqa: E402


CLS_COLORS = [SERIES[2], SERIES[3], SERIES[4]]  # 评测类别 1/2/3（橙色留给“被选中的切片”）


def draw(ax, img, slices, sel, scores, dets, title, show_scores, dim_skipped=False):
    sel_set = set(sel)
    shown = img
    if dim_skipped:  # 未推理的切片压暗：一张图讲清“我跳过了什么”
        shown = img.copy()
        for k, (x1, y1, x2, y2) in enumerate(slices):
            if k not in sel_set:
                sub = shown[y1:y2, x1:x2]
                sub[:] = (sub * 0.45 + 60).astype(np.uint8)
    ax.imshow(shown)
    for k, (x1, y1, x2, y2) in enumerate(slices):
        if show_scores:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=SERIES[0],
                                   alpha=0.45 * float(1 - scores[k]), edgecolor="none"))
        on = k in sel_set
        ax.add_patch(Rectangle((x1 + 3, y1 + 3), x2 - x1 - 6, y2 - y1 - 6, fill=False,
                               edgecolor=SERIES[1] if on else "white", lw=1.8 if on else 0.6,
                               alpha=1.0 if on else 0.5))
    _boxes(ax, dets)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.axis("off")


def _boxes(ax, dets):
    for x1, y1, x2, y2, s, c in dets:
        if int(c) in R.COCO_TO_EVAL and s >= 0.3:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.8,
                                   edgecolor=CLS_COLORS[R.COCO_TO_EVAL[int(c)] - 1]))


def draw_heat(ax, img, heat, slices, sel, dets, title):
    """第三张：检测热图（粗检证据）。回答“为什么选这些片”，是解释方法的关键证据。"""
    ax.imshow(img)
    disp = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    ax.imshow(disp, cmap="jet", alpha=0.5, vmin=0, vmax=max(float(heat.max()), 1e-6))
    sel_set = set(sel)
    for k, (x1, y1, x2, y2) in enumerate(slices):
        if k in sel_set:
            ax.add_patch(Rectangle((x1 + 3, y1 + 3), x2 - x1 - 6, y2 - y1 - 6, fill=False,
                                   edgecolor=SERIES[1], lw=1.8))
    _boxes(ax, dets)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--op", type=float, default=0.9)
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--prior", default="det+edge",
                    help="det / edge / spectral / det+edge / det+spectral / uncertain(+edge) / max(+edge) / heatmap(+edge)")
    args = ap.parse_args()
    R.set_dataset(args.dataset)
    CACHE, RES = R.CACHE, R.RES

    cfg = GlanceConfig(img_weight=args.img_weight)
    cache = pickle.loads(CACHE.read_bytes())
    recs = {r["id"]: r for r in cache["images"]}
    gt = json.loads(R.GT.read_text())
    centers = gt_centers(gt)
    legend = "，".join(f"{n} = {c['name']}" for n, c in zip(["绿", "黄", "粉"], gt["categories"]))

    ids = args.ids
    if not ids:
        # 按“置信检测(≥0.3)是否保住”自动挑 3 张：成功（省得多且几乎不丢）/ 典型 / 失败（丢得最多）
        rows = []
        for r in cache["images"]:
            n = len(r["slices"])
            sel = select_slices(slice_scores(r, cfg, *R.prior_kinds(args.prior))[0], "threshold", args.op, 0)
            a = merge(r, np.arange(n), cfg)[0]
            b = merge(r, sel, cfg)[0]
            na, nb = int((a[:, 4] >= 0.3).sum()), int((b[:, 4] >= 0.3).sum())
            rows.append((r["id"], len(sel) / n, na, nb))
        g = pd.DataFrame(rows, columns=["image_id", "frac", "na", "nb"])
        ok = g[(g.na >= 15) & (g.nb >= 0.95 * g.na)]
        ok = ok if len(ok) else g
        ids = [int(ok.sort_values(["frac", "na"], ascending=[True, False]).iloc[0].image_id),
               int(g.iloc[(g.frac - g.frac.median()).abs().argsort().iloc[0]].image_id),
               int(g.assign(lost=g.na - g.nb).sort_values("lost").iloc[-1].image_id)]

    out = RES / "vis"
    out.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        rec = recs[iid]
        img = load_rgb(R.IMAGES / rec["file_name"])
        n = len(rec["slices"])
        scores = slice_scores(rec, cfg, *R.prior_kinds(args.prior))[0]
        sel = select_slices(scores, "threshold", args.op, 0)
        d_sahi, _ = merge(rec, np.arange(n), cfg)
        d_gl, _ = merge(rec, sel, cfg)
        na, nb = int((d_sahi[:, 4] >= 0.3).sum()), int((d_gl[:, 4] >= 0.3).sum())
        t_sahi = rec["t_glance"] + rec["t_slice"].sum()
        t_gl = rec["t_glance"] + slice_scores(rec, cfg, *R.prior_kinds(args.prior))[1] + rec["t_slice"][sel].sum()
        c = centers.get(iid, np.zeros((0, 3)))
        h, w = rec["hw"]
        heat = heatmap_prior(rec["glance"][:, :4], rec["glance"][:, 4], rec["hw"], rec["slices"],
                             cfg.img_map_size, cfg.heat_sigma)[1]
        fig, axes = plt.subplots(1, 3, figsize=(21, 21 * h / w / 3 + 0.9))
        draw(axes[0], img, rec["slices"], np.arange(n), scores, d_sahi,
             f"SAHI 均匀切片：{n}/{n} 片，≈{1000 * t_sahi:.0f} ms", False)
        draw(axes[1], img, rec["slices"], sel, scores, d_gl,
             f"Glance-SAHI[{args.prior}]：{len(sel)}/{n} 片（橙框，未选压暗），≈{1000 * t_gl:.0f} ms",
             True, dim_skipped=True)
        draw_heat(axes[2], img, heat, rec["slices"], sel, d_gl,
                  "检测热图（粗检证据）：越红 = 附近弱检测置信度之和越大，橙框 = 被选中细看的切片")
        fig.suptitle(f"{rec['file_name']}  |  真值目标 {len(c)} 个，其中小目标 {int(c[:, 2].sum()) if len(c) else 0} 个"
                     f"  |  检测框：{legend}（显示置信度 ≥ 0.3）", color=INK, fontsize=10)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(out / f"compare_{iid}.jpg", dpi=110)
        plt.close(fig)
        print(f"{rec['file_name']}: SAHI {n} slices / {len(d_sahi)} dets; Glance {len(sel)} slices / {len(d_gl)} dets")


if __name__ == "__main__":
    main()
