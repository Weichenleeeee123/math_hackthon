"""受控实验：把“目标占比”变成唯一自变量，量出“越稀疏 → 越省”这条机制。

定位：**受控实验 / 机制验证**，不是主结果。主结论仍然来自真实数据集（VisDrone 548 张、
DOTA 458 张）。本脚本的作用是补上 REPORT 4.4 里如实承认的那句“4K 未实测”——
用可控的自造画布把稀疏度推到 4K 量级，直接量出切片用量与上限。

构造（借自参照实现的 Sparse4K 思路）：
- 画布 3840×2160 = 3×3 格，每格 1280×720；
- k 格放真实 VisDrone 图（标注同步缩放、平移），其余格放“去目标”的真实航拍背景
  —— 背景是把另一张 VisDrone 图的目标框 inpaint 抹掉，**保留道路/屋顶/停车线等困难负样本**，
  不是纯色假图；目标覆盖面积 > 25% 的图不进背景池（抹得太多会糊成一片）；
- 于是“目标占比”只由 k 决定，可以直接量三类与检测器无关的量：
    1) SAHI 切片数（随面积线性增长）
    2) 空切片比例（不含任何真值目标）
    3) Oracle 可省比例（只跑含目标的切片 = 可省上界）
  以及一条**不需要检测器**的选片质量曲线：仅边缘先验的“覆盖率 vs 切片比例”，配同数量随机对照。

--save 会把画布按 VisDrone 格式落盘，之后可以用检测器补上真实 AP（可选，需要 GPU）：
  python scripts/prepare_data.py --dataset sparse4k
  python scripts/run_eval.py cache --dataset sparse4k
  python scripts/run_eval.py sim   --dataset sparse4k
  python scripts/run_eval.py buckets --dataset sparse4k

  python scripts/sparsity_sweep.py                 # 只算几何/图像先验 → results/sparsity_*.csv + fig8
  python scripts/sparsity_sweep.py --save          # 同时把画布写到 datasets/VisDrone-Sparse4K/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sahi.slicing import get_slice_bboxes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures as MF  # noqa: E402
import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.imageio import imread, imwrite  # noqa: E402
from glance_sahi.saliency import image_prior_map, region_prior  # noqa: E402
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

plt = MF.plt

CW, CH, G = 1280, 720, 3          # 每格尺寸 × 3×3 = 3840×2160（4K）
THS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SMALL = 32 * 32
SEEDS = 2


def load_ann(ann_dir: Path, stem: str) -> np.ndarray:
    rows = []
    for line in (ann_dir / f"{stem}.txt").read_text().splitlines():
        v = [int(x) for x in line.strip().strip(",").split(",")[:8]]
        if len(v) >= 6 and v[2] > 0 and v[3] > 0:
            rows.append(v + [0] * (8 - len(v)))
    return np.array(rows, np.int32).reshape(-1, 8)


def background(path: Path, ann: np.ndarray):
    """抹掉所有目标（含忽略区），只把被抹区域贴回去；返回 (背景图, 目标覆盖比例)。"""
    src = imread(path)
    if src is None:
        return None, 1.0
    img = cv2.resize(src, (CW, CH), interpolation=cv2.INTER_AREA)
    sx, sy = CW / src.shape[1], CH / src.shape[0]
    mask = np.zeros((CH, CW), np.uint8)
    for x, y, w, h, *_ in ann:
        cv2.rectangle(mask, (int(x * sx) - 4, int(y * sy) - 4),
                      (int((x + w) * sx) + 4, int((y + h) * sy) + 4), 255, -1)
    small = cv2.resize(img, (CW // 2, CH // 2))
    m2 = cv2.resize(mask, (CW // 2, CH // 2), interpolation=cv2.INTER_NEAREST)
    filled = cv2.resize(cv2.inpaint(small, m2, 5, cv2.INPAINT_TELEA), (CW, CH))
    img[mask > 0] = filled[mask > 0]
    return img, float(mask.mean()) / 255.0


def compose(rng, k, files, anns, fg_pool, bgs):
    canvas = np.zeros((CH * G, CW * G, 3), np.uint8)
    cells = rng.permutation(G * G)
    for c in cells[k:]:
        r, q = divmod(int(c), G)
        canvas[r * CH:(r + 1) * CH, q * CW:(q + 1) * CW] = bgs[rng.integers(len(bgs))]
    lines, boxes = [], []
    for c in cells[:k]:
        r, q = divmod(int(c), G)
        p = fg_pool[rng.integers(len(fg_pool))]
        src = imread(p)
        if src is None:
            continue
        canvas[r * CH:(r + 1) * CH, q * CW:(q + 1) * CW] = cv2.resize(src, (CW, CH), interpolation=cv2.INTER_AREA)
        sx, sy = CW / src.shape[1], CH / src.shape[0]
        for x, y, w, h, s, cat, tr, oc in anns[p.stem]:
            X, Y = int(round(x * sx)) + q * CW, int(round(y * sy)) + r * CH
            W, H = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
            lines.append(f"{X},{Y},{W},{H},{s},{cat},{tr},{oc}")
            if cat != 0 and int(s) != 0:  # 忽略区不算目标（与评测口径一致）
                boxes.append([X, Y, X + W, Y + H, cat])
    return canvas, lines, np.array(boxes, np.float32).reshape(-1, 5)


def cover(gt, slices, sel):
    """被选切片覆盖的 GT 中心掩码（与 run_eval.covered 同口径）。"""
    if len(gt) == 0:
        return np.zeros(0, bool)
    cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
    hit = np.zeros(len(gt), bool)
    for k in sel:
        x1, y1, x2, y2 = slices[k]
        hit |= (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4,6,8", help="每张画布放几格真实目标图")
    ap.add_argument("--per-k", type=int, default=6)
    ap.add_argument("--bg-pool", type=int, default=60)
    ap.add_argument("--save", action="store_true", help="同时写 datasets/VisDrone-Sparse4K/（供之后跑检测器）")
    a = ap.parse_args()

    R.set_dataset("visdrone")
    ann_dir = Path(R.IMAGES).parent / "annotations"
    files = sorted(Path(R.IMAGES).glob("*.jpg"))
    anns = {f.stem: load_ann(ann_dir, f.stem) for f in files}
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(files))
    bg_pool, fg_pool = [files[i] for i in perm[: a.bg_pool]], [files[i] for i in perm[a.bg_pool:]]

    bgs = []
    for p in bg_pool:
        img, frac = background(p, anns[p.stem])
        if img is not None and frac < 0.25:  # 目标覆盖太高的图抹不干净，不适合当背景
            bgs.append(img)
    print(f"背景池 {len(bgs)}/{len(bg_pool)} 张（其余目标占比过高，inpaint 后不适合当背景）")

    cfg = GlanceConfig()
    save_dir = ROOT / "datasets" / "VisDrone-Sparse4K"
    if a.save:
        (save_dir / "images").mkdir(parents=True, exist_ok=True)
        (save_dir / "annotations").mkdir(parents=True, exist_ok=True)

    summary, curve = [], []
    for k in [int(v) for v in a.ks.split(",")]:
        agg = dict(images=0, area=0.0, slices=0, empty=0, oracle=0, tot_gt=0, tot_small=0)
        acc = {}
        for j in range(a.per_k):
            canvas, lines, gt = compose(rng, k, files, anns, fg_pool, bgs)
            h, w = canvas.shape[:2]
            slices = get_slice_bboxes(h, w, cfg.slice_size, cfg.slice_size, False,
                                      cfg.overlap_ratio, cfg.overlap_ratio)
            gt_wh = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0, np.float32)
            area = float(gt_wh.sum())
            small = gt_wh < SMALL

            agg["images"] += 1
            agg["area"] += area / (h * w)
            agg["slices"] += len(slices)
            agg["tot_gt"] += len(gt)
            agg["tot_small"] += int(small.sum())
            occupied = set()
            for si, (x1, y1, x2, y2) in enumerate(slices):
                if len(gt) and cover(gt, slices, [si]).any():
                    occupied.add(si)
            agg["empty"] += len(slices) - len(occupied)
            agg["oracle"] += len(occupied)

            # 仅边缘先验：不需要检测器的选片质量（覆盖-切片比例曲线）
            sal, scale = image_prior_map(canvas, "edge", cfg.img_map_size)
            s_img = region_prior(sal, scale, slices)
            for th in THS:
                sel = select_slices(s_img, "threshold", th, 0)
                hit = cover(gt, slices, sel)
                c = acc.setdefault(("edge", th), dict(n_run=0, n_sl=0, hit=0, n_gt=0, hit_s=0, n_s=0))
                c["n_run"] += len(sel)
                c["n_sl"] += len(slices)
                c["hit"] += int(hit.sum())
                c["n_gt"] += len(gt)
                if len(gt):
                    c["hit_s"] += int((hit & small).sum())
                    c["n_s"] += int(small.sum())
                for seed in range(SEEDS):
                    rnd = random_slices(len(slices), len(sel), np.random.default_rng(seed))
                    hr = cover(gt, slices, rnd)
                    c = acc.setdefault((f"random#{seed}", th), dict(n_run=0, n_sl=0, hit=0, n_gt=0, hit_s=0, n_s=0))
                    c["n_run"] += len(rnd)
                    c["n_sl"] += len(slices)
                    c["hit"] += int(hr.sum())
                    c["n_gt"] += len(gt)
                    if len(gt):
                        c["hit_s"] += int((hr & small).sum())
                        c["n_s"] += int(small.sum())

            if a.save:
                name = f"s4k_k{k}_{j:03d}"
                imwrite(save_dir / "images" / f"{name}.jpg", canvas, quality=92)
                (save_dir / "annotations" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        n_img = agg["images"]
        summary.append(dict(k=k, canvases=n_img, obj_area_frac=agg["area"] / n_img,
                            slices_per_img=agg["slices"] / n_img, empty_slice_frac=agg["empty"] / agg["slices"],
                            oracle_slice_frac=agg["oracle"] / agg["slices"],
                            gt_per_img=agg["tot_gt"] / n_img, small_frac=agg["tot_small"] / max(agg["tot_gt"], 1)))
        for (method, th), c in acc.items():
            curve.append(dict(k=k, method=method, threshold=th, slice_frac=c["n_run"] / c["n_sl"],
                              cov_all=c["hit"] / max(c["n_gt"], 1), cov_small=c["hit_s"] / max(c["n_s"], 1)))
        print(f"k={k}: 目标面积占比 {summary[-1]['obj_area_frac']:.2%}  "
              f"切片/图 {summary[-1]['slices_per_img']:.1f}  "
              f"空切片 {summary[-1]['empty_slice_frac']:.1%}  "
              f"Oracle 需跑 {summary[-1]['oracle_slice_frac']:.1%}", flush=True)

    R.RES.mkdir(exist_ok=True)
    (R.RES / "figures").mkdir(exist_ok=True)
    sdf, cdf = pd.DataFrame(summary), pd.DataFrame(curve)
    sdf.to_csv(R.RES / "sparsity_summary.csv", index=False)
    cdf.to_csv(R.RES / "sparsity_curve.csv", index=False)
    print(f"wrote {R.RES / 'sparsity_summary.csv'} 与 {R.RES / 'sparsity_curve.csv'}")
    if a.save:
        print(f"画布已写入 {save_dir}：可接着跑 "
              f"prepare_data.py --dataset sparse4k / run_eval.py cache|sim|buckets --dataset sparse4k")

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(sdf.k, sdf.empty_slice_frac * 100, "-o", color=MF.SERIES[1], label="空切片比例（SAHI 白跑的部分）",
            markeredgecolor=MF.SURFACE)
    ax.plot(sdf.k, (1 - sdf.oracle_slice_frac) * 100, "-s", color=MF.SERIES[6], label="Oracle 可省比例（上界）",
            markeredgecolor=MF.SURFACE)
    for _, r in sdf.iterrows():
        ax.annotate(f"{r.slices_per_img:.0f} 片/图", (r.k, (1 - r.oracle_slice_frac) * 100),
                    textcoords="offset points", xytext=(0, -14), ha="center", color=MF.INK2, fontsize=8)
    ax.set_xlabel("画布里放了几格真实目标图（3×3 = 9 格，其余是去目标的真实背景）")
    ax.set_ylabel("占全部切片的比例（%）")
    ax.set_title("受控实验：图越稀疏，可省的比例越大", loc="left", color=MF.INK)
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1]
    for k, color in zip(sorted(cdf.k.unique()), MF.SERIES):
        d = cdf[(cdf.k == k) & (cdf.method == "edge")].sort_values("slice_frac")
        ax.plot(d.slice_frac * 100, d.cov_all * 100, "-o", color=color, label=f"k={k}",
                markeredgecolor=MF.SURFACE, markeredgewidth=1.0)
    rnd = cdf[cdf.method.str.startswith("random")].groupby("threshold", as_index=False)[["slice_frac", "cov_all"]].mean()
    ax.plot(rnd.slice_frac.sort_values() * 100, rnd.sort_values("slice_frac").cov_all * 100, "--", color=MF.MUTED,
            label="随机选片（对照）")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("实际推理的切片比例（%）")
    ax.set_ylabel("真值目标覆盖率（%）")
    ax.set_title("仅边缘先验（与检测器无关）：每片预算下的覆盖率", loc="left", color=MF.INK)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(R.RES / "figures" / "fig8_sparsity.png")
    plt.close(fig)
    print(f"wrote {R.RES / 'figures' / 'fig8_sparsity.png'}")


if __name__ == "__main__":
    main()
