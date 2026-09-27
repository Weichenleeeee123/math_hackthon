"""漏检归因分解：把漏掉的目标拆成"检测器能力"和"选片代价"（REPORT 3.13）。

对每个非 crowd 的评测目标，问三个问题：
  1. 最终输出里有匹配的检测框吗？（命中）
  2. 任意切片（**含被丢弃的**）里有匹配框吗？（这个目标本来检得出来吗）
  3. 被选中的切片里有匹配框吗？

据此把漏检拆成三类（这是 3.5 的 small_cov 只回答了"覆盖面"、没回答"责任"的那一半）：
  A 检测器能力：任何切片都检不出 → 换任何选片策略都救不回来，不是选片的锅
  B 选片代价：**只有被丢弃的切片**里能检出 → 这才是选片造成的真实损失
  C 选中却漏：被选切片里检出了但没进最终输出（被 NMS 合并/过滤）→ 白白跑了，应接近 0

输出：results/attribution.csv、results/figures/fig10_attribution.png
用法：python scripts/attribute.py [--dataset visdrone|dota] [--ths 0.9 0.99]
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

from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.router import gt_targets  # noqa: E402,F401  （同口径，active_eval 经 AT.gt_targets 复用）
from glance_sahi.selector import select_slices  # noqa: E402

import run_eval as R  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def det_arrays(dets: np.ndarray):
    """(N,6) 检测 → (中心, 评测类别)；不可映射到评测类别的框返回 -1 类别。"""
    if len(dets) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, int)
    d = np.asarray(dets, np.float32).reshape(-1, 6)
    centers = np.stack([(d[:, 0] + d[:, 2]) / 2.0, (d[:, 1] + d[:, 3]) / 2.0], axis=1)
    cls = np.array([R.COCO_TO_EVAL.get(int(round(c)), -1) for c in d[:, 5]], dtype=int)
    return centers, cls


def hit_mask(ref_centers, ref_cls, gt, sel_mask=None, det_slice=None):
    """某个 GT 是否被 ref 里的检测命中（可选：只考虑 sel_mask 为真的那些检测）。"""
    cx, cy, r, gcls, _ = gt
    m = ref_cls == gcls
    if sel_mask is not None:
        m = m & sel_mask
    if not m.any():
        return False
    d = np.hypot(ref_centers[m, 0] - cx, ref_centers[m, 1] - cy)
    return bool((d <= r).any())


def analyze(cache, targets, cfg, method, select_fn):
    """统计一个方法下的 命中 / A / B / C，全部与小目标分开计。"""
    stats = {k: dict(n=0, hit=0, miss_detector=0, miss_selection=0, miss_selected=0)
             for k in ("all", "small")}
    for rec in cache["images"]:
        sel = select_fn(rec)
        sel_arr = np.asarray(sel, dtype=int)
        sel_mask = np.isin(np.arange(len(rec["slice_preds"])), sel_arr)

        all_c, all_cls, all_slice = [], [], []
        for k, p in enumerate(rec["slice_preds"]):
            c, cl = det_arrays(p)
            all_c.append(c), all_cls.append(cl)
            all_slice.append(np.full(len(c), k, dtype=int))
        slice_c = np.concatenate(all_c) if all_c else np.zeros((0, 2), np.float32)
        slice_cls = np.concatenate(all_cls) if all_cls else np.zeros(0, int)
        slice_of = np.concatenate(all_slice) if all_slice else np.zeros(0, int)
        slice_sel = sel_mask[slice_of] if len(slice_of) else np.zeros(0, bool)

        fin, _ = R.merge(rec, sel, cfg)
        fin_c, fin_cls = det_arrays(fin)

        for gt in targets.get(rec["id"], []):
            bucket = "small" if gt[4] else "all"
            keys = ("all", bucket) if bucket == "small" else ("all",)
            for key in keys:
                stats[key]["n"] += 1
            if hit_mask(fin_c, fin_cls, gt):
                for key in keys:
                    stats[key]["hit"] += 1
            elif hit_mask(slice_c, slice_cls, gt, sel_mask=slice_sel):
                for key in keys:
                    stats[key]["miss_selected"] += 1
            elif hit_mask(slice_c, slice_cls, gt):
                for key in keys:
                    stats[key]["miss_selection"] += 1
            else:
                for key in keys:
                    stats[key]["miss_detector"] += 1
    rows = []
    for key, s in stats.items():
        rows.append(dict(method=method, subset=key, n_gt=s["n"], hit=s["hit"],
                         miss_detector=s["miss_detector"], miss_selection=s["miss_selection"],
                         miss_selected=s["miss_selected"]))
    return rows


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import make_figures as MF

    d = df[df.subset == "small"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    parts = [("hit", "命中", MF.SERIES[2]),
             ("miss_detector", "漏：检测器能力（任何切片都检不出）", MF.SERIES[3]),
             ("miss_selection", "漏：选片代价（只在丢弃的切片里能检出）", MF.SERIES[1]),
             ("miss_selected", "漏：选中却漏（应≈0）", MF.SERIES[7])]
    left = np.zeros(len(d))
    for col, lab, color in parts:
        share = d[col].to_numpy() / d["n_gt"].to_numpy() * 100
        ax.barh(d.method, share, left=left, color=color, label=lab, height=0.55)
        for y, (l, s) in enumerate(zip(left, share)):
            if s > 3:
                ax.text(l + s / 2, y, f"{s:.1f}", ha="center", va="center",
                        color="white", fontsize=8.5)
        left = left + share
    ax.set_xlabel("占小目标真值的比例（%）")
    ax.set_title("小目标漏检归因：多少是选片的代价，多少是检测器本身检不出", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"写出 {out_path}")


def main(args):
    R.set_dataset(args.dataset)
    cache = pickle.loads(R.CACHE.read_bytes())
    if args.limit:
        cache = {**cache, "images": cache["images"][: args.limit]}
    targets = gt_targets(json.loads(R.GT.read_text()))
    cfg = GlanceConfig(img_weight=args.img_weight)

    methods = {"sahi_uniform": lambda r: np.arange(len(r["slices"]))}
    for th in args.ths:
        methods[f"glance@{th}"] = (lambda th: lambda r: select_slices(
            R.slice_scores(r, cfg, "noisyor", "edge")[0], "threshold", th, 0))(th)

    rows = []
    for name, fn in methods.items():
        rows += analyze(cache, targets, cfg, name, fn)
    df = pd.DataFrame(rows)
    df.to_csv(R.RES / "attribution.csv", index=False)

    for subset, title in (("all", "全部评测目标"), ("small", "小目标（<32²）")):
        print(f"\n=== {title} ===")
        print(f"{'方法':>16} {'目标数':>8} {'命中':>7} {'漏-检测器':>10} "
              f"{'漏-选片代价':>12} {'漏-选中却漏':>12}")
        for _, r in df[df.subset == subset].iterrows():
            print(f"{r['method']:>16} {int(r['n_gt']):>8} {r['hit'] / r['n_gt']:>7.1%} "
                  f"{r['miss_detector'] / r['n_gt']:>10.1%} "
                  f"{r['miss_selection'] / r['n_gt']:>12.2%} "
                  f"{r['miss_selected'] / r['n_gt']:>12.2%}")

    make_figure(df, R.RES / "figures" / "fig10_attribution.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone", choices=list(R.datasets.DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ths", type=float, nargs="*", default=[0.9, 0.99])
    ap.add_argument("--img-weight", type=float, default=0.3)
    main(ap.parse_args())
