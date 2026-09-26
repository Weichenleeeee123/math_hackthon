"""粗检测置信度的分箱标定实验（对应 REPORT 3.12）。

奇偶拆分：奇数序号的图拟合标定器（调参集），偶数序号的图做留出验证（留出集），
与 REPORT 3.8 的诊断协议一致。全程离线读 `results/cache.pkl`，不需要 GPU/检测器。

输出：
  results/calibration.csv        每个表观尺度箱的样本量与映射（"改了什么"）
  results/calib_reliability.csv  标定前后的 ECE 与可靠性表（"结果如何"之一）
  results/calib_holdout.csv      留出集上 标定 vs 未标定 的 AP / 切片比例 / 小目标覆盖率

用法（PowerShell）：
  & $py scripts/calibrate.py                          # VisDrone
  & $py scripts/calibrate.py --dataset dota
  & $py scripts/calibrate.py --min-bin 80 --grid 201
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

# Windows 控制台默认 GBK，中文与 p̂ 这类组合字符会报 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from glance_sahi.calibration import (  # noqa: E402
    DetectionCalibrator, expected_calibration_error, inside_any_box, match_to_gt,
)
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.saliency import detection_prior, fuse  # noqa: E402
from glance_sahi.selector import select_slices  # noqa: E402

import run_eval as R  # noqa: E402  （复用缓存路径、COCO 评测与合并逻辑，保证口径完全一致）


# --------------------------------------------------------------------------- 真值索引
def gt_index(gt: dict):
    """按 (图像, 评测类别) 索引非 crowd 目标的中心与对角线；crowd 框单独存。

    标定拟合的标签定义**与评测口径解耦**：这里只关心"这个粗检测是不是真的"，
    所以用中心距离匹配；crowd/忽略区里的检测不参与拟合（避免脏标签）。
    """
    objs: dict[int, dict[int, list]] = {}
    crowds: dict[int, list] = {}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        if a["iscrowd"]:
            crowds.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
            continue
        objs.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(
            (x + w / 2.0, y + h / 2.0, float(np.hypot(w, h))))
    packed = {
        iid: {c: (np.array([(p[0], p[1]) for p in v], np.float32),
                np.array([p[2] for p in v], np.float32)) for c, v in per.items()}
        for iid, per in objs.items()
    }
    crowd_boxes = {k: np.asarray(v, np.float32).reshape(-1, 4) for k, v in crowds.items()}
    return packed, crowd_boxes


def gather_detections(cache, objs, crowds, ids, cal: DetectionCalibrator):
    """把若干张图的粗检测汇总成 (表观尺度, 置信度, 是否真阳) 三个数组。"""
    sizes, scores, labels = [], [], []
    for rec in cache["images"]:
        if rec["id"] not in ids:
            continue
        g = rec["glance"]
        if len(g) == 0:
            continue
        boxes, sc, cls = g[:, :4], g[:, 4], g[:, 5]
        scale = cal.scale_for(rec["hw"])
        sz = cal.apparent_size(boxes, scale)
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                            (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
        ignore = inside_any_box(centers, crowds.get(rec["id"], np.zeros((0, 4), np.float32)))
        ev = np.array([R.COCO_TO_EVAL.get(int(round(c)), -1) for c in cls])
        lab = np.zeros(len(g), dtype=bool)
        per_img = objs.get(rec["id"], {})
        for ec in np.unique(ev):
            if ec < 0 or ec not in per_img:
                continue
            m = ev == ec
            gc, gd = per_img[ec]
            lab[m] = match_to_gt(centers[m], gc, gd)
        keep = ~ignore
        sizes.append(sz[keep]), scores.append(sc[keep]), labels.append(lab[keep])
    if not sizes:
        return np.zeros(0), np.zeros(0), np.zeros(0, bool)
    return (np.concatenate(sizes), np.concatenate(scores), np.concatenate(labels))


# --------------------------------------------------------------------------- 打分
def calib_scores(rec, cfg: GlanceConfig, cal: DetectionCalibrator, img_kind: str | None):
    """标定后的切片分数：p̂ 代替原始 c 进入 noisy-OR，其余与主方法完全一致。"""
    boxes, raw = rec["glance"][:, :4], rec["glance"][:, 4]
    p = cal.transform(boxes, raw, cal.scale_for(rec["hw"]))
    s_det = detection_prior(boxes, p, rec["slices"], cfg.det_margin, "noisyor")
    if img_kind is None:
        return s_det
    return fuse(s_det, rec[f"prior_{img_kind}"], cfg.img_weight)


# --------------------------------------------------------------------------- 出图
def make_figure(rel: pd.DataFrame, df: pd.DataFrame, ece_raw: float, ece_cal: float,
                out_path: Path) -> None:
    """fig9：左=可靠性图（标定前 vs 标定后），右=同预算下的精度-切片比例曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import make_figures as MF  # 复用报告统一的配色，并在导入时套好中文字体（Microsoft YaHei 等）

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MF.MUTED, label="完美标定")
    for stage, mark, color, lab in (("raw", "o", MF.SERIES[7], f"未标定 c（ECE={ece_raw:.3f}）"),
                                    ("calibrated", "s", MF.SERIES[0], f"标定后概率（ECE={ece_cal:.3f}）")):
        d = rel[rel.stage == stage].sort_values("mean_conf")
        ax.plot(d.mean_conf, d.empirical_rate, marker=mark, color=color, label=lab)
    ax.set_xlabel("预测为真目标的概率")
    ax.set_ylabel("实际为真目标的比例")
    ax.set_title("可靠性图（留出集）：先验概率能不能当概率用", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1]
    b = df[df["family"].astype(str).str.contains("budget")]
    for fam, mark, color, lab in (("glance_raw_budget", "o", MF.SERIES[7], "Glance 未标定"),
                                  ("glance_calib_budget", "s", MF.SERIES[0], "Glance 标定后"),
                                  ("random_budget", "^", MF.SERIES[3], "随机对照（3 种子均值）")):
        d = b[b["family"] == fam]
        if d.empty:
            continue
        g = d.groupby("slice_frac", as_index=False)["AP"].mean().sort_values("slice_frac")
        ax.plot(g.slice_frac * 100, g.AP, marker=mark, color=color, ls="-" if "random" not in fam else "--",
                label=lab)
    sahi = df[df["method"] == "sahi_uniform"]
    if not sahi.empty:
        ax.axhline(float(sahi.AP.iloc[0]), color=MF.INK2, ls=":", lw=1.2,
                   label=f"SAHI 全切（AP={float(sahi.AP.iloc[0]):.4f}）")
    ax.set_xlabel("切片比例（%）")
    ax.set_ylabel("COCO AP")
    ax.set_title("同预算下的排序质量（留出集）", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"写出 {out_path}")


# --------------------------------------------------------------------------- 只重算产物
def rebuild_only(args):
    """不重跑评测：从已保存的 calibrator.pkl 与两张 CSV 重新生成分箱表和图 9。

    评测部分（COCOeval 在高切片预算下很慢）已经落盘，改表格/改图时不必再跑一遍。
    """
    R.set_dataset(args.dataset)
    cal = DetectionCalibrator.load(R.RES / "calibrator.pkl")
    pd.DataFrame(cal.summary()).to_csv(R.RES / "calibration.csv", index=False)
    rel = pd.read_csv(R.RES / "calib_reliability.csv")
    df = pd.read_csv(R.RES / "calib_holdout.csv")
    ece = {}
    for stage, d in rel.groupby("stage"):
        ece[stage] = float((d.n / d.n.sum() * d.gap.abs()).sum())
    make_figure(rel, df, ece["raw"], ece["calibrated"],
                R.RES / "figures" / "fig9_calibration.png")
    print(f"重建完成：ECE(raw)={ece['raw']:.4f} ECE(calibrated)={ece['calibrated']:.4f}")


# --------------------------------------------------------------------------- 主流程
def main(args):
    R.set_dataset(args.dataset)
    cache = pickle.loads(R.CACHE.read_bytes())
    gt = json.loads(R.GT.read_text())
    objs, crowds = gt_index(gt)
    centers = R.gt_centers(gt)

    images = cache["images"]
    if args.limit:
        images = images[: args.limit]
        cache = {**cache, "images": images}
    fit_ids = {rec["id"] for i, rec in enumerate(images) if i % 2 == 1}
    hold_ids = {rec["id"] for i, rec in enumerate(images) if i % 2 == 0}
    print(f"图数 {len(images)}：调参集（奇数）{len(fit_ids)}，留出集（偶数）{len(hold_ids)}")

    cal = DetectionCalibrator(model_input=args.model_input, grid_size=args.grid,
                              min_bin=args.min_bin)
    sz, sc, lab = gather_detections(cache, objs, crowds, fit_ids, cal)
    print(f"调参集粗检测 {len(sc)} 条，真阳率 {lab.mean():.3f}")
    cal.fit(sz, sc, lab)

    # --- 1) 表观尺度分箱：映射长什么样 ---
    cal_rows = cal.summary()
    pd.DataFrame(cal_rows).to_csv(R.RES / "calibration.csv", index=False)
    print("\n表观尺度分箱（模型输入像素）与标定映射：")
    print(f"{'箱':>3} {'区间':>14} {'样本':>7} {'回退全局':>8} "
          f"{'p̂(c=.05)':>9} {'p̂(c=.10)':>9} {'p̂(c=.90)':>9}")
    for r in cal_rows:
        hi = "inf" if np.isinf(r["hi"]) else f"{r['hi']:.0f}"
        print(f"{r['bin']:>3} [{r['lo']:>3.0f},{hi:>4}) {r['n']:>7} "
              f"{str(r['used_global']):>8} {r['p_at_c05']:>9.3f} "
              f"{r['p_at_c10']:>9.3f} {r['p_at_c90']:>9.3f}")

    # --- 2) 可靠性：标定前后的 ECE（在留出集上算，避免自证） ---
    sz_h, sc_h, lab_h = gather_detections(cache, objs, crowds, hold_ids, cal)
    ece_raw, ece_cal, rows = cal.reliability(sz_h, sc_h, lab_h, n_bins=args.ece_bins)
    # 原始 c 的可靠性表（不经过标定器），与标定后的并排列出
    ece_raw, rows_raw = expected_calibration_error(sc_h, lab_h, n_bins=args.ece_bins)
    rel = pd.concat([pd.DataFrame(rows_raw).assign(stage="raw"),
                     pd.DataFrame(rows).assign(stage="calibrated")], ignore_index=True)
    rel.to_csv(R.RES / "calib_reliability.csv", index=False)
    print(f"\n留出集 {len(sc_h)} 条粗检测：ECE(raw)={ece_raw:.4f} → ECE(calibrated)={ece_cal:.4f} "
          f"(下降 {100 * (1 - ece_cal / max(ece_raw, 1e-9)):.1f}%)")

    # --- 3) 留出集上的端到端对比：标定前 vs 标定后 ---
    hold = {**cache, "images": [r for r in images if r["id"] in hold_ids]}
    cfg = GlanceConfig(img_weight=args.img_weight)
    results = []

    def add(name, fn, **tags):
        m, _ = R.run_method(hold, R.GT, centers, name, fn, cfg)
        m.update(tags)
        results.append(m)
        print(f"{name:34s} AP={m['AP']:.4f} AP50={m['AP50']:.4f} APs={m['APs']:.4f} "
              f"slices={m['slices_per_img']:.2f} ({m['slice_frac']:.1%}) "
              f"cov_small={m['small_cov']:.3f}")

    add("full_image", lambda r: np.zeros(0, int), family="full")
    add("sahi_uniform", lambda r: np.arange(len(r["slices"])), family="sahi")

    for th in args.ths:
        def fn_raw(r, th=th):
            return select_slices(R.slice_scores(r, cfg, "noisyor", "edge")[0], "threshold", th, 0)
        add(f"glance_raw@{th}", fn_raw, family="glance_raw", threshold=th)

        def fn_cal(r, th=th):
            return select_slices(calib_scores(r, cfg, cal, "edge"), "threshold", th, 0)
        add(f"glance_calib@{th}", fn_cal, family="glance_calib", threshold=th)

        def fn_cal_det(r, th=th):  # 只用检测先验（标定后），不含边缘先验
            return select_slices(calib_scores(r, cfg, cal, None), "threshold", th, 0)
        add(f"glance_calib_det@{th}", fn_cal_det, family="glance_calib_det", threshold=th)

    # 阈值模式对"标定后的绝对概率"不公平（标定把弱检测整体抬高、可阈值区间被压缩）。
    # 同预算下比**排序质量**才是标定的正面检验：同样跑 x% 的切片，谁的 AP/覆盖率更高。
    for b in args.budgets:
        def fn_raw_b(r, b=b):
            return select_slices(R.slice_scores(r, cfg, "noisyor", "edge")[0], "budget", 0.0, b)
        add(f"glance_raw_budget@{b}", fn_raw_b, family="glance_raw_budget", budget=b)

        def fn_cal_b(r, b=b):
            return select_slices(calib_scores(r, cfg, cal, "edge"), "budget", 0.0, b)
        add(f"glance_calib_budget@{b}", fn_cal_b, family="glance_calib_budget", budget=b)

        for seed in range(3):
            rng = np.random.default_rng(seed)

            def fn_rand_b(r, b=b, rng=rng):
                n = len(r["slices"])
                return np.sort(rng.choice(n, size=min(int(round(b * n)), n), replace=False))
            add(f"random_matched_budget@{b}#s{seed}", fn_rand_b,
                family="random_budget", budget=b, seed=seed)

    df = pd.DataFrame(results)
    df.to_csv(R.RES / "calib_holdout.csv", index=False)
    cal.save(R.RES / "calibrator.pkl")
    make_figure(rel, df, ece_raw, ece_cal, R.RES / "figures" / "fig9_calibration.png")
    print(f"\n写出：{R.RES / 'calibration.csv'}、{R.RES / 'calib_reliability.csv'}、"
          f"{R.RES / 'calib_holdout.csv'}、{R.RES / 'calibrator.pkl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone", choices=list(R.datasets.DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-input", type=int, default=640)
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--min-bin", type=int, default=50)
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--ths", type=float, nargs="*", default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--budgets", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--rebuild-only", action="store_true",
                    help="只用已保存的 calibrator.pkl 与 CSV 重建分箱表和图 9，不重跑评测")
    a = ap.parse_args()
    rebuild_only(a) if a.rebuild_only else main(a)
