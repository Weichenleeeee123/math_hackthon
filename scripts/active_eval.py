"""主动式两轮选片 + 可计算停机判据 E 的评测（REPORT 3.14）。

全程离线读 `results/cache.pkl`（每一片的检测结果都在缓存里，"跑第 k 片"= 读缓存），
否则"决策依赖观测"就没法离线复现。不需要 GPU。

对比方式与 REPORT 的主图一致：**画曲线**（AP ~ 切片比例），而不是挑一个工作点比数——
参数是扫出来的一族工作点，避免"挑最好看的那个"。留出集固定为偶数序号 274 张
（与 3.12/3.13 同一套拆分），标定器仍用奇数 274 张拟合好的那个。

关键对照是**每图同数量**的严格匹配：主动式第 1 轮 x% + 第 2 轮追加的片数，正好等于一次性
选片跑同样多的片数，于是问题变成"第 2 轮用观测挑出来的片，是不是比盲选下一批更好"。

输出：
  results/active_holdout.csv    一次性 / 两轮主动 / E 停机 / 随机 的曲线数据
  results/active_E.csv          每张图的 E 与实际选片代价（校验 D 的判据是否真的预测损失）
  results/figures/fig11_active.png

用法：python scripts/active_eval.py [--dataset visdrone]
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

from glance_sahi.active import (  # noqa: E402
    active_select, calibrated_scores, estop_select, expected_missed,
)
from glance_sahi.calibration import DetectionCalibrator  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

import attribute as AT  # noqa: E402  （复用 3.13 的匹配口径，避免两套定义打架）
import run_eval as R  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def oneshot_fn(cfg, cal, budget):
    def fn(rec):
        base, _ = calibrated_scores(rec, cfg, cal)
        return select_slices(base, "budget", 0.0, budget)
    return fn


def active_fn(cfg, cal, budget1, theta2, extra_frac, sigma, gamma):
    def fn(rec):
        sel, _ = active_select(rec, cfg, cal, budget1, theta2, extra_frac, sigma, gamma)
        return sel
    return fn


def estop_fn(cfg, cal, eps):
    def fn(rec):
        sel, _ = estop_select(rec, cfg, cal, eps)
        return sel
    return fn


def matched_oneshot_fn(cfg, cal, reference_fn):
    """**每图同数量**的一次性选片：与 reference 在每张图上跑一样多的切片。

    比"按比例设 budget"严格——按比例会因 round() 系统性少跑（8 片图取 0.9 → 只有 7 片）。
    """
    def fn(rec):
        base, _ = calibrated_scores(rec, cfg, cal)
        k = len(reference_fn(rec))
        return select_slices(base, "budget", 0.0, k / max(len(rec["slices"]), 1))
    return fn


def matched_random_fn(reference_fn, seed):
    rng = np.random.default_rng(seed)

    def fn(rec):
        return random_slices(len(rec["slices"]), len(reference_fn(rec)), rng)
    return fn


def eval_E(cache, cfg, cal, targets, fn):
    """逐图算 E 与实际"选片代价"目标数，检验 E 是不是真的预测了损失（D 的校验）。"""
    rows = []
    for rec in cache["images"]:
        sel = fn(rec)
        _, s_det = calibrated_scores(rec, cfg, cal)
        E = expected_missed(s_det, sel)
        sel_mask = np.isin(np.arange(len(rec["slice_preds"])), np.asarray(sel, int))

        all_c, all_cls, all_slice = [], [], []
        for k, p in enumerate(rec["slice_preds"]):
            c, cl = AT.det_arrays(p)
            all_c.append(c), all_cls.append(cl)
            all_slice.append(np.full(len(c), k, dtype=int))
        sc = np.concatenate(all_c) if all_c else np.zeros((0, 2), np.float32)
        scl = np.concatenate(all_cls) if all_cls else np.zeros(0, int)
        sl = np.concatenate(all_slice) if all_slice else np.zeros(0, int)
        s_sel = sel_mask[sl] if len(sl) else np.zeros(0, bool)

        cost = cost_small = 0
        for gt in targets.get(rec["id"], []):
            if AT.hit_mask(sc, scl, gt, sel_mask=s_sel):
                continue
            if AT.hit_mask(sc, scl, gt):        # 只在被丢弃的切片里能检出 → 选片代价
                cost += 1
                cost_small += int(gt[4])
        rows.append(dict(image_id=rec["id"], E=E, sel_cost=cost, sel_cost_small=cost_small,
                         n_run=len(sel), n_slices=len(rec["slices"])))
    df = pd.DataFrame(rows)
    rho = float(df[["E", "sel_cost_small"]].corr(method="spearman").iloc[0, 1]) \
        if df.E.nunique() > 1 and df.sel_cost_small.nunique() > 1 else float("nan")
    return df, rho


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import make_figures as MF

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for fam, style, color, lab in (
        ("oneshot", "o-", MF.SERIES[7], "一次性选片（标定后）"),
        ("active", "s-", MF.SERIES[0], "两轮主动选片（观测驱动）"),
        ("estop", "d-", MF.SERIES[2], "E 停机（预期漏看 ≤ ε）"),
        ("random", "^--", MF.SERIES[3], "随机对照（3 种子均值）"),
    ):
        d = df[df.family == fam]
        if d.empty:
            continue
        g = d.groupby("slice_frac", as_index=False)["AP"].mean().sort_values("slice_frac")
        ax.plot(g.slice_frac * 100, g.AP, style, color=color, label=lab)
    sahi = df[df.method == "sahi_uniform"]
    if not sahi.empty:
        ax.axhline(float(sahi.AP.iloc[0]), color=MF.INK2, ls=":", lw=1.2,
                   label=f"SAHI 全切（AP={float(sahi.AP.iloc[0]):.4f}）")
    ax.set_xlabel("切片比例（%）")
    ax.set_ylabel("COCO AP")
    ax.set_title("同预算下：主动选片 / E 停机 / 一次性（留出集 274 张）", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"写出 {out_path}")


def main(args):
    R.set_dataset(args.dataset)
    cache = pickle.loads(R.CACHE.read_bytes())
    images = cache["images"][: args.limit] if args.limit else cache["images"]
    even = [r for i, r in enumerate(images) if i % 2 == 0]
    hold = {**cache, "images": even}
    cal = DetectionCalibrator.load(R.RES / "calibrator.pkl")
    cfg = GlanceConfig(img_weight=args.img_weight)
    gt_json = json.loads(R.GT.read_text())
    targets = AT.gt_targets(gt_json)
    centers = R.gt_centers(gt_json)
    print(f"留出集（偶数序号）{len(even)} 张")

    results = []

    def add(name, fn, **tags):
        m, _ = R.run_method(hold, R.GT, centers, name, fn, cfg)
        m.update(tags)
        results.append(m)
        print(f"{name:34s} AP={m['AP']:.4f} APs={m['APs']:.4f} "
              f"slices={m['slice_frac']:.1%} cov_small={m['small_cov']:.3f}")
        return m

    add("full_image", lambda r: np.zeros(0, int), family="full")
    add("sahi_uniform", lambda r: np.arange(len(r["slices"])), family="sahi")

    for b in args.budgets:
        add(f"oneshot_budget@{b}", oneshot_fn(cfg, cal, b), family="oneshot", budget=b)
    for b in args.budgets:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            add(f"random_budget@{b}#s{seed}",
                (lambda rng, b=b: lambda r: random_slices(
                    len(r["slices"]), int(round(b * len(r["slices"]))), rng))(rng),
                family="random", budget=b, seed=seed)

    print("\n两轮主动选片：")
    ref = active_fn(cfg, cal, args.report_budget1, args.report_theta2, args.report_extra,
                    args.sigma, args.gamma)
    for b1 in args.budget1s:
        for ef in args.extra_fracs:
            for th2 in args.theta2s:
                add(f"active@b1{b1}_ef{ef}_t2{th2}",
                    active_fn(cfg, cal, b1, th2, ef, args.sigma, args.gamma),
                    family="active", budget1=b1, extra_frac=ef, theta2=th2)
    for seed in range(3):
        add(f"random_matched#{seed}", matched_random_fn(ref, seed), family="random_matched")
    add("oneshot_matched", matched_oneshot_fn(cfg, cal, ref), family="oneshot_matched")

    print("\nE 停机（每图预算由判据自适应）：")
    for eps in args.epsilons:
        add(f"estop@eps{eps}", estop_fn(cfg, cal, eps), family="estop", epsilon=eps)

    df = pd.DataFrame(results)
    df.to_csv(R.RES / "active_holdout.csv", index=False)

    # ---- D 的校验：E 能不能预测真实损失 ----
    print("\nE（预期漏看目标数）与实际选片代价（小目标）的校验：")
    e_rows = []
    for label, fn in (("oneshot_report", oneshot_fn(cfg, cal, args.report_budget1 + args.report_extra)),
                      ("active_report", ref)):
        d, rho = eval_E(hold, cfg, cal, targets, fn)
        e_rows.append(d.assign(method=label))
        print(f"{label:16s} ΣE={d.E.sum():8.1f}  实际选片代价(小目标)={int(d.sel_cost_small.sum()):5d}  "
              f"逐图 Spearman(E, 代价)={rho:.3f}")
    pd.concat(e_rows).to_csv(R.RES / "active_E.csv", index=False)

    make_figure(df, R.RES / "figures" / "fig11_active.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone", choices=list(R.datasets.DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="只在前 N 张上跑（冒烟用，别出报告数字）")
    ap.add_argument("--budgets", type=float, nargs="*", default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--budget1s", type=float, nargs="*", default=[0.6, 0.8])
    ap.add_argument("--extra-fracs", type=float, nargs="*", default=[0.05, 0.15])
    ap.add_argument("--theta2s", type=float, nargs="*", default=[0.5, 0.7])
    ap.add_argument("--epsilons", type=float, nargs="*", default=[1.0, 5.0, 20.0, 100.0])
    ap.add_argument("--report-budget1", type=float, default=0.6)
    ap.add_argument("--report-extra", type=float, default=0.15)
    ap.add_argument("--report-theta2", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=600.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--img-weight", type=float, default=0.3)
    main(ap.parse_args())
