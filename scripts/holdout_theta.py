"""工作点 θ 的留出验证：在奇数号图上选 θ，在偶数号图上报告，并给 AP 差按图配对的 bootstrap 区间。

原主结果的 θ=0.9、λ=0.3 是在同一份 val 上看结果选的，会被问"是不是在测试集上调的"。
这里把选择规则写死（预注册 docs/PREREG-2026-09-26.md）：
  在调参集（奇数序号，与 calibrate.py 相同的拆分）上，取"保住 SAHI 小目标增益 ≥ 98%"的 θ 中
  切片比例最小的一个；保住比例 = (APs_θ − APs_整图) / (APs_SAHI − APs_整图)。
然后在留出集（偶数序号）上报告 SAHI / 整图 / Glance@θ* / 同数量随机（3 种子），
ΔAP 与耗时比相对 SAHI，按图配对 bootstrap。全程离线读 cache.pkl，不需要 GPU。

  python scripts/holdout_theta.py --dataset visdrone
  python scripts/holdout_theta.py --dataset dota15            # OBB 检测器的缓存
"""

import argparse
import contextlib
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval as R  # noqa: E402
from glance_sahi.bootstrap import PreparedEval, paired_bootstrap  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

THS = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
RETAIN = 0.98  # 预注册：保住 SAHI 小目标增益的比例下限


def evaluate(records, select, cfg):
    """对一组缓存记录跑一个选片策略，返回 (coco 检测列表, 逐图耗时, 逐图切片数)。耗时口径同 run_eval.run_method。"""
    dets, times, n_run = [], [], []
    for rec in records:
        sel, t_extra = select(rec)
        merged, t_nms = R.merge(rec, sel, cfg)
        dets += R.to_coco_dets(rec["id"], merged)
        times.append(rec["t_glance"] + t_extra + rec["t_slice"][sel].sum() + t_nms)
        n_run.append(len(sel) / max(len(rec["slices"]), 1))
    return dets, np.array(times), np.array(n_run)


def main(args):
    from pycocotools.coco import COCO

    R.set_dataset(args.dataset, args.res_tag)
    cache = pickle.loads(R.CACHE.read_bytes())
    cfg = GlanceConfig(img_weight=args.img_weight)
    det_kind, img_kind = R.prior_kinds(args.prior)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(R.GT))
    images = cache["images"]
    split = {"tune": [r for i, r in enumerate(images) if i % 2 == 1],
             "holdout": [r for i, r in enumerate(images) if i % 2 == 0]}

    def glance(th):
        def fn(rec):
            s, t = R.slice_scores(rec, cfg, det_kind, img_kind)
            return select_slices(s, "threshold", th, 0), t
        return fn

    full = lambda rec: (np.zeros(0, int), 0.0)  # noqa: E731
    sahi = lambda rec: (np.arange(len(rec["slices"])), 0.0)  # noqa: E731

    def ap_of(records, select):
        dets, t, frac = evaluate(records, select, cfg)
        ids = [r["id"] for r in records]
        return PreparedEval(gt, dets, ids, R.DS["max_dets"]).ap(), t, frac

    # ---- 调参集：扫 θ，按预注册规则选 θ*
    tune = split["tune"]
    a_full, _, _ = ap_of(tune, full)
    a_sahi, _, _ = ap_of(tune, sahi)
    gain = a_sahi["APs"] - a_full["APs"]
    curve = []
    for th in THS:
        a, t, frac = ap_of(tune, glance(th))
        curve.append({"split": "tune", "theta": th, "AP": 100 * a["AP"], "APs": 100 * a["APs"],
                      "slice_frac": frac.mean(), "retain": (a["APs"] - a_full["APs"]) / gain if gain > 0 else np.nan})
        print(f"tune θ={th:<5} AP={100 * a['AP']:.2f} APs={100 * a['APs']:.2f} "
              f"slices={frac.mean():.1%} retain={curve[-1]['retain']:.3f}")
    ok = [c for c in curve if c["retain"] >= RETAIN]
    if not ok:
        raise SystemExit(f"调参集上没有 θ 能保住 ≥{RETAIN:.0%} 的小目标增益；按预注册如实报告这一点")
    best = min(ok, key=lambda c: (c["slice_frac"], -c["theta"]))
    th_star = best["theta"]
    print(f"\n调参集（{len(tune)} 张）选出 θ* = {th_star}（保住 {best['retain']:.1%}，切片 {best['slice_frac']:.1%}）")

    # ---- 留出集：报告 θ*，与 SAHI 配对比较
    hold = split["holdout"]
    ids = [r["id"] for r in hold]
    methods = {"full_image": full, "sahi_uniform": sahi, f"glance@{th_star}": glance(th_star)}
    for seed in range(3):
        rng = np.random.default_rng(seed)

        def rnd(rec, rng=rng):
            k = len(glance(th_star)(rec)[0])
            return random_slices(len(rec["slices"]), k, rng), 0.0
        methods[f"random@{th_star}#s{seed}"] = rnd
    prepared, times, fracs = {}, {}, {}
    for name, fn in methods.items():
        dets, t, frac = evaluate(hold, fn, cfg)
        prepared[name] = PreparedEval(gt, dets, ids, R.DS["max_dets"])
        times[name], fracs[name] = t, frac.mean()
    df = pd.DataFrame(paired_bootstrap(prepared, times, "sahi_uniform", args.boot, args.seed))
    df["slice_frac"] = df.method.map(fracs)
    p_full, p_sahi = prepared["full_image"].ap(), prepared["sahi_uniform"].ap()
    g = p_sahi["APs"] - p_full["APs"]
    df["retain"] = df.method.map({n: (prepared[n].ap()["APs"] - p_full["APs"]) / g if g > 0 else np.nan
                                  for n in prepared})
    df.insert(0, "split", "holdout")
    df["theta_star"] = th_star
    out = R.RES / "holdout_theta.csv"
    pd.concat([pd.DataFrame(curve), df], ignore_index=True).to_csv(out, index=False)

    cols = ["method", "slice_frac", "ms_per_img", "AP", "APs", "retain", "dAP", "dAP_lo", "dAP_hi",
            "dAPs", "dAPs_lo", "dAPs_hi", "time_ratio", "time_ratio_lo", "time_ratio_hi"]
    pd.set_option("display.width", 200)
    print(f"\n留出集（{len(hold)} 张），相对 SAHI，{args.boot} 次按图配对 bootstrap；耗时为缓存离线口径：")
    print(df[cols].round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--res-tag", default="")
    ap.add_argument("--prior", default="det+edge", help="打分：det+edge（默认主方法）/ edge / det ...")
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
