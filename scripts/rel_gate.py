"""相对化门控（饱和鲁棒）离线评测：绝对 θ vs 图像内相对 θ。

为什么：检测器越强，扫视弱证据越普遍，noisy-OR 手工门整图饱和
（VisDrone-ft 上平均分 0.94、ECE 0.61，θ=0.9 只省 9% 切片，见 docs/RESULTS-router-ft）。
saliency.relativize 把分数减去本图背景再归一，θ 变成"本图内的相对突出程度"。
本脚本回答：相对化后，同样的 AP 下能多省多少切片？排序/标定是否恢复？

协议与 3.15 / docs/RESULTS-router-ft 一致：奇数图调参（为每个方法选 θ*，使调参集
切片比例匹配绝对门 θ=0.9 / 0.99 的锚点），偶数图留出报告 AP/APs/切片比例；
另输出各方法在留出集的 θ 扫描 Pareto 与切片级 AUC/PR-AUC/ECE（gain 标签）。

输出（results/<dataset>/，--tag 为后缀）：
  rel_gate<tag>.csv           留出集端到端：方法 × θ（Pareto 扫描 + 锚点工作点）
  rel_gate_metrics<tag>.csv   切片级 AUC / PR-AUC / ECE
用法：
  & $py scripts/rel_gate.py --dataset visdrone_ft
  & $py scripts/rel_gate.py --dataset visdrone --limit 100 --quick   # 冒烟
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from glance_sahi import router as RT  # noqa: E402
from glance_sahi.calibration import expected_calibration_error  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.saliency import relativize  # noqa: E402
from glance_sahi.selector import select_slices  # noqa: E402

import run_eval as R  # noqa: E402

THS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
ANCHOR_THS = (0.9, 0.99)
VARIANTS = {"fusion_abs": "none", "fusion_relq25": "q25", "fusion_relmed": "median", "fusion_rank": "rank"}


def scores_all(images, cfg, kind):
    out = []
    for rec in images:
        s = R.slice_scores(rec, cfg, "noisyor", "edge")[0]
        out.append(relativize(s, kind))
    return out


def slice_frac_at(scores_list, th):
    tot = sum(len(s) for s in scores_list)
    run = sum(len(select_slices(s, "threshold", th, 0)) for s in scores_list)
    return run / max(tot, 1)


def main(a):
    R.set_dataset(a.dataset)
    cache = pickle.loads(R.CACHE.read_bytes())
    images = cache["images"][: a.limit] if a.limit else cache["images"]
    gt = json.loads(R.GT.read_text())
    targets = RT.gt_targets(gt)
    centers = R.gt_centers(gt)
    cfg = GlanceConfig(img_weight=a.img_weight)
    fit_idx, hold_idx = RT.split_images(images, "oddeven")
    fit = [images[i] for i in fit_idx]
    hold = [images[i] for i in hold_idx]
    print(f"[oddeven] 调参 {len(fit)} 图 / 留出 {len(hold)} 图，dataset={a.dataset}")

    S_fit = {n: scores_all(fit, cfg, k) for n, k in VARIANTS.items()}
    S_hold = {n: scores_all(hold, cfg, k) for n, k in VARIANTS.items()}
    pid = {rec["id"]: i for i, rec in enumerate(hold)}

    # --- 1) 切片级排序 / 标定（留出集，gain 标签） ---
    yh = []
    for rec in hold:
        yh.append(RT.utility_labels(rec, targets.get(rec["id"], []), R.COCO_TO_EVAL, cfg.output_conf, "gain"))
    yh = np.concatenate(yh)
    mrows = []
    for n, s in S_hold.items():
        s_all = np.concatenate(s)
        ece = expected_calibration_error(np.clip(s_all, 0, 1), yh)[0]
        mrows.append(dict(method=n, auc=RT.roc_auc(yh, s_all), pr_auc=RT.average_precision(yh, s_all),
                          ece=ece, mean_score=float(np.mean(s_all))))
        print(f"  {n:14s} AUC={mrows[-1]['auc']:.4f} PR-AUC={mrows[-1]['pr_auc']:.4f} "
              f"ECE={ece:.4f} mean={mrows[-1]['mean_score']:.3f}")

    # --- 2) 调参集锚点：绝对门 θ=0.9 / 0.99 的切片比例 ---
    anchors = {th: slice_frac_at(S_fit["fusion_abs"], th) for th in ANCHOR_THS}
    print("锚点（fusion_abs 调参集切片比例）：", {f"θ={k}": f"{v:.1%}" for k, v in anchors.items()})

    hold_cache = {**cache, "images": hold}
    rows = []

    def add(name, th, role):
        m, _ = R.run_method(hold_cache, R.GT, centers, name,
                            lambda r, n=name, t=th: select_slices(S_hold[n][pid[r["id"]]],
                                                                  "threshold", t, 0), cfg)
        m.update(method=name, th=th, role=role)
        rows.append(m)
        print(f"  [{role}] {name:14s} θ={th:<5} slices={m['slice_frac']:.1%} "
              f"AP={m['AP']:.4f} APs={m['APs']:.4f}")

    t0 = time.time()
    for name in VARIANTS:                      # 留出 Pareto：全方法 × 全 θ
        for th in THS:
            add(name, th, "pareto")
    stars = {}                                 # 锚点工作点：θ* 在调参集上匹配锚点切片比例
    for target in anchors.values():
        for name in VARIANTS:
            th_star = min(THS, key=lambda t: abs(slice_frac_at(S_fit[name], t) - target))
            stars[(name, target)] = th_star
            add(name, th_star, f"anchor@{target:.3f}")
    print(f"COCO 评测 {len(rows)} 次用时 {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    out = R.RES / f"rel_gate{a.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(mrows).to_csv(R.RES / f"rel_gate_metrics{a.tag}.csv", index=False)
    print(f"写出 {out}（{len(df)} 行）与 rel_gate_metrics{a.tag}.csv")

    # --- 3) 摘要：切片比例对齐锚点时的 AP 对比 ---
    print("\n=== 摘要（留出集：同切片比例下 AP 对比，θ* 由调参集选出） ===")
    for anchor_th, target in anchors.items():
        base = df[(df.method == "fusion_abs") & (df.th == anchor_th)]
        b = base.iloc[0]
        print(f"锚点 fusion_abs@{anchor_th}: slices={b.slice_frac:.1%} AP={b.AP:.4f} APs={b.APs:.4f}")
        for name in VARIANTS:
            if name == "fusion_abs":
                continue
            r = df[(df.method == name) & (df.th == stars[(name, target)])].iloc[0]
            print(f"  {name:14s} θ*={stars[(name, target)]:<5} slices={r.slice_frac:.1%} "
                  f"AP={r.AP:.4f}（Δ={r.AP - b.AP:+.4f}） APs={r.APs:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone_ft")
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="冒烟：缩小 θ 网格")
    args = ap.parse_args()
    if args.quick:
        THS = [0.7, 0.9]
    main(args)
