"""配对 bootstrap 置信区间（REPORT 3.16）：稀疏激活到底比稠密差多少、比随机好多少，是否显著。

  & $py scripts/bootstrap_ci.py                 # 留出集（偶数序号 274 图）：含可学习路由
  & $py scripts/bootstrap_ci.py --scope all     # 全部 548 图：只比零训练方法（路由在奇数图上训练过）
  & $py scripts/bootstrap_ci.py --replot        # 只按已有 CSV 重画 fig13（不重跑 bootstrap）

输出 results/bootstrap_ci<_scope>.csv（点估计 + 95% CI）、bootstrap_delta<_scope>.csv（配对差值）、
figures/fig13_ci<_scope>.png（ΔAP 森林图）。
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

from pycocotools.coco import COCO  # noqa: E402

from glance_sahi import router as RT  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.evalboot import CachedEval, paired_bootstrap  # noqa: E402
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

import run_eval as R  # noqa: E402

def labels(th: float = 0.9) -> dict:
    return {
        "sahi_uniform": "SAHI（稠密，全激活）",
        f"fusion_thr@{th}": f"手工稀疏门 θ={th}",
        f"router_matched@{th}": "可学习路由（同每图 k）",
        "router_global@op": "可学习路由（全局阈值）",
        "router_global@0.5": "可学习路由（全局阈值，50% 预算）",
        "fusion_budget@0.5": "手工稀疏门（每图 top-50%）",
        f"random_matched@{th}": "随机（同每图 k，3 种子）",
        "oracle_gt_small": "Oracle（真值，上界）",
        "full_image": "整图一次（无切片）",
    }


# 这些方法的预算不是手工门 θ 的“同每图 k”，不能和 random_matched@θ 直接比
OFF_BUDGET = {"router_global@0.5", "fusion_budget@0.5"}


def main(a):
    import contextlib
    import io

    R.set_dataset(a.dataset)
    th = a.th
    rm = f"random_matched@{th}"
    suf = f"_{a.scope}"
    if a.replot:
        images = pickle.loads(R.CACHE.read_bytes())["images"]
        n = len(RT.split_images(images, a.split)[1]) if a.scope == "holdout" else len(images)
        fig(pd.read_csv(R.RES / f"bootstrap_delta{suf}.csv"), pd.read_csv(R.RES / f"bootstrap_ci{suf}.csv"),
            R.RES / "figures" / f"fig13_ci{suf}.png", a.scope, n, th)
        return
    cache = pickle.loads(R.CACHE.read_bytes())
    images = cache["images"]
    gt = json.loads(R.GT.read_text())
    centers = R.gt_centers(gt)
    cfg = GlanceConfig(img_weight=a.img_weight)
    if a.scope == "holdout":
        _, hold_idx = RT.split_images(images, a.split)
        images = [images[i] for i in hold_idx]
    ids = [r["id"] for r in images]
    print(f"scope={a.scope}：{len(images)} 图，B={a.B}")

    fused = {r["id"]: R.slice_scores(r, cfg, "noisyor", "edge")[0] for r in images}
    fus = lambda r: select_slices(fused[r["id"]], "threshold", th, 0)

    def oracle(r):
        c = centers.get(r["id"], np.zeros((0, 3), np.float32))
        c = c[c[:, 2] > 0] if len(c) else c
        return np.array([k for k in range(len(r["slices"])) if R.covered(c, r["slices"], [k]).any()], int)

    methods = {"full_image": lambda r: np.zeros(0, int),
               "sahi_uniform": lambda r: np.arange(len(r["slices"])),
               f"fusion_thr@{th}": fus, "oracle_gt_small": oracle}
    for s in range(3):
        rng = np.random.default_rng(s)
        methods[f"{rm}#s{s}"] = lambda r, rng=rng: random_slices(len(r["slices"]), len(fus(r)), rng)
    if a.scope == "holdout":
        router = RT.load_router(str(R.RES / f"router{a.router_tag}.json"))
        P = {r["id"]: router.predict_proba(RT.features_from_rec(r, cfg)) for r in images}
        methods[f"router_matched@{th}"] = lambda r: RT.route(P[r["id"]], "matched", k=len(fus(r)))
        methods["router_global@op"] = lambda r: RT.route(P[r["id"]], "global", thr=router.default_threshold)
        # 半预算：μ 取训练集上激活率 50% 的分位点（router.json 里存好的，留出集不参与）
        mu50 = router.meta["thresholds_for_fraction"]["0.5"]
        methods["router_global@0.5"] = lambda r: RT.route(P[r["id"]], "global", thr=mu50)
        methods["fusion_budget@0.5"] = lambda r: select_slices(fused[r["id"]], "budget", 0, 0.5)
        for s in range(3):
            rng = np.random.default_rng(100 + s)
            methods[f"random_budget@0.5#s{s}"] = lambda r, rng=rng: RT.route(rng.random(len(r["slices"])),
                                                                              "topk", rho=0.5)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(R.GT))
    evals, frac = {}, {}
    for name, fn in methods.items():
        t0 = time.time()
        dets, run, tot = [], 0, 0
        for r in images:
            sel = fn(r)
            run, tot = run + len(sel), tot + len(r["slices"])
            dets += R.to_coco_dets(r["id"], R.merge(r, sel, cfg)[0])
        evals[name] = CachedEval(R.GT, dets, ids, R.DS["max_dets"], coco_gt=coco_gt)
        frac[name] = run / tot
        print(f"  {name:26s} AP={evals[name].ap():.4f} APs={evals[name].ap(None, 'small'):.4f} "
              f"slices={frac[name]:.1%} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    refs = ["sahi_uniform", rm, f"fusion_thr@{th}"]
    groups = {rm: [f"{rm}#s{s}" for s in range(3)]}
    if a.scope == "holdout":
        refs += ["fusion_budget@0.5", "random_budget@0.5"]
        groups["random_budget@0.5"] = [f"random_budget@0.5#s{s}" for s in range(3)]
    ci, delta, _ = paired_bootstrap(evals, len(ids), B=a.B, seed=a.seed, ref=refs, group_avg=groups)
    print(f"bootstrap {a.B} 次用时 {time.time() - t0:.1f}s")
    for g, members in groups.items():
        frac[g] = frac[members[0]]
    ci["slice_frac"] = ci.method.map(frac)
    ci.to_csv(R.RES / f"bootstrap_ci{suf}.csv", index=False)
    delta.to_csv(R.RES / f"bootstrap_delta{suf}.csv", index=False)
    print(ci[ci.area == "all"].to_string(index=False))
    print(delta.to_string(index=False))
    fig(delta, ci, R.RES / "figures" / f"fig13_ci{suf}.png", a.scope, len(ids), th)


def fig(delta, ci, out, scope, n, th: float = 0.9):
    import make_figures as MF
    plt = MF.plt

    LABELS = labels(th)
    fig_, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ref, title in ((axes[0], "sahi_uniform", "相对稠密 SAHI 的 ΔAP（越接近 0 越好）"),
                           (axes[1], f"random_matched@{th}", "相对随机激活的 ΔAP（>0 = 路由有效）")):
        d = delta[(delta.vs == ref) & delta.method.isin(LABELS)]
        order = [m for m in LABELS if m in set(d.method) and m != ref and
                 not (ref.startswith("random_matched") and m in OFF_BUDGET)]
        for j, area in enumerate(("all", "small")):
            dd = d[d.area == area].set_index("method").reindex(order)
            y = np.arange(len(order)) + (0.15 if j == 0 else -0.15)
            ax.errorbar(dd.delta * 100, y, xerr=[(dd.delta - dd.lo) * 100, (dd.hi - dd.delta) * 100],
                        fmt="o" if j == 0 else "s", color=MF.SERIES[0] if j == 0 else MF.SERIES[1], capsize=3,
                        label="AP" if j == 0 else "AP_small")
        ax.axvline(0, color=MF.INK, lw=1)
        ax.set_yticks(np.arange(len(order)), [LABELS[m] for m in order])
        ax.set_xlabel("Δ（百分点），95% 配对 bootstrap 区间")
        ax.set_title(title, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig_.legend(h, lab, fontsize=9, loc="upper right", ncol=2, frameon=False)
    fig_.suptitle(f"图像级配对 bootstrap（{'留出集' if scope == 'holdout' else '全集'} {n} 图）", x=0.01, ha="left")
    fig_.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(out)
    plt.close(fig_)
    print(f"写出 {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--scope", default="holdout", choices=["holdout", "all"])
    ap.add_argument("--split", default="oddeven", choices=["oddeven", "sequence"])
    ap.add_argument("--router-tag", default="")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--th", type=float, default=0.9, help="手工门主工作点 θ（DOTA 用 0.5 与 --img-weight 1.0，见 REPORT 3.7）")
    ap.add_argument("--replot", action="store_true", help="只读已有 bootstrap_*.csv 重画图")
    main(ap.parse_args())
