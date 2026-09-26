"""可学习稀疏路由器：训练 + 留出评测（REPORT 3.15）。全程 CPU、离线读 results/cache.pkl。

为什么：手工稀疏门 S = 1−(1−S_det)(1−λS_img) 的形式和 λ、θ 都是人定的；它估计的是“片里有没有目标”，
      而真正该估计的是“跑这片检测器**有没有增量**”（整图一眼已经看清的目标，再切一次是白花算力）。
改了什么：把选片写成稀疏路由 z_k = 1[g_φ(x_k) ≥ μ]，g_φ 是 22 维扫视特征上的小 MLP（~1k 参数），
      标签 = 该片是否检出扫视没检出的真值（gain），损失 = BCE + α·平均激活（L1 稀疏）。
结果如何：留出集上与 dense（SAHI 全切）、手工稀疏门、随机、oracle 在同一切片预算下比 AP / APs。

协议：奇数序号图训练（其内再做 5 折按图分组交叉验证选 α、隐层宽度），偶数序号图留出评测
（与 calibrate.py 同）；--split sequence 按视频序列划分，检查相邻帧泄漏。

输出（results/，--tag 为后缀）：
  router.json               集成权重 + 元信息（默认工作点 = 与手工门主工作点 θ = --ths[0] 同切片比例的全局阈值）
  router_cv.csv             交叉验证：每个 (hidden, α) 的切片级 PR-AUC
  router_holdout.csv        留出集：各方法 × 各预算的 AP/APs/切片比例/覆盖率
  router_metrics.csv        切片级 AUC / PR-AUC / ECE（融合分 vs 学习路由）
  router_features.csv       置换特征重要性（留出集 PR-AUC 下降）
  router_sparsity.csv       稀疏激活统计（每图激活率分布、与目标数的相关）
  figures/fig12_router.png  AP / APs – 切片比例曲线
  figures/fig14_router_sparsity.png  激活率 vs 目标数、门控分数分布（α=0 vs α*）

用法：
  & $py scripts/train_router.py                      # 完整（约 10–20 分钟，主要是 COCOeval）
  & $py scripts/train_router.py --limit 80 --seeds 1 --quick   # 冒烟
  & $py scripts/train_router.py --split sequence --tag _seq --no-ablations
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
from glance_sahi.selector import select_slices  # noqa: E402

import run_eval as R  # noqa: E402


# ------------------------------------------------------------------------------ 数据
def build_xy(images, cfg, targets, label):
    X, Y, G = [], [], []
    for i, rec in enumerate(images):
        X.append(RT.features_from_rec(rec, cfg))
        Y.append(RT.utility_labels(rec, targets.get(rec["id"], []), R.COCO_TO_EVAL, cfg.output_conf, label))
        G.append(np.full(len(rec["slices"]), i))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def cv_select(X, y, g, grid_hidden, grid_alpha, epochs, folds=5, seed=0):
    """按图分组的 K 折交叉验证，指标 = 切片级 PR-AUC（排序质量，与预算无关）。"""
    ug = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of = {v: i % folds for i, v in enumerate(perm)}
    f = np.array([fold_of[v] for v in g])
    rows = []
    for h in grid_hidden:
        for a in grid_alpha:
            sc = []
            for k in range(folds):
                tr, va = f != k, f == k
                p = RT.train_router(X[tr], y[tr], hidden=h, alpha=a, epochs=epochs, seed=seed)
                sc.append(RT.average_precision(y[va], RT.NumpyRouter([p]).predict_proba(X[va])))
            rows.append(dict(hidden=h, alpha=a, pr_auc=float(np.mean(sc)), pr_auc_sd=float(np.std(sc))))
            print(f"  cv hidden={h:>3} alpha={a:<6} PR-AUC={np.mean(sc):.4f} ± {np.std(sc):.4f}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ 主流程
def main(a):
    R.set_dataset(a.dataset)
    cache = pickle.loads(R.CACHE.read_bytes())
    images = cache["images"][: a.limit] if a.limit else cache["images"]
    gt = json.loads(R.GT.read_text())
    targets = RT.gt_targets(gt)
    centers = R.gt_centers(gt)
    cfg = GlanceConfig(img_weight=a.img_weight)
    fit_idx, hold_idx = RT.split_images(images, a.split)
    fit = [images[i] for i in fit_idx]
    hold = [images[i] for i in hold_idx]
    print(f"[{a.split}] 训练 {len(fit)} 图 / 留出 {len(hold)} 图，标签={a.label}")

    t0 = time.time()
    Xf, yf, gf = build_xy(fit, cfg, targets, a.label)
    Xh, yh, gh = build_xy(hold, cfg, targets, a.label)
    print(f"特征 {Xf.shape[1]} 维；训练切片 {len(yf)}（正例率 {yf.mean():.3f}），留出切片 {len(yh)}；"
          f"{time.time() - t0:.1f}s")

    # --- 1) 交叉验证选超参（只用训练图） ---
    cv = cv_select(Xf, yf, gf, a.hidden, a.alphas, a.epochs, seed=0)
    best = cv.sort_values("pr_auc", ascending=False).iloc[0]
    h_star, a_star = int(best.hidden), float(best.alpha)
    print(f"选中 hidden={h_star} alpha={a_star}")

    # --- 2) 最终模型：多种子集成 ---
    def fit_ens(X, y, hidden, alpha, seeds):
        t = time.time()
        ms = [RT.train_router(X, y, hidden=hidden, alpha=alpha, epochs=a.epochs, seed=s) for s in range(seeds)]
        return RT.NumpyRouter(ms), ms, time.time() - t

    router, members, t_train = fit_ens(Xf, yf, h_star, a_star, a.seeds)
    print(f"训练 {a.seeds} 个种子用时 {t_train:.1f}s，参数量 {router.n_params}/成员")
    variants = {"router": router}
    if not a.no_ablations:
        # 稀疏正则消融：α=0 与“强稀疏” α=1.0（CV 网格里的 α 太小，看不出对门控分布的影响）
        variants["router_alpha0"] = fit_ens(Xf, yf, h_star, 0.0, a.seeds)[0]
        variants["router_alphaHi"] = fit_ens(Xf, yf, h_star, max(1.0, max(a.alphas)), a.seeds)[0]
        # 容量消融：CV 选中线性就补一个 MLP，选中 MLP 就补一个线性
        if h_star == 0:
            variants["router_mlp"] = fit_ens(Xf, yf, max(a.hidden), a_star, a.seeds)[0]
        else:
            variants["router_linear"] = fit_ens(Xf, yf, 0, a_star, a.seeds)[0]
        if a.label != "gt":
            Xg, yg, _ = build_xy(fit, cfg, targets, "gt")
            variants["router_label_gt"] = fit_ens(Xg, yg, h_star, a_star, a.seeds)[0]

    # 每张留出图的分数（按图切开）
    def per_img(r, X, g, n):
        p = r.predict_proba(X)
        return [p[g == i] for i in range(n)]

    P = {name: per_img(r, Xh, gh, len(hold)) for name, r in variants.items()}
    P_fit = router.predict_proba(Xf)
    fused = {rec["id"]: R.slice_scores(rec, cfg, "noisyor", "edge")[0] for rec in hold}
    pid = {rec["id"]: i for i, rec in enumerate(hold)}

    # --- 3) 切片级指标 ---
    mrows = []
    fused_all = np.concatenate([fused[rec["id"]] for rec in hold])
    for name, s in [("fusion(det+edge)", fused_all), ("det_noisyor", Xh[:, 0])] + \
                   [(n, np.concatenate(p)) for n, p in P.items()]:
        ece = expected_calibration_error(np.clip(s, 0, 1), yh)[0]
        mrows.append(dict(scorer=name, auc=RT.roc_auc(yh, s), pr_auc=RT.average_precision(yh, s), ece=ece,
                          mean_score=float(np.mean(s))))
        print(f"  {name:22s} AUC={mrows[-1]['auc']:.4f} PR-AUC={mrows[-1]['pr_auc']:.4f} ECE={ece:.4f}")
    pd.DataFrame(mrows).to_csv(R.RES / f"router_metrics{a.tag}.csv", index=False)

    # --- 4) 置换特征重要性（留出集 PR-AUC 的下降，5 次平均） ---
    base = RT.average_precision(yh, router.predict_proba(Xh))
    rng = np.random.default_rng(0)
    frows = []
    for j, fn in enumerate(RT.FEATURE_NAMES):
        drops = []
        for _ in range(5):
            Xp = Xh.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - RT.average_precision(yh, router.predict_proba(Xp)))
        frows.append(dict(feature=fn, pr_auc_drop=float(np.mean(drops)), sd=float(np.std(drops))))
    fi = pd.DataFrame(frows).sort_values("pr_auc_drop", ascending=False)
    fi.to_csv(R.RES / f"router_features{a.tag}.csv", index=False)
    print("最重要的 6 个特征：", ", ".join(f"{r.feature}({r.pr_auc_drop:.3f})" for r in fi.head(6).itertuples()))

    # --- 5) 留出集端到端 AP：同预算对比 ---
    hold_cache = {**cache, "images": hold}
    results = []
    sel_log = {}

    def add(name, fn, **tags):
        m, pi = R.run_method(hold_cache, R.GT, centers, name, fn, cfg)
        m.update(tags)
        results.append(m)
        sel_log[name] = pi
        print(f"{name:34s} AP={m['AP']:.4f} APs={m['APs']:.4f} slices={m['slice_frac']:.1%} "
              f"cov_small={m['small_cov']:.3f}")

    add("full_image", lambda r: np.zeros(0, int), family="full", budget=0.0)
    add("sahi_uniform", lambda r: np.arange(len(r["slices"])), family="sahi", budget=1.0)

    def oracle(r):
        c = centers.get(r["id"], np.zeros((0, 3), np.float32))
        c = c[c[:, 2] > 0] if len(c) else c
        return np.array([k for k in range(len(r["slices"])) if R.covered(c, r["slices"], [k]).any()], int)
    add("oracle_gt_small", oracle, family="oracle")

    # 手工稀疏门的工作点 --ths（VisDrone 0.9/0.99，DOTA 0.5/0.9），和“同每图 k”的路由；第一个是主工作点
    th0 = a.ths[0]
    for th in a.ths:
        fus_th = lambda r, th=th: select_slices(fused[r["id"]], "threshold", th, 0)
        add(f"fusion_thr@{th}", fus_th, family="fusion_thr", threshold=th)
        add(f"router_matched@{th}", lambda r, f=fus_th: RT.route(P["router"][pid[r["id"]]], "matched",
                                                                   k=len(f(r))),
            family="router_matched", threshold=th)
    # 默认工作点：全局阈值，使训练集激活率 = 手工门主工作点在训练集上的切片比例
    rho_op = sum(len(select_slices(R.slice_scores(r, cfg, "noisyor", "edge")[0], "threshold", th0, 0))
                 for r in fit) / sum(len(r["slices"]) for r in fit)
    thr_op = RT.global_threshold(P_fit, rho_op)
    add("router_global@op", lambda r: RT.route(P["router"][pid[r["id"]]], "global", thr=thr_op),
        family="router_global_op", budget=rho_op)

    for b in a.budgets:
        add(f"fusion_budget@{b}", lambda r, b=b: select_slices(fused[r["id"]], "budget", 0, b),
            family="fusion_budget", budget=b)
        for name in P:
            add(f"{name}_topk@{b}", lambda r, b=b, name=name: RT.route(P[name][pid[r["id"]]], "topk", rho=b),
                family=f"{name}_topk", budget=b)
        mu = RT.global_threshold(P_fit, b)
        add(f"router_global@{b}", lambda r, mu=mu: RT.route(P["router"][pid[r["id"]]], "global", thr=mu),
            family="router_global", budget=b, mu=mu)
        for seed in range(3):
            rng_b = np.random.default_rng(seed)
            add(f"random_budget@{b}#s{seed}",
                lambda r, b=b, rng_b=rng_b: RT.route(rng_b.random(len(r["slices"])), "topk", rho=b),
                family="random_budget", budget=b, seed=seed)

    df = pd.DataFrame(results)
    df.to_csv(R.RES / f"router_holdout{a.tag}.csv", index=False)

    # --- 6) 稀疏激活统计（同为工作点附近的切片比例） ---
    srows = []
    for name in ["sahi_uniform", f"fusion_thr@{th0}", f"router_matched@{th0}", "router_global@op"] + \
                [f"fusion_budget@{b}" for b in a.budgets] + [f"router_global@{b}" for b in a.budgets]:
        if name not in sel_log:
            continue
        pi = sel_log[name]
        st = RT.activation_stats(pi.n_run / pi.n_slices, pi.n_gt)
        st.update(method=name, slice_frac=float(pi.n_run.sum() / pi.n_slices.sum()))
        srows.append(st)
    sp = pd.DataFrame(srows)
    sp.to_csv(R.RES / f"router_sparsity{a.tag}.csv", index=False)
    print(sp[["method", "slice_frac", "rate_mean", "rate_std", "rate_gini", "spearman_rate_ngt"]]
          .to_string(index=False))

    # 每图激活率明细（画图用）
    pd.concat([sel_log[n].assign(method=n) for n in (f"fusion_thr@{th0}", "router_global@op", f"router_matched@{th0}")
               if n in sel_log]).to_csv(R.RES / f"router_per_image{a.tag}.csv", index=False)

    # --- 7) 保存路由器 ---
    meta = dict(dataset=a.dataset, split=a.split, label=a.label, hidden=h_star, alpha=a_star, seeds=a.seeds,
                n_train_images=len(fit), n_train_slices=int(len(yf)), pos_rate=float(yf.mean()),
                default_threshold=float(thr_op), default_rho=float(rho_op),
                thresholds_for_fraction={str(b): RT.global_threshold(P_fit, b) for b in a.budgets},
                img_weight=a.img_weight, train_seconds=t_train, n_params=router.n_params)
    RT.save_router(members, meta, R.RES / f"router{a.tag}.json")
    gates = {n: np.concatenate(P[n]).tolist() for n in ("router", "router_alpha0", "router_alphaHi") if n in P}
    (R.RES / f"router_gates{a.tag}.json").write_text(json.dumps(gates))
    cv.to_csv(R.RES / f"router_cv{a.tag}.csv", index=False)
    print(f"写出 {R.RES / f'router{a.tag}.json'}（默认全局阈值 μ={thr_op:.4f}，训练激活率 ρ={rho_op:.3f}）")

    import make_router_figures as MRF
    MRF.main(a.tag, R.RES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--split", default="oddeven", choices=["oddeven", "sequence"])
    ap.add_argument("--label", default="gain", choices=["gain", "gt", "gt_small"])
    ap.add_argument("--hidden", type=int, nargs="*", default=[0, 16, 32])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.01, 0.1])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--ths", type=float, nargs="*", default=[0.9, 0.99])
    ap.add_argument("--budgets", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--quick", action="store_true", help="冒烟：缩小超参网格与预算")
    args = ap.parse_args()
    if args.quick:
        args.hidden, args.alphas, args.budgets, args.epochs = [16], [0.0, 0.1], [0.5], 150
    main(args)
