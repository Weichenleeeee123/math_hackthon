"""分辨率路由评测（REPORT 3.18）：每张图先看一眼 640，再决定是否/放大到多少再看一次。

读 strong_baselines.py run 的逐图结果（full@640…2560 的检测与实测耗时），离线拼出任意"每图选一个分辨率"
策略的数据集 AP 与平均耗时，不需要 GPU。协议与 3.15 相同：奇数序号图训练、偶数序号图留出，
λ（每毫秒值多少 AP）只在训练图上选。

  & $py scripts/res_route.py --dataset visdrone_ft --tag _fp16b16

输出 results/<dataset>/：res_route<tag>.csv（留出集各策略）、res_route_boot<tag>.csv（配对 bootstrap）、
figures/fig15_res_route<tag>.png
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from glance_sahi import resroute as RR  # noqa: E402
from glance_sahi.bootstrap import PreparedEval  # noqa: E402

import run_eval as R  # noqa: E402


def main(a):
    from pycocotools.coco import COCO

    R.set_dataset(a.dataset)
    data = pickle.loads((R.RES / f"strong{a.tag}.pkl").read_bytes())
    ids = [im["id"] for im in data["images"]]
    hw = {im["id"]: im["hw"] for im in data["images"]}
    opts = [f"full@{s}" for s in a.sizes if f"full@{s}" in data["methods"]]
    base = opts[0]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(R.GT))
    prepared = {}
    for n in opts + [m for m in data["methods"] if m.startswith(("sahi@", "glance@"))]:
        dets = [d for i, arr in zip(ids, data["dets"][n]) for d in R.to_coco_dets(i, arr)]
        prepared[n] = PreparedEval(gt, dets, ids, R.DS["max_dets"])
    mixed = RR.MixedEval(prepared)
    ms = {n: 1000 * np.asarray(data["times"][n]) for n in prepared}
    pos = {i: j for j, i in enumerate(ids)}

    # 级联耗时：640 永远跑（它就是扫视）；选 r ≠ 640 时再加一次 full@r
    cost = np.stack([ms[base] + (0 if o == base else ms[o]) for o in opts], 1)
    X = np.stack([RR.glance_features(d[d[:, 4] >= a.output_conf], hw[i]) for i, d in zip(ids, data["dets"][base])])

    fit = [i for j, i in enumerate(ids) if j % 2 == 1]
    hold = [i for j, i in enumerate(ids) if j % 2 == 0]
    fi, hi = np.array([pos[i] for i in fit]), np.array([pos[i] for i in hold])
    print(f"{a.dataset}{a.tag}：训练 {len(fit)} 图 / 留出 {len(hold)} 图，选项 {opts}")

    # 标签：逐图边际 ΔAP（相对 base），训练集和留出集各自在自己的图集合上算
    G = {}
    for name, sub in (("fit", fit), ("hold", hold)):
        G[name] = np.stack([np.zeros(len(sub))] + [RR.marginal_gain(mixed, sub, base, o) for o in opts[1:]], 1)
    print("训练集逐图边际 ΔAP 均值（×100）：", dict(zip(opts, G["fit"].mean(0).round(4))))

    # 回归：每个选项一个岭回归（α 按训练集留一交叉验证的简单网格选）
    models = [None]
    for k in range(1, len(opts)):
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            err = []
            for f in range(5):
                tr, va = np.arange(len(fit)) % 5 != f, np.arange(len(fit)) % 5 == f
                m = RR.Ridge(alpha).fit(X[fi][tr], G["fit"][tr, k])
                err.append(np.mean((m.predict(X[fi][va]) - G["fit"][va, k]) ** 2))
            if best is None or np.mean(err) < best[0]:
                best = (np.mean(err), alpha)
        models.append(RR.Ridge(best[1]).fit(X[fi], G["fit"][:, k]))

    def pred(idx):
        return np.stack([np.zeros(len(idx))] + [m.predict(X[idx]) for m in models[1:]], 1)

    P_fit, P_hold = pred(fi), pred(hi)

    def evaluate(sub, idx, sel):
        choice = {i: opts[s] for i, s in zip(sub, sel)}
        r = mixed.ap(choice)
        return dict(AP=100 * r["AP"], APs=100 * r["APs"], ms_per_img=float(cost[idx, sel].mean()),
                    **{f"frac_{o}": float(np.mean(sel == k)) for k, o in enumerate(opts)})

    rows = []
    # 固定分辨率（注意：固定 full@r 不必先跑 640，耗时就是它自己；这里如实用单跑耗时）
    for o in prepared:
        choice = {i: o for i in hold}
        r = mixed.ap(choice)
        rows.append(dict(policy=o, family="fixed" if o.startswith("full") else o.split("@")[0], AP=100 * r["AP"],
                         APs=100 * r["APs"], ms_per_img=float(ms[o][hi].mean())))
    # λ 扫描
    lams = np.r_[0.0, np.geomspace(1e-5, 1e-1, 41)]
    fit_curve = []
    for lam in lams:
        sel_f = RR.choose(P_fit, cost[fi], lam)
        fit_curve.append((lam, evaluate(fit, fi, sel_f)))
        rows.append(dict(policy=f"route@{lam:.2e}", family="route", lam=lam,
                         **evaluate(hold, hi, RR.choose(P_hold, cost[hi], lam))))
    # 单特征规则对照：中位表观尺寸 < t 就放大到最大尺寸，否则停在 640（t 同样在训练集上扫）
    app50 = X[:, RR.FEATURE_NAMES.index("app_p50")]
    big = len(opts) - 1
    rule_ts = np.unique(np.percentile(app50[fi], np.linspace(0, 100, 21)))
    for t in rule_ts:
        rows.append(dict(policy=f"rule_app50<{t:.1f}", family="rule", t=t,
                         **evaluate(hold, hi, np.where(app50[hi] < t, big, 0))))
    # Oracle：用留出集真实边际收益选（不可部署，上界参考），λ 同样扫
    for lam in lams:
        rows.append(dict(policy=f"oracle@{lam:.2e}", family="oracle", lam=lam,
                         **evaluate(hold, hi, RR.choose(G["hold"], cost[hi], lam))))
    # 不扫视的 Oracle：假设能凭空知道每张图该用哪个分辨率，只跑那一次（耗时 = t(r)，没有 640 那一眼）。
    # 这是任何"不先看一眼"的分辨率选择器的上界；它都赢不了固定分辨率，级联就更不可能
    direct = np.stack([ms[o] for o in opts], 1)
    for lam in lams:
        sel = RR.choose(G["hold"], direct[hi], lam)
        e = evaluate(hold, hi, sel)
        e["ms_per_img"] = float(direct[hi, sel].mean())
        rows.append(dict(policy=f"oracle_direct@{lam:.2e}", family="oracle_direct", lam=lam, **e))
    df = pd.DataFrame(rows)

    # 工作点：只用训练集选 λ —— (a) 训练集耗时不超过 full@1280；(b) 训练集 AP 不低于 full@1920 − 0.1
    fixed_fit = {o: (100 * mixed.ap({i: o for i in fit})["AP"], float(ms[o][fi].mean())) for o in opts}
    ops = {}
    for key, ref, ok in (("op_time", "full@1280", lambda e, r: e["ms_per_img"] <= fixed_fit[r][1]),
                         ("op_ap", "full@1920", lambda e, r: e["AP"] >= fixed_fit[r][0] - 0.1)):
        if ref not in fixed_fit:
            continue
        cand = [(lam, e) for lam, e in fit_curve if ok(e, ref)]
        if not cand:
            continue
        lam = max(cand, key=lambda c: c[1]["AP"])[0] if key == "op_time" \
            else min(cand, key=lambda c: c[1]["ms_per_img"])[0]
        ops[key] = (lam, ref)
    print("训练集上选出的工作点：", {k: f"λ={v[0]:.2e}（对照 {v[1]}）" for k, v in ops.items()})

    # 配对 bootstrap（留出集）：工作点 vs 对照的固定分辨率
    rng = np.random.default_rng(a.seed)
    hold_arr = np.array(hold)
    brows = []
    for key, (lam, ref) in ops.items():
        sel = RR.choose(P_hold, cost[hi], lam)
        ch = {i: opts[s] for i, s in zip(hold, sel)}
        chr_ = {i: ref for i in hold}
        t_pol, t_ref = cost[hi, sel], ms[ref][hi]
        d_ap, d_aps, ratio = [], [], []
        for _ in range(a.boot):
            b = rng.integers(0, len(hold), len(hold))
            p1, p0 = mixed.ap(ch, hold_arr[b]), mixed.ap(chr_, hold_arr[b])
            d_ap.append(100 * (p1["AP"] - p0["AP"]))
            d_aps.append(100 * (p1["APs"] - p0["APs"]))
            ratio.append(t_pol[b].mean() / t_ref[b].mean())
        p1, p0 = mixed.ap(ch), mixed.ap(chr_)
        brows.append(dict(op=key, lam=lam, ref=ref, AP=100 * p1["AP"], AP_ref=100 * p0["AP"],
                          dAP=100 * (p1["AP"] - p0["AP"]), dAP_lo=np.percentile(d_ap, 2.5),
                          dAP_hi=np.percentile(d_ap, 97.5), dAPs=100 * (p1["APs"] - p0["APs"]),
                          dAPs_lo=np.percentile(d_aps, 2.5), dAPs_hi=np.percentile(d_aps, 97.5),
                          ms=float(t_pol.mean()), ms_ref=float(t_ref.mean()), time_ratio=float(t_pol.mean() / t_ref.mean()),
                          time_ratio_lo=np.percentile(ratio, 2.5), time_ratio_hi=np.percentile(ratio, 97.5),
                          **{f"frac_{o}": float(np.mean(sel == k)) for k, o in enumerate(opts)}))
    boot = pd.DataFrame(brows)

    df.to_csv(R.RES / f"res_route{a.tag}.csv", index=False)
    boot.to_csv(R.RES / f"res_route_boot{a.tag}.csv", index=False)
    print(df[df.family.isin(["fixed", "sahi", "glance"])][["policy", "AP", "APs", "ms_per_img"]].round(3)
          .to_string(index=False))
    if len(boot):
        print(boot.round(3).to_string(index=False))
    fig(df, boot, R.RES / "figures" / f"fig15_res_route{a.tag}.png", a.dataset, len(hold))


def fig(df, boot, out, dataset, n):
    import make_figures as MF

    plt = MF.plt
    f, ax = plt.subplots(figsize=(7.5, 4.8))
    fx = df[df.family == "fixed"].sort_values("ms_per_img")
    ax.plot(fx.ms_per_img, fx.AP, "o-", color=MF.MUTED, label="固定整图分辨率")
    for _, r in fx.iterrows():
        ax.annotate(r.policy.split("@")[1], (r.ms_per_img, r.AP), textcoords="offset points", xytext=(4, -10),
                    fontsize=8, color=MF.INK2)
    for fam, lab, col, mk in (("route", "分辨率路由（学出来的，λ 扫描）", MF.SERIES[0], "."),
                              ("rule", "单特征规则：中位表观尺寸 < t 就放大", MF.SERIES[1], "."),
                              ("oracle", "Oracle（按真实收益选，上界）", MF.SERIES[5], "."),
                              ("oracle_direct", "Oracle，且不付 640 那一眼的钱", MF.SERIES[4], ".")):
        d = df[df.family == fam]
        d = d.drop_duplicates(["AP", "ms_per_img"]).sort_values("ms_per_img")
        ax.plot(d.ms_per_img, d.AP, mk + "-", color=col, label=lab, lw=1.4, ms=5,
                alpha=0.9 if fam != "oracle" else 0.6)
    for fam, col, mk in (("sahi", MF.SERIES[3], "s"), ("glance", MF.SERIES[2], "D")):
        d = df[df.family == fam]
        ax.plot(d.ms_per_img, d.AP, mk, color=col, ls="none", label="SAHI" if fam == "sahi" else "Glance-SAHI")
    for _, r in boot.iterrows():
        ax.plot(r.ms, r.AP, "*", color=MF.SERIES[0], ms=15, markeredgecolor=MF.INK, markeredgewidth=0.8)
        ax.annotate("工作点（λ 在训练图上选）", (r.ms, r.AP), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("每图耗时（ms，含 640 扫视）")
    ax.set_ylabel("COCO AP（%）")
    ax.set_title(f"{dataset}：先看 640，再决定放大多少（留出 {n} 张）", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    f.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(out)
    plt.close(f)
    print(f"写出 {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone_ft")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sizes", type=int, nargs="*", default=[640, 960, 1280, 1600, 1920])
    ap.add_argument("--output-conf", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
