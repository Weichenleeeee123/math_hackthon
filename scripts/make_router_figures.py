"""可学习稀疏路由器的报告图（fig12、fig14），由 train_router.py 调用，也可单独重画：

  python scripts/make_router_figures.py [tag]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as MF  # noqa: E402  （统一配色与中文字体）

plt = MF.plt


def _curve(df, fam, y):
    d = df[df.family == fam]
    if d.empty:
        return d
    return d.groupby("budget", as_index=False)[["slice_frac", y]].mean().sort_values("slice_frac")


def fig_router(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    series = [("router_global", "可学习路由·全局阈值（稀疏门，本节）", "o", MF.SERIES[0], "-"),
              ("router_topk", "可学习路由·每图 top-k", "s", MF.SERIES[6], "-"),
              ("fusion_budget", "手工稀疏门（det+edge 融合）·每图 top-k", "^", MF.SERIES[1], "-"),
              ("router_mlp_topk", "MLP 门控（容量消融）·top-k", "v", MF.SERIES[3], ":"),
              ("router_linear_topk", "线性门控（消融）·top-k", "v", MF.SERIES[3], ":"),
              ("router_label_gt_topk", "标签=“有无目标”（消融）·top-k", "d", MF.SERIES[4], ":"),
              ("random_budget", "随机激活（3 种子均值）", "x", MF.MUTED, "--")]
    sahi = df[df.method == "sahi_uniform"].iloc[0]
    orc = df[df.method == "oracle_gt_small"]
    ref = _curve(df, "router_topk", "AP")
    # 手工门主工作点 = train_router.py --ths 的第一个（VisDrone 0.9，DOTA 0.5）
    thr_rows = df[df.family == "fusion_thr"]
    th = f"{thr_rows.threshold.iloc[0]:g}" if len(thr_rows) else "0.9"
    for ax, y, yl in ((axes[0], "AP", "COCO AP（%）"), (axes[1], "APs", "AP_small（%）")):
        for fam, lab, mk, col, ls in series:
            c = _curve(df, fam, y)
            if c.empty:
                continue
            if fam != "router_topk" and fam.endswith("_topk") and len(ref) == len(c) and \
                    np.allclose(_curve(df, fam, "AP")["AP"].values, ref["AP"].values):
                continue  # 消融变体与主模型完全相同（如 CV 本就选中线性）时不重复画
            ax.plot(c.slice_frac * 100, c[y] * 100, ls=ls, marker=mk, color=col, label=lab,
                    markeredgecolor=MF.SURFACE, markeredgewidth=1.2)
        ax.axhline(sahi[y] * 100, color=MF.INK, lw=1.2, ls=":")
        ax.annotate(f"Dense：SAHI 全激活 {sahi[y] * 100:.2f}", (0.01, sahi[y] * 100),
                    xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, 4),
                    ha="left", color=MF.INK2, fontsize=9)
        if len(orc):
            o = orc.iloc[0]
            ax.plot(o.slice_frac * 100, o[y] * 100, "*", color=MF.SERIES[5], markersize=14, linestyle="none",
                    label="Oracle（按真值激活，上界）")
        for m, mk in ((f"fusion_thr@{th}", "D"), (f"router_matched@{th}", "P")):
            r = df[df.method == m]
            if len(r):
                r = r.iloc[0]
                ax.plot(r.slice_frac * 100, r[y] * 100, mk, color=MF.SERIES[1] if "fusion" in m else MF.SERIES[0],
                        markersize=9, linestyle="none", markeredgecolor=MF.INK, markeredgewidth=0.8,
                        label=f"手工门 θ={th} 工作点" if "fusion" in m else f"路由·同每图 k（θ={th} 配对）")
        ax.set_xlabel("激活的切片专家比例（%）")
        ax.set_ylabel(yl)
    axes[0].set_title("稀疏激活 vs 稠密：同预算下的精度（留出集）", loc="left")
    axes[1].set_title("小目标精度（留出集）", loc="left")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print(f"写出 {out}")


def fig_sparsity(per_img: pd.DataFrame, gates: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    thr = per_img.method[per_img.method.str.startswith("fusion_thr@")]
    fm = thr.iloc[0] if len(thr) else "fusion_thr@0.9"
    for (m, lab), col in zip(((fm, f"手工门 θ={fm.split('@')[1]}"), ("router_global@op", "可学习路由（全局阈值）")),
                             (MF.SERIES[1], MF.SERIES[0])):
        d = per_img[per_img.method == m]
        if d.empty:
            continue
        rate = d.n_run / d.n_slices
        bins = pd.cut(d.n_gt, [-1, 10, 30, 60, 100, 200, 10 ** 6],
                      labels=["≤10", "11–30", "31–60", "61–100", "101–200", ">200"])
        g = rate.groupby(bins, observed=False).agg(["mean", "std", "count"])
        q = rate.groupby(bins, observed=False).quantile([0.25, 0.75]).unstack()
        x = np.arange(len(g))
        off = -0.18 if "fusion" in m else 0.18
        # 误差棒 = 四分位距（激活率落在 [0,1]，均值 ± 标准差会画出负数）
        yerr = np.vstack([(g["mean"] - q[0.25]).clip(lower=0), (q[0.75] - g["mean"]).clip(lower=0)]) * 100
        ax.bar(x + off, g["mean"] * 100, width=0.36, color=col, label=lab, yerr=yerr,
               error_kw=dict(ecolor=MF.MUTED, lw=1))
        for xi, n in zip(x, g["count"]):
            if off > 0:
                ax.annotate(f"n={int(n)}", (xi, 0), textcoords="offset points", xytext=(0, -24),
                            ha="center", fontsize=7.5, color=MF.INK2, annotation_clip=False)
        ax.set_xticks(x, g.index.astype(str))
    ax.set_xlabel("图中目标数", labelpad=14)
    ax.set_ylabel("每图激活的切片比例（%）")
    ax.set_title("稀疏激活跟着内容走：目标越多，激活越多", loc="left")
    ax.legend(fontsize=8.5, loc="upper left", title="柱 = 均值，竖线 = 四分位距", title_fontsize=8)

    ax = axes[1]
    same = "router_alpha0" in gates and "router" in gates and \
        np.allclose(np.asarray(gates["router_alpha0"]), np.asarray(gates["router"]))
    for (k, lab), col in zip((("router_alpha0", "α=0（无稀疏正则）"), ("router", "α*（交叉验证选中）"),
                              ("router_alphaHi", "α=1（强 L1 稀疏）")), (MF.MUTED, MF.SERIES[0], MF.SERIES[7])):
        if k not in gates or (same and k == "router_alpha0"):
            continue
        if same and k == "router":
            lab = "α*=0（交叉验证选中，即无稀疏正则）"
        ax.hist(np.asarray(gates[k]), bins=40, range=(0, 1), histtype="step", lw=2, color=col,
                label=f"{lab}，均值 {np.mean(gates[k]):.2f}")
    ax.set_xlabel("门控输出 g_φ(x_k)")
    ax.set_ylabel("切片数")
    ax.set_title("L1 稀疏正则如何改变门控分布（留出集）" if len(ax.patches) > 1 else "门控输出分布（留出集）",
                 loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"写出 {out}")


def main(tag: str = "", res: Path = MF.RES):
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / f"router_holdout{tag}.csv")
    fig_router(df, fig_dir / f"fig12_router{tag}.png")
    pi = res / f"router_per_image{tag}.csv"
    gp = res / f"router_gates{tag}.json"
    if pi.exists() and gp.exists():
        fig_sparsity(pd.read_csv(pi), json.loads(gp.read_text()), fig_dir / f"fig14_router_sparsity{tag}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
