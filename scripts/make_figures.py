"""从 results/sweep.csv、results/per_image.csv 生成报告图。"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

# 参考调色板（分类色按固定顺序使用）与图表墨色
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "font.size": 10, "figure.dpi": 110,
})


def curve(df, family, x, y):
    d = df[df.family == family]
    if family == "random":
        d = d.groupby("threshold", as_index=False)[[x, y]].mean()
    return d.sort_values(x)


def point(ax, df, method, x, y, label, marker, color, dx=6, dy=-12):
    r = df[df.method == method].iloc[0]
    ax.plot(r[x], r[y], marker=marker, color=color, markersize=14 if marker == "*" else 9, linestyle="none",
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5, label=label)
    ax.annotate(label, (r[x], r[y]), textcoords="offset points", xytext=(dx, dy), color=INK2, fontsize=9)


def fig_pareto(df, x, xlabel, fname, op):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    g = curve(df, "glance_det+edge", x, "APs")
    r = curve(df, "random", x, "APs")
    ax.plot(g[x], g.APs * 100, "-o", color=SERIES[0], label="Glance-SAHI（显著性选片）",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(r[x], r.APs * 100, "-o", color=SERIES[1], label="随机选同样数量的切片（对照）",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    opr = g[g.threshold == op]
    if len(opr):
        ax.annotate(f"工作点 θ={op}", (opr[x].iloc[0], opr.APs.iloc[0] * 100), textcoords="offset points",
                    xytext=(10, -26), color=SERIES[0], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=1))
    d = df.copy()
    d["APs"] *= 100
    point(ax, d, "sahi_uniform", x, "APs", "SAHI 均匀切片", "s", INK)
    point(ax, d, "full_image", x, "APs", "整图直接推理", "D", MUTED, dy=8)
    point(ax, d, "oracle_gt_small", x, "APs", "Oracle（按真值选片，上界）", "*", SERIES[6], dy=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AP_small（%）")
    ax.set_title(f"小目标精度 vs 计算量（{DS_NAME}）", loc="left", color=INK)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def fig_ablation(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    # 证据来源 + 打分函数变体（后者来自 run_eval.py 的 ABLATIONS；老 csv 里没有就自动跳过）
    variants = [("det+edge", "检测先验 + 边缘显著性（默认）", "o"), ("det", "仅检测先验", "s"),
                ("uncertain+edge", "检测先验（4p(1−p) 加权）+ 边缘", "^"),
                ("max+edge", "检测先验（v0：置信度取最大）+ 边缘", "v"),
                ("heatmap+edge", "高斯热图先验 + 边缘", "D"),
                ("edge", "仅边缘显著性", "o"), ("spectral", "仅谱残差显著性", "o"),
                ("det+spectral", "检测先验 + 谱残差", "^")]
    variants = [v for v in variants if f"glance_{v[0]}" in set(df.family)]
    sahi = df[df.method == "sahi_uniform"].iloc[0]
    for i, ax in enumerate(axes):
        for (key, label, mk), color in zip(variants, SERIES):
            c = curve(df, f"glance_{key}", "slice_frac", "APs")
            ax.plot(c.slice_frac * 100, c.APs * 100, "-", marker=mk, color=color, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        r = curve(df, "random", "slice_frac", "APs")
        ax.plot(r.slice_frac * 100, r.APs * 100, "--", color=MUTED, label="随机选片（对照）")
        ax.axhline(sahi.APs * 100, color=INK, lw=1, ls=":")
        ax.set_xlabel("实际推理的切片占 SAHI 全部切片的比例（%）")
        ax.set_ylabel("AP_small（%）")
        if i == 1:  # 右图：放大 60%~100% 区间，看清几条“检测先验”曲线
            ax.set_xlim(58, 101)
            base = curve(df, "glance_det", "slice_frac", "APs")
            lo = min(r.APs.min(), base.APs.min() if len(base) else r.APs.min()) * 100
            ax.set_ylim(lo - 0.3, sahi.APs * 100 + 0.2)
            ax.set_title("放大：切片比例 60%–100%", loc="left", color=INK)
        else:
            ax.annotate("SAHI 均匀切片", (2, sahi.APs * 100), textcoords="offset points", xytext=(0, 4),
                        color=INK2, fontsize=9)
            ax.set_title("消融：显著性证据来源", loc="left", color=INK)
            ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ablation.png")
    plt.close(fig)


def fig_det_variants(df):
    """只画“检测先验权重取法”的消融：noisy-OR vs 4p(1−p) vs 取最大（v0）。"""
    fams = [("det", "noisy-OR：1 − Π(1 − c_j)（本文）", "o"),
            ("uncertain", "不确定度加权：4p(1−p)（消融）", "^"),
            ("max", "取最大置信度（v0，消融对照）", "v"),
            ("heatmap", "高斯热图先验（消融）", "D"),
            ("evidence", "证据质量阈值 Σc ≥ τ（τ 扫描）", "P")]
    fams = [f for f in fams if f"glance_{f[0]}" in set(df.family)]
    if not fams:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (key, label, mk), color in zip(fams, SERIES):
        c = curve(df, f"glance_{key}", "slice_frac", "APs")
        ax.plot(c.slice_frac * 100, c.APs * 100, "-", marker=mk, color=color, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    r = curve(df, "random", "slice_frac", "APs")
    ax.plot(r.slice_frac * 100, r.APs * 100, "--", color=MUTED, label="随机选片（对照）")
    sahi = df[df.method == "sahi_uniform"].iloc[0]
    ax.axhline(sahi.APs * 100, color=INK, lw=1, ls=":")
    ax.set_xlabel("实际推理的切片占 SAHI 全部切片的比例（%）")
    ax.set_ylabel("AP_small（%）")
    ax.set_title("打分函数消融：检测先验怎么加权", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_det_weights.png")
    plt.close(fig)


def fig_per_image(pi, op):
    g = pi[pi.method == f"glance[det+edge]@{op}"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(g.n_gt, 100 * g.n_run / g.n_slices, s=22, color=SERIES[0], alpha=0.6, edgecolors=SURFACE,
               linewidths=0.6)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("图中目标数（person + vehicle，对数刻度）")
    ax.set_ylabel("实际推理的切片比例（%）")
    ax.set_ylim(-3, 103)
    ax.set_title("每张图的切片用量随“图里有多少东西”自适应", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_per_image.png")
    plt.close(fig)


BUCKET_ORDER = ["[0,20)", "[20,50)", "[50,100)", "[100,inf)"]


def fig_buckets():
    """按“图中目标数”分桶：既报切片比例、也报**该桶的精度**（读 run_eval.py buckets 的输出）。"""
    path = RES / "buckets.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    uni = df[df.method == "sahi_uniform"].set_index("bucket")
    gl = df[df.method.str.startswith("glance[")].set_index("bucket")
    rd = df[df.method.str.startswith("random_matched")].groupby("bucket").mean(numeric_only=True)
    if gl.empty:
        return
    order = [b for b in BUCKET_ORDER if b in gl.index]
    x, w = np.arange(len(order)), 0.27
    series = [(uni, "SAHI 均匀切片", MUTED), (gl, "Glance-SAHI（本项目）", SERIES[0]),
              (rd, "随机选同样数量的切片", SERIES[1])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, col, ylab, title in [
        (axes[0], "slice_frac", "实际推理的切片比例（%）", "切片用量的自适应"),
        (axes[1], "APs", "AP_small（%）", "分桶精度：稀疏图上省得对不对"),
    ]:
        for (d, lab, color), off in zip(series, (-w, 0, w)):
            v = [100 * float(d.loc[b, col]) for b in order]
            ax.bar(x + off, v, w, color=color, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n{int(gl.loc[b, 'images'])} 张" for b in order], fontsize=9)
        ax.set_xlabel("图中目标数（person + vehicle）")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", color=INK)
    axes[1].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_buckets.png")
    plt.close(fig)


DS_NAME = "VisDrone2019-DET-val"

if __name__ == "__main__":
    op = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    if len(sys.argv) > 2 and sys.argv[2] != "visdrone":  # make_figures.py 0.9 dota
        RES = RES / sys.argv[2]
        FIG = RES / "figures"
        DS_NAME = {"dota": "DOTA-v1.0 val"}.get(sys.argv[2], sys.argv[2])
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "sweep.csv")
    abl = RES / "sweep_abl.csv"  # run_eval.py sim 的打分函数消融（单独文件，不改动 sweep.csv 旧行）
    if abl.exists():
        a = pd.read_csv(abl)
        a = a[~a.family.isin(set(df.family))]
        df = pd.concat([df, a], ignore_index=True)
    fig_pareto(df, "slices_per_img", "每张图的切片推理次数", "fig1_pareto_slices.png", op)
    fig_pareto(df, "ms_per_img", "每张图耗时（ms，RTX 3050 Laptop）", "fig2_pareto_time.png", op)
    fig_ablation(df)
    fig_det_variants(df)
    fig_buckets()
    fig_per_image(pd.read_csv(RES / "per_image.csv"), op)
    print(f"figures -> {FIG}")
