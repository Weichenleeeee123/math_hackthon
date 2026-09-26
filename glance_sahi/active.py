"""主动式（两轮）选片 + 可计算的停机判据 E。

为什么改
--------
现有选片是**一次性**的：先验分数算完就定了，跑出什么结果都不影响后面的决策——
SAHI 是静态均匀切，Glance-SAHI 只是把"切哪些"从尺寸换成内容，仍然只算一次。
SaccadeNet 的注视循环是"边看边决定"：每一眼的观测都更新信念，再决定下一眼看哪。
这里把那个循环压成**两轮**，只保留"决策依赖观测"这一核心，不引入它的对数极坐标视网膜。

    第 1 轮：用标定后的分数按**预算**跑最靠前的 x% 片（预算模式，不碰阈值——因为标定后的
            分数在密集场景会饱和，见 REPORT 3.12 结果四，阈值模式没有可用的选择区间）；
    第 2 轮：把第 1 轮的**实测结果**扩散成新证据——
            · 某片检出非空 ⇒ 该片附近目标成簇，邻居片值得复核（正证据）；
            · 某片被选中却检空 ⇒ 那里只是粗检的假象，该区域降温（负证据）。
            在未选切片里按修正后的分数追加 top-m 片（m 由预算余量 cap 决定，且要有正信号）。

**公平对照**是"每图同数量的一次性选片"：第 1 轮 x% + 第 2 轮追加的片数，正好等于一次性
选片跑同样多的片数，于是问题变成——第 2 轮用观测挑出来的片，是不是比"盲选下一批"更好？

D：可计算的停机判据 E
--------------------
标定之后每片的 S_det(k) 是"这片含目标"的后验概率，于是

    E = Σ_{k ∉ 已跑} S_det(k)

就是"还没看的地方预期漏掉多少个目标"——不需要真值、跨图可比的绝对量。它给出第三种选片
方式（与 θ、τ 并列）：**按分数从高到低追加，直到 E ≤ ε 就停**，每张图的预算自适应。
3.13 的归因表说明选片的代价上限本来就很小，所以这里的目标不是把 AP 抬多高，而是把
"什么时候可以停"从拍脑袋的常数变成可以算的量。
"""

from __future__ import annotations

import numpy as np

from .saliency import detection_prior, fuse
from .selector import select_slices


def slice_centers(slices) -> np.ndarray:
    s = np.asarray(slices, float).reshape(-1, 4)
    return np.stack([(s[:, 0] + s[:, 2]) / 2.0, (s[:, 1] + s[:, 3]) / 2.0], axis=1)


def neighbor_weights(centers: np.ndarray, sigma: float) -> np.ndarray:
    """片中心之间的高斯相似度（对角置 0，避免自己给自己投票）。"""
    d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    np.fill_diagonal(w, 0.0)
    return w


def observation_signal(rec, ran) -> np.ndarray:
    """已跑切片的实测信号：有检出 +1，被选中却检空 −1，未跑 0。"""
    sig = np.zeros(len(rec["slices"]), dtype=float)
    for k in ran:
        sig[int(k)] = 1.0 if len(rec["slice_preds"][int(k)]) else -1.0
    return sig


def refresh_scores(base: np.ndarray, rec, ran, sigma: float, gamma: float) -> np.ndarray:
    """用实测信号扩散修正先验分数：S' = clip(S + γ·Σ_j K_ij·signal_j / Σ_j K_ij)。

    分母按已跑切片归一，因此 R 落在 [−1, 1]：邻居全都检空 → −1，全都检出 → +1。
    """
    if not ran:
        return np.asarray(base, float).copy()
    w = neighbor_weights(slice_centers(rec["slices"]), sigma)
    ran = np.asarray(ran, int)
    sig = observation_signal(rec, ran)
    denom = w[:, ran].sum(1)
    num = w[:, ran] @ sig[ran]
    R = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-9)
    return np.clip(np.asarray(base, float) + gamma * R, 0.0, 1.0)


def expected_missed(s_det: np.ndarray, ran) -> float:
    """E = Σ_{未跑切片} S_det(k)：还没看的地方预期漏掉多少个目标（可计算的停机判据）。"""
    mask = np.ones(len(s_det), dtype=bool)
    mask[np.asarray(ran, int)] = False
    return float(np.asarray(s_det, float)[mask].sum())


def calibrated_scores(rec, cfg, cal):
    """返回 (融合后的选片分数, 每片 S_det)。cal 为 None 时退化成原始置信度。"""
    boxes, raw = rec["glance"][:, :4], rec["glance"][:, 4]
    p = cal.transform(boxes, raw, cal.scale_for(rec["hw"])) if cal is not None else raw
    s_det = detection_prior(boxes, p, rec["slices"], cfg.det_margin, "noisyor")
    s_img = rec.get("prior_edge")
    return (fuse(s_det, s_img, cfg.img_weight) if s_img is not None else s_det), s_det


def active_select(rec, cfg, cal, budget1: float, theta2: float, extra_frac: float = 0.10,
                  sigma: float = 600.0, gamma: float = 0.5, min_gain: int = 1):
    """两轮主动选片。返回 (选中切片下标, 诊断信息)。

    budget1  第 1 轮按预算保留的比例。
    theta2   第 2 轮的门槛（作用在"修正后的分数"上，只有正信号才可能过线）。
    extra_frac  第 2 轮最多追加的比例（相对总切片数）——上限保证"主动"不无限膨胀预算。
    """
    slices = rec["slices"]
    n = len(slices)
    base, s_det = calibrated_scores(rec, cfg, cal)

    ran = list(select_slices(base, "budget", 0.0, budget1).tolist())
    info = {"round1": len(ran), "round2_added": 0, "E_round1": expected_missed(s_det, ran)}

    cap = int(round(extra_frac * n))
    if ran and cap >= min_gain and n > len(ran):
        refreshed = refresh_scores(base, rec, ran, sigma, gamma)
        rest = np.setdiff1d(np.arange(n), np.asarray(ran, int))
        order = rest[np.argsort(-refreshed[rest], kind="stable")]
        cand = [int(k) for k in order[:cap] if refreshed[k] >= theta2]
        if len(cand) >= min_gain:
            ran.extend(cand)
            info["round2_added"] = len(cand)

    info["E_after"] = expected_missed(s_det, ran)
    info["n_run"] = len(ran)
    return np.array(sorted(set(ran)), dtype=int), info


def estop_select(rec, cfg, cal, eps: float):
    """按 E 停机：分数从高到低追加切片，直到 E = Σ_{未选} S_det ≤ ε。

    与 θ / τ 并列的第三种选片方式，特点是**每张图的预算自适应**，且判据是概率量（可跨图比）。
    """
    base, s_det = calibrated_scores(rec, cfg, cal)
    order = np.argsort(-np.asarray(base, float), kind="stable")
    remaining = float(np.asarray(s_det, float).sum())
    sel: list[int] = []
    for k in order:
        sel.append(int(k))
        remaining -= float(s_det[k])
        if remaining <= eps:
            break
    return np.array(sorted(sel), dtype=int), {"n_run": len(sel), "E_after": remaining}
