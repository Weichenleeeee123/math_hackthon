"""粗检测置信度的分箱标定：把"只在序关系上可靠"的 c_j 变成"可以当概率用"的 p̂_j。

为什么改（对应 REPORT 2.2 已承认的局限）
--------------------------------------
检测先验把粗检测置信度 c_j **直接当作** P(该处有目标 | 粗检测 j)，再用 noisy-OR 聚合。
但 c_j 来自"整图缩小推理"（模型输入 640）：
- 同一张图里，目标被下采样后越小，响应越弱；
- 不同图的缩放比也不同（1360×765→640 是 0.47×，2000×1500→640 是 0.32×）。

所以同一个 c 在不同图、不同目标尺度下的含义并不相同，c 只在**序关系**上可靠
（REPORT 2.2 原文：c_j 只在序关系上可靠，检测器未在缩略图尺度上标定）。

改了什么（借鉴 SaccadeNet 的"按混淆变量分箱标定"）
--------------------------------------------------
SaccadeNet 用偏心率 e 分箱标定 d′(e)=(μ1−μ0)/σ（见 saccadenet docs/report.md）。
这里把**混淆变量**取成"粗检测框在模型输入尺度下的表观尺度"：

    t = scale · √(w·h),   scale = model_input / max(H, W)

完全可部署——只用粗检测框，不需要真值。按 t 分箱后，在标定集上用保序回归（PAVA）
拟合单调映射 p̂ = f_b(c)，使 p̂ ≈ P(检测为真 | c, 箱 b)。

与 SaccadeNet 的 d′(e) 同构：都是"按一个可观测的混淆变量分箱、再重新标定置信度"。
区别：它标定每个候选的 log-odds，我们标定 noisy-OR 的**输入概率**；它需要重新训练/标定网络，
我们只对**已经拿到的粗检测**做一次后处理，检测器与切片流程一行不改。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# 表观尺度（模型输入像素，≈ 检测框等效边长）的分箱边界；最后一箱为 +inf
DEFAULT_BIN_EDGES = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf]


def pava(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators：返回与 y 等长、单调不减的最小二乘拟合。

    保序回归是概率标定的标准做法（isotonic calibration）：它**不假设**任何参数形式，
    只要求 p̂ 随 c 单调不减——这正是我们想要的，因为 c 的序关系是可靠的。
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    blocks: list[list[float]] = []  # [sum_y, count, start, end]
    for i in range(n):
        cur = [float(y[i]), 1.0, i, i]
        while blocks and blocks[-1][0] / blocks[-1][1] > cur[0] / cur[1]:
            prev = blocks.pop()
            cur = [prev[0] + cur[0], prev[1] + cur[1], prev[2], cur[3]]
        blocks.append(cur)
    out = np.empty(n, dtype=float)
    for s, c, start, end in blocks:
        out[start:end + 1] = s / c
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """ECE（期望标定误差）+ 可靠性表的逐箱明细。

    可靠性表回答的是："模型说 0.3 的地方，实际有多少比例是真的？"
    未标定的 c 会系统性偏离对角线；标定后应贴近对角线、ECE 明显下降。
    """
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    if len(probs) == 0:
        return 0.0, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        conf = float(probs[m].mean())
        acc = float(labels[m].mean())
        ece += n / len(probs) * abs(acc - conf)
        rows.append(dict(bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), n=n,
                         mean_conf=conf, empirical_rate=acc, gap=acc - conf))
    return float(ece), rows


def match_to_gt(det_centers: np.ndarray, gt_centers: np.ndarray, gt_diag: np.ndarray,
                tol: float = 6.0) -> np.ndarray:
    """把检测框中心匹配到真值中心：距离 ≤ max(半个 GT 框对角线, tol) 记为真阳。

    对小目标用"中心距离"而不是 IoU 匹配更稳：VisDrone 里 20px 的行人，框只有几十个像素，
    IoU 对 1–2 像素的偏移非常敏感。这是标定拟合用的标签定义，与评测口径无关。
    """
    hit = np.zeros(len(det_centers), dtype=bool)
    if len(gt_centers) == 0 or len(det_centers) == 0:
        return hit
    d = np.linalg.norm(det_centers[:, None, :] - gt_centers[None, :, :], axis=2)
    radius = np.maximum(0.5 * gt_diag, tol)[None, :]
    hit = (d <= radius).any(axis=1)
    return hit


def inside_any_box(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """点是否落在任一框内（用于剔除落在 crowd/忽略区里的检测，避免给标定制造脏标签）。"""
    if len(boxes) == 0 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    b = np.asarray(boxes, float).reshape(-1, 4)
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    return ((x >= b[:, 0]) & (x < b[:, 2]) & (y >= b[:, 1]) & (y < b[:, 3])).any(axis=1)


class DetectionCalibrator:
    """按"表观尺度"分箱的保序回归标定器。

    每个箱存一条 101 点的单调映射（在 [0,1] 的置信度网格上取值），用线性插值调用。
    样本太少的箱（< min_bin）回退到**全局**映射，避免小箱过拟合。
    """

    def __init__(self, bin_edges=None, model_input: int = 640, grid_size: int = 101,
                 min_bin: int = 50):
        self.bin_edges = np.asarray(
            DEFAULT_BIN_EDGES if bin_edges is None else bin_edges, dtype=float)
        if not np.isinf(self.bin_edges[-1]):
            raise ValueError("分箱边界的最后一箱必须是 +inf")
        self.model_input = int(model_input)
        self.grid = np.linspace(0.0, 1.0, int(grid_size))
        self.min_bin = int(min_bin)
        self.maps: list[np.ndarray] = []
        self.counts: list[int] = []
        self.global_map: np.ndarray | None = None
        self.n_fit_samples = 0

    # ---------------------------------------------------------------- 尺度
    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1

    def scale_for(self, hw) -> float:
        """整图缩小推理的缩放比（长边对齐模型输入）。"""
        h, w = float(hw[0]), float(hw[1])
        return self.model_input / max(h, w, 1.0)

    def apparent_size(self, boxes: np.ndarray, scale: float) -> np.ndarray:
        b = np.asarray(boxes, float).reshape(-1, 4)
        if len(b) == 0:
            return np.zeros(0, dtype=float)
        wh = np.clip(b[:, 2] - b[:, 0], 1e-6, None) * np.clip(b[:, 3] - b[:, 1], 1e-6, None)
        return scale * np.sqrt(wh)

    def bin_index(self, sizes: np.ndarray) -> np.ndarray:
        s = np.asarray(sizes, float)
        return np.clip(np.searchsorted(self.bin_edges, s, side="right") - 1, 0, self.n_bins - 1)

    # ---------------------------------------------------------------- 拟合
    @staticmethod
    def _fit_one(scores: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """在 (score → label) 上跑 PAVA 并重采样到固定网格，得到单调不减的映射。"""
        order = np.argsort(scores, kind="stable")
        c, y = np.asarray(scores, float)[order], np.asarray(labels, float)[order]
        fit = pava(y)
        # c 可能有并列值；PAVA 在并列处取常数，np.interp 不会有歧义
        return np.interp(grid, c, fit)

    def fit(self, sizes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> "DetectionCalibrator":
        sizes = np.asarray(sizes, float)
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, float)
        if len(scores) == 0:
            raise ValueError("标定集为空，无法拟合")
        self.n_fit_samples = int(len(scores))
        self.global_map = self._fit_one(scores, labels, self.grid)
        bidx = self.bin_index(sizes)
        self.maps, self.counts = [], []
        for b in range(self.n_bins):
            m = bidx == b
            self.counts.append(int(m.sum()))
            self.maps.append(self._fit_one(scores[m], labels[m], self.grid)
                             if m.sum() > 1 else self.global_map.copy())
        return self

    # ---------------------------------------------------------------- 应用
    def transform(self, boxes: np.ndarray, scores: np.ndarray, scale: float) -> np.ndarray:
        """把粗检测置信度映射成标定概率 p̂。样本不足的箱回退到全局映射。"""
        scores = np.asarray(scores, float)
        if len(scores) == 0:
            return np.zeros(0, dtype=float)
        if self.global_map is None:
            raise RuntimeError("标定器尚未拟合（先调用 fit）")
        bidx = self.bin_index(self.apparent_size(boxes, scale))
        out = np.empty(len(scores), dtype=float)
        for i, (b, c) in enumerate(zip(bidx, scores)):
            m = self.maps[b] if self.counts[b] >= self.min_bin else self.global_map
            out[i] = float(np.interp(c, self.grid, m))
        return np.clip(out, 0.0, 0.999)

    def reliability(self, sizes, scores, labels, n_bins: int = 15):
        """返回 (原始 c 的 ECE, 标定后 p̂ 的 ECE, 标定后可靠性表)。

        sizes 是"已乘过 scale 的表观尺度"（见 apparent_size），因此这里直接用 size 分箱。
        """
        bidx = self.bin_index(sizes)
        p = np.empty(len(scores), dtype=float)
        for i, (b, c) in enumerate(zip(bidx, scores)):
            m = self.maps[b] if self.counts[b] >= self.min_bin else self.global_map
            p[i] = float(np.interp(c, self.grid, m))
        ece_raw, _ = expected_calibration_error(scores, labels, n_bins)
        ece_cal, rows = expected_calibration_error(np.clip(p, 0, 0.999), labels, n_bins)
        return ece_raw, ece_cal, rows

    # ---------------------------------------------------------------- 持久化
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "DetectionCalibrator":
        return pickle.loads(Path(path).read_bytes())

    def summary(self) -> list[dict]:
        """每个箱的样本量与**实际生效**的映射（用于打印/进表）。

        样本不足的箱在 transform 时会回退到全局映射，所以这里也显示全局映射的值，
        否则表格会让人以为那些箱有自己的映射。
        """
        rows = []
        for b in range(self.n_bins):
            used_global = self.counts[b] < self.min_bin
            m = self.global_map if used_global else self.maps[b]
            rows.append(dict(
                bin=b, lo=float(self.bin_edges[b]), hi=float(self.bin_edges[b + 1]),
                n=self.counts[b], used_global=used_global,
                p_at_c05=float(np.interp(0.05, self.grid, m)),
                p_at_c10=float(np.interp(0.10, self.grid, m)),
                p_at_c30=float(np.interp(0.30, self.grid, m)),
                p_at_c90=float(np.interp(0.90, self.grid, m)),
            ))
        return rows
