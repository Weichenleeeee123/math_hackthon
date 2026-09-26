"""扫视阶段的显著性证据：检测先验 + 图像先验。

两路证据都在“一眼缩略图”上计算，成本与切片数无关：
- 检测先验：整图缩小推理（SAHI 本来就要做的 standard prediction）在低阈值下给出的粗检测。
  每个粗检测把置信度 c 当成“该处有目标”的概率，切片内“至少有一个目标”的概率由 noisy-OR 给出。
- 图像先验：缩略图上的局部纹理/显著性，兜住缩小后检测器完全看不见的极小目标。

检测先验的权重取法（`kind`）是打分函数的消融变体，见 `det_weights` 与 REPORT 3.8。
"""

import cv2
import numpy as np


def image_prior_map(image: np.ndarray, kind: str, map_size: int) -> tuple[np.ndarray, float]:
    """返回 (缩略图尺度的显著性图, 缩放比例 scale=map/原图)。显著性已按分位数归一化到 [0,1]。"""
    h, w = image.shape[:2]
    scale = map_size / max(h, w)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if kind == "edge":
        # 小目标在缩略图上表现为局部高频（边缘密集）；再减去大尺度平均，抑制大片均匀纹理（草地、屋顶）
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        local = cv2.blur(mag, (5, 5))
        broad = cv2.blur(mag, (31, 31))
        sal = np.maximum(local - broad, 0) + 0.5 * local
    elif kind == "spectral":
        # Hou & Zhang 2007 谱残差显著性
        f = np.fft.fft2(cv2.resize(gray, (64, 64)))
        log_amp = np.log(np.abs(f) + 1e-8)
        residual = log_amp - cv2.blur(log_amp, (3, 3))
        sal64 = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
        sal = cv2.resize(cv2.GaussianBlur(sal64.astype(np.float32), (0, 0), 2.5), gray.shape[::-1])
    else:
        raise ValueError(kind)

    lo, hi = np.percentile(sal, [5, 99])
    return np.clip((sal - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32), scale


# 检测先验的权重取法（打分函数的消融变体，见 REPORT 3.8）：
#   "noisyor"   w = c        置信度即“该处有目标”的概率，弱证据可累积（默认，主方法）
#   "uncertain" w = 4c(1−c)  不确定度加权：c ≈ 0.5 的“似是而非”框最热，已确定/c≈0 的框变冷
#   "max"       不做 noisy-OR 累积，直接取片内最大 c —— v0 的直觉做法，用作反面消融
DET_WEIGHT_KINDS = ("noisyor", "uncertain", "max")


def det_weights(scores: np.ndarray, kind: str = "noisyor") -> np.ndarray:
    """把粗检测置信度映射成 [0,1] 的“该处有目标”的证据强度 w_j。"""
    c = np.clip(scores, 0, 0.999)
    if kind in ("noisyor", "max"):
        return c
    if kind == "uncertain":
        # 4p(1−p)：p=0.5 时取 1，p→0 或 p→1 时取 0。整图已经看清的地方不必再细看。
        return 4.0 * c * (1.0 - c)
    raise ValueError(f"unknown det weight kind {kind!r}, expected one of {DET_WEIGHT_KINDS}")


def detection_prior(boxes: np.ndarray, scores: np.ndarray, slices: list[list[int]], margin: int,
                    kind: str = "noisyor") -> np.ndarray:
    """每个切片的检测先验 —— 把粗检测当作独立证据，估计“切片里有目标”的后验概率。

        S_det(k) = 1 − Π_{j: 中心 ∈ R_k ⊕ margin} (1 − w_j),   w_j = det_weights(c_j, kind)

    kind="max" 时退化为片内证据的最大值（v0 的直觉做法），用于消融对照。

    boxes: (N,4) 原图坐标 xyxy；scores: (N,)；slices: SAHI 的切片列表 [x1,y1,x2,y2]。
    """
    if len(boxes) == 0:
        return np.zeros(len(slices), dtype=np.float32)
    w = det_weights(scores, kind)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    out = np.empty(len(slices), dtype=np.float32)
    if kind == "max":
        for k, (x1, y1, x2, y2) in enumerate(slices):
            inside = (cx >= x1 - margin) & (cx < x2 + margin) & (cy >= y1 - margin) & (cy < y2 + margin)
            out[k] = float(w[inside].max()) if inside.any() else 0.0
        return out
    for k, (x1, y1, x2, y2) in enumerate(slices):
        inside = (cx >= x1 - margin) & (cx < x2 + margin) & (cy >= y1 - margin) & (cy < y2 + margin)
        out[k] = 1.0 - float(np.prod(1.0 - w[inside]))  # 片内无证据时 prod(空)=1 → 0
    return out


def evidence_mass(boxes: np.ndarray, scores: np.ndarray, slices: list[list[int]]) -> np.ndarray:
    """每个切片的“证据量” = 中心落在片内的弱检测置信度之和。

    刻意**不归一化**：τ 因此是一个能在整个数据集（乃至跨数据集）统一标定的绝对量，
    不像 θ 作用在相对分数上、换图像先验权重 λ 就整体漂移（REPORT 3.7 的痛点）。
    """
    if len(boxes) == 0:
        return np.zeros(len(slices), dtype=np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    s = np.asarray(slices, np.float32).reshape(-1, 4)
    inside = ((cx[None, :] >= s[:, 0:1]) & (cx[None, :] < s[:, 2:3]) &
              (cy[None, :] >= s[:, 1:2]) & (cy[None, :] < s[:, 3:4]))
    return (inside * np.asarray(scores, np.float32)[None, :]).sum(1).astype(np.float32)


def coarse_heatmap(shape_hw, boxes, scores, sigma: float = 6.0, normalize: bool = False) -> np.ndarray:
    """粗检热图：框中心撒点 + 一次高斯模糊（boxes 为缩略图坐标）。

    卷积**保持质量**：全图之和 ≈ Σc_j，因此像素值就是“该像素附近的弱检测置信度之和”，
    可以直接在切片内求和再用 1 − exp(−Σ) 饱和。这一点是有意与参考实现不同的：
    参考实现乘了 2πσ² 让单个 blob 的**峰值**等于 c（便于看热图），但那使像素值随 σ 漂移，
    不适合做逐片求和；我们改为保质量，显示时用 `normalize=True`（按最大值归一，仅用于看图）。

    `normalize=True` 只做显示归一化，不改变任何选片用的数值。
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    heat = np.zeros((h, w), np.float32)
    if len(boxes):
        b = np.asarray(boxes, np.float32).reshape(-1, 4)
        cx = np.clip(((b[:, 0] + b[:, 2]) / 2).astype(int), 0, w - 1)
        cy = np.clip(((b[:, 1] + b[:, 3]) / 2).astype(int), 0, h - 1)
        np.add.at(heat, (cy, cx), np.asarray(scores, np.float32))
        heat = cv2.GaussianBlur(heat, (0, 0), sigma)
    if normalize:
        m = float(heat.max())
        heat = heat / m if m > 0 else heat
    return heat


def heatmap_prior(boxes: np.ndarray, scores: np.ndarray, hw: tuple[int, int], slices: list[list[int]],
                  map_size: int = 512, sigma: float = 6.0):
    """热图变体（`det_prior="heatmap"`）：原图坐标的粗检测 → 缩略图热图 → 每片先验。

        S_det(k) = 1 − exp(− Σ_{pixel ∈ R_k} heat)，  heat = G_σ * Σ_j c_j δ(中心_j)

    与 noisy-OR 同族：稀疏弱证据下 Σc 很小，1 − exp(−Σc) ≈ 1 − Π(1 − c_j)。
    好处是框的贡献按高斯**软扩散**到邻近切片（不必像 noisy-OR 那样靠 det_margin 硬扩边界），
    且成本是 O(像素) 的一次卷积，与框数无关（DOTA 44.7 片/图 × 每片多框时差别明显）。
    返回 (s_det, heat, scale)，heat 是归一化后的显示用热图（3.6 的热图面板）。
    """
    h, w = hw
    scale = map_size / max(h, w)
    th = max(int(round(h * scale)), 1)
    tw = max(int(round(w * scale)), 1)
    b = np.asarray(boxes, np.float32).reshape(-1, 4) * scale
    raw = coarse_heatmap((th, tw), b, scores, sigma, normalize=False)
    mass = np.empty(len(slices), dtype=np.float32)
    for k, (x1, y1, x2, y2) in enumerate(slices):
        ya, xa = int(y1 * scale), int(x1 * scale)
        yb = min(max(int(np.ceil(y2 * scale)), ya + 1), th)
        xb = min(max(int(np.ceil(x2 * scale)), xa + 1), tw)
        mass[k] = float(raw[ya:yb, xa:xb].sum())
    return 1.0 - np.exp(-mass), coarse_heatmap((th, tw), b, scores, sigma, normalize=True), scale


def region_prior(sal: np.ndarray, scale: float, slices: list[list[int]]) -> np.ndarray:
    """每个切片的图像先验：切片内显著性的 95 分位（对“一小撮亮点”敏感，对均值不敏感）。"""
    out = np.empty(len(slices), dtype=np.float32)
    for k, (x1, y1, x2, y2) in enumerate(slices):
        patch = sal[int(y1 * scale):max(int(y2 * scale), int(y1 * scale) + 1),
                    int(x1 * scale):max(int(x2 * scale), int(x1 * scale) + 1)]
        out[k] = np.percentile(patch, 95)
    return out


def fuse(s_det: np.ndarray, s_img: np.ndarray | None, img_weight: float) -> np.ndarray:
    """noisy-OR 融合：S = 1 - (1 - S_det)(1 - λ·S_img)。"""
    if s_img is None or img_weight <= 0:
        return s_det
    return 1.0 - (1.0 - s_det) * (1.0 - img_weight * s_img)
