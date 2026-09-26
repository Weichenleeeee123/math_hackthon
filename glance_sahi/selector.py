"""按切片分数选出需要细看的切片。"""

import numpy as np


def select_slices(scores: np.ndarray, mode: str, threshold: float, budget: float,
                  evidence: np.ndarray | None = None, tau: float = 0.0, min_slices: int = 0) -> np.ndarray:
    """返回被选中切片的下标（升序，保持 SAHI 原有的切片顺序）。

    mode="threshold": 分数 ≥ θ（相对分数，θ ∈ [0,1]）
    mode="budget":    保留分数最高的 round(budget·N) 片
    mode="evidence":  证据量 ≥ τ（**绝对量、未归一化**，可跨图/跨数据集统一标定；见 REPORT 3.9）

    min_slices > 0 时，选中数不足就按分数从高到低补齐——避免“阈值一高一片不剩”的退化
    （坏处是把“这张图确实没有目标”也强制切一片，所以默认 0 = 关闭，保持既有结果不变）。
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=int)
    if mode == "threshold":
        keep = set(np.flatnonzero(scores >= threshold).tolist())
    elif mode == "budget":
        k = int(round(budget * n))
        keep = set(np.argsort(-scores, kind="stable")[:k].tolist())
    elif mode == "evidence":
        if evidence is None:
            raise ValueError("mode='evidence' 需要 evidence")
        keep = set(np.flatnonzero(np.asarray(evidence) >= tau).tolist())
        min_slices = max(min_slices, 1)  # 证据法的保底：至少细看一片
    else:
        raise ValueError(mode)
    if len(keep) < min_slices:
        for i in np.argsort(-scores, kind="stable"):
            if len(keep) >= min_slices:
                break
            keep.add(int(i))
    return np.array(sorted(keep), dtype=int)


def random_slices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """对照组：从同一网格里随机选 k 片。"""
    return np.sort(rng.choice(n, size=min(k, n), replace=False)) if k > 0 else np.zeros(0, dtype=int)
