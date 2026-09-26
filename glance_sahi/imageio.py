"""Windows 中文路径安全的图像读写。

`cv2.imread` / `cv2.imwrite` 在含中文的路径上会静默返回 None（本项目工作区路径就含中文），
改用 `np.fromfile` / `ndarray.tofile` 配合 `cv2.imdecode` / `cv2.imencode`。
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path) -> np.ndarray | None:
    """读成 OpenCV 约定（BGR）；失败返回 None。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imread_rgb(path) -> np.ndarray | None:
    bgr = imread(path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite(path, img, quality: int = 92) -> bool:
    ext = Path(path).suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return bool(ok)
