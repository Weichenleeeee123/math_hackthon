"""Demo 与报告用的绘图小工具（只依赖 cv2 / numpy）。图像均为 RGB uint8。"""

import cv2
import numpy as np

CLASS_COLORS = {0: (42, 120, 214), 2: (235, 104, 52), 5: (27, 175, 122), 7: (237, 161, 0),
                8: (74, 58, 167), 4: (232, 123, 164)}


def draw_slices(img, slices, selected, scores=None, dim: float = 0.55, show_scores: bool = True):
    """画 SAHI 切片网格：被激活（选中）的切片保持原亮度并描粗边，未激活的切片变暗。"""
    out = img.copy()
    sel = set(int(k) for k in selected)
    mask = np.zeros(img.shape[:2], bool)
    for k in sel:
        x1, y1, x2, y2 = slices[k]
        mask[y1:y2, x1:x2] = True
    out[~mask] = (out[~mask] * (1 - dim)).astype(np.uint8)
    t = max(2, round(max(img.shape[:2]) / 600))
    for k, (x1, y1, x2, y2) in enumerate(slices):
        on = k in sel
        cv2.rectangle(out, (x1 + t, y1 + t), (x2 - t, y2 - t), (40, 220, 90) if on else (150, 150, 150),
                      2 * t if on else t)
        if show_scores and scores is not None:
            txt = f"{float(scores[k]):.2f}"
            fs = 0.5 * t
            cv2.putText(out, txt, (x1 + 6 * t, y1 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * t)
            cv2.putText(out, txt, (x1 + 6 * t, y1 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255) if on else (190, 190, 190), t)
    return out


def draw_dets(img, dets, min_score: float = 0.25):
    """dets: (N,6) [x1,y1,x2,y2,score,coco_cls]。"""
    out = img.copy()
    t = max(1, round(max(img.shape[:2]) / 900))
    for x1, y1, x2, y2, s, c in np.asarray(dets).reshape(-1, 6):
        if s < min_score:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS.get(int(c), (230, 60, 60)), t)
    return out


def overlay_heat(img, slices, scores, alpha: float = 0.45):
    """把每片分数铺成热图叠在原图上（重叠区域取最大值）。"""
    h, w = img.shape[:2]
    heat = np.zeros((h, w), np.float32)
    for (x1, y1, x2, y2), s in zip(slices, scores):
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(s))
    col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return (img * (1 - alpha) + col * alpha).astype(np.uint8)
