"""Glance-SAHI 主流程：可直接替换 sahi.predict.get_sliced_prediction。

SAHI 原流程：                               Glance-SAHI：
  1. 生成 N 个均匀切片                         1. 同一个 N 切片网格（sahi.slicing.get_slice_bboxes）
  2. 对 N 片逐一推理                           2. 整图缩小推理一次（= SAHI 的 standard pred，放低阈值）
  3. 整图推理一次（standard pred）             3. 用粗检测 + 图像显著性给每片打分
  4. 合并（GREEDYNMM / NMS）                   4. 只对分数高的 k 片推理（k ≤ N）
                                               5. 合并（与 SAHI 完全相同的后处理）
"""

import time
from dataclasses import dataclass, field

import numpy as np
from sahi.predict import POSTPROCESS_NAME_TO_CLASS, filter_predictions, get_prediction
from sahi.prediction import PredictionResult
from sahi.slicing import get_slice_bboxes

from .config import GlanceConfig
from .saliency import detection_prior, evidence_mass, fuse, heatmap_prior, image_prior_map, region_prior
from .selector import random_slices, select_slices


@dataclass
class GlanceStats:
    n_slices_total: int = 0
    n_slices_run: int = 0
    slice_scores: np.ndarray | None = None
    selected: np.ndarray | None = None
    slices: list = field(default_factory=list)
    evidence: np.ndarray | None = None   # 每片的绝对证据量（mode="evidence" 用）
    heat: np.ndarray | None = None       # det_prior="heatmap" 时的缩略图热图（可视化用）
    t_glance: float = 0.0     # 整图缩小推理
    t_saliency: float = 0.0   # 打分 + 选片
    t_slices: float = 0.0     # 切片推理
    t_post: float = 0.0       # 合并

    @property
    def t_total(self) -> float:
        return self.t_glance + self.t_saliency + self.t_slices + self.t_post


def make_postprocess(cfg: GlanceConfig):
    return POSTPROCESS_NAME_TO_CLASS[cfg.postprocess_type](
        match_threshold=cfg.postprocess_match_threshold,
        match_metric=cfg.postprocess_match_metric,
        class_agnostic=False,
    )


def _predict_slices(image, model, slices, idx, exclude_ids):
    preds = []
    h, w = image.shape[:2]
    for k in idx:
        x1, y1, x2, y2 = slices[k]
        res = get_prediction(image[y1:y2, x1:x2], model, shift_amount=[x1, y1], full_shape=[h, w],
                             exclude_classes_by_id=exclude_ids)
        preds.extend(p.get_shifted_object_prediction() for p in res.object_prediction_list)
    return preds


def glance_sliced_prediction(
    image: np.ndarray,
    model,
    cfg: GlanceConfig,
    exclude_classes_by_id: list[int] | None = None,
    force_select: np.ndarray | None = None,
    random_k_rng: np.random.Generator | None = None,
) -> tuple[PredictionResult, GlanceStats]:
    """image: RGB ndarray。

    force_select: 直接指定要跑的切片下标（用于“全选”正确性自检）。
    random_k_rng: 给定时，先按 cfg 算出 k，再从网格里随机选 k 片（随机对照组）。
    """
    h, w = image.shape[:2]
    st = GlanceStats()
    slices = get_slice_bboxes(h, w, cfg.slice_size, cfg.slice_size, False, cfg.overlap_ratio, cfg.overlap_ratio)
    st.slices, st.n_slices_total = slices, len(slices)

    # 1) 扫视：整图缩小推理一次，低阈值
    t0 = time.perf_counter()
    glance = get_prediction(image, model, exclude_classes_by_id=exclude_classes_by_id,
                            confidence_threshold=cfg.glance_conf).object_prediction_list
    st.t_glance = time.perf_counter() - t0

    # 2) 打分 + 选片
    t0 = time.perf_counter()
    if force_select is not None:
        sel = np.asarray(force_select, dtype=int)
        scores = np.ones(len(slices), dtype=np.float32)
    else:
        boxes = np.array([p.bbox.to_xyxy() for p in glance], dtype=np.float32).reshape(-1, 4)
        confs = np.array([p.score.value for p in glance], dtype=np.float32)
        if cfg.det_prior == "heatmap":
            s_det, st.heat, _ = heatmap_prior(boxes, confs, (h, w), slices, cfg.img_map_size, cfg.heat_sigma)
        else:
            s_det = detection_prior(boxes, confs, slices, cfg.det_margin, cfg.det_prior)
        s_img = None
        if cfg.img_prior != "none" and cfg.img_weight > 0:
            sal, scale = image_prior_map(image, cfg.img_prior, cfg.img_map_size)
            s_img = region_prior(sal, scale, slices)
        scores = fuse(s_det, s_img, cfg.img_weight)
        st.evidence = evidence_mass(boxes, confs, slices)
        sel = select_slices(scores, cfg.mode, cfg.threshold, cfg.budget,
                            st.evidence, cfg.tau, cfg.min_slices)
        if random_k_rng is not None:
            sel = random_slices(len(slices), len(sel), random_k_rng)
    st.slice_scores, st.selected, st.n_slices_run = scores, sel, len(sel)
    st.t_saliency = time.perf_counter() - t0

    # 3) 只对选中切片推理（输出阈值与 SAHI 相同）
    t0 = time.perf_counter()
    preds = _predict_slices(image, model, slices, sel, exclude_classes_by_id)
    st.t_slices = time.perf_counter() - t0

    # 4) 合并：扫视结果中 ≥ 输出阈值的部分，就是 SAHI 的 standard prediction
    t0 = time.perf_counter()
    preds.extend(p for p in glance if p.score.value >= cfg.output_conf)
    if len(preds) > 1:
        preds = make_postprocess(cfg)(preds)
    st.t_post = time.perf_counter() - t0

    return PredictionResult(image=image, object_prediction_list=preds,
                            durations_in_seconds={"prediction": st.t_glance + st.t_slices,
                                                  "postprocess": st.t_post}), st


def sahi_uniform_prediction(image, model, cfg: GlanceConfig, exclude_classes_by_id=None):
    """官方 SAHI 基线（直接调用 sahi.predict.get_sliced_prediction），返回 (result, seconds, n_slices)。"""
    from sahi.predict import get_sliced_prediction

    t0 = time.perf_counter()
    res = get_sliced_prediction(
        image, model,
        slice_height=cfg.slice_size, slice_width=cfg.slice_size,
        overlap_height_ratio=cfg.overlap_ratio, overlap_width_ratio=cfg.overlap_ratio,
        perform_standard_pred=True,
        postprocess_type=cfg.postprocess_type,
        postprocess_match_metric=cfg.postprocess_match_metric,
        postprocess_match_threshold=cfg.postprocess_match_threshold,
        force_postprocess_type=True,
        exclude_classes_by_id=exclude_classes_by_id,
        verbose=0,
    )
    dt = time.perf_counter() - t0
    n = len(get_slice_bboxes(image.shape[0], image.shape[1], cfg.slice_size, cfg.slice_size, False,
                             cfg.overlap_ratio, cfg.overlap_ratio))
    return res, dt, n


def full_image_prediction(image, model, exclude_classes_by_id=None):
    t0 = time.perf_counter()
    res = get_prediction(image, model, exclude_classes_by_id=exclude_classes_by_id)
    return res, time.perf_counter() - t0
