"""predict.py 管线的单元检查：用一个确定性的假检测器代替 YOLO（不需要 GPU、权重或数据集）。

相当于 run_eval.py e2e --check-all 的离线版：全选时 Glance-SAHI 必须与官方 get_sliced_prediction 一致，
且批推理只改变调用方式、不改变结果（假检测器没有批内浮点误差）。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahi.postprocess.combine import NMSPostprocess  # noqa: E402
from sahi.prediction import ObjectPrediction  # noqa: E402

from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.predict import glance_sliced_prediction, make_postprocess, sahi_uniform_prediction  # noqa: E402


class BlobModel:
    """假检测器：每个亮块（连通域）是一个目标，分数 = 块内最大亮度 / 255。"""

    def __init__(self, is_obb=False):
        self.confidence_threshold = 0.05
        self.is_obb = is_obb
        self.batches = []

    def perform_inference(self, image):
        self.perform_batch_inference([image])

    def perform_batch_inference(self, images):
        self._images = list(images)
        self.batches.append(len(images))

    def convert_original_predictions(self, shift_amount=None, full_shape=None):
        if np.ndim(shift_amount) == 1:   # get_prediction 传单张的 [x, y] / [h, w]
            shift_amount, full_shape = [shift_amount], [full_shape]
        self._per_image = [self._detect(im, s, fs) for im, s, fs in zip(self._images, shift_amount, full_shape)]

    @property
    def object_prediction_list(self):
        return self._per_image[0]

    @property
    def object_prediction_list_per_image(self):
        return self._per_image

    def _detect(self, img, shift, full_shape):
        g = img.max(axis=2)
        n, lab, stats, _ = cv2.connectedComponentsWithStats((g > 0).astype(np.uint8))
        out = []
        for i in range(1, n):
            x, y, w, h, _ = (int(v) for v in stats[i])
            score = float(g[lab == i].max()) / 255.0
            if score >= self.confidence_threshold:
                out.append(ObjectPrediction(bbox=[x, y, x + w, y + h], category_id=0, category_name="blob",
                                            score=score, shift_amount=list(shift), full_shape=list(full_shape)))
        return out


def _scene(h=1100, w=1500):
    img = np.zeros((h, w, 3), np.uint8)
    # (x, y, 边长, 亮度)：强目标、弱目标、低于输出阈值的“扫视弱证据”，含一个跨切片边界的块
    for x, y, s, v in [(60, 80, 30, 255), (700, 500, 12, 60), (1300, 900, 8, 200), (1200, 150, 6, 5),
                       (400, 395, 40, 180), (900, 1000, 10, 30)]:
        img[y:y + s, x:x + s] = v
    return img


def _arr(res):
    return np.array(sorted(p.bbox.to_xyxy() + [p.score.value, p.category.id] for p in res.object_prediction_list),
                    np.float64).reshape(-1, 6)


@pytest.mark.parametrize("bs", [1, 3])
def test_force_all_equals_official_sahi(bs):
    img, cfg = _scene(), GlanceConfig(batch_size=bs)
    ref, _, n = sahi_uniform_prediction(img, BlobModel(), cfg)
    m = BlobModel()
    res, st = glance_sliced_prediction(img, m, cfg, force_select=np.arange(n))
    assert st.n_slices_run == n > 1
    assert np.array_equal(_arr(res), _arr(ref))
    assert max(m.batches) == bs, "切片按 batch_size 成批送检测器（第一批是扫视的整图）"


def test_batch_size_does_not_change_selection_result():
    img = _scene()
    r1, s1 = glance_sliced_prediction(img, BlobModel(), GlanceConfig(threshold=0.5, batch_size=1))
    r4, s4 = glance_sliced_prediction(img, BlobModel(), GlanceConfig(threshold=0.5, batch_size=4))
    assert 0 < s1.n_slices_run < s1.n_slices_total
    assert np.array_equal(s1.selected, s4.selected)
    assert np.array_equal(_arr(r1), _arr(r4))


def test_single_slice_image_matches_sahi_and_runs_once():
    img = _scene(300, 400)
    ref, _, n = sahi_uniform_prediction(img, BlobModel(), GlanceConfig())
    m = BlobModel()
    res, st = glance_sliced_prediction(img, m, GlanceConfig(threshold=0.99))
    assert n == 1 and st.n_slices_run == 1
    assert m.batches == [1], "唯一的切片就是整图：不再额外扫视一次"
    assert np.array_equal(_arr(res), _arr(ref))


def test_obb_model_forces_nms_like_sahi():
    cfg = GlanceConfig(postprocess_type="GREEDYNMM", postprocess_match_metric="IOS")
    assert isinstance(make_postprocess(cfg, BlobModel(is_obb=True)), NMSPostprocess)
    assert not isinstance(make_postprocess(cfg, BlobModel()), NMSPostprocess)


def test_learned_scorer_rejects_evidence_mode():
    with pytest.raises(ValueError):
        glance_sliced_prediction(_scene(), BlobModel(), GlanceConfig(scorer="learned", mode="evidence"))


# ---------------------------------------------------------------- 覆盖感知去冗余（REPORT 3.19）
def test_prune_removes_near_duplicate_edge_slice_keeps_union():
    from sahi.slicing import get_slice_bboxes

    from glance_sahi.selector import prune_redundant

    s = get_slice_bboxes(1080, 1920, 512, 512, False, 0.2, 0.2)   # 贴边片 x=1408 与 x=1228 大面积重叠
    allk = np.arange(len(s))
    kept = prune_redundant(s, allk, np.zeros(len(s)), 0.1)
    assert 0 < len(kept) < len(s)
    # 保证：每删一片，覆盖并集最多少 min_new·片面积
    cov = np.zeros((1080 // 8, 1920 // 8), bool)
    for x1, y1, x2, y2 in np.asarray(s)[kept] // 8:
        cov[y1:y2, x1:x2] = True
    lost = (~cov).sum()
    assert lost <= (len(s) - len(kept)) * 0.1 * (512 // 8) ** 2
    assert np.array_equal(prune_redundant(s, allk, np.zeros(len(s)), 0.0), allk), "min_new=0 = 关闭"
    assert list(prune_redundant(s, [3], np.zeros(len(s)), 0.5)) == [3], "单片不删"


def test_prune_drops_lower_score_of_two_duplicates():
    from glance_sahi.selector import prune_redundant

    s = [[0, 0, 100, 100], [4, 0, 104, 100], [300, 0, 400, 100]]
    assert list(prune_redundant(s, [0, 1, 2], np.array([0.9, 0.5, 0.1]), 0.1)) == [0, 2]
    assert list(prune_redundant(s, [0, 1, 2], np.array([0.5, 0.9, 0.1]), 0.1)) == [1, 2]


def test_prune_in_pipeline_default_off():
    img = _scene()
    r0, s0 = glance_sliced_prediction(img, BlobModel(), GlanceConfig(threshold=0.0))
    r1, s1 = glance_sliced_prediction(img, BlobModel(), GlanceConfig(threshold=0.0, prune_min_new=0.1))
    assert s0.n_slices_run == s0.n_slices_total
    assert s1.n_slices_run < s0.n_slices_run and set(s1.selected) <= set(s0.selected)
