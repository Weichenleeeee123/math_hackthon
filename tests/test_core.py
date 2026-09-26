"""不需要检测器的单元检查：python -m pytest tests（或直接 python tests/test_core.py）。

移植自参照实现（saliency_sahi/tests/test_core.py）的 7 项检查，并按 glance_sahi 的 API 改写：
- 网格一致性测试被简化——glance_sahi 直接调用 sahi.slicing.get_slice_bboxes，
  这里只做“与原图边界一致 + 与 SAHI 的 slice_image 逐片一致”的核对；
- 新增打分函数三种变体（noisy-OR / 4p(1−p) 不确定度 / 取最大）的行为测试，对应 REPORT 3.8。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahi.slicing import get_slice_bboxes, slice_image  # noqa: E402

from glance_sahi import data as datasets  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.data.dota import COCO_TO_EVAL as DOTA_COCO_TO_EVAL, DOTA_TO_EVAL  # noqa: E402
from glance_sahi.data.visdrone import (  # noqa: E402
    COCO_TO_EVAL, IGNORE_VISDRONE, VISDRONE_TO_EVAL, convert,
)
from glance_sahi.rules import find_illegal_parking, ground_point, load_zones, suggest_zone  # noqa: E402
from glance_sahi.saliency import (  # noqa: E402
    DET_WEIGHT_KINDS, coarse_heatmap, det_weights, detection_prior, evidence_mass, fuse,
    heatmap_prior, image_prior_map, region_prior,
)
from glance_sahi.selector import random_slices, select_slices  # noqa: E402


def grid(h, w, size=512, overlap=0.2):
    return get_slice_bboxes(h, w, size, size, False, overlap, overlap)


# --------------------------------------------------------------- 切片网格
def test_grid_in_bounds_and_matches_sahi():
    for h, w in [(1080, 1920), (765, 1360), (1500, 2000), (480, 360)]:
        cells = grid(h, w)
        assert cells, "网格不应为空"
        for x1, y1, x2, y2 in cells:
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        # 与 SAHI 自己切出来的每一片逐片一致（起点 + 切片尺寸）
        img = np.zeros((h, w, 3), np.uint8)
        ref = slice_image(img, slice_height=512, slice_width=512,
                          overlap_height_ratio=0.2, overlap_width_ratio=0.2)
        assert cells == [[x, y, x + si.shape[1], y + si.shape[0]]
                         for (x, y), si in zip(ref.starting_pixels, ref.images)]


# --------------------------------------------------------------- 检测先验（noisy-OR）
def _box(cx, cy, half=10.0):
    return [cx - half, cy - half, cx + half, cy + half]


def test_noisy_or_detection_prior_accumulates_weak_evidence():
    slices = [[0, 0, 256, 256], [256, 0, 512, 256]]
    boxes = np.array([_box(100, 100), _box(160, 100), _box(200, 100)], np.float32)
    s = detection_prior(boxes, np.full(3, 0.5, np.float32), slices, margin=16)
    assert abs(float(s[0]) - (1 - 0.5 ** 3)) < 1e-6, "三个 p=0.5 的证据应给出 0.875"
    assert s[1] == 0.0

    # 二十个 p=0.05 的弱证据能累积到 0.64，单个 p=0.3 只有 0.3：这正是 noisy-OR 相比取最大的价值
    weak = np.array([_box(60 + 8 * i, 120) for i in range(20)], np.float32)
    s_weak = detection_prior(weak, np.full(20, 0.05, np.float32), slices, margin=16)
    s_strong = detection_prior(np.array([_box(120, 120)], np.float32), np.array([0.3], np.float32), slices, margin=16)
    assert abs(float(s_weak[0]) - (1 - 0.95 ** 20)) < 1e-6
    assert float(s_weak[0]) > float(s_strong[0])

    # 空检测 → 全零；margin 内的边缘目标算进去，太远的不算
    assert detection_prior(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), slices, 16).tolist() == [0.0, 0.0]
    near = detection_prior(np.array([_box(128, 250)], np.float32), np.array([0.4], np.float32), slices, 16)
    far = detection_prior(np.array([_box(128, 300)], np.float32), np.array([0.4], np.float32), slices, 16)
    assert near[0] > 0 and far[0] == 0


def test_uncertain_weight_peaks_at_half():
    assert float(det_weights(np.array([0.5]), "uncertain")[0]) == 1.0
    assert float(det_weights(np.array([0.0]), "uncertain")[0]) == 0.0
    assert abs(float(det_weights(np.array([0.9]), "uncertain")[0]) - 0.36) < 1e-6
    # 置信度越高反而越冷 —— v1 改动的直觉：整图已经看清的地方不必再细看
    assert det_weights(np.array([0.5]), "uncertain")[0] > det_weights(np.array([0.95]), "uncertain")[0]
    assert det_weights(np.array([0.5]), "uncertain")[0] > det_weights(np.array([0.05]), "uncertain")[0]
    # noisyor / max 直接用置信度
    assert det_weights(np.array([0.9]), "noisyor")[0] == det_weights(np.array([0.9]), "max")[0] == 0.9


def test_det_prior_variants_and_invalid_kind():
    slices = [[0, 0, 256, 256]]
    boxes = np.array([_box(100, 100)], np.float32)

    assert float(detection_prior(boxes, np.array([0.5], np.float32), slices, 16, "noisyor")[0]) == 0.5
    assert float(detection_prior(boxes, np.array([0.5], np.float32), slices, 16, "uncertain")[0]) == 1.0
    assert float(detection_prior(boxes, np.array([0.5], np.float32), slices, 16, "max")[0]) == 0.5

    # 一个很确定 + 一个很弱的框：noisy-OR 保住了 0.91，取最大丢掉了弱证据，不确定度加权把两者都压冷
    two = np.array([_box(80, 80), _box(200, 200)], np.float32)
    sc = np.array([0.9, 0.1], np.float32)
    v_or = float(detection_prior(two, sc, slices, 16, "noisyor")[0])
    v_un = float(detection_prior(two, sc, slices, 16, "uncertain")[0])
    v_max = float(detection_prior(two, sc, slices, 16, "max")[0])
    assert abs(v_or - 0.91) < 1e-6 and abs(v_un - (1 - 0.64 ** 2)) < 1e-6 and abs(v_max - 0.9) < 1e-6
    assert v_or > v_max > v_un
    for kind in DET_WEIGHT_KINDS:
        assert detection_prior(two, sc, slices, 16, kind).shape == (1,)
    try:
        detection_prior(two, sc, slices, 16, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("未知的 kind 应抛 ValueError")


# --------------------------------------------------------------- 融合与图像先验
def test_fuse_noisy_or():
    s_det = np.array([0.0, 0.5, 0.9], np.float32)
    s_img = np.array([1.0, 0.5, 0.2], np.float32)
    assert fuse(s_det, None, 0.3) is s_det
    assert fuse(s_det, s_img, 0.0) is s_det
    assert np.allclose(fuse(np.zeros(3, np.float32), np.ones(3, np.float32), 1.0), 1.0)
    # 1 − (1 − 0.5)(1 − 0.3·0.5) = 0.575
    assert abs(float(fuse(s_det, s_img, 0.3)[1]) - 0.575) < 1e-6
    out = fuse(s_det, s_img, 0.7)
    assert out.min() >= 0 and out.max() <= 1
    assert (out >= s_det).all()  # 图像先验只加证据，不减证据


def test_region_prior_is_patch_quantile():
    sal = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    slices = [[0, 0, 5, 5], [5, 5, 10, 10]]
    got = region_prior(sal, 1.0, slices)
    want = [np.percentile(sal[0:5, 0:5], 95), np.percentile(sal[5:10, 5:10], 95)]
    assert np.allclose(got, want, atol=1e-6)
    assert got.shape == (2,)


def test_image_prior_bounds_and_kinds():
    img = np.random.default_rng(0).integers(0, 255, (128, 160, 3), dtype=np.uint8)
    for kind in ("edge", "spectral"):
        sal, scale = image_prior_map(img, kind, map_size=64)
        assert abs(scale - 64 / 160) < 1e-9
        assert sal.shape == (round(128 * scale), 64)
        assert sal.min() >= 0 and sal.max() <= 1 + 1e-6
        assert np.isfinite(sal).all()
    try:
        image_prior_map(img, "bogus", 64)
    except ValueError:
        pass
    else:
        raise AssertionError("未知的图像先验应抛 ValueError")


# --------------------------------------------------------------- 选片
def test_select_slices_threshold_and_budget():
    scores = (np.arange(20) / 20).astype(np.float32)  # 0, 0.05, ..., 0.95
    assert select_slices(scores, "threshold", 0.5, 0).tolist() == list(range(10, 20))
    assert select_slices(scores, "threshold", 2.0, 0).tolist() == []
    for budget, k in [(0.25, 5), (0.5, 10), (0.75, 15), (1.0, 20)]:
        keep = select_slices(scores, "budget", 0.9, budget)
        assert len(keep) == k and keep.tolist() == list(range(20 - k, 20))
    assert select_slices(np.zeros(0, np.float32), "threshold", 0.1, 0).tolist() == []
    try:
        select_slices(scores, "bogus", 0.5, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("未知的 mode 应抛 ValueError")


def test_random_control_is_matched_and_reproducible():
    rng = np.random.default_rng(0)
    keep = random_slices(10, 3, rng)
    assert len(keep) == 3 and len(set(keep.tolist())) == 3 and keep.tolist() == sorted(keep.tolist())
    assert random_slices(10, 3, np.random.default_rng(0)).tolist() == keep.tolist()
    assert random_slices(10, 0, np.random.default_rng(0)).tolist() == []
    assert len(random_slices(10, 99, np.random.default_rng(0))) == 10


# --------------------------------------------------------------- 类别映射与数据注册表
def test_category_maps_and_dataset_registry():
    assert COCO_TO_EVAL == {0: 1, 2: 2, 5: 2, 7: 2}
    assert VISDRONE_TO_EVAL == {1: 1, 2: 1, 4: 2, 5: 2, 6: 2, 9: 2}
    assert IGNORE_VISDRONE == {0, 11}
    assert DOTA_TO_EVAL == {10: 1, 9: 1, 1: 2, 0: 3}
    assert DOTA_COCO_TO_EVAL == {2: 1, 5: 1, 7: 1, 8: 2, 4: 3}

    v = datasets.get("visdrone")
    assert v["exclude_coco_ids"] == [i for i in range(80) if i not in COCO_TO_EVAL]
    assert 0 not in v["exclude_coco_ids"] and v["max_dets"] == 500
    assert datasets.get("dota")["max_dets"] == 2000
    assert set(datasets.DATASETS) == {"visdrone", "dota", "sparse4k"}  # sparse4k = 受控实验画布
    sp = datasets.get("sparse4k")
    assert sp["coco_to_eval"] == COCO_TO_EVAL and sp["gt"].name == "coco_eval.json"

    cfg = GlanceConfig()
    assert cfg.det_prior == "noisyor" and cfg.img_prior == "edge" and cfg.threshold == 0.9


def test_visdrone_convert_marks_ignores():
    import cv2
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "images").mkdir()
        (root / "annotations").mkdir()
        cv2.imwrite(str(root / "images" / "a.jpg"), np.zeros((40, 60, 3), np.uint8))
        (root / "annotations" / "a.txt").write_text(
            "1,1,10,10,1,1,0,0\n"      # pedestrian -> person
            "5,5,10,10,1,4,0,0\n"      # car        -> vehicle
            "0,0,50,50,0,0,0,0\n"      # ignored region -> 两个类别各一条 iscrowd
            "9,9,5,5,1,3,0,0\n"        # bicycle：不在评测类别内，丢弃
            "2,2,4,4,1,11,0,0\n"       # others -> iscrowd
        )
        out = root / "coco.json"
        convert(root, out)
        coco = json.loads(out.read_text())

    # 数据目录已删除，下面只对解析出来的内容做断言
    assert coco["categories"] == [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]
    assert [a["category_id"] for a in coco["annotations"] if not a["iscrowd"]] == [1, 2]
    assert sum(a["iscrowd"] for a in coco["annotations"]) == 4  # ignored region + others，各两类
    assert coco["images"] == [{"id": 1, "file_name": "a.jpg", "width": 60, "height": 40}]


# --------------------------------------------------------------- 热图变体（撒点 + 一次卷积）
def test_coarse_heatmap_conserves_mass_and_is_soft():
    shape = (64, 64)
    boxes = np.array([[20 + 4 * i, 45, 24 + 4 * i, 49] for i in range(8)], np.float32)
    raw = coarse_heatmap(shape, boxes, np.full(8, 0.1, np.float32))
    assert abs(float(raw.sum()) - 0.8) < 0.02, "卷积应保质量：全图之和 ≈ Σc"
    assert coarse_heatmap(shape, np.zeros((0, 4), np.float32), np.zeros(0, np.float32)).max() == 0.0
    assert abs(float(coarse_heatmap(shape, boxes, np.full(8, 0.1, np.float32), normalize=True).max()) - 1.0) < 1e-6

    slices = [[0, 0, 32, 32], [32, 0, 64, 32]]
    edge = np.array([[30, 10, 34, 14]], np.float32)  # 框中心 (32,12) 正好压在切片边界上
    sc = np.array([0.9], np.float32)
    hm, _, _ = heatmap_prior(edge, sc, (64, 64), slices, map_size=64, sigma=3.0)
    assert hm[0] > 0 and hm[1] > 0, "软边界：相邻两片都分到证据"
    hard = detection_prior(edge, sc, slices, 0)
    assert hard[0] == 0.0 and hard[1] > 0, "硬边界（margin=0）：只有一片拿得到"


def test_heatmap_prior_matches_noisy_or_when_sparse():
    slices = [[0, 0, 256, 256], [256, 0, 512, 256]]
    boxes = np.array([_box(120 + 3 * i, 120) for i in range(20)], np.float32)  # 20 个 0.05 的弱框
    sc = np.full(20, 0.05, np.float32)
    s_or = float(detection_prior(boxes, sc, slices, 16)[0])
    s_hm = float(heatmap_prior(boxes, sc, (256, 512), slices, map_size=512, sigma=6.0)[0][0])
    assert abs(s_or - (1 - 0.95 ** 20)) < 1e-6
    assert abs(s_hm - s_or) < 0.02, "稀疏弱证据下 1 − exp(−Σc) ≈ 1 − Π(1 − c_j)"
    assert s_hm > 0.6 > s_hm - 0.1  # 弱证据同样被累积（对比“取最大”只能得到 0.05）


def test_evidence_mass_is_absolute_and_margin_free():
    slices = [[0, 0, 256, 256], [256, 0, 512, 256]]
    boxes = np.array([_box(100, 100), _box(150, 100), _box(200, 100)], np.float32)
    m = evidence_mass(boxes, np.full(3, 0.5, np.float32), slices)
    assert abs(float(m[0]) - 1.5) < 1e-6 and m[1] == 0.0, "证据量是未归一化的绝对量 Σc"
    assert float(m[0]) > 1.0, "可以超过 1，这正是它能跨图统一设阈值的原因"
    edge = evidence_mass(np.array([_box(256, 100)], np.float32), np.array([0.9], np.float32), slices)
    assert edge[0] == 0.0 and abs(float(edge[1]) - 0.9) < 1e-6, "证据量不设 margin"
    assert evidence_mass(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), slices).tolist() == [0.0, 0.0]


def test_select_evidence_mode_and_min_slices():
    scores = np.array([0.1, 0.9, 0.2, 0.8], np.float32)
    ev = np.array([0.0, 2.0, 0.3, 1.5], np.float32)
    assert select_slices(scores, "evidence", 0, 0, ev, tau=1.0).tolist() == [1, 3]
    assert select_slices(scores, "evidence", 0, 0, ev, tau=0.25).tolist() == [1, 2, 3]
    assert select_slices(scores, "evidence", 0, 0, ev, tau=5.0).tolist() == [1], "保底：一片都不剩时按分数补 1 片"
    assert select_slices(scores, "evidence", 0, 0, ev, tau=0.0, min_slices=3).tolist() == [0, 1, 2, 3]
    assert select_slices(np.full(5, 0.1, np.float32), "threshold", 0.9, 0, min_slices=2).tolist() == [0, 1]
    assert select_slices(scores, "threshold", 0.5, 0).tolist() == [1, 3], "默认 min_slices=0，行为不变"
    try:
        select_slices(scores, "evidence", 0, 0, None, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("evidence 模式缺 evidence 应抛 ValueError")


# --------------------------------------------------------------- 业务规则层（违停）
def test_illegal_parking_rules(tmp_path=None):
    zone = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    dets = np.array([[10, 10, 30, 40, 0.9, 2],    # 车辆：着地点 (20,40) 在禁停区内 → 命中
                     [90, 90, 110, 130, 0.8, 2],  # 车辆：着地点 (100,130) 在区外 → 不命中
                     [10, 10, 30, 40, 0.9, 0]],   # person：规则层不判 → 不命中
                    np.float32)
    assert ground_point([10, 10, 30, 40]) == (20.0, 40.0)
    hits = find_illegal_parking(dets, [zone], vehicle_ids=[2])
    assert len(hits) == 1 and hits[0]["ground_point"] == (20.0, 40.0) and hits[0]["zone"] == 0
    assert find_illegal_parking(dets, [], vehicle_ids=[2]) == []
    assert find_illegal_parking(np.zeros((0, 6), np.float32), [zone], [2]) == []

    one = np.array([[10, 10, 30, 40, 0.9, 2]], np.float32)
    zs = suggest_zone(one, [2], (200, 200), rel=0.1)
    assert len(zs) == 1 and zs[0].shape == (4, 2)
    assert cv2.pointPolygonTest(zs[0].astype(np.float32), ground_point(one[0, :4]), False) >= 0
    assert suggest_zone(np.zeros((0, 6), np.float32), [2], (200, 200)) == []

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "zones.json"
        p.write_text('[{"points": [[0, 0], [10, 0], [10, 10]]}, [[1, 1], [2, 1], [2, 2], [1, 2]]]')
        zones = load_zones(p)
    assert len(zones) == 2 and zones[0].shape == (3, 2)


# --------------------------------------------------------------- 评测脚本的先验名解析
def test_prior_name_parsing():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_eval import ABLATIONS, prior_kinds

    assert prior_kinds("det") == ("noisyor", None)
    assert prior_kinds("edge") == (None, "edge")
    assert prior_kinds("det+spectral") == ("noisyor", "spectral")
    assert prior_kinds("uncertain+edge") == ("uncertain", "edge")
    assert prior_kinds("heatmap") == ("heatmap", None)
    # 消融表里每个名字都必须能解析（否则 sim 扫到一半才会报错）
    assert all(prior_kinds(name) == (k, i) for name, k, i in ABLATIONS)
    try:
        prior_kinds("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("未知先验名应抛 ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
