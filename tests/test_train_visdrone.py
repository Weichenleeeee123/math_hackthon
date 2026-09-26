"""train_visdrone 的数据转换：类别合并、忽略区域抹灰、裁块坐标与 YOLO 归一化。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_visdrone as T  # noqa: E402


def test_parse_merges_classes_and_collects_ignores():
    text = "10,20,30,40,1,1,0,0\n5,5,8,8,1,2,0,0\n100,100,50,20,1,4,0,0\n1,1,9,9,0,4,0,0\n" \
           "0,0,64,64,0,0,0,0\n7,7,5,5,1,11,0,0\n3,3,6,6,1,10,0,0\n9,9,0,4,1,1,0,0\n"
    boxes, ignore = T.parse_annotation(text)
    assert boxes == [(10, 20, 30, 40, 0), (5, 5, 8, 8, 0), (100, 100, 50, 20, 1)]  # 摩托(10)丢弃，零宽丢弃
    assert ignore == [(1, 1, 9, 9), (0, 0, 64, 64), (7, 7, 5, 5)]  # score=0、ignored、others


def test_yolo_lines_clip_and_normalize():
    boxes = [(100, 100, 40, 20, 1), (630, 300, 20, 20, 0), (700, 700, 10, 10, 0)]
    lines = T.yolo_lines(boxes, 0, 0, 640, 640, 1.0)
    c, cx, cy, w, h = map(float, lines[0].split())
    assert c == 1 and np.isclose(cx, 120 / 640) and np.isclose(cy, 110 / 640)
    assert np.isclose(w, 40 / 640) and np.isclose(h, 20 / 640)
    assert len(lines) == 2  # 第二个框裁掉一半（=50% ≥ 40%）保留；第三个在窗口外
    assert np.isclose(float(lines[1].split()[3]), 10 / 640)
    half = T.yolo_lines([(0, 0, 100, 50, 0)], 0, 0, 1000, 500, 0.5)[0].split()
    assert np.allclose([float(v) for v in half[1:]], [0.05, 0.05, 0.1, 0.1])


def test_make_samples_masks_ignored_and_sizes():
    rng = np.random.default_rng(0)
    img = np.full((765, 1360, 3), 200, np.uint8)
    samples = T.make_samples(img, [(600, 300, 30, 30, 1)], [(0, 0, 50, 50)], rng)
    (n1, full, l1), (n2, crop, l2) = samples
    assert (n1, n2) == ("full", "crop")
    assert max(full.shape[:2]) == 640 and crop.shape[:2] == (640, 640)
    assert full[2, 2, 0] == T.GRAY and img[2, 2, 0] == 200  # 抹灰不改原图
    assert len(l1) == 1 and len(l2) == 1  # 80% 概率以目标为中心，裁块里应含这个目标（种子 0）
