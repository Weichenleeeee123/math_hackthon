"""glance_sahi.resroute：按图拼接的 AP 必须与单方法 AP 一致（容差 1e-12）；逐图拼接等于把检测结果拼起来再评。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("pycocotools")

from glance_sahi import resroute as RR  # noqa: E402
from glance_sahi.bootstrap import PreparedEval  # noqa: E402
from test_bootstrap import _coco, _synthetic  # noqa: E402


def test_mixed_eval_equals_single_method_and_concatenated_dets(tmp_path):
    path, ids, good, bad = _synthetic(tmp_path)
    gt = _coco(path)
    prep = {"good": PreparedEval(gt, good, ids, 500), "bad": PreparedEval(gt, bad, ids, 500)}
    mixed = RR.MixedEval(prep)
    # 向量化 accumulate 与 pycocotools 只差求平均的浮点顺序（~1e-16）
    for m in ("good", "bad"):
        got, ref = mixed.ap({i: m for i in ids}), prep[m].ap()
        for key in ("AP", "AP50", "APs"):
            assert got[key] == pytest.approx(ref[key], abs=1e-12)
    # 前一半图用 good、后一半用 bad = 把两部分检测结果拼起来整体评
    half = set(ids[: len(ids) // 2])
    choice = {i: "good" if i in half else "bad" for i in ids}
    dets = [d for d in good if d["image_id"] in half] + [d for d in bad if d["image_id"] not in half]
    assert mixed.ap(choice)["AP"] == pytest.approx(PreparedEval(gt, dets, ids, 500).ap()["AP"], abs=1e-12)


def test_marginal_gain_sign_and_zero_for_same_method(tmp_path):
    path, ids, good, bad = _synthetic(tmp_path)
    gt = _coco(path)
    mixed = RR.MixedEval({"good": PreparedEval(gt, good, ids, 500), "bad": PreparedEval(gt, bad, ids, 500)})
    assert np.all(RR.marginal_gain(mixed, ids, "good", "good") == 0)
    assert RR.marginal_gain(mixed, ids, "bad", "good").mean() > 0  # 换成更好的方法，平均边际收益为正


def test_glance_features_and_ridge():
    d = np.array([[0, 0, 10, 10, 0.9, 0], [100, 100, 104, 104, 0.1, 2]], np.float32)
    x = RR.glance_features(d, (1000, 2000))
    assert x.shape == (len(RR.FEATURE_NAMES),) and np.isfinite(x).all()
    assert x[RR.FEATURE_NAMES.index("frac_app_lt8")] == 1.0  # 640/2000 缩放后两个框都 < 8 像素
    assert np.isfinite(RR.glance_features(np.zeros((0, 6), np.float32), (100, 100))).all()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 2 * X[:, 0] - X[:, 2] + 0.5
    assert np.allclose(RR.Ridge(1e-6).fit(X, y).predict(X), y, atol=1e-4)
    assert list(RR.choose(np.array([[0, 1.0], [0, 0.1]]), np.array([[0, 5.0], [0, 5.0]]), 0.1)) == [1, 0]
