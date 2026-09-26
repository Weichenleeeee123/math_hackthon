"""glance_sahi.bootstrap：恒等重采样必须与 run_eval.coco_eval 逐位一致；配对差值的基本性质。"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("pycocotools")

import run_eval as R  # noqa: E402
from glance_sahi.bootstrap import PreparedEval, paired_bootstrap  # noqa: E402


def _synthetic(tmp_path, n_img=12, seed=0):
    rng = np.random.default_rng(seed)
    images, anns, dets_good, dets_bad = [], [], [], []
    aid = 1
    for i in range(1, n_img + 1):
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 800, "height": 600})
        for _ in range(rng.integers(3, 12)):
            w, h = rng.uniform(8, 60, 2)
            x, y = rng.uniform(0, 700), rng.uniform(0, 500)
            cat = int(rng.integers(1, 3))
            anns.append({"id": aid, "image_id": i, "category_id": cat, "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
            jit = rng.normal(0, 2, 4)
            if rng.random() < 0.8:
                dets_good.append({"image_id": i, "category_id": cat, "score": float(rng.uniform(0.3, 1)),
                                  "bbox": [x + jit[0], y + jit[1], w + jit[2], h + jit[3]]})
            if rng.random() < 0.5:
                dets_bad.append({"image_id": i, "category_id": cat, "score": float(rng.uniform(0.05, 0.6)),
                                 "bbox": [x + 3 * jit[0], y + 3 * jit[1], w, h]})
        dets_good.append({"image_id": i, "category_id": 1, "score": 0.2, "bbox": [1, 1, 20, 20]})  # 误检
    gt = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    return path, [im["id"] for im in images], dets_good, dets_bad


def _coco(path):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def test_identity_matches_coco_eval(tmp_path):
    path, ids, good, _ = _synthetic(tmp_path)
    ref = R.coco_eval(path, good, ids)
    got = PreparedEval(_coco(path), good, ids, R.DS["max_dets"]).ap()
    assert got["AP"] == pytest.approx(ref["AP"], abs=1e-12)
    assert got["AP50"] == pytest.approx(ref["AP50"], abs=1e-12)
    assert got["APs"] == pytest.approx(ref["APs"], abs=1e-12)


def test_resample_is_repeatable_and_order_free(tmp_path):
    path, ids, good, _ = _synthetic(tmp_path)
    pe = PreparedEval(_coco(path), good, ids, 500)
    base = pe.ap()
    assert pe.ap(list(reversed(ids))) == base  # AP 与图的顺序无关
    sample = [ids[0], ids[0], ids[3], ids[5], ids[5], ids[5]]
    a1 = pe.ap(sample)
    pe.ap()  # 中间插一次别的序列，不应污染下一次
    assert pe.ap(sample) == a1


def test_paired_bootstrap_self_difference_is_zero(tmp_path):
    path, ids, good, bad = _synthetic(tmp_path)
    gt = _coco(path)
    prepared = {"good": PreparedEval(gt, good, ids, 500), "bad": PreparedEval(gt, bad, ids, 500),
                "empty": PreparedEval(gt, [], ids, 500)}
    times = {"good": np.full(len(ids), 0.2), "bad": np.full(len(ids), 0.1), "empty": np.full(len(ids), 0.05)}
    rows = {(r["ref"], r["method"]): r for r in paired_bootstrap(prepared, times, ["good", "bad"], n_boot=30)}
    same = rows["good", "good"]
    assert same["dAP"] == 0 and same["dAP_lo"] == 0 and same["dAP_hi"] == 0 and same["time_ratio"] == 1
    worse = rows["good", "bad"]
    assert worse["dAP"] < 0 and worse["dAP_lo"] <= worse["dAP"] <= worse["dAP_hi"]
    assert worse["time_ratio"] == pytest.approx(0.5)
    assert rows["good", "empty"]["AP"] == 0
    assert rows["bad", "good"]["dAP"] == pytest.approx(-worse["dAP"])
