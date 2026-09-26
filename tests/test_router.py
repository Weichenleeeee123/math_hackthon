"""可学习稀疏路由器 / bootstrap / 设备回退的单元检查（不需要检测器、GPU 或数据集）。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahi.slicing import get_slice_bboxes  # noqa: E402

from glance_sahi import router as RT  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.saliency import detection_prior, fuse, image_prior_map, region_prior  # noqa: E402

torch = pytest.importorskip("torch")


def _rec(seed=0, h=1080, w=1920, n=40):
    rng = np.random.default_rng(seed)
    slices = get_slice_bboxes(h, w, 512, 512, False, 0.2, 0.2)
    xy = rng.uniform([0, 0], [w - 20, h - 20], (n, 2))
    g = np.concatenate([xy, xy + rng.uniform(5, 40, (n, 2)), rng.uniform(0.01, 0.9, (n, 1)),
                        rng.choice([0, 2], (n, 1))], 1).astype(np.float32)
    return {"id": seed, "hw": (h, w), "slices": slices, "glance": g,
            "prior_edge": rng.uniform(0, 1, len(slices)).astype(np.float32),
            "slice_preds": [np.zeros((0, 6), np.float32) for _ in slices]}


# ---------------------------------------------------------------- 特征
def test_features_shape_finite_deterministic():
    cfg = GlanceConfig()
    r = _rec()
    X = RT.features_from_rec(r, cfg)
    assert X.shape == (len(r["slices"]), len(RT.FEATURE_NAMES))
    assert np.isfinite(X).all()
    assert np.array_equal(X, RT.features_from_rec(r, cfg))
    # 第 0 列就是 noisy-OR 检测先验，fused 列就是手工门分数
    assert np.allclose(X[:, 0], detection_prior(r["glance"][:, :4], r["glance"][:, 4], r["slices"], cfg.det_margin))
    fused = fuse(X[:, 0], r["prior_edge"], cfg.img_weight)
    assert np.allclose(X[:, RT.FEATURE_NAMES.index("fused")], fused)


def test_features_empty_glance():
    r = _rec()
    r["glance"] = np.zeros((0, 6), np.float32)
    X = RT.features_from_rec(r, GlanceConfig())
    assert np.isfinite(X).all()
    for f in ("det_noisyor", "det_max", "log_mass", "log_n_weak", "det_heatmap"):
        assert (X[:, RT.FEATURE_NAMES.index(f)] == 0).all()


def test_online_features_equal_offline():
    """predict.score_slices(learned) 与离线 features_from_rec 走同一条特征路径。"""
    cfg = GlanceConfig()
    rng = np.random.default_rng(1)
    img = (rng.random((700, 1300, 3)) * 255).astype(np.uint8)
    r = _rec(h=700, w=1300)
    sal, scale = image_prior_map(img, "edge", cfg.img_map_size)
    r["prior_edge"] = region_prior(sal, scale, r["slices"])
    a = RT.slice_features(r["glance"], (700, 1300), r["slices"], r["prior_edge"], cfg)
    assert np.array_equal(a, RT.features_from_rec(r, cfg))


# ---------------------------------------------------------------- 标签
def test_gain_label_counts_only_new_hits():
    r = _rec(n=0)
    r["glance"] = np.array([[100, 100, 120, 120, 0.8, 0]], np.float32)   # 扫视已命中 GT#0
    k = 0
    r["slice_preds"][k] = np.array([[100, 100, 120, 120, 0.9, 0]], np.float32)
    targets = [(110.0, 110.0, 10.0, 1, 1.0)]
    y = RT.utility_labels(r, targets, {0: 1, 2: 2}, 0.05, "gain")
    assert y.sum() == 0, "扫视已经检出的目标，切片再检出不算增量"
    targets.append((300.0, 300.0, 10.0, 1, 1.0))
    r["slice_preds"][k] = np.array([[295, 295, 305, 305, 0.9, 0]], np.float32)
    y = RT.utility_labels(r, targets, {0: 1, 2: 2}, 0.05, "gain")
    assert y[k] == 1 and y.sum() == 1
    # 类别不同不算命中
    r["slice_preds"][k][:, 5] = 2
    assert RT.utility_labels(r, targets, {0: 1, 2: 2}, 0.05, "gain").sum() == 0


def test_gt_label_is_center_in_slice():
    r = _rec(n=0)
    y = RT.utility_labels(r, [(10.0, 10.0, 6.0, 1, 1.0)], {0: 1}, 0.05, "gt")
    assert y[0] == 1 and y.sum() == 1


# ---------------------------------------------------------------- 训练 / 推理
def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(RT.FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 3] - 0.3 * X[:, 7] + 0.3 * rng.normal(size=n) > 0).astype(np.float32)
    return X, y


def test_router_learns_separable_and_numpy_equals_torch():
    X, y = _toy()
    p = RT.train_router(X, y, hidden=16, epochs=200)
    nr = RT.NumpyRouter([p])
    s = nr.predict_proba(X)
    assert RT.roc_auc(y, s) > 0.9
    assert np.allclose(s, RT.torch_forward(p, X), atol=1e-5)
    lin = RT.train_router(X, y, hidden=0, epochs=200)
    assert RT.roc_auc(y, RT.NumpyRouter([lin]).predict_proba(X)) > 0.9


def test_sparsity_penalty_lowers_mean_activation():
    X, y = _toy()
    p0 = RT.NumpyRouter([RT.train_router(X, y, hidden=16, alpha=0.0, epochs=200)]).predict_proba(X).mean()
    p1 = RT.NumpyRouter([RT.train_router(X, y, hidden=16, alpha=0.5, epochs=200)]).predict_proba(X).mean()
    assert p1 < p0


def test_save_load_roundtrip(tmp_path):
    X, y = _toy(400)
    ms = [RT.train_router(X, y, hidden=8, epochs=50, seed=s) for s in range(2)]
    path = tmp_path / "r.json"
    RT.save_router(ms, {"default_threshold": 0.3}, path)
    r = RT.load_router(str(path))
    assert np.allclose(r.predict_proba(X), RT.NumpyRouter(ms).predict_proba(X))
    assert r.default_threshold == 0.3
    d = json.loads(path.read_text())
    d["feature_names"] = d["feature_names"][:-1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        RT.load_router(str(bad))


# ---------------------------------------------------------------- 路由
def test_route_modes():
    p = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    assert list(RT.route(p, "topk", rho=0.4)) == [1, 3]
    assert list(RT.route(p, "topk", rho=1.0)) == [0, 1, 2, 3, 4], "ρ=1 = 稠密（SAHI 全激活）"
    assert list(RT.route(p, "global", thr=0.5)) == [1, 2, 3]
    assert list(RT.route(p, "matched", k=1)) == [1]
    assert len(RT.route(p, "matched", k=0)) == 0
    q = np.random.default_rng(0).random(10000)
    mu = RT.global_threshold(q, 0.3)
    assert abs((q >= mu).mean() - 0.3) < 0.01


def test_gini_and_activation_stats():
    assert RT.gini(np.ones(10)) == pytest.approx(0.0)
    assert RT.gini(np.array([0, 0, 0, 1.0])) == pytest.approx(0.75)
    st = RT.activation_stats([0.1, 0.5, 0.9], [1, 5, 9])
    assert st["spearman_rate_ngt"] == pytest.approx(1.0)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert RT.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert RT.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert RT.average_precision(y, [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_split_sequence_no_leak():
    ims = [{"file_name": f"{s:07d}_{f:05d}_d_0.jpg"} for s in range(6) for f in range(4)]
    fit, hold = RT.split_images(ims, "sequence")
    seq = lambda i: ims[i]["file_name"][:7]
    assert not ({seq(i) for i in fit} & {seq(i) for i in hold})
    assert sorted(fit + hold) == list(range(len(ims)))
    fit, hold = RT.split_images([{"file_name": "P0001.png"}] * 4, "sequence")   # 无序列 → 奇偶
    assert fit == [1, 3] and hold == [0, 2]


# ---------------------------------------------------------------- bootstrap（与 pycocotools 逐位一致）
def _tiny_coco(tmp_path):
    rng = np.random.default_rng(0)
    images, anns, dets = [], [], []
    aid = 1
    for i in range(1, 9):
        images.append({"id": i, "width": 400, "height": 300, "file_name": f"{i}.jpg"})
        for _ in range(6):
            x, y = rng.uniform(0, 350), rng.uniform(0, 250)
            w, h = rng.uniform(8, 60, 2)
            c = int(rng.integers(1, 3))
            anns.append({"id": aid, "image_id": i, "category_id": c, "bbox": [x, y, w, h], "area": w * h,
                         "iscrowd": 0})
            aid += 1
            if rng.random() < 0.8:
                dets.append({"image_id": i, "category_id": c, "score": float(rng.random()),
                             "bbox": [x + rng.normal(0, 2), y + rng.normal(0, 2), w, h]})
        dets.append({"image_id": i, "category_id": 1, "score": float(rng.random()), "bbox": [5, 5, 20, 20]})
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"images": images, "annotations": anns,
                             "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}))
    return p, dets, [im["id"] for im in images]


def test_bootstrap_identity_equals_pycocotools(tmp_path):
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from glance_sahi.evalboot import CachedEval, paired_bootstrap

    gt_path, dets, ids = _tiny_coco(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = ids
        ev.params.maxDets = [1, 100, 500]
        ev.evaluate(), ev.accumulate(), ev.summarize()
    E = CachedEval(gt_path, dets, ids, 500)
    assert E.ap() == pytest.approx(ev.stats[0], abs=1e-9)
    assert E.ap(None, "small") == pytest.approx(ev.stats[3], abs=1e-9)
    # 重复图像的多重集：同一张图出现两次 ≠ 出现一次（pycocotools 做不到，这里必须能）
    assert np.isfinite(E.ap(np.array([0, 0, 1, 2]))), "多重集重采样应可计算"
    E2 = CachedEval(gt_path, dets[::2], ids, 500)
    ci, delta, _ = paired_bootstrap({"a": E, "b": E2}, len(ids), B=50, seed=0, ref="a")
    ci2, _, _ = paired_bootstrap({"a": E, "b": E2}, len(ids), B=50, seed=0, ref="a")
    assert ci.equals(ci2), "同种子可复现"
    row = ci[(ci.method == "a") & (ci.area == "all")].iloc[0]
    assert row.lo <= row.AP <= row.hi


# ---------------------------------------------------------------- 设备回退
def test_resolve_device(monkeypatch):
    from glance_sahi.detector import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda:0"
    assert resolve_device("cpu") == "cpu"


def test_default_config_unchanged():
    cfg = GlanceConfig()
    assert cfg.scorer == "fusion" and cfg.threshold == 0.9 and cfg.img_weight == 0.3
