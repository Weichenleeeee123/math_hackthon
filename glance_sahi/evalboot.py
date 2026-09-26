"""图像级配对 bootstrap 的 COCO AP 置信区间（REPORT 3.16）。

pycocotools 的 imgIds 会去重，没法直接“有放回重采样”。做法：
  1) 每个方法只跑一次 COCOeval.evaluate()，缓存逐图的 evalImgs（匹配结果与 IoU 无关于重采样）；
  2) 自己实现 accumulate：把重采样得到的图像多重集的 evalImgs 拼起来（重复的图就重复拼），
     逐类别、逐 IoU 阈值算插值 PR 曲线，AP = 101 点精度的平均 —— 与 pycocotools 逐行同构；
  3) 所有方法在同一次重采样里共享同一组图像下标（配对），差值 Δ 的方差因此大幅降低。

恒等重采样（每张图恰好一次）的结果必须与 coco_eval 完全一致（tests 里有检查）。
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

AREA_INDEX = {"all": 0, "small": 1, "medium": 2, "large": 3}
# pycocotools 的 summarize 口径：stats[0]（AP）用 maxDets=100，stats[3..5]（APs/m/l）用 maxDets[-1]。
# 为了与 run_eval.coco_eval 报的数完全一致，这里逐图截断到同样的上限。
AP_CAP_ALL = 100


class CachedEval:
    """一个方法的逐图匹配结果；ap(img_idx, area) 在任意图像多重集上重算 AP。"""

    def __init__(self, gt_path, dets: list, img_ids: list, max_dets: int, coco_gt=None):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        self.img_ids = list(img_ids)
        with contextlib.redirect_stdout(io.StringIO()):
            gt = coco_gt if coco_gt is not None else COCO(str(gt_path))
            ev = COCOeval(gt, gt.loadRes(dets) if dets else gt.loadRes([_dummy(gt, self.img_ids[0])]), "bbox")
            ev.params.imgIds = self.img_ids
            ev.params.maxDets = [1, 100, max_dets]
            ev.evaluate()
        p = ev.params
        self.recThrs = p.recThrs
        self.n_iou = len(p.iouThrs)
        self.cat_ids = p.catIds
        I, A = len(p.imgIds), len(p.areaRng)
        # 预打包：per[(k, a)][i] = (scores, dtm(T,D), dtIg(T,D), n_pos_gt) 或 None
        self.order = {iid: i for i, iid in enumerate(p.imgIds)}   # pycocotools 内部会排序去重
        self.per = {}
        for k in range(len(p.catIds)):
            for a in range(A):
                rows = []
                for i in range(I):
                    e = ev.evalImgs[k * A * I + a * I + i]
                    if e is None:
                        rows.append(None)
                        continue
                    sc = np.asarray(e["dtScores"][:max_dets], float)
                    rows.append((sc, e["dtMatches"][:, :max_dets] > 0, e["dtIgnore"][:, :max_dets].astype(bool),
                                 int((np.asarray(e["gtIgnore"]) == 0).sum())))
                self.per[(k, a)] = rows
        self.empty = not dets
        self.max_dets = max_dets

    def idx_of(self, img_ids) -> np.ndarray:
        return np.array([self.order[i] for i in img_ids], int)

    def ap(self, img_idx=None, area: str = "all") -> float:
        """img_idx：pycocotools 内部顺序下的图像下标多重集（None = 全部各一次）。"""
        if self.empty:
            return 0.0
        a = AREA_INDEX[area]
        cap = AP_CAP_ALL if area == "all" else self.max_dets
        n = len(self.order)
        idx = np.arange(n) if img_idx is None else np.asarray(img_idx, int)
        vals = []
        for k in range(len(self.cat_ids)):
            rows = [self.per[(k, a)][i] for i in idx]
            rows = [r for r in rows if r is not None]
            if not rows:
                continue
            npig = sum(r[3] for r in rows)
            if npig == 0:
                continue
            sc = np.concatenate([r[0][:cap] for r in rows])
            o = np.argsort(-sc, kind="mergesort")
            dtm = np.concatenate([r[1][:, :cap] for r in rows], axis=1)[:, o]
            dtig = np.concatenate([r[2][:, :cap] for r in rows], axis=1)[:, o]
            tps = np.cumsum(dtm & ~dtig, axis=1, dtype=float)
            fps = np.cumsum(~dtm & ~dtig, axis=1, dtype=float)
            for t in range(self.n_iou):
                tp, fp = tps[t], fps[t]
                q = np.zeros(len(self.recThrs))
                if len(tp):
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = np.maximum.accumulate(pr[::-1])[::-1]
                    inds = np.searchsorted(rc, self.recThrs, side="left")
                    ok = inds < len(pr)
                    q[ok] = pr[inds[ok]]
                vals.append(q)
        return float(np.mean(vals)) if vals else -1.0


def _dummy(gt, img_id):
    cid = gt.getCatIds()[0]
    return {"image_id": img_id, "category_id": cid, "bbox": [0, 0, 1, 1], "score": 1e-9}


def paired_bootstrap(evals: dict, n_img: int, B: int = 1000, seed: int = 0, ref: str = "sahi_uniform",
                     areas=("all", "small"), group_avg: dict | None = None):
    """evals: {方法名: CachedEval}（同一组图像）。group_avg: {新名: [方法名...]}，每次重采样取组内平均
    （如 3 个随机种子）。返回 (点估计+CI 表, Δ 表, 原始重采样样本 dict)。"""
    import pandas as pd

    rng = np.random.default_rng(seed)
    names = list(evals)
    group_avg = group_avg or {}
    samples = {(m, a): np.empty(B) for m in names for a in areas}
    point = {(m, a): evals[m].ap(None, a) for m in names for a in areas}
    for b in range(B):
        idx = rng.integers(0, n_img, n_img)
        for m in names:
            for a in areas:
                samples[(m, a)][b] = evals[m].ap(idx, a)
    for g, members in group_avg.items():
        for a in areas:
            samples[(g, a)] = np.mean([samples[(m, a)] for m in members], axis=0)
            point[(g, a)] = float(np.mean([point[(m, a)] for m in members]))
    all_names = [m for m in names if not any(m in v for v in group_avg.values())] + list(group_avg)

    rows = []
    for m in all_names:
        for a in areas:
            s = samples[(m, a)]
            rows.append(dict(method=m, area=a, AP=point[(m, a)], lo=float(np.percentile(s, 2.5)),
                             hi=float(np.percentile(s, 97.5)), se=float(s.std(ddof=1))))
    drows = []
    for m in all_names:
        for r in ([ref] if isinstance(ref, str) else ref):
            if m == r or (r, areas[0]) not in samples:
                continue
            for a in areas:
                d = samples[(m, a)] - samples[(r, a)]
                drows.append(dict(method=m, vs=r, area=a, delta=point[(m, a)] - point[(r, a)],
                                  lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                                  p_le0=float((d <= 0).mean())))
    return pd.DataFrame(rows), pd.DataFrame(drows), samples
