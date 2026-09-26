"""按图配对的 bootstrap：给"方法 A 与方法 B 的 AP 差 / 耗时比"一个误差范围。

COCO AP 是整个数据集上的量，不能逐图相加，所以做法是：每个方法只做一次 COCOeval.evaluate()
（逐图匹配，最贵的一步），再把逐图结果按重采样的图序列重新 accumulate。所有方法共用同一组
重采样下标，差值就是配对的。

恒等重采样（原图序列）得到的 AP 与 run_eval.coco_eval 完全相同（tests/test_bootstrap.py 检查）。
"""

import contextlib
import io

import numpy as np


def _area_index(params, label):
    return params.areaRngLbl.index(label)


class PreparedEval:
    """一个方法在一份真值上做完 evaluate() 的结果；ap(ids) 在任意图序列（可重复）上重算 AP。"""

    def __init__(self, gt, dets, img_ids, max_dets):
        from pycocotools.cocoeval import COCOeval

        self.img_ids = list(img_ids)
        self.empty = not dets  # 与 run_eval.coco_eval 一致：没有任何检测时 AP 记 0
        if self.empty:
            return
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dets), "bbox")
            ev.params.imgIds = self.img_ids
            ev.params.maxDets = [1, 100, max_dets]
            ev.evaluate()
        self.ev = ev
        p = ev.params
        n_img, n_area = len(p.imgIds), len(p.areaRng)
        self.table = {(k, a, img): ev.evalImgs[k * n_area * n_img + a * n_img + i]
                      for k in range(len(p.catIds)) for a in range(n_area) for i, img in enumerate(p.imgIds)}
        self.a_all, self.a_small = _area_index(p, "all"), _area_index(p, "small")

    def ap(self, ids=None) -> dict:
        """在图序列 ids（默认原序列）上重算 AP / AP50 / APs（与 COCOeval.summarize 同口径，×1）。"""
        if self.empty:
            return dict(AP=0.0, AP50=0.0, APs=0.0)
        ids = self.img_ids if ids is None else list(ids)
        ev, p = self.ev, self.ev.params
        n_area = len(p.areaRng)
        p.imgIds = ev._paramsEval.imgIds = ids
        ev.evalImgs = [self.table[k, a, img] for k in range(len(p.catIds)) for a in range(n_area) for img in ids]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.accumulate()
        prec = ev.eval["precision"]  # [T, R, K, A, M]
        m = len(p.maxDets) - 1

        def mean_valid(x):
            x = x[x > -1]
            return float(x.mean()) if x.size else -1.0

        t50 = int(np.where(np.isclose(p.iouThrs, 0.5))[0][0])
        return dict(AP=mean_valid(prec[:, :, :, self.a_all, m]),
                    AP50=mean_valid(prec[t50, :, :, self.a_all, m]),
                    APs=mean_valid(prec[:, :, :, self.a_small, m]))


def paired_bootstrap(prepared: dict, times: dict | None, refs, n_boot: int = 200, seed: int = 0,
                     metrics=("AP", "APs")) -> list[dict]:
    """prepared: {方法名: PreparedEval}（同一份真值、同一图序列）；times: {方法名: 逐图秒数 ndarray}。

    每个方法、每个参照 ref 各出一行：点估计与 95% 百分位区间，
    ΔAP = 方法 − ref（AP 点，×100），耗时比 = 方法 / ref。所有方法、所有 ref 共用同一组重采样。
    """
    refs = [refs] if isinstance(refs, str) else list(refs)
    names = list(prepared)
    ids = np.array(prepared[refs[0]].img_ids)
    point = {n: prepared[n].ap() for n in names}
    rng = np.random.default_rng(seed)
    boot_ap = {n: {m: np.empty(n_boot) for m in metrics} for n in names}
    boot_t = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        for n in names:
            a = prepared[n].ap(ids[idx])
            for m in metrics:
                boot_ap[n][m][b] = 100 * a[m]
            if times is not None:
                boot_t[n][b] = times[n][idx].mean()
    rows = []
    for ref in refs:
        for n in names:
            row = {"method": n, "ref": ref}
            for m in metrics:
                row[m] = 100 * point[n][m]
                row[f"d{m}"] = 100 * (point[n][m] - point[ref][m])
                row[f"d{m}_lo"], row[f"d{m}_hi"] = np.percentile(boot_ap[n][m] - boot_ap[ref][m], [2.5, 97.5])
            if times is not None:
                row["ms_per_img"] = 1000 * times[n].mean()
                row["time_ratio"] = times[n].mean() / times[ref].mean()
                row["time_ratio_lo"], row["time_ratio_hi"] = np.percentile(boot_t[n] / boot_t[ref], [2.5, 97.5])
            rows.append(row)
    return rows
