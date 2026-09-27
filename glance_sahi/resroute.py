"""分辨率路由（REPORT 3.18）：先用 640 看一眼整图，再按内容决定"就这样 / 放大到多少再看一次"。

为什么：实验 A 显示在 VisDrone 这种约 1 百万像素的图上，高分辨率整图推理同时支配 SAHI 与 Glance-SAHI。
      那么"先看一眼再决定"这个想法该用在**分辨率**上，而不是切片上：目标大、看得清的图 640 就够，
      目标小而密的图才值得放大到 1920。

级联：
  第 0 级  full@640（= 扫视，永远跑；输出直接可用）
  第 1 级  按扫视特征选一个 r ∈ {640, 960, …}：r = 640 就停，否则再跑一次 full@r，输出 full@r 的结果
  耗时    t(640) + [r ≠ 640]·t(r)

评测的关键：COCO 的 evaluate() 是逐图的，所以"每张图用不同方法"的数据集 AP 可以直接由各方法逐图
匹配结果拼起来再 accumulate（`MixedEval`），不必重跑检测器；与只用一个方法时的 AP 一致到 1e-12（有单测）。
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "log_n05", "log_n25", "log_n50", "conf_mean", "conf_sum_log",
    "app_p10", "app_p50", "app_p90", "frac_app_lt8", "frac_app_lt16", "frac_app_lt32",
    "low_conf_mass", "log_long_side", "aspect", "person_frac",
)


def glance_features(dets: np.ndarray, hw, glance_size: int = 640) -> np.ndarray:
    """dets: (N,6) [x1,y1,x2,y2,score,cls]，640 扫视的检测结果（原图坐标，score ≥ 输出阈值）。

    表观尺寸 = 检测器输入里框的边长（√面积 × glance_size / 长边）：小目标在 640 下只有几个像素，
    这正是"需要放大"的信号（与 3.15 路由器里最重要的特征 log_app_size 同一含义）。
    """
    h, w = hw
    d = np.asarray(dets, np.float32).reshape(-1, 6)
    s = d[:, 4]
    app = np.sqrt(np.maximum((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]), 1.0)) * glance_size / max(h, w)
    n = len(d)

    def q(p):
        return float(np.percentile(app, p)) if n else 0.0

    def frac(t):
        return float((app < t).mean()) if n else 0.0

    return np.array([
        np.log1p(n), np.log1p((s >= 0.25).sum()), np.log1p((s >= 0.5).sum()),
        float(s.mean()) if n else 0.0, np.log1p(float(s.sum())),
        q(10), q(50), q(90), frac(8), frac(16), frac(32),
        float(s[s < 0.25].sum()) / max(n, 1), np.log(max(h, w)), max(h, w) / min(h, w),
        float((d[:, 5] == 0).mean()) if n else 0.0,
    ], np.float32)


class MixedEval:
    """把多个方法的逐图 COCO 匹配结果按"每张图选哪个方法"拼起来，重算数据集 AP。

    prepared: {方法名: glance_sahi.bootstrap.PreparedEval}，同一份真值、同一图序列。
    逐图匹配结果先打包成 numpy（与 evalboot.CachedEval 同法），accumulate 向量化，
    单次重算约几十毫秒（pycocotools.accumulate 约 0.5 s），逐图边际收益与 bootstrap 才跑得动。
    """

    def __init__(self, prepared: dict):
        self.prepared = prepared
        self.ref = next(p for p in prepared.values() if not p.empty)
        p = self.ref.ev.params
        self.recThrs, self.iouThrs = p.recThrs, p.iouThrs
        self.K = len(p.catIds)
        self.cap = p.maxDets[-1]
        self.areas = {"all": self.ref.a_all, "small": self.ref.a_small}
        self.t50 = int(np.where(np.isclose(p.iouThrs, 0.5))[0][0])
        self.pack = {}
        for name, pe in prepared.items():
            for k in range(self.K):
                for a in self.areas.values():
                    for img in pe.img_ids:
                        e = None if pe.empty else pe.table[k, a, img]
                        if e is None:
                            # 没有检测：该图只贡献真值数（从参照方法拿，真值与方法无关）
                            r = self.ref.table[k, a, img]
                            e_np = None if r is None else (np.zeros(0), np.zeros((len(self.iouThrs), 0), bool),
                                                           np.zeros((len(self.iouThrs), 0), bool),
                                                           int((np.asarray(r["gtIgnore"]) == 0).sum()))
                        else:
                            c = self.cap
                            e_np = (np.asarray(e["dtScores"][:c], float), e["dtMatches"][:, :c] > 0,
                                    e["dtIgnore"][:, :c].astype(bool), int((np.asarray(e["gtIgnore"]) == 0).sum()))
                        self.pack[name, k, a, img] = e_np

    def ap(self, choice: dict, ids=None) -> dict:
        """choice: {图 id: 方法名}；ids：参与评测的图序列（可重复，用于 bootstrap），默认 = choice 的键。"""
        ids = list(choice) if ids is None else list(ids)
        out = {}
        for label, a in self.areas.items():
            q = self._precision(choice, ids, a)  # (K, T, R)，缺类别为 None
            vals = [x for x in q if x is not None]
            out["AP" if label == "all" else "APs"] = float(np.mean(vals)) if vals else -1.0
            if label == "all":
                v50 = [x[self.t50] for x in vals]
                out["AP50"] = float(np.mean(v50)) if v50 else -1.0
        return out

    def _precision(self, choice, ids, a):
        R = len(self.recThrs)
        res = []
        for k in range(self.K):
            rows = [self.pack[choice[img], k, a, img] for img in ids]
            rows = [r for r in rows if r is not None]
            npig = sum(r[3] for r in rows)
            if not rows or npig == 0:
                res.append(None)  # 与 pycocotools 相同：该类别不计入平均
                continue
            sc = np.concatenate([r[0] for r in rows])
            o = np.argsort(-sc, kind="mergesort")
            dtm = np.concatenate([r[1] for r in rows], axis=1)[:, o]
            dtig = np.concatenate([r[2] for r in rows], axis=1)[:, o]
            tps = np.cumsum(dtm & ~dtig, axis=1, dtype=float)
            fps = np.cumsum(~dtm & ~dtig, axis=1, dtype=float)
            q = np.zeros((len(self.iouThrs), R))
            for t in range(len(self.iouThrs)):
                tp, fp = tps[t], fps[t]
                if not len(tp):
                    continue
                rc = tp / npig
                pr = np.maximum.accumulate((tp / (fp + tp + np.spacing(1)))[::-1])[::-1]
                inds = np.searchsorted(rc, self.recThrs, side="left")
                ok = inds < len(pr)
                q[t, ok] = pr[inds[ok]]
            res.append(q)
        return res


def marginal_gain(mixed: MixedEval, ids, base: str, cand: str) -> np.ndarray:
    """每张图的边际 ΔAP：其余图都用 base，只把这一张换成 cand 时数据集 AP 的变化（×100）。

    这是逐图的"切换价值"标签：比单图 AP 稳（单图 AP 在只有几个目标的图上跳变很大），
    而且口径就是最终要优化的数据集 AP。
    """
    ids = list(ids)
    choice = {i: base for i in ids}
    ap0 = mixed.ap(choice)["AP"]
    out = np.empty(len(ids))
    for j, i in enumerate(ids):
        choice[i] = cand
        out[j] = 100 * (mixed.ap(choice)["AP"] - ap0)
        choice[i] = base
    return out


# =============================================================================== 岭回归（纯 numpy）
class Ridge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        R = self.alpha * np.eye(Z.shape[1])
        R[0, 0] = 0.0  # 截距不正则
        self.w = np.linalg.solve(Z.T @ Z + R, Z.T @ np.asarray(y, np.float64))
        return self

    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (np.asarray(X, np.float64) - self.mu) / self.sd]
        return Z @ self.w


def choose(pred_gain: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    """pred_gain, cost: (N, M)，每张图每个选项的预测收益（ΔAP×100，相对第 0 个选项）与耗时（ms）。
    返回每张图选中的选项下标：argmax 收益 − λ·耗时（λ = 每毫秒值多少 AP，扫它得到一条前沿）。"""
    return np.argmax(pred_gain - lam * cost, axis=1)
