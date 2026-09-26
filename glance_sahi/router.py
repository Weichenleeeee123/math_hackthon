"""可学习稀疏路由器（Learned Sparse Router）—— 把 Glance-SAHI 写成“条件计算”的形式。

对应关系（REPORT §2.5 的 MoE 对照表）：
  token        ↔ SAHI 网格里的一个切片 R_k
  router       ↔ 整图一眼（glance）+ 本模块的小网络 g_φ(x_k) ∈ [0,1]
  expert       ↔ 在切片 R_k 上跑一次检测器（权重共享的“空间专家”，不是参数 MoE）
  sparse top-k ↔ 每图只激活 k = ρN 个切片（或一个全局阈值 μ：跨图共享的“算力影子价格”）
  dense model  ↔ SAHI（所有切片全激活）

门控网络只吃**推理时就拿得到**的特征（扫视粗检测 + 缩略图边缘先验 + 切片几何），
离线（读 results/cache.pkl）与在线（predict.py）共用同一个 `slice_features`，保证口径一致。

训练在 CPU 上几秒钟完成；推理只需 numpy（`NumpyRouter`），部署不依赖 torch。
权重存 JSON（.gitignore 忽略 *.pt，JSON 也便于审阅、无 pickle 安全问题）。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .saliency import detection_prior, fuse, heatmap_prior

WEAK_CONF = 0.05     # = GlanceConfig.output_conf：低于它的扫视框是“弱证据”，不进最终输出
PERSON_COCO = 0

FEATURE_NAMES = (
    # --- 检测证据（扫视粗检测） ---
    "det_noisyor", "det_uncertain", "det_max", "log_mass", "log_n_weak", "log_n_strong",
    "mass_person", "mass_other", "det_heatmap", "log_app_size",
    # --- 图像先验 ---
    "edge", "fused",
    # --- 上下文 ---
    "nb_max_det", "nb_mean_edge", "img_log_mass", "img_rank",
    # --- 几何 ---
    "cx", "cy", "border", "area_frac", "log_n_slices", "scale",
)


# =============================================================================== 特征
def slice_features(glance, hw, slices, prior_edge, cfg, model_input: int = 640) -> np.ndarray:
    """每个切片一行特征，(K, len(FEATURE_NAMES))。

    glance: (N,6) [x1,y1,x2,y2,score,coco_cls]，扫视阶段（conf ≥ glance_conf）的粗检测，原图坐标
    prior_edge: (K,) 缩略图 Sobel 边缘先验（region_prior 的 95 分位）
    """
    s = np.asarray(slices, np.float32).reshape(-1, 4)
    K = len(s)
    if K == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    h, w = hw
    g = np.asarray(glance, np.float32).reshape(-1, 6)
    boxes, conf, cls = g[:, :4], g[:, 4], g[:, 5]
    edge = np.asarray(prior_edge, np.float32).reshape(-1)

    d_no = detection_prior(boxes, conf, slices, cfg.det_margin, "noisyor")
    d_un = detection_prior(boxes, conf, slices, cfg.det_margin, "uncertain")
    d_mx = detection_prior(boxes, conf, slices, cfg.det_margin, "max")
    d_hm = heatmap_prior(boxes, conf, hw, slices, cfg.img_map_size, cfg.heat_sigma)[0].astype(np.float32)

    scale = model_input / max(h, w)
    if len(g):
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        inside = ((cx[None] >= s[:, 0:1]) & (cx[None] < s[:, 2:3]) &
                  (cy[None] >= s[:, 1:2]) & (cy[None] < s[:, 3:4])).astype(np.float32)
        weak = (conf < WEAK_CONF).astype(np.float32)
        mass = inside @ conf
        n_weak = inside @ weak
        n_strong = inside @ (1.0 - weak)
        person = inside @ (conf * (cls == PERSON_COCO))
        app = np.log1p(np.sqrt(np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0)) * scale)
        log_app = (inside @ (conf * app)) / np.maximum(mass, 1e-6)
        img_mass = float(conf.sum())
    else:
        mass = n_weak = n_strong = person = log_app = np.zeros(K, np.float32)
        img_mass = 0.0

    fused = fuse(d_no, edge, cfg.img_weight)

    # 邻域：切比雪夫中心距 ≤ 1.1 × 切片边长（即 8 邻域）
    scx, scy = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2
    size = float(np.median(np.maximum(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])))
    nb = (np.abs(scx[:, None] - scx[None]) <= 1.1 * size) & (np.abs(scy[:, None] - scy[None]) <= 1.1 * size)
    np.fill_diagonal(nb, False)
    cnt = nb.sum(1)
    nb_max = np.where(cnt > 0, np.max(np.where(nb, d_no[None], 0.0), axis=1), 0.0)
    nb_edge = np.where(cnt > 0, (nb * edge[None]).sum(1) / np.maximum(cnt, 1), 0.0)

    rank = (np.argsort(np.argsort(fused, kind="stable"), kind="stable") / max(K - 1, 1)) if K > 1 else np.ones(1)
    border = ((s[:, 0] <= 0) | (s[:, 1] <= 0) | (s[:, 2] >= w) | (s[:, 3] >= h)).astype(np.float32)
    area = (s[:, 2] - s[:, 0]) * (s[:, 3] - s[:, 1]) / float(h * w)
    ones = np.ones(K, np.float32)

    X = np.stack([
        d_no, d_un, d_mx, np.log1p(mass), np.log1p(n_weak), np.log1p(n_strong),
        person, mass - person, d_hm, log_app,
        edge, fused,
        nb_max, nb_edge, np.log1p(img_mass) * ones, rank,
        scx / w, scy / h, border, area, np.log(K) * ones, scale * ones,
    ], axis=1)
    return X.astype(np.float32)


def features_from_rec(rec: dict, cfg) -> np.ndarray:
    """离线：直接从 cache.pkl 的一条记录算特征（与在线 predict.py 同一路径）。"""
    return slice_features(rec["glance"], rec["hw"], rec["slices"], rec["prior_edge"], cfg)


# =============================================================================== 标签
def gt_targets(gt: dict) -> dict:
    """每个非 crowd 目标：(cx, cy, 匹配半径, 评测类别, 是否小目标)。与 scripts/attribute.py 同口径。"""
    out: dict[int, list] = {}
    for a in gt["annotations"]:
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        out.setdefault(a["image_id"], []).append(
            (x + w / 2.0, y + h / 2.0, max(0.5 * float(np.hypot(w, h)), 6.0),
             int(a["category_id"]), float(a["area"] < 32 * 32)))
    return out


def _hits(dets: np.ndarray, T: np.ndarray, coco_to_eval: dict) -> np.ndarray:
    """T 里每个真值是否被 dets 中同类、中心距 ≤ 半径 的检测命中。"""
    d = np.asarray(dets, np.float32).reshape(-1, 6)
    if len(d) == 0 or len(T) == 0:
        return np.zeros(len(T), bool)
    dcls = np.array([coco_to_eval.get(int(round(c)), -1) for c in d[:, 5]])
    dcx, dcy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
    dist = np.hypot(dcx[None] - T[:, 0:1], dcy[None] - T[:, 1:2])
    return ((dist <= T[:, 2:3]) & (dcls[None] == T[:, 3:4])).any(1)


def utility_labels(rec: dict, targets: list, coco_to_eval: dict, output_conf: float = 0.05,
                   kind: str = "gain") -> np.ndarray:
    """切片的“专家效用”标签 y_k ∈ {0,1}。

    kind="gain"（默认）：这片自己的检测命中了**扫视标准预测没命中**的真值 —— “这个专家有增量”。
    kind="gt"：片内有任意真值中心；kind="gt_small"：片内有小目标中心（= oracle 的定义）。
    """
    s = np.asarray(rec["slices"], np.float32).reshape(-1, 4)
    y = np.zeros(len(s), np.float32)
    T = np.asarray(targets or [], np.float32).reshape(-1, 5)
    if len(T) == 0:
        return y
    if kind in ("gt", "gt_small"):
        t = T if kind == "gt" else T[T[:, 4] > 0]
        if len(t) == 0:
            return y
        inside = ((t[None, :, 0] >= s[:, 0:1]) & (t[None, :, 0] < s[:, 2:3]) &
                  (t[None, :, 1] >= s[:, 1:2]) & (t[None, :, 1] < s[:, 3:4]))
        return inside.any(1).astype(np.float32)
    if kind != "gain":
        raise ValueError(kind)
    g = rec["glance"]
    base = _hits(g[g[:, 4] >= output_conf], T, coco_to_eval)
    todo = ~base
    if not todo.any():
        return y
    T2 = T[todo]
    for k, p in enumerate(rec["slice_preds"]):
        y[k] = float(_hits(p, T2, coco_to_eval).any())
    return y


# =============================================================================== 数据划分
def split_images(images: list, kind: str = "oddeven") -> tuple[list[int], list[int]]:
    """返回 (调参集下标, 留出集下标)。

    oddeven：奇数序号调参、偶数序号留出（与 calibrate.py 同协议）。
    sequence：按视频序列划分（VisDrone 文件名 <seq>_<frame>_...），同一视频的帧不会跨集合，
              用来检查 oddeven 相邻帧泄漏是否抬高了结果；无序列信息的数据集回退 oddeven。
    """
    if kind == "oddeven":
        return [i for i in range(len(images)) if i % 2 == 1], [i for i in range(len(images)) if i % 2 == 0]
    if kind != "sequence":
        raise ValueError(kind)
    seqs = [Path(im["file_name"]).stem.split("_")[0] if "_" in Path(im["file_name"]).stem else None
            for im in images]
    if any(q is None for q in seqs):
        return split_images(images, "oddeven")
    uniq = sorted(set(seqs))
    fit_seq = set(uniq[1::2])
    fit = [i for i, q in enumerate(seqs) if q in fit_seq]
    return fit, [i for i, q in enumerate(seqs) if q not in fit_seq]


# =============================================================================== 模型
def _build_net(n_in: int, hidden: int):
    import torch.nn as nn

    if not hidden:
        return nn.Sequential(nn.Linear(n_in, 1))            # 逻辑回归（消融）
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, max(hidden // 2, 1)), nn.ReLU(),
                         nn.Linear(max(hidden // 2, 1), 1))


def train_router(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, *, hidden: int = 32,
                 alpha: float = 0.0, beta: float = 0.0, budget_rho: float | None = None,
                 epochs: int = 400, lr: float = 3e-3, wd: float = 1e-4, seed: int = 0) -> dict:
    """全批量 Adam 训练门控网络，返回可 JSON 序列化的参数。

        L = BCE(g_φ(x), y)                                   专家效用的概率估计
          + α · mean_k g_φ(x_k)                               L1 稀疏：压低平均激活率（预算的软松弛）
          + β · mean_img ( mean_{k∈img} g_φ(x_k) − ρ )²       软预算：每图激活率向 ρ 靠拢（可选）

    不加 pos_weight：保持输出是“有增量”的概率（ECE 有意义，全局阈值可解释）。
    标准化参数只在训练集上拟合。CPU 上确定性。
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd)
    yt = torch.from_numpy(y)
    net = _build_net(X.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    if groups is not None and beta > 0 and budget_rho is not None:
        _, gidx = np.unique(np.asarray(groups), return_inverse=True)
        gidx_t = torch.from_numpy(gidx.astype(np.int64))
        n_g = int(gidx.max()) + 1
        cnt = torch.zeros(n_g).index_add_(0, gidx_t, torch.ones(len(gidx)))
    else:
        gidx_t = None
    for _ in range(epochs):
        z = net(Xt).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(z, yt)
        p = torch.sigmoid(z)
        if alpha > 0:
            loss = loss + alpha * p.mean()
        if gidx_t is not None:
            per = torch.zeros(n_g).index_add_(0, gidx_t, p) / cnt
            loss = loss + beta * ((per - budget_rho) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    layers = [(m.weight.detach().numpy().tolist(), m.bias.detach().numpy().tolist())
              for m in net if isinstance(m, torch.nn.Linear)]
    return {"mu": mu.tolist(), "sd": sd.tolist(), "layers": layers, "hidden": hidden,
            "alpha": alpha, "beta": beta, "seed": seed}


def torch_forward(params: dict, X: np.ndarray) -> np.ndarray:
    """用 torch 重建网络做前向（测试 numpy 推理与 torch 一致用）。"""
    import torch

    net = _build_net(len(params["mu"]), params["hidden"])
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for m, (W, b) in zip(lin, params["layers"]):
            m.weight.copy_(torch.tensor(W))
            m.bias.copy_(torch.tensor(b))
        x = torch.from_numpy(((np.asarray(X, np.float32) - np.array(params["mu"], np.float32))
                              / np.array(params["sd"], np.float32)))
        return torch.sigmoid(net(x).squeeze(1)).numpy()


class NumpyRouter:
    """纯 numpy 推理的门控网络（可为多个种子的集成，输出取平均概率）。"""

    def __init__(self, members: list[dict], meta: dict | None = None):
        self.members = members
        self.meta = meta or {}
        self._np = [(np.array(m["mu"], np.float32), np.array(m["sd"], np.float32),
                     [(np.array(W, np.float32), np.array(b, np.float32)) for W, b in m["layers"]])
                    for m in members]

    @staticmethod
    def _forward(mu, sd, layers, X):
        x = (np.asarray(X, np.float32) - mu) / sd
        for i, (W, b) in enumerate(layers):
            x = x @ W.T + b
            if i < len(layers) - 1:
                x = np.maximum(x, 0)
        return 1.0 / (1.0 + np.exp(-x[:, 0]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32).reshape(-1, len(FEATURE_NAMES))
        if len(X) == 0:
            return np.zeros(0, np.float32)
        return np.mean([self._forward(mu, sd, L, X) for mu, sd, L in self._np], axis=0).astype(np.float32)

    def features(self, glance, hw, slices, prior_edge, cfg) -> np.ndarray:
        return slice_features(glance, hw, slices, prior_edge, cfg)

    @property
    def default_threshold(self) -> float:
        return float(self.meta.get("default_threshold", 0.5))

    @property
    def n_params(self) -> int:
        return int(sum(np.asarray(W).size + np.asarray(b).size for W, b in self.members[0]["layers"]))


def save_router(members: list[dict], meta: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"feature_names": list(FEATURE_NAMES), "members": members, "meta": meta},
                                     ensure_ascii=False))


@lru_cache(maxsize=4)
def load_router(path) -> NumpyRouter:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(d["feature_names"]) != FEATURE_NAMES:
        raise ValueError(f"{path} 的特征列表与当前代码不一致，请重新训练（scripts/train_router.py）")
    return NumpyRouter(d["members"], d["meta"])


# =============================================================================== 路由（选片）
def global_threshold(p_train: np.ndarray, rho: float) -> float:
    """全局阈值 μ：让训练集上 p ≥ μ 的比例 ≈ ρ（跨图共享的容量 / 算力影子价格）。"""
    p = np.asarray(p_train, float)
    if rho >= 1:
        return -np.inf
    if rho <= 0:
        return np.inf
    return float(np.quantile(p, 1.0 - rho))


def route(p: np.ndarray, mode: str = "topk", rho: float | None = None, thr: float | None = None,
          k: int | None = None) -> np.ndarray:
    """返回被激活切片的下标（升序）。topk：每图 round(ρN)；global：p ≥ μ；matched：给定 k。"""
    p = np.asarray(p, float)
    n = len(p)
    if mode == "global":
        return np.flatnonzero(p >= thr).astype(int)
    if mode == "topk":
        k = int(round(rho * n))
    elif mode != "matched":
        raise ValueError(mode)
    k = max(0, min(int(k), n))
    return np.sort(np.argsort(-p, kind="stable")[:k]).astype(int)


# =============================================================================== 指标
def roc_auc(y, s) -> float:
    """Mann–Whitney U 形式的 ROC-AUC（含并列的平均秩）。"""
    y = np.asarray(y) > 0.5
    s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    import pandas as pd

    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s) -> float:
    """PR 曲线下面积（average precision，按分数降序逐个正例取精度的平均）。"""
    y = np.asarray(y) > 0.5
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-np.asarray(s, float), kind="stable")
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float(prec[yy].mean())


def gini(x) -> float:
    """基尼系数：0 = 激活完全均匀（每图一样多），→1 = 算力集中在少数图上。"""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(2 * (i * x).sum() / (n * x.sum()) - (n + 1) / n)


def activation_stats(rates, n_gt) -> dict:
    """稀疏激活的统计：每图激活率的分布，以及它是否“跟着内容走”（与目标数的 Spearman 相关）。"""
    import pandas as pd

    r = np.asarray(rates, float)
    g = np.asarray(n_gt, float)
    # Spearman = 秩的 Pearson（不依赖 scipy）
    rr, rg = pd.Series(r).rank().to_numpy(), pd.Series(g).rank().to_numpy()
    rho = float(np.corrcoef(rr, rg)[0, 1]) if rr.std() > 0 and rg.std() > 0 else float("nan")
    return dict(rate_mean=float(r.mean()), rate_std=float(r.std()), rate_gini=gini(r),
                frac_img_zero=float((r == 0).mean()), frac_img_all=float((r >= 1).mean()),
                spearman_rate_ngt=rho)
