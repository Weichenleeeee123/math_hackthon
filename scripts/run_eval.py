"""在 VisDrone / DOTA 上评测：整图 / SAHI 均匀切片 / Glance-SAHI / 随机选片 / Oracle。

四个子命令：
  cache    每张图只跑一次“扫视 + 全部切片”，把每片的检测结果和耗时存盘。
           任何选片策略的结果都等于“扫视结果 + 被选切片的结果 → 同一个 NMS”，
           所以扫参、消融、随机对照都可以离线精确复现，不必重复推理。
  sim      读缓存，扫阈值/消融（含 det_prior 四种变体、绝对证据量 τ）/随机对照
           → results/sweep.csv、results/per_image.csv
  e2e      真实端到端计时：full / 官方 SAHI get_sliced_prediction / glance_sliced_prediction
           → results/e2e.csv（同时用来核对 sim 的 AP 与真实运行一致）
  buckets  按“图中目标数”分桶报精度与切片比例 → results/buckets.csv（只看稀疏图上值不值）
"""

import argparse
import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glance_sahi import data as datasets  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.saliency import (  # noqa: E402
    detection_prior, evidence_mass, fuse, heatmap_prior, image_prior_map, region_prior,
)
from glance_sahi.selector import random_slices, select_slices  # noqa: E402

# 当前数据集（set_dataset 切换）；VisDrone 结果放 results/，其它数据集放 results/<name>/
DS = IMAGES = GT = RES = CACHE = COCO_TO_EVAL = EXCLUDE = None


def set_dataset(name="visdrone", res_tag=""):
    """res_tag：结果目录后缀（如换检测器/网格后另起目录，不覆盖原有 cache 与 csv）。"""
    global DS, IMAGES, GT, RES, CACHE, COCO_TO_EVAL, EXCLUDE
    DS = datasets.get(name)
    IMAGES, GT, COCO_TO_EVAL, EXCLUDE = DS["images"], DS["gt"], DS["coco_to_eval"], DS["exclude_coco_ids"]
    RES = ROOT / "results" if name == "visdrone" and not res_tag else ROOT / "results" / f"{name}{res_tag}"
    CACHE = RES / "cache.pkl"


set_dataset("visdrone")


from glance_sahi.imageio import imread_rgb as load_rgb  # noqa: E402  （Windows 中文路径安全：fromfile + imdecode）


def preds_to_np(preds):
    return np.array([p.bbox.to_xyxy() + [p.score.value, p.category.id] for p in preds],
                    dtype=np.float32).reshape(-1, 6)


def warmup(model, images, n=3, slice_size=512):
    """用真实尺寸的整图和 512 切片预热（首次遇到新输入尺寸时 CUDA/cuDNN 会有一次性开销，不应计入耗时）。"""
    from sahi.predict import get_prediction

    for im in images[:n]:
        img = load_rgb(IMAGES / im["file_name"])
        get_prediction(img, model)
        get_prediction(img[:slice_size, :slice_size], model)


# ----------------------------------------------------------------------------------------- cache
def cmd_cache(args):
    from sahi.predict import get_prediction
    from sahi.slicing import get_slice_bboxes

    from glance_sahi.detector import build_model

    EXCLUDE_COCO_IDS = EXCLUDE
    cfg = GlanceConfig(slice_size=args.slice_size)
    coco = json.loads(GT.read_text())
    images = coco["images"][: args.limit] if args.limit else coco["images"]
    model = build_model(args.weights, conf=cfg.output_conf, device=args.device, image_size=args.imgsz)
    warmup(model, images, slice_size=cfg.slice_size)

    cache = {"weights": args.weights, "imgsz": args.imgsz, "cfg": cfg.__dict__, "images": []}
    for im in tqdm(images, desc="cache"):
        img = load_rgb(IMAGES / im["file_name"])
        h, w = img.shape[:2]
        slices = get_slice_bboxes(h, w, cfg.slice_size, cfg.slice_size, False, cfg.overlap_ratio, cfg.overlap_ratio)

        t0 = time.perf_counter()
        glance = get_prediction(img, model, exclude_classes_by_id=EXCLUDE_COCO_IDS,
                                confidence_threshold=cfg.glance_conf).object_prediction_list
        t_glance = time.perf_counter() - t0

        slice_preds, t_slice = [], []
        for x1, y1, x2, y2 in slices:
            t0 = time.perf_counter()
            r = get_prediction(img[y1:y2, x1:x2], model, shift_amount=[x1, y1], full_shape=[h, w],
                               exclude_classes_by_id=EXCLUDE_COCO_IDS)
            ps = [p.get_shifted_object_prediction() for p in r.object_prediction_list]
            t_slice.append(time.perf_counter() - t0)
            slice_preds.append(preds_to_np(ps))

        priors = {}
        for kind in ("edge", "spectral"):
            t0 = time.perf_counter()
            sal, scale = image_prior_map(img, kind, cfg.img_map_size)
            s = region_prior(sal, scale, slices)
            priors[kind] = (s, time.perf_counter() - t0)

        cache["images"].append({
            "id": im["id"], "file_name": im["file_name"], "hw": (h, w), "slices": slices,
            "glance": preds_to_np(glance), "t_glance": t_glance,
            "slice_preds": slice_preds, "t_slice": np.array(t_slice),
            "prior_edge": priors["edge"][0], "t_prior_edge": priors["edge"][1],
            "prior_spectral": priors["spectral"][0], "t_prior_spectral": priors["spectral"][1],
        })
    RES.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(cache))
    print(f"cached {len(cache['images'])} images -> {CACHE}")


# ------------------------------------------------------------------------------------------- sim
def to_coco_dets(image_id, dets):
    out = []
    for x1, y1, x2, y2, s, c in dets:
        if int(c) in COCO_TO_EVAL:
            out.append({"image_id": image_id, "category_id": COCO_TO_EVAL[int(c)],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return out


def coco_eval(gt_path, dets, img_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        if not dets:
            return dict(AP=0, AP50=0, AP75=0, APs=0, APm=0, APl=0)
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.params.maxDets = [1, 100, DS["max_dets"]]  # 航拍图单图目标常超过 100
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return dict(AP=s[0], AP50=s[1], AP75=s[2], APs=s[3], APm=s[4], APl=s[5])


def merge(rec, sel, cfg):
    from sahi.postprocess.combine import batched_nms

    parts = [rec["slice_preds"][k] for k in sel]
    g = rec["glance"]
    parts.append(g[g[:, 4] >= cfg.output_conf])
    dets = np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)
    t0 = time.perf_counter()
    if len(dets) > 1:
        dets = dets[batched_nms(dets, match_metric=cfg.postprocess_match_metric,
                                match_threshold=cfg.postprocess_match_threshold)]
    return dets, time.perf_counter() - t0


# 扫参/消融的变体：(方法名, 检测先验权重 kind（None=不用检测先验）, 图像先验种类（None=不用）)
# 前 5 个是主方法（noisy-OR）；后 4 个是打分函数消融（REPORT 3.8）：
# "uncertain" = 4p(1−p) 不确定度加权；"max" = v0 的“置信度 + 取最大”反面消融。
ABLATIONS = [
    ("det", "noisyor", None),
    ("edge", None, "edge"),
    ("spectral", None, "spectral"),
    ("det+edge", "noisyor", "edge"),
    ("det+spectral", "noisyor", "spectral"),
    ("uncertain", "uncertain", None),
    ("uncertain+edge", "uncertain", "edge"),
    ("max", "max", None),
    ("max+edge", "max", "edge"),
    ("heatmap", "heatmap", None),          # 热图变体：O(像素) 一次卷积 + 软边界（REPORT 3.9）
    ("heatmap+edge", "heatmap", "edge"),
]
# 新增的消融变体只在关心的阈值上扫，控制 COCO 评测次数（主要看高阈值/低切片预算区间）
ABLATION_THS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
# 绝对证据量 τ（mode="evidence"）：未归一化、可跨图统一标定，与 θ 是两种旋钮（REPORT 3.9）
EVIDENCE_TAUS = [0.2, 0.5, 1.0, 2.0, 4.0]


def slice_scores(rec, cfg, det_kind, img_kind):
    """返回 (每个切片的分数, 该分数额外的计算耗时)。det_kind=None 表示只用图像先验。

    det_kind="heatmap" 走热图路径（从缓存里的粗检测重建，不需要图像）；
    其余走 noisy-OR / 4p(1−p) / max 的逐片写法。
    """
    s_det, t_det = None, 0.0
    if det_kind == "heatmap":
        t0 = time.perf_counter()
        s_det = heatmap_prior(rec["glance"][:, :4], rec["glance"][:, 4], rec["hw"], rec["slices"],
                              cfg.img_map_size, cfg.heat_sigma)[0]
        t_det = time.perf_counter() - t0
    elif det_kind is not None:
        s_det = detection_prior(rec["glance"][:, :4], rec["glance"][:, 4], rec["slices"],
                                cfg.det_margin, det_kind)
    if img_kind is None:
        return s_det, t_det
    s_img, t_img = rec[f"prior_{img_kind}"], rec[f"t_prior_{img_kind}"]
    if s_det is None:
        return s_img, t_img
    return fuse(s_det, s_img, cfg.img_weight), t_det + t_img


def prior_kinds(prior: str):
    """把 "det" / "edge" / "det+edge" 这类写法的先验名解析成 (det_kind, img_kind)。

    兼容新增变体："uncertain+edge" → ("uncertain", "edge")、"heatmap" → ("heatmap", None)。
    供 coverage.py / lambda_check.py / edge_vs_random.py / visualize.py / pick_cases.py 复用。
    """
    head, _, tail = prior.partition("+")
    det_kind = head if head in ("noisyor", "uncertain", "max", "heatmap", "det") else None
    if det_kind == "det":
        det_kind = "noisyor"
    img_kind = tail or (None if det_kind is not None else head)
    if img_kind not in (None, "edge", "spectral"):
        raise ValueError(f"无法解析的先验名：{prior!r}")
    return det_kind, img_kind


def gt_centers(gt):
    by_img = {}
    for a in gt["annotations"]:
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append((x + w / 2, y + h / 2, a["area"] < 32 * 32))
    return {k: np.array(v, dtype=np.float32) for k, v in by_img.items()}


def covered(centers, slices, sel):
    """被选切片覆盖的 GT 中心比例（全部 / 小目标）。"""
    if len(centers) == 0:
        return np.zeros(0, bool)
    hit = np.zeros(len(centers), bool)
    for k in sel:
        x1, y1, x2, y2 = slices[k]
        hit |= (centers[:, 0] >= x1) & (centers[:, 0] < x2) & (centers[:, 1] >= y1) & (centers[:, 1] < y2)
    return hit


def run_method(cache, gt_path, centers, name, select_fn, cfg, t_extra_fn=lambda r: 0.0):
    dets_all, rows = [], []
    for rec in cache["images"]:
        sel = select_fn(rec)
        dets, t_nms = merge(rec, sel, cfg)
        dets_all += to_coco_dets(rec["id"], dets)
        c = centers.get(rec["id"], np.zeros((0, 3), np.float32))
        hit = covered(c, rec["slices"], sel)
        t_extra = t_extra_fn(rec)
        rows.append({
            "image_id": rec["id"], "n_slices": len(rec["slices"]), "n_run": len(sel),
            "n_gt": len(c), "n_gt_small": int(c[:, 2].sum()) if len(c) else 0,
            "cov_small": int((hit & (c[:, 2] > 0)).sum()) if len(c) else 0,
            "t_extra": t_extra,
            "time": rec["t_glance"] + t_extra + rec["t_slice"][sel].sum() + t_nms,
        })
    pi = pd.DataFrame(rows)
    m = coco_eval(gt_path, dets_all, [r["id"] for r in cache["images"]])
    # ms_extra = 该方法的打分开销，单独成列（图像先验/热图都是真实成本，进表比只在正文写一句有说服力）
    m.update(method=name, slices_per_img=pi.n_run.mean(), slice_frac=pi.n_run.sum() / pi.n_slices.sum(),
             ms_per_img=1000 * pi.time.mean(), ms_extra=1000 * pi.t_extra.mean(),
             small_cov=pi.cov_small.sum() / max(pi.n_gt_small.sum(), 1))
    return m, pi


def cmd_sim(args):
    cache = pickle.loads(CACHE.read_bytes())
    if args.limit:  # 快速冒烟：只在缓存的前 N 张上扫（AP 会更噪，别用来出报告数字）
        cache["images"] = cache["images"][: args.limit]
    gt_path = GT
    gt = json.loads(gt_path.read_text())
    centers = gt_centers(gt)
    cfg = GlanceConfig(img_weight=args.img_weight)
    results, per_image = [], {}
    names = {s.strip() for s in args.only.split(",") if s.strip()}
    abls = [v for v in ABLATIONS if not names or v[0] in names]
    if names and not abls:
        raise SystemExit(f"--only {args.only} 没匹配到任何变体，可选：{[v[0] for v in ABLATIONS]}")

    def add(name, fn, extra=lambda r: 0.0, **tags):
        m, pi = run_method(cache, gt_path, centers, name, fn, cfg, extra)
        m.update(tags)
        results.append(m)
        per_image[name] = pi
        print(f"{name:32s} AP={m['AP']:.4f} AP50={m['AP50']:.4f} APs={m['APs']:.4f} "
              f"slices={m['slices_per_img']:.2f} ({m['slice_frac']:.1%}) {m['ms_per_img']:.1f}ms "
              f"small_cov={m['small_cov']:.3f}")

    add("full_image", lambda r: np.zeros(0, int), family="full")
    add("sahi_uniform", lambda r: np.arange(len(r["slices"])), family="sahi")

    # “为什么改”：SAHI 切片中不含任何目标的比例
    n_empty = n_tot = 0
    for rec in cache["images"]:
        c = centers.get(rec["id"], np.zeros((0, 3), np.float32))
        for k in range(len(rec["slices"])):
            n_tot += 1
            n_empty += int(not covered(c, rec["slices"], [k]).any())
    print(f"SAHI slices with no GT object: {n_empty}/{n_tot} = {n_empty / n_tot:.1%}")

    # Oracle：只跑含小目标的切片（上界，需要 GT，不可部署）
    def oracle(r):
        c = centers.get(r["id"], np.zeros((0, 3), np.float32))
        c = c[c[:, 2] > 0] if len(c) else c
        return np.array([k for k in range(len(r["slices"])) if covered(c, r["slices"], [k]).any()], int)
    add("oracle_gt_small", oracle, family="oracle")

    ths = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    for name, det_kind, img_kind in abls:
        for th in (ths if det_kind == "noisyor" else ABLATION_THS):
            def fn(r, det_kind=det_kind, img_kind=img_kind, th=th):
                return select_slices(slice_scores(r, cfg, det_kind, img_kind)[0], "threshold", th, 0)
            add(f"glance[{name}]@{th}", fn,
                lambda r, det_kind=det_kind, img_kind=img_kind: slice_scores(r, cfg, det_kind, img_kind)[1],
                family=f"glance_{name}", threshold=th)

    # 绝对证据量 τ：与 θ 并列的第二个旋钮，值未归一化、可跨图（乃至跨数据集）统一标定
    for tau in EVIDENCE_TAUS:
        def fn(r, tau=tau):
            ev = evidence_mass(r["glance"][:, :4], r["glance"][:, 4], r["slices"])
            base = slice_scores(r, cfg, "noisyor", None)[0]
            return select_slices(base, "evidence", 0, 0, ev, tau)
        add(f"glance_evidence@{tau}", fn, family="glance_evidence", tau=tau)

    # 随机对照：每张图随机选与 glance[det+edge]（主方法）相同数量的切片
    for th in ([] if args.no_random else (ths if any(v[1] == "noisyor" for v in abls) else ABLATION_THS)):
        for seed in range(3):
            rng = np.random.default_rng(seed)

            def fn(r, th=th, rng=rng):
                k = len(select_slices(slice_scores(r, cfg, "noisyor", "edge")[0], "threshold", th, 0))
                return random_slices(len(r["slices"]), k, rng)
            add(f"random_matched@{th}#s{seed}", fn, family="random", threshold=th, seed=seed)

    df = pd.DataFrame(results)
    df.to_csv(RES / f"sweep{args.tag}.csv", index=False)
    keep = ("sahi_uniform", f"glance[det+edge]@{args.op}", "full_image")
    parts = [v.assign(method=k) for k, v in per_image.items() if k in keep]
    if parts:
        pd.concat(parts).to_csv(RES / f"per_image{args.tag}.csv", index=False)
    print(f"wrote {RES / f'sweep{args.tag}.csv'}")


# ------------------------------------------------------------------------------------------- e2e
def cmd_e2e(args):
    from sahi.predict import get_prediction

    from glance_sahi.detector import build_model
    from glance_sahi.predict import full_image_prediction, glance_sliced_prediction, sahi_uniform_prediction

    EXCLUDE_COCO_IDS = EXCLUDE
    cfg = GlanceConfig(threshold=args.op, img_prior="edge", img_weight=args.img_weight, slice_size=args.slice_size)
    gt_path = GT
    coco = json.loads(gt_path.read_text())
    images = coco["images"][: args.limit] if args.limit else coco["images"]
    model = build_model(args.weights, conf=cfg.output_conf, device=args.device, image_size=args.imgsz)
    warmup(model, images, slice_size=cfg.slice_size)

    dets = {"full_image": [], "sahi_uniform": [], "glance_sahi": [], "glance_all_slices": []}
    times = {k: [] for k in dets}
    n_run = {k: [] for k in dets}
    for im in tqdm(images, desc="e2e"):
        img = load_rgb(IMAGES / im["file_name"])
        r, dt = full_image_prediction(img, model, EXCLUDE_COCO_IDS)
        dets["full_image"] += to_coco_dets(im["id"], preds_to_np(r.object_prediction_list))
        times["full_image"].append(dt), n_run["full_image"].append(0)

        r, dt, n = sahi_uniform_prediction(img, model, cfg, EXCLUDE_COCO_IDS)
        dets["sahi_uniform"] += to_coco_dets(im["id"], preds_to_np(r.object_prediction_list))
        times["sahi_uniform"].append(dt), n_run["sahi_uniform"].append(n)

        t0 = time.perf_counter()
        r, st = glance_sliced_prediction(img, model, cfg, EXCLUDE_COCO_IDS)
        times["glance_sahi"].append(time.perf_counter() - t0), n_run["glance_sahi"].append(st.n_slices_run)
        dets["glance_sahi"] += to_coco_dets(im["id"], preds_to_np(r.object_prediction_list))

        if args.check_all:  # 正确性自检：全选时应与官方 SAHI 完全一致
            t0 = time.perf_counter()
            r, st = glance_sliced_prediction(img, model, cfg, EXCLUDE_COCO_IDS,
                                             force_select=np.arange(st.n_slices_total))
            times["glance_all_slices"].append(time.perf_counter() - t0)
            n_run["glance_all_slices"].append(st.n_slices_total)
            dets["glance_all_slices"] += to_coco_dets(im["id"], preds_to_np(r.object_prediction_list))

    rows = []
    for k in dets:
        if not times[k]:
            continue
        m = coco_eval(gt_path, dets[k], [im["id"] for im in images])
        m.update(method=k, ms_per_img=1000 * np.mean(times[k]), fps=1 / np.mean(times[k]),
                 slices_per_img=np.mean(n_run[k]))
        rows.append(m)
        print(f"{k:20s} AP={m['AP']:.4f} AP50={m['AP50']:.4f} APs={m['APs']:.4f} "
              f"{m['ms_per_img']:.1f}ms/img slices={m['slices_per_img']:.2f}")
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / f"e2e{args.tag}.csv", index=False)


# ----------------------------------------------------------------------------------------- buckets
# 按“图里有多少目标”分桶报**精度**（不只是切片比例）：回答“自适应机制在稀疏图上值不值”。
BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 10 ** 9)]


def cmd_buckets(args):
    cache = pickle.loads(CACHE.read_bytes())
    gt_path = GT
    centers = gt_centers(json.loads(gt_path.read_text()))
    cfg = GlanceConfig(threshold=args.op, img_weight=args.img_weight)

    main = lambda r: select_slices(slice_scores(r, cfg, "noisyor", "edge")[0], "threshold", args.op, 0)
    methods = {"sahi_uniform": lambda r: np.arange(len(r["slices"])),
               f"glance[det+edge]@{args.op}": main}
    for seed in range(3):
        rng = np.random.default_rng(seed)

        def fn(r, rng=rng):
            return random_slices(len(r["slices"]), len(main(r)), rng)
        methods[f"random_matched@{args.op}#s{seed}"] = fn

    rows = []
    for name, fn in methods.items():
        dets_by_img, info = {}, {}
        for rec in cache["images"]:
            sel = fn(rec)
            dets, t_nms = merge(rec, sel, cfg)
            dets_by_img[rec["id"]] = to_coco_dets(rec["id"], dets)
            c = centers.get(rec["id"], np.zeros((0, 3), np.float32))
            hit = covered(c, rec["slices"], sel)
            info[rec["id"]] = dict(n_run=len(sel), n_slices=len(rec["slices"]), n_gt=len(c),
                                   n_gt_small=int(c[:, 2].sum()) if len(c) else 0,
                                   cov_small=int((hit & (c[:, 2] > 0)).sum()) if len(c) else 0,
                                   time=rec["t_glance"] + rec["t_slice"][sel].sum() + t_nms)
        for lo, hi in BUCKETS:
            ids = [i for i, v in info.items() if lo <= v["n_gt"] < hi]
            if not ids:
                continue
            m = coco_eval(gt_path, [d for i in ids for d in dets_by_img[i]], ids)
            n_gt_small = sum(info[i]["n_gt_small"] for i in ids)
            row = dict(method=name, bucket=f"[{lo},{hi if hi < 10 ** 8 else 'inf'})", images=len(ids),
                       slice_frac=sum(info[i]["n_run"] for i in ids) / sum(info[i]["n_slices"] for i in ids),
                       ms_per_img=1000 * float(np.mean([info[i]["time"] for i in ids])),
                       small_cov=sum(info[i]["cov_small"] for i in ids) / max(n_gt_small, 1), **m)
            rows.append(row)
            print(f"{name:28s} {row['bucket']:10s} imgs={len(ids):4d} slices={row['slice_frac']:6.1%} "
                  f"AP={m['AP']:.4f} AP50={m['AP50']:.4f} APs={m['APs']:.4f} cov_small={row['small_cov']:.3f}")
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "buckets.csv", index=False)
    print(f"wrote {RES / 'buckets.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "sim", "e2e", "buckets"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--device", default="auto", help="auto = 有 CUDA 用 cuda:0，否则 cpu")
    ap.add_argument("--op", type=float, default=0.9, help="Glance-SAHI 的工作点阈值")
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--check-all", action="store_true")
    ap.add_argument("--tag", default="", help="输出文件名后缀（sim 写 sweep<tag>.csv / per_image<tag>.csv）")
    ap.add_argument("--only", default="", help="sim 只跑这些消融变体（逗号分隔，如 uncertain,max,heatmap）；留空=全部")
    ap.add_argument("--no-random", action="store_true", help="sim 跳过随机对照（快速冒烟用）")
    ap.add_argument("--dataset", default="visdrone", choices=list(datasets.DATASETS))
    ap.add_argument("--slice-size", type=int, default=512, help="cache/e2e 的切片边长（SAHI 基线与 Glance 共用）")
    ap.add_argument("--imgsz", type=int, default=640, help="检测器输入尺寸（OBB 检测器用 1024）")
    ap.add_argument("--res-tag", default="", help="结果目录后缀：results/<dataset><res-tag>/，不覆盖原结果")
    a = ap.parse_args()
    set_dataset(a.dataset, a.res_tag)
    {"cache": cmd_cache, "sim": cmd_sim, "e2e": cmd_e2e, "buckets": cmd_buckets}[a.cmd](a)
