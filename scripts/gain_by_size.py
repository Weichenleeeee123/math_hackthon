"""SAHI 比整图推理多找回的目标，按"扫视图中的表观边长"分在哪里；Glance-SAHI 又保住了其中多少。

回答：Glance 的盲区（缩略图里看不见的极小目标）在 SAHI 的增益里占多大分量。
"找到" = 同类检测框与真值 IoU ≥ 0.5 且分数 ≥ --score（贪心一对一匹配，按分数从高到低）。
整图 = 缓存里的扫视结果中 ≥ 输出阈值的部分（就是 SAHI 的 standard pred）；SAHI = 全部切片；
Glance = det+edge@θ 选中的切片。全程离线读 cache.pkl。

  python scripts/gain_by_size.py dota15 --theta 0.9
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval as R  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.selector import select_slices  # noqa: E402

EDGES = [0, 4, 8, 16, 32, np.inf]


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def found(gt_xyxy, gt_cat, dets, score):
    """贪心匹配，返回每个真值是否被找到。dets: [x1,y1,x2,y2,score,cls]，cls 已映射到评测类别。"""
    hit = np.zeros(len(gt_xyxy), bool)
    d = dets[dets[:, 4] >= score]
    d = d[np.argsort(-d[:, 4])]
    for x1, y1, x2, y2, _, c in d:
        cand = np.where((gt_cat == c) & ~hit)[0]
        if not len(cand):
            continue
        ious = iou(np.array([x1, y1, x2, y2]), gt_xyxy[cand])
        k = int(np.argmax(ious))
        if ious[k] >= 0.5:
            hit[cand[k]] = True
    return hit


def main(args):
    R.set_dataset(args.dataset, args.res_tag)
    cache = pickle.loads(R.CACHE.read_bytes())
    imgsz = cache.get("imgsz", 640)
    gt = json.loads(R.GT.read_text())
    by_img = {}
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            by_img.setdefault(a["image_id"], []).append((*a["bbox"], a["category_id"]))
    cfg = GlanceConfig(img_weight=args.img_weight)
    to_eval = R.COCO_TO_EVAL

    rows = []
    for rec in cache["images"]:
        g = np.array(by_img.get(rec["id"], []), dtype=np.float64).reshape(-1, 5)
        if not len(g):
            continue
        xyxy = np.c_[g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]]
        side = np.sqrt(g[:, 2] * g[:, 3]) * imgsz / max(rec["hw"])
        sel = select_slices(R.slice_scores(rec, cfg, *R.prior_kinds(args.prior))[0], "threshold", args.theta, 0)
        res = {}
        for name, s in (("full", np.zeros(0, int)), ("sahi", np.arange(len(rec["slices"]))), ("glance", sel)):
            d, _ = R.merge(rec, s, cfg)
            d = d[np.isin(d[:, 5].astype(int), list(to_eval))].astype(np.float64)
            d[:, 5] = [to_eval[int(c)] for c in d[:, 5]]
            res[name] = found(xyxy, g[:, 4], d, args.score)
        for i in range(len(g)):
            rows.append({"side": side[i], **{k: bool(v[i]) for k, v in res.items()}})
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.side, EDGES, right=False).astype(str)
    df["gain"] = df.sahi & ~df.full           # SAHI 找到、整图没找到：SAHI 的增益
    df["gain_kept"] = df.gain & df.glance     # 其中 Glance 也找到
    agg = df.groupby("bin", sort=False).agg(n=("side", "size"), full=("full", "mean"), sahi=("sahi", "mean"),
                                             glance=("glance", "mean"), gain=("gain", "sum"),
                                             gain_kept=("gain_kept", "sum")).reset_index()
    agg["gain_share"] = agg.gain / agg.gain.sum()
    agg["kept_frac"] = agg.gain_kept / agg.gain.clip(lower=1)
    out = R.RES / f"gain_by_size_theta{args.theta:g}.csv"
    agg.to_csv(out, index=False)
    print(f"{args.dataset}：扫视输入 {imgsz}，θ={args.theta}，分数 ≥ {args.score}，目标 {len(df)} 个")
    print(agg.round(3).to_string(index=False))
    print(f"\nSAHI 相对整图多找回 {int(df.gain.sum())} 个目标，Glance 保住 {int(df.gain_kept.sum())} 个"
          f"（{df.gain_kept.sum() / max(df.gain.sum(), 1):.1%}）；"
          f"其中 <8px 占增益 {agg[agg.bin.isin(['[0.0, 4.0)', '[4.0, 8.0)'])].gain.sum() / max(df.gain.sum(), 1):.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="dota15")
    ap.add_argument("--res-tag", default="")
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--prior", default="det+edge")
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--score", type=float, default=0.25)
    main(ap.parse_args())
