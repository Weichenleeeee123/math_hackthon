"""细分指标：person / car / truck / bus 四类分别报 AP，而不是只报 person / vehicle 两个超类。

动机：把 car、bus、truck 合并成一个“vehicle”再评测，等于把三类之间的差异全抹掉，
信息量少一半——报告里会看不出“大车更难还是小车更难”。做法是 **GT 与检测同时映射到细分
类别，再重跑 COCOeval 并按类别过滤**（不是只改 GT），两边口径一致才可比。

  python scripts/per_class.py                     # VisDrone，默认工作点 0.9
  python scripts/per_class.py --limit 60          # 快速冒烟（AP 会更噪）
输出 results/per_class.csv（方法 × 类别），GT json 写到 datasets/<ds>/coco_fine.json。
"""

import argparse
import contextlib
import io
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
from glance_sahi.data.visdrone import COCO_TO_FINE, convert_fine  # noqa: E402
from glance_sahi.selector import random_slices, select_slices  # noqa: E402


def to_fine_dets(image_id, dets):
    """把缓存里的 COCO 类 id 映射到细分类别（人 / 轿车 / 卡车 / 客车）。"""
    out = []
    for x1, y1, x2, y2, s, c in dets:
        if int(c) in COCO_TO_FINE:
            out.append({"image_id": image_id, "category_id": COCO_TO_FINE[int(c)],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return out


def eval_by_class(gt_path, dets, img_ids, max_dets, cat_ids):
    """GT 与 det 都已在细分类别上，按 catIds 过滤后重跑 COCOeval。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(dets) if dets else None
        out = {}
        for cid in cat_ids:
            if dt is None:
                out[cid] = dict(AP=0.0, AP50=0.0, APs=0.0)
                continue
            ev = COCOeval(gt, dt, "bbox")
            ev.params.imgIds = list(img_ids)
            ev.params.catIds = [cid]
            ev.params.maxDets = [1, 100, max_dets]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            out[cid] = dict(AP=ev.stats[0], AP50=ev.stats[1], APs=ev.stats[3])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", type=float, default=0.9)
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R.set_dataset(a.dataset)
    if a.dataset != "visdrone":
        raise SystemExit("细分评测目前只定义了 VisDrone 的类别映射")

    fine_path = Path(R.DS["dir"]) / "coco_fine.json"
    coco_fine = convert_fine(Path(R.DS["dir"]), fine_path, a.limit)
    names = {c["id"]: c["name"] for c in coco_fine["categories"]}
    n_gt = {cid: sum(1 for an in coco_fine["annotations"] if not an["iscrowd"] and an["category_id"] == cid)
            for cid in names}
    print("每类真值数：", {names[k]: v for k, v in n_gt.items()})

    cache = pickle.loads(R.CACHE.read_bytes())
    imgs = cache["images"][: a.limit] if a.limit else cache["images"]
    cfg = GlanceConfig(threshold=a.op, img_weight=a.img_weight)
    main_sel = lambda r: select_slices(R.slice_scores(r, cfg, "noisyor", "edge")[0], "threshold", a.op, 0)
    rng = np.random.default_rng(a.seed)
    methods = {"sahi_uniform": lambda r: np.arange(len(r["slices"])),
               f"glance[det+edge]@{a.op}": main_sel,
               f"random_matched@{a.op}#s{a.seed}": lambda r: random_slices(len(r["slices"]), len(main_sel(r)), rng)}

    rows = []
    for name, fn in methods.items():
        dets = []
        for rec in imgs:
            d, _ = R.merge(rec, fn(rec), cfg)
            dets += to_fine_dets(rec["id"], d)
        m = eval_by_class(fine_path, dets, [r["id"] for r in imgs], R.DS["max_dets"], list(names))
        for cid, met in m.items():
            rows.append(dict(method=name, category=names[cid], n_gt=n_gt[cid], **met))
            print(f"{name:26s} {names[cid]:8s} n_gt={n_gt[cid]:6d} "
                  f"AP={met['AP']:.4f} AP50={met['AP50']:.4f} APs={met['APs']:.4f}", flush=True)
        avg = float(np.mean([met["AP50"] for met in m.values()]))
        print(f"{name:26s} {'4类平均':8s}           AP50={avg:.4f}（这些类别不再被合并成 vehicle）")
    out = R.RES
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_class.csv", index=False)
    print(f"wrote {out / 'per_class.csv'}")


if __name__ == "__main__":
    main()
