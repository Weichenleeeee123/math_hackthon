"""精度测量：COCO AP（pycocotools），供 Demo 的单图质量指标与评测脚本共用。

只用于"算分"，不参与计时：COCOeval 单次约 70 ms，夹在耗时测量里会把速度对比带偏。
口径与 scripts/run_eval.py 完全一致（maxDets[2] 重算 AP，见 coco_eval 注释）。
"""

import contextlib
import io
from pathlib import Path

import numpy as np

_COCO_CACHE: dict = {}


def merge_preds(rec_or_parts, sel=None, cfg=None):
    """把选中的切片预测 + 扫视检测合并成一组框（batched NMS）。

    两种调用方式：
      merge_preds(cache_rec, sel, cfg)            —— 评测脚本用缓存
      merge_preds([pred_arrays...], cfg=cfg)      —— Demo 用实时结果
    """
    from sahi.postprocess.combine import batched_nms

    if cfg is None and sel is not None and not isinstance(sel, (list, range, tuple, np.ndarray)):
        sel, cfg = None, sel
    if isinstance(rec_or_parts, dict):
        parts = [rec_or_parts["slice_preds"][k] for k in sel]
        g = rec_or_parts["glance"]
        parts.append(g[g[:, 4] >= cfg.output_conf])
    else:
        parts = list(rec_or_parts)
    dets = np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)
    if len(dets) > 1:
        dets = dets[batched_nms(dets, match_metric=cfg.postprocess_match_metric,
                                match_threshold=cfg.postprocess_match_threshold)]
    return dets


def to_coco_dets(image_id, dets, coco_to_eval):
    out = []
    for x1, y1, x2, y2, s, c in np.asarray(dets).reshape(-1, 6):
        if int(c) in coco_to_eval:
            out.append({"image_id": image_id, "category_id": coco_to_eval[int(c)],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)})
    return out


def coco_eval(gt_path, dets, img_ids, max_dets=500):
    """返回 AP / AP50 / APs。dets 为空或该图无真值时返回 None 字典。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = str(gt_path)
    if gt_path not in _COCO_CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            _COCO_CACHE[gt_path] = COCO(gt_path)
    gt = _COCO_CACHE[gt_path]
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "APs": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.params.maxDets = [1, 100, max_dets]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    # stats[0] 固定按每图 100 个检测算；航拍图常超过 100 个目标，这里按 maxDets[2] 重算，与报告同口径
    prec = ev.eval["precision"][:, :, :, 0, -1]
    ap = float(prec[prec > -1].mean()) if (prec > -1).any() else None
    return {"AP": ap, "AP50": float(s[1]), "APs": float(s[3])}


def gt_boxes(gt_path, image_id=None, file_name=None):
    """取某张图的真值框 (M,5) [x1,y1,x2,y2,cat]，没有就返回空。"""
    import json

    gt_path = Path(gt_path)
    if not gt_path.exists():
        return np.zeros((0, 5), np.float32)
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    iid = image_id
    if iid is None and file_name:
        iid = next((i["id"] for i in data["images"] if i.get("file_name") == file_name), None)
    if iid is None:
        return np.zeros((0, 5), np.float32)
    rows = []
    for a in data["annotations"]:
        if a["image_id"] != iid or a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        rows.append([x, y, x + w, y + h, a["category_id"]])
    return np.array(rows, np.float32).reshape(-1, 5)


def gt_image_id(gt_path, file_name):
    """按文件名查 image_id；不在真值里返回 None（自己上传的图通常如此）。"""
    import json

    gt_path = Path(gt_path)
    if not gt_path.exists() or not file_name:
        return None
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    return next((i["id"] for i in data["images"] if i.get("file_name") == file_name), None)
