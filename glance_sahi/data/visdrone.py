"""VisDrone2019-DET 原始标注 -> COCO json。

VisDrone 标注每行：bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
category: 0 ignored-region, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
          7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others

检测器是 COCO 预训练的 YOLO（零训练），因此只评测两类能跨数据集对上的目标：
  person  <- pedestrian, people           （COCO: person）
  vehicle <- car, van, truck, bus         （COCO: car, bus, truck）
ignored-region / others / score=0 的框记为 iscrowd=1：落在里面的检测不算误检（近似 VisDrone 官方规则）。
其余类别（自行车、三轮车、摩托）不参与评测，检测器对应的 COCO 类也在推理时排除。
"""

import json
from pathlib import Path

from ..imageio import imread

EVAL_CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]
VISDRONE_TO_EVAL = {1: 1, 2: 1, 4: 2, 5: 2, 6: 2, 9: 2}
IGNORE_VISDRONE = {0, 11}

# COCO(80 类, ultralytics 0-based id) -> 评测类别
COCO_TO_EVAL = {0: 1, 2: 2, 5: 2, 7: 2}  # person, car, bus, truck


# 细分评测：把 VisDrone 的 10 类按“COCO 预训练检测器能输出的 4 类”归并，
# 这样 truck / bus / car 的差异不会被合并成一个大类抹掉（见 REPORT 3.11）。
FINE_CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"},
                   {"id": 3, "name": "truck"}, {"id": 4, "name": "bus"}]
VISDRONE_TO_FINE = {1: 1, 2: 1, 4: 2, 5: 2, 6: 3, 9: 4}
# COCO(ultralytics 0-based) -> 细分类别：person, car, truck, bus
COCO_TO_FINE = {0: 1, 2: 2, 7: 3, 5: 4}


def convert_fine(root: Path, out_json: Path, limit: int | None = None) -> dict:
    """同 convert()，但保留 4 个细分评测类别（person / car / truck / bus）。"""
    img_dir, ann_dir = root / "images", root / "annotations"
    images, annotations = [], []
    ann_id = 1
    files = sorted(img_dir.glob("*.jpg"))
    if limit:
        files = files[:limit]
    for img_id, img_path in enumerate(files, start=1):
        h, w = imread(img_path).shape[:2]
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        for line in (ann_dir / f"{img_path.stem}.txt").read_text().splitlines():
            vals = [int(v) for v in line.strip().strip(",").split(",")[:8]]
            if len(vals) < 6:
                continue
            x, y, bw, bh, score, cat = vals[:6]
            if bw <= 0 or bh <= 0:
                continue
            if cat in IGNORE_VISDRONE or score == 0:
                for fine_cat in (1, 2, 3, 4):  # 忽略区域对四个细分类别都生效
                    annotations.append({"id": ann_id, "image_id": img_id, "category_id": fine_cat,
                                        "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 1})
                    ann_id += 1
            elif cat in VISDRONE_TO_FINE:
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": VISDRONE_TO_FINE[cat],
                                    "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": FINE_CATEGORIES}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    return coco


def convert(root: Path, out_json: Path, limit: int | None = None) -> dict:
    img_dir, ann_dir = root / "images", root / "annotations"
    images, annotations = [], []
    ann_id = 1
    files = sorted(img_dir.glob("*.jpg"))
    if limit:
        files = files[:limit]
    for img_id, img_path in enumerate(files, start=1):
        h, w = imread(img_path).shape[:2]
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        for line in (ann_dir / f"{img_path.stem}.txt").read_text().splitlines():
            vals = [int(v) for v in line.strip().strip(",").split(",")[:8]]
            if len(vals) < 6:
                continue
            x, y, bw, bh, score, cat = vals[:6]
            if bw <= 0 or bh <= 0:
                continue
            if cat in IGNORE_VISDRONE or score == 0:
                # 忽略区域对两个评测类别都生效
                for eval_cat in (1, 2):
                    annotations.append({"id": ann_id, "image_id": img_id, "category_id": eval_cat,
                                        "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 1})
                    ann_id += 1
            elif cat in VISDRONE_TO_EVAL:
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": VISDRONE_TO_EVAL[cat],
                                    "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": EVAL_CATEGORIES}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    return coco
