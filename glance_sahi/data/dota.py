"""DOTA-v1.0 val（ultralytics 发布的 YOLO-OBB 格式）-> COCO json（水平框）。

标签每行：cls x1 y1 x2 y2 x3 y3 x4 y4（归一化）。ultralytics DOTAv1 类别：
  0 plane, 1 ship, 2 storage tank, 3 baseball diamond, 4 tennis court, 5 basketball court,
  6 ground track field, 7 harbor, 8 bridge, 9 large vehicle, 10 small vehicle, 11 helicopter,
  12 roundabout, 13 soccer ball field, 14 swimming pool

只评测 COCO 检测器能对上的三类：
  vehicle <- small vehicle, large vehicle   （COCO: car, bus, truck）
  ship    <- ship                            （COCO: boat）
  plane   <- plane                           （COCO: airplane）
"""

import json
from pathlib import Path

import numpy as np

from ..imageio import imread

EVAL_CATEGORIES = [{"id": 1, "name": "vehicle"}, {"id": 2, "name": "ship"}, {"id": 3, "name": "plane"}]
DOTA_TO_EVAL = {10: 1, 9: 1, 1: 2, 0: 3}
COCO_TO_EVAL = {2: 1, 5: 1, 7: 1, 8: 2, 4: 3}  # car, bus, truck, boat, airplane


# 换成在 DOTA 上训练过的 OBB 检测器（yolo11*-obb.pt，15 类）后，全部 15 类都能评测：
# 类别 id = DOTA id + 1；旋转框取外接水平框（DOTA Task2 的 HBB 口径），检测端用 SAHI 给的 obb.xyxy。
ALL_CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate([
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
    "roundabout", "soccer ball field", "swimming pool"])]
DOTA_TO_ALL = {i: i + 1 for i in range(15)}
OBB_TO_EVAL = dict(DOTA_TO_ALL)  # 检测器输出的类别 id 与 DOTA id 相同


def convert_all(root: Path, out_json: Path, limit: int | None = None) -> dict:
    return convert(root, out_json, limit, mapping=DOTA_TO_ALL, categories=ALL_CATEGORIES)


def convert(root: Path, out_json: Path, limit: int | None = None,
            mapping: dict = DOTA_TO_EVAL, categories: list = EVAL_CATEGORIES) -> dict:
    img_dir, lab_dir = root / "images" / "val", root / "labels" / "val"
    images, annotations = [], []
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if limit:
        files = files[:limit]
    ann_id = 1
    for img_id, p in enumerate(files, start=1):
        h, w = imread(p).shape[:2]
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lab = lab_dir / f"{p.stem}.txt"
        if not lab.exists():
            continue
        for line in lab.read_text().splitlines():
            v = line.split()
            if len(v) < 9 or int(v[0]) not in mapping:
                continue
            pts = np.array(v[1:9], dtype=np.float32).reshape(4, 2) * [w, h]
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw < 1 or bh < 1:
                continue
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": mapping[int(v[0])],
                                "bbox": [float(x1), float(y1), bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_json.write_text(json.dumps(coco))
    return coco
