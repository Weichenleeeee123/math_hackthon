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


def convert(root: Path, out_json: Path, limit: int | None = None) -> dict:
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
            if len(v) < 9 or int(v[0]) not in DOTA_TO_EVAL:
                continue
            pts = np.array(v[1:9], dtype=np.float32).reshape(4, 2) * [w, h]
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw < 1 or bh < 1:
                continue
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": DOTA_TO_EVAL[int(v[0])],
                                "bbox": [float(x1), float(y1), bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": EVAL_CATEGORIES}
    out_json.write_text(json.dumps(coco))
    return coco
