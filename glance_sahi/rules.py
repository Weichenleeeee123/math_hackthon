"""业务规则层：把“检测框”变成“业务事件”。

以违停为例：车辆框的**着地点**（底边中点）落在禁停多边形内 → 判为“车停在禁停区”。
走查核对：静帧只能判断“车在禁停区”，真正的违停还需要停留时长，视频里要配跟踪做
dwell-time 判定（见 REPORT 4.3）。这一层是纯几何，无模型/GPU 依赖，可直接单测。

规则层与选片逻辑解耦：选片少切了哪片，只影响“有没有检出这辆车”，不影响这里怎么判。
"""

import json
from pathlib import Path

import cv2
import numpy as np


def ground_point(box) -> tuple[float, float]:
    """车辆着地点 = 框底边中点（斜视/俯视画面里比框中心更接近车辆真实地面位置）。"""
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box, np.float32).reshape(-1)[:4])
    return (x1 + x2) / 2.0, y2


def load_zones(path) -> list[np.ndarray]:
    """禁停区：json 可以是 `[[[x,y],...], ...]`，也可以是 `{"zones": [{"points": [...]}, ...]}`。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    polys = data.get("zones", []) if isinstance(data, dict) else data
    out = []
    for p in polys:
        pts = p["points"] if isinstance(p, dict) else p
        arr = np.asarray(pts, np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
            out.append(arr)
    return out


def find_illegal_parking(dets, zones, vehicle_ids) -> list[dict]:
    """dets: (N,6) xyxy, score, class_id；zones: 像素坐标多边形列表。返回命中列表（含所在 zone）。"""
    hits = []
    vids = set(int(v) for v in vehicle_ids)
    for row in np.asarray(dets, np.float32).reshape(-1, 6):
        box, score, cls = row[:4], float(row[4]), int(row[5])
        if cls not in vids:
            continue
        pt = ground_point(box)
        for zi, poly in enumerate(zones):
            if cv2.pointPolygonTest(np.asarray(poly, np.float32), pt, False) >= 0:
                hits.append({"box": [float(v) for v in box], "score": score, "cls": cls,
                             "zone": zi, "ground_point": pt})
                break
    return hits


def suggest_zone(dets, vehicle_ids, shape_hw, rel: float = 0.10) -> list[np.ndarray]:
    """演示用：在车辆（按着地点）最密集处自动生成一个矩形禁停条带。

    真实部署时应由人工在画面上框选（或用电子围栏/标定信息），这里只是让 demo 能一键跑通。
    """
    h, w = shape_hw
    vids = set(int(v) for v in vehicle_ids)
    pts = [ground_point(r[:4]) for r in np.asarray(dets, np.float32).reshape(-1, 6) if int(r[5]) in vids]
    if not pts:
        return []
    cx, cy = np.median(np.array(pts, np.float32), 0)
    hw, hh = w * rel, h * rel * 0.6
    poly = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                     [cx + hw, cy + hh], [cx - hw, cy + hh]], np.float32)
    return [np.clip(poly, [0, 0], [w - 1, h - 1])]
