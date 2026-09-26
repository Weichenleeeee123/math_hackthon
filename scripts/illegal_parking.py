"""违停判定 demo：Glance-SAHI 检测 → 规则层（着地点 × 禁停多边形）。

用法（PowerShell）：
  $py scripts/illegal_parking.py --image datasets/VisDrone2019-DET-val/images/0000100_00504_d_0000004.jpg
  $py scripts/illegal_parking.py --image <图> --zones zones.json --op 0.9     # 指定禁停区（像素坐标）
  $py scripts/illegal_parking.py --image <图> --auto-zone                     # 没给 zones：自动挑一块演示

不传 --zones 也不加 --auto-zone 时，默认自动生成一块（仅为跑通流程）。

走查核对：**静帧只能判断“车在禁停区”**；真正的违停还需要停留时长（dwell-time），
视频里要配跟踪：连续 N 帧同一辆车落在同一禁停区且位移小于阈值才告警。见 REPORT 4.3。
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures as MF  # noqa: E402,F401  （字体与配色）
from run_eval import preds_to_np, load_rgb  # noqa: E402

from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.detector import EXCLUDE_COCO_IDS, build_model  # noqa: E402
from glance_sahi.predict import glance_sliced_prediction  # noqa: E402
from glance_sahi.rules import find_illegal_parking, ground_point, load_zones, suggest_zone  # noqa: E402

# COCO 类 id：2=car, 5=bus, 7=truck（detector.py 的 COCO_TO_EVAL 里映射到 vehicle）
VEHICLE_COCO_IDS = [2, 5, 7]
SERIES_RED = "#e34948"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--zones", help="禁停区 json：[{\"points\": [[x,y],...]}] 或 [[[x,y],...]]")
    ap.add_argument("--auto-zone", action="store_true", help="没给 zones 时自动挑一块（演示用）")
    ap.add_argument("--op", type=float, default=0.9)
    ap.add_argument("--img-weight", type=float, default=0.3)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "results" / "vis"))
    a = ap.parse_args()

    cfg = GlanceConfig(threshold=a.op, img_weight=a.img_weight)
    model = build_model(a.weights, conf=cfg.output_conf, device=a.device)
    img = load_rgb(a.image)
    h, w = img.shape[:2]
    result, st = glance_sliced_prediction(img, model, cfg, EXCLUDE_COCO_IDS)
    dets = preds_to_np(result.object_prediction_list)

    zones = load_zones(a.zones) if a.zones else []
    if not zones:
        zones = suggest_zone(dets, VEHICLE_COCO_IDS, (h, w), rel=0.10)
        print("未指定禁停区，已自动生成一块演示用区域（真实部署请框选或接入电子围栏）")
    hits = find_illegal_parking(dets, zones, VEHICLE_COCO_IDS)

    n_veh = int(np.isin(dets[:, 5].astype(int), VEHICLE_COCO_IDS).sum()) if len(dets) else 0
    print(f"{Path(a.image).name}: 切片 {st.n_slices_run}/{st.n_slices_total}，"
          f"{1000 * st.t_total:.0f} ms，检出 {len(dets)} 框（其中车辆 {n_veh}），"
          f"禁停区内 {len(hits)} 辆")
    for hit in hits:
        x, y = hit["ground_point"]
        print(f"  zone{hit['zone']}  score={hit['score']:.2f}  着地点=({x:.0f},{y:.0f})  "
              f"box={[round(v) for v in hit['box']]}")
    print("提示：静帧只能判“车在禁停区”；违停还需要 dwell-time（视频 + 跟踪）。")

    fig, ax = plt.subplots(figsize=(12, 12 * h / w))
    ax.imshow(img)
    for i, poly in enumerate(zones):
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=SERIES_RED, alpha=0.18,
                                edgecolor=SERIES_RED, lw=2, label="禁停区" if i == 0 else None))
    for x1, y1, x2, y2, s, c in dets:
        if s < 0.3:
            continue
        veh = int(c) in VEHICLE_COCO_IDS
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0,
                               edgecolor=MF.SERIES[1] if veh else MF.SERIES[2]))
    for hit in hits:
        x1, y1, x2, y2 = hit["box"]
        ax.add_patch(Rectangle((x1 - 3, y1 - 3), x2 - x1 + 6, y2 - y1 + 6, fill=False, lw=2.4,
                               edgecolor=SERIES_RED))
        gx, gy = hit["ground_point"]
        ax.plot([gx], [gy], marker="o", ms=5, color=SERIES_RED)
    ax.set_title(f"{Path(a.image).name}：Glance-SAHI {st.n_slices_run}/{st.n_slices_total} 片，"
                 f"违停判定 {len(hits)} 辆（红框；红点 = 着地点）｜静帧不含 dwell-time",
                 loc="left", color=MF.INK, fontsize=10)
    ax.axis("off")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"illegal_parking_{Path(a.image).stem}.jpg", dpi=110)
    plt.close(fig)
    print(f"→ {out / f'illegal_parking_{Path(a.image).stem}.jpg'}")


if __name__ == "__main__":
    main()
