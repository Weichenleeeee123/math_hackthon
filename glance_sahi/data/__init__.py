"""数据集注册表：图像目录、评测 json、COCO→评测类别映射。"""

from pathlib import Path

from . import dota, visdrone

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "visdrone": {
        "dir": ROOT / "datasets" / "VisDrone2019-DET-val",
        "images": ROOT / "datasets" / "VisDrone2019-DET-val" / "images",
        "coco_to_eval": visdrone.COCO_TO_EVAL,
        "max_dets": 500,
    },
    "dota": {
        "dir": ROOT / "datasets" / "DOTAv1",
        "images": ROOT / "datasets" / "DOTAv1" / "images" / "val",
        "coco_to_eval": dota.COCO_TO_EVAL,
        "max_dets": 2000,  # DOTA 单图目标可上千
    },
    # 同一份 DOTA val，但评全部 15 类，配 DOTA 上训练过的 OBB 检测器（yolo11s-obb.pt）使用。
    # 真值另存一个文件，不覆盖 COCO 检测器用的 3 类 coco_eval.json。
    "dota15": {
        "dir": ROOT / "datasets" / "DOTAv1",
        "images": ROOT / "datasets" / "DOTAv1" / "images" / "val",
        "gt_name": "coco_eval_dota15.json",
        "coco_to_eval": dota.OBB_TO_EVAL,
        "max_dets": 2000,
    },
    # 受控实验画布（scripts/sparsity_sweep.py --save 生成）：3×3 拼接的 4K 图，
    # 用来在 GPU 上补“稀疏度 → 精度/耗时”的真实 AP（可选，见 REPORT 3.10）
    "sparse4k": {
        "dir": ROOT / "datasets" / "VisDrone-Sparse4K",
        "images": ROOT / "datasets" / "VisDrone-Sparse4K" / "images",
        "coco_to_eval": visdrone.COCO_TO_EVAL,
        "max_dets": 500,
    },
}


def get(name: str) -> dict:
    d = dict(DATASETS[name])
    d["gt"] = d["dir"] / d.get("gt_name", "coco_eval.json")
    d["exclude_coco_ids"] = [i for i in range(80) if i not in d["coco_to_eval"]]
    return d
