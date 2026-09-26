"""检测器：SAHI 的 ultralytics 封装 + COCO 预训练 YOLO11（零训练）。"""

from sahi import AutoDetectionModel

from .data.visdrone import COCO_TO_EVAL

# 推理时排除与评测无关的 COCO 类别（SAHI 基线和 Glance-SAHI 用同一个排除表）
EXCLUDE_COCO_IDS = [i for i in range(80) if i not in COCO_TO_EVAL]


def resolve_device(device: str = "auto") -> str:
    """"auto"：有 CUDA 用 cuda:0，否则回退 cpu（评委电脑 / 无显卡笔记本也能跑 Demo）。其他值原样返回。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_model(weights: str = "yolo11s.pt", conf: float = 0.05, device: str = "auto", image_size: int = 640):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=conf,
        device=resolve_device(device),
        image_size=image_size,
    )
