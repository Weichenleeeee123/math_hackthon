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


def build_model(weights: str = "yolo11s.pt", conf: float = 0.05, device: str = "auto", image_size: int = 640,
                half: bool = False):
    """half=True：FP16 推理（只在 CUDA 上生效）。RTX 3050 上整图 @1920 55 → 34 ms、15 片批推理 200 → 133 ms；
    检测结果不再与 FP32 逐位一致，AP 差见 REPORT 3.18。默认 False，已有结果全部是 FP32。"""
    device = resolve_device(device)
    m = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=conf,
        device=device,
        image_size=image_size,
    )
    if half and device != "cpu":
        # ultralytics ≥ 8.4 用 quantize=16 取代 half；SAHI 每次调用都经 model(...) 走 predictor，会读到这个 override
        m.model.overrides["quantize"] = 16
        m.model.predictor = None
    return m
