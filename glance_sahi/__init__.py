"""Glance-SAHI: 先缩小看全图、只切可疑区域的 SAHI 切片推理。"""

from .config import GlanceConfig
from .predict import glance_sliced_prediction

__all__ = ["GlanceConfig", "glance_sliced_prediction"]
