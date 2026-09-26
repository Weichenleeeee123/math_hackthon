from dataclasses import dataclass


@dataclass
class GlanceConfig:
    # --- 与 SAHI 基线完全一致的切片与后处理参数 ---
    slice_size: int = 512
    overlap_ratio: float = 0.2
    # 输出阈值 0.05 < 0.1 时，SAHI 默认会自动切到 NMS/IOU（避免 GREEDYNMM 把低分框合并变大），
    # 这里显式固定为 NMS/IOU，SAHI 基线与 Glance-SAHI 共用同一个后处理。
    postprocess_type: str = "NMS"
    postprocess_match_metric: str = "IOU"
    postprocess_match_threshold: float = 0.5

    # --- 置信度 ---
    output_conf: float = 0.05   # 最终输出的检测阈值（SAHI 基线相同）
    glance_conf: float = 0.01   # 扫视阶段放低阈值：弱检测也是“这里可能有东西”的证据

    # --- 显著性打分 ---
    det_margin: int = 16        # 粗检测框中心落在切片外扩 margin 像素内也算
    # 检测先验的权重取法（打分函数消融，见 REPORT 3.8、3.9）：
    #   "noisyor"   1 − Π(1 − c_j)：置信度当作概率，弱证据可累积（默认，主方法）
    #   "uncertain" 4c(1−c) 加权后再 noisy-OR：c≈0.5 的框最热
    #   "max"       片内最大置信度（不做累积）：v0 反面消融
    #   "heatmap"   撒点 + 一次高斯模糊 + 1 − exp(−Σc)：O(像素)，无硬边界，分数是绝对量
    det_prior: str = "noisyor"
    heat_sigma: float = 6.0     # 热图变体的高斯核（缩略图像素）：决定框的贡献软扩散多远
    img_prior: str = "edge"     # "edge" | "spectral" | "none"
    img_weight: float = 0.3     # 图像先验在 noisy-OR 融合中的权重 λ（检测器不认识该场景时调到 ~1.0，见 REPORT 3.7）
    img_map_size: int = 512     # 计算图像先验时的缩略图长边

    # --- 选片规则 ---
    mode: str = "threshold"     # "threshold" | "budget" | "evidence"
    threshold: float = 0.9      # 切片分数 ≥ threshold 才细看（VisDrone 默认工作点；0.99 更激进）
    budget: float = 0.5         # mode="budget" 时保留分数最高的比例
    tau: float = 1.0            # mode="evidence"：片内证据量（弱检测置信度之和）≥ τ 才细看，绝对值可跨图标定
    min_slices: int = 0         # 保底切片数：不足时按分数补齐（0=关闭；evidence 模式自动至少 1）
