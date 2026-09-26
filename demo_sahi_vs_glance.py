"""Glance-SAHI 现场 Demo（Gradio）：同一张图，SAHI 全切 vs Glance-SAHI 只切可疑区域，三栏同屏对比。

用法：
  pip install gradio
  python demo_sahi_vs_glance.py                 # 浏览器打开 http://127.0.0.1:7860
  python demo_sahi_vs_glance.py --device cpu    # 没有 GPU 时（很慢，只适合看选片效果）

三栏：左 SAHI 全部切片；中 Glance-SAHI 选中的切片（跳过的压暗）；右 扫视打分 S(k)（越亮越可疑）+ 扫视弱框。
耗时为本机单次实测：每种（检测器，图像尺寸）第一次运行前会先预热一次，不计入耗时。
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from glance_sahi.config import GlanceConfig
from glance_sahi import data as datasets
from glance_sahi.detector import build_model
from glance_sahi.predict import glance_sliced_prediction, sahi_uniform_prediction

ROOT = Path(__file__).resolve().parent

# 预设 = 论文里用过的三组设置（检测器、输入尺寸、切片大小、默认 θ / λ）
PRESETS = {
    "VisDrone · COCO 零训练 YOLO11s": dict(weights="yolo11s.pt", imgsz=640, slice=512, theta=0.9, lam=0.3,
                                          exclude=datasets.get("visdrone")["exclude_coco_ids"]),
    "VisDrone · 微调 YOLO11s": dict(weights=str(ROOT / "weights" / "yolo11s-visdrone-ft.pt"), imgsz=640, slice=512,
                                   theta=0.99, lam=0.3, exclude=None),
    "DOTA · 官方 YOLO11s-OBB": dict(weights="yolo11s-obb.pt", imgsz=1024, slice=1024, theta=0.5, lam=0.3, exclude=None),
}
DEFAULT = next(iter(PRESETS))

_VD = ROOT / "datasets" / "VisDrone2019-DET-val" / "images"
EXAMPLES = [
    (_VD / "0000165_04325_d_0000105.jpg", DEFAULT),  # 成功案例：天空和楼宇被跳过
    # 失败案例（REPORT 3.5）：斜视远景、低对比，扫视看不到就选不到；换微调检测器后盲区变小
    (_VD / "0000001_04527_d_0000008.jpg", DEFAULT),
    (_VD / "0000001_04527_d_0000008.jpg", "VisDrone · 微调 YOLO11s"),
    (ROOT / "datasets" / "DOTAv1" / "images" / "val" / "P0179.jpg", "DOTA · 官方 YOLO11s-OBB"),
]

DISPLAY = 1100  # 显示图的长边（像素）
ORANGE, WHITE, GREEN = (235, 104, 52), (255, 255, 255), (27, 175, 122)

_models, _warm = {}, set()


def get_model(name, device):
    p = PRESETS[name]
    if name not in _models:
        _models[name] = build_model(p["weights"], conf=GlanceConfig().output_conf, device=device, image_size=p["imgsz"])
    return _models[name]


def warmup(name, model, img, cfg, exclude):
    """首次遇到新的（检测器，输入尺寸）时 CUDA/cuDNN 有一次性开销，不应计入耗时。"""
    key = (name, img.shape[:2])
    if key not in _warm:  # 完整跑一遍 SAHI：边缘切片尺寸不同，只预热 512 方片不够
        sahi_uniform_prediction(img, model, cfg, exclude)
        _warm.add(key)


# ------------------------------------------------------------------------------------ 画图
def _scaled(img):
    s = DISPLAY / max(img.shape[:2])
    s = min(s, 1.0)
    return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA), s


def _rect(canvas, box, s, color, t, inset=0):
    x1, y1, x2, y2 = (int(round(v * s)) for v in box)
    cv2.rectangle(canvas, (x1 + inset, y1 + inset), (x2 - inset, y2 - inset), color, t, cv2.LINE_AA)


def _dets(canvas, preds, s):
    for p in preds:
        _rect(canvas, p.bbox.to_xyxy(), s, GREEN, 2)


def _label(canvas, text):
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 18 + 11 * len(text)), 30), (20, 40, 90), -1)
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)


def draw_sahi(img, slices, preds):
    c, s = _scaled(img)
    for b in slices:
        _rect(c, b, s, ORANGE, 1, inset=2)
    _dets(c, preds, s)
    _label(c, f"SAHI: {len(slices)}/{len(slices)} slices")
    return c


def draw_glance(img, st, preds):
    c, s = _scaled(img)
    sel = set(int(k) for k in st.selected)
    for k, b in enumerate(st.slices):  # 未推理的切片压暗：一眼看出跳过了什么
        if k not in sel:
            x1, y1, x2, y2 = (int(round(v * s)) for v in b)
            sub = c[y1:y2, x1:x2]
            sub[:] = (sub * 0.4 + 40).astype(np.uint8)
    for k in sel:
        _rect(c, st.slices[k], s, ORANGE, 3, inset=2)
    _dets(c, preds, s)
    _label(c, f"Glance-SAHI: {len(sel)}/{st.n_slices_total} slices")
    return c


def draw_scores(img, st, theta):
    """每个像素取覆盖它的切片分数最大值；越亮越可疑，过门槛的切片描橙边。"""
    c, s = _scaled(img)
    heat = np.zeros(c.shape[:2], np.float32)
    for k, b in enumerate(st.slices):
        x1, y1, x2, y2 = (int(round(v * s)) for v in b)
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], st.slice_scores[k])
    gray = cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32)
    warm = np.array(ORANGE, np.float32)
    h = heat[..., None]
    c = (gray * (0.25 + 0.45 * h) + warm * 0.45 * h).clip(0, 255).astype(np.uint8)
    if st.glance_boxes is not None:  # 扫视阶段的弱框：点越大置信度越高
        for x1, y1, x2, y2, conf in st.glance_boxes:
            cx, cy = int((x1 + x2) / 2 * s), int((y1 + y2) / 2 * s)
            cv2.circle(c, (cx, cy), 2 + int(6 * conf), (255, 225, 90), -1, cv2.LINE_AA)
    for k, b in enumerate(st.slices):
        if st.slice_scores[k] >= theta:
            _rect(c, b, s, ORANGE, 2, inset=2)
    if len(st.slices) <= 40:
        for k, (x1, y1, x2, y2) in enumerate(st.slices):
            cx, cy = int((x1 + x2) / 2 * s), int((y1 + y2) / 2 * s)
            cv2.putText(c, f"{st.slice_scores[k]:.2f}", (cx - 18, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        WHITE, 1, cv2.LINE_AA)
    _label(c, f"glance score S(k), theta={theta:g}")
    return c


# ------------------------------------------------------------------------------------ 推理
def run(img, name, theta, lam, device):
    if img is None:
        raise gr.Error("先上传一张图，或点下面的示例")
    p = PRESETS[name]
    model = get_model(name, device)
    cfg = GlanceConfig(slice_size=p["slice"])
    gcfg = replace(cfg, threshold=float(theta), img_weight=float(lam))
    warmup(name, model, img, cfg, p["exclude"])

    sahi_res, t_sahi, n_all = sahi_uniform_prediction(img, model, cfg, p["exclude"])
    g_res, st = glance_sliced_prediction(img, model, gcfg, p["exclude"])
    sahi_preds, g_preds = sahi_res.object_prediction_list, g_res.object_prediction_list

    t_g = st.t_total
    k, n = st.n_slices_run, st.n_slices_total
    md = f"""
| | SAHI 全切 | Glance-SAHI |
|---|---:|---:|
| 切片推理次数 | {n_all} | **{k}**（{k / max(n, 1):.0%}） |
| 输出检测框 | {len(sahi_preds)} | {len(g_preds)} |
| 耗时（ms/图） | {t_sahi * 1000:.0f} | **{t_g * 1000:.0f}**（{t_g / t_sahi:.0%}） |

Glance 耗时拆分：扫视 {st.t_glance * 1000:.0f} ms · 打分选片 {st.t_saliency * 1000:.1f} ms ·
切片推理 {st.t_slices * 1000:.0f} ms · 合并 {st.t_post * 1000:.1f} ms。
图 {img.shape[1]}×{img.shape[0]}，检测器 `{Path(p['weights']).name}`，切片 {p['slice']}，θ = {theta:g}，λ = {lam:g}，设备 {device}。
"""
    return (draw_sahi(img, [tuple(b) for b in st.slices], sahi_preds), draw_glance(img, st, g_preds),
            draw_scores(img, st, float(theta)), md)


def build_ui(device):
    with gr.Blocks(title="Glance-SAHI Demo") as demo:
        gr.Markdown("## Glance-SAHI：先扫一眼全图，只切可疑区域\n"
                    "整图缩小推理一次（阈值 0.01）→ 每片用 noisy-OR 攒弱证据打分 S(k) → 只对 S(k) ≥ θ 的切片放大细看。"
                    "θ = 0 时与官方 SAHI 逐位一致。")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Image(type="numpy", label="输入图像", height=320)
            with gr.Column(scale=1):
                preset = gr.Dropdown(list(PRESETS), value=DEFAULT, label="检测器预设")
                theta = gr.Slider(0.0, 1.0, value=PRESETS[DEFAULT]["theta"], step=0.01, label="θ：切片分数门槛（越大越省）")
                lam = gr.Slider(0.0, 1.0, value=PRESETS[DEFAULT]["lam"], step=0.05, label="λ：图像先验权重")
                btn = gr.Button("运行对比", variant="primary")
        ex = [[str(pth), nm] for pth, nm in EXAMPLES if pth.exists()]
        if ex:
            gr.Examples(ex, inputs=[inp, preset], label="示例（来自本地数据集）")
        with gr.Row():  # 主对比放大并排，方便投影
            out_sahi = gr.Image(label="SAHI：全部切片", height=460)
            out_glance = gr.Image(label="Glance-SAHI：只跑橙框切片（其余压暗）", height=460)
        with gr.Row():
            out_heat = gr.Image(label="扫视打分 S(k)：越亮越可疑，黄点为扫视弱框", height=340)
            stats = gr.Markdown()

        preset.change(lambda nm: (PRESETS[nm]["theta"], PRESETS[nm]["lam"]), preset, [theta, lam])
        btn.click(lambda im, nm, th, lm: run(im, nm, th, lm, device), [inp, preset, theta, lam],
                  [out_sahi, out_glance, out_heat, stats])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui(args.device).launch(server_name="127.0.0.1", server_port=args.port)
