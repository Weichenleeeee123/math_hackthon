"""Glance-SAHI 交互 Demo：稠密切片（SAHI） vs 手工稀疏门 vs 可学习稀疏路由器。

  & .\\.venv\\Scripts\\python.exe app.py                 # 默认用最新微调模型（weights/yolo11s-visdrone-ft.pt）
  & .\\.venv\\Scripts\\python.exe app.py --weights yolo11s.pt          # 切回 COCO 零训练模型
  & .\\.venv\\Scripts\\python.exe app.py --dataset dota --port 7860 --share

三种方法都调用与评测完全相同的函数（sahi_uniform_prediction / glance_sliced_prediction），
界面上的切片数与耗时就是真实运行值，不是模拟。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from glance_sahi import data as datasets  # noqa: E402
from glance_sahi.ap import coco_eval, gt_boxes, gt_image_id, merge_preds, to_coco_dets  # noqa: E402
from glance_sahi.config import GlanceConfig  # noqa: E402
from glance_sahi.detector import build_model, resolve_device  # noqa: E402
from glance_sahi.predict import glance_sliced_prediction, preds_to_np as to_np, sahi_uniform_prediction  # noqa: E402
from glance_sahi.viz import draw_dets, draw_slices, overlay_heat  # noqa: E402

EXAMPLES = ROOT / "datasets" / "VisDrone2019-DET-val" / "images"
EXAMPLE_NAMES = ["0000001_02999_d_0000005.jpg", "0000022_01036_d_0000006.jpg", "0000242_00001_d_0000001.jpg",
                 "0000330_00801_d_0000804.jpg", "0000001_03999_d_0000007.jpg"]
DOTA_EXAMPLES = ["P1029.jpg", "P1179.jpg"]   # 中等幅面 28 片 / 大幅面 6.5K×6.6K 256 片

# 默认用仓库里最新的微调模型；没有就退回 COCO 零训练权重
FT_WEIGHTS = ROOT / "weights" / "yolo11s-visdrone-ft.pt"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --ink: #16202b; --ink-2: #4a5663; --ink-3: #8a95a1;
  --line: #e6e9ed; --paper: #ffffff; --canvas: #f5f6f4;
  --accent: #0f766e; --accent-soft: #e6f2f0;
  --act: #28dc5a;  /* 与 viz.draw_slices 激活切片的 (40, 220, 90) 一致 */
  --good: #15803d; --good-bg: #ecf7ef; --warn: #b45309; --warn-bg: #fdf4e7;
  --mono: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace;
}
.gradio-container { background: var(--canvas) !important; max-width: 1480px !important; margin: 0 auto !important; }
.gradio-container, .gradio-container * { font-family: 'Inter', 'PingFang SC', 'HarmonyOS Sans SC',
  'Microsoft YaHei', system-ui, sans-serif; }

/* 顶部 */
.hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap;
  padding: 22px 4px 18px; margin-bottom: 6px; border-bottom: 1px solid var(--line); }
.hero .eyebrow { font-family: var(--mono) !important; font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 8px; }
.hero h1 { margin: 0; font-size: 28px; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.hero h1 span { color: var(--ink-3); font-weight: 500; }
.hero p { margin: 8px 0 0; font-size: 14px; line-height: 1.7; color: var(--ink-2); max-width: 720px; }
.hero p b { color: var(--ink); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12.5px; color: var(--ink-2); }
.legend i { display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend .l-act { border: 2.5px solid var(--act); }
.legend .l-skip { background: #4b5159; border: 1px solid #969696; }
.legend .l-det { background: linear-gradient(90deg, #2a78d6 0 33%, #eb6834 33% 66%, #1baf7a 66%); }
.legend .l-heat { background: linear-gradient(90deg, #000004, #b53679, #fcffa4); }
.meta-line { font-family: var(--mono) !important; font-size: 11.5px; color: var(--ink-3); margin-top: 10px; width: 100%; }

/* 卡片容器 */
.panel { background: var(--paper) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
  padding: 18px !important; box-shadow: 0 1px 2px rgba(22, 32, 43, .04) !important; }
.sec-title { font-size: 12px; font-weight: 600; color: var(--ink-3); letter-spacing: .08em; text-transform: uppercase;
  margin: 2px 0 4px; }
.sec-title b { color: var(--ink); font-size: 14px; letter-spacing: 0; text-transform: none; margin-right: 8px; }
.run-btn button, button.run-btn { height: 48px !important; font-size: 15px !important; font-weight: 600 !important;
  background: var(--accent) !important; border: none !important; border-radius: 10px !important; }
.run-btn button:hover, button.run-btn:hover { background: #0b5f58 !important; }

/* KPI */
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 0 0 12px; }
@media (max-width: 1200px) { .kpis { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 760px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
.kpi { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  animation: kpin .4s ease-out both; }
@keyframes kpin { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .kpi { animation: none; } }
.kpi:nth-child(2) { animation-delay: .05s; }
.kpi:nth-child(3) { animation-delay: .1s; }
.kpi:nth-child(4) { animation-delay: .15s; }
.kpi .k-label { font-size: 12.5px; color: var(--ink-2); display: flex; align-items: center; gap: 7px; }
.kpi .k-label::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--ink-3); }
.kpi .k-value { font-family: var(--mono) !important; font-size: 30px; font-weight: 600; color: var(--ink);
  margin: 6px 0 2px; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.kpi .k-sub { font-size: 12px; color: var(--ink-3); }
.kpi.good .k-label::before { background: var(--good); }
.kpi.warn .k-label::before { background: var(--warn); }
.kpi.good .k-value { color: var(--good); }
.kpi.warn .k-value { color: var(--warn); }
.kpi.info .k-label::before { background: var(--accent); }
.kpi.hl { background: var(--accent-soft); border-color: #bfe0db; }
.kpi.empty .k-value { color: var(--ink-3); }

.badges { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 4px; }
.badge { font-family: var(--mono) !important; display: inline-block; padding: 3px 9px; border-radius: 6px;
  font-size: 11.5px; background: var(--paper); color: var(--ink-2); border: 1px solid var(--line); }
.badge.ok { color: var(--good); background: var(--good-bg); border-color: #cfe9d6; }
.badge.off { color: #b91c1c; background: #fdecec; border-color: #f5cccc; }

/* 精度：验证集 AP 表 + 单图 AP 卡 */
.bench { background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px; margin-bottom: 12px; }
.bench table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
.bench th { text-align: left; font-weight: 600; color: var(--ink-3); font-size: 12px;
  border-bottom: 1px solid var(--line); padding: 6px 8px; }
.bench td { padding: 7px 8px; border-bottom: 1px solid #f1f3f4; color: var(--ink-2); }
.bench td.num { text-align: right; font-family: var(--mono) !important; font-variant-numeric: tabular-nums; }
.bench tr.base { background: #f7f8f9; }
.bench tr.base td { color: var(--ink); font-weight: 600; }
.bench tr.ok td:nth-child(5) { color: var(--good); font-weight: 600; }
.bench tr.bad td:nth-child(5) { color: var(--bad, #b91c1c); font-weight: 600; }
.bench tr:last-child td { border-bottom: none; }
.bench-src { font-family: var(--mono) !important; font-size: 11px; color: var(--ink-3); margin-top: 8px; }

.qcards { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
.qcard { flex: 1 1 150px; background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
  padding: 11px 14px; animation: kpin .4s ease-out both; }
@media (prefers-reduced-motion: reduce) { .qcard { animation: none; } }
.qcard:nth-child(2) { animation-delay: .05s; }
.qcard:nth-child(3) { animation-delay: .1s; }
.qcard:nth-child(4) { animation-delay: .15s; }
.qcard .q-label { font-size: 12.5px; color: var(--ink-2); }
.qcard .q-value { font-family: var(--mono) !important; font-size: 24px; font-weight: 600; color: var(--ink);
  margin: 4px 0 2px; font-variant-numeric: tabular-nums; }
.qcard .q-sub { font-size: 11.5px; color: var(--ink-3); }
.qcard.hl { background: var(--accent-soft); border-color: #bfe0db; }
.qcard.dim { background: #fafbfc; }
.qcard.dim .q-value { color: var(--ink-3); }

/* 结果图 */
.result-img > div:first-child { display: none !important; }
.result-img { border-radius: 0 !important; border: none !important; background: none !important;
  box-shadow: none !important; padding: 0 !important; }
.rcard { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
  margin-bottom: 14px; animation: rc .35s ease-out both; }
@keyframes rc { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .rcard { animation: none; } }
.rcard:hover { border-color: #cdd3d9; box-shadow: 0 4px 14px rgba(22, 32, 43, .08); }
.rcard img { width: 100%; height: 330px; object-fit: contain; background: #0e1217; display: block; }
.rcap { display: flex; align-items: baseline; gap: 8px; padding: 9px 14px;
  border-top: 1px solid var(--line); flex-wrap: wrap; font-size: 13px; }
.rcap .no { font-family: var(--mono) !important; font-size: 11px; font-weight: 600; color: var(--paper);
  background: var(--ink); border-radius: 4px; padding: 1px 6px; }
.rcap .t { font-weight: 600; color: var(--ink); }
.rcap .d { color: var(--ink-3); font-size: 12px; }
.rcap .m { font-family: var(--mono) !important; font-size: 12px; color: var(--accent); margin-left: auto; }

.guide { font-size: 12.5px; line-height: 1.75; color: var(--ink-2); }
.guide b { color: var(--ink); }
.guide code { font-family: var(--mono) !important; font-size: 12px; background: #f1f3f5;
  padding: 1px 5px; border-radius: 4px; }
.guide ol, .guide ul { margin: 6px 0; padding-left: 20px; }
.guide li { margin: 3px 0; }
.foot-note, .foot-note p { font-size: 12.5px !important; color: var(--ink-3) !important; line-height: 1.7; }
.empty-hint { border: 1px dashed #cfd5db; border-radius: 12px; padding: 22px; text-align: center;
  color: var(--ink-3); font-size: 13.5px; background: var(--paper); margin-bottom: 12px; }
.empty-hint b { color: var(--ink); }
.hint { font-size: 12px; color: var(--ink-3); line-height: 1.6; margin: 0 2px 4px; }
.result-img img { background: #0e1217; object-fit: contain; }
.tbl table { font-size: 13px !important; }
.tbl td, .tbl th { font-variant-numeric: tabular-nums; }

/* 顶部亮点条：AP 与其他优势，任何 Tab 下都可见 */
.strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 14px 0 6px; }
@media (max-width: 1100px) { .strip { grid-template-columns: repeat(2, 1fr); } }
.strip .s { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px;
  border-left: 4px solid var(--accent); }
.strip .s.ap { background: linear-gradient(135deg, #e6f2f0, #ffffff 70%); border-left-color: var(--accent); }
.strip .s-v { font-family: var(--mono) !important; font-size: 26px; font-weight: 700; color: var(--ink);
  font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.strip .s-v small { font-size: 13px; color: var(--ink-3); font-weight: 500; margin-left: 4px; }
.strip .s-v .up { color: var(--good); }
.strip .s-t { font-size: 13px; font-weight: 600; color: var(--ink); margin-top: 2px; }
.strip .s-d { font-size: 11.5px; color: var(--ink-3); margin-top: 2px; line-height: 1.5; }

/* 优势页 */
.adv { display: flex; flex-direction: column; gap: 14px; }
.adv-card { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.adv-card h3 { margin: 0 0 4px; font-size: 16px; color: var(--ink); }
.adv-card .lead { font-size: 13px; color: var(--ink-2); line-height: 1.7; margin: 0 0 10px; }
.adv-card .lead b { color: var(--ink); }
.adv-grid { display: grid; grid-template-columns: 3fr 2fr; gap: 14px; }
@media (max-width: 1100px) { .adv-grid { grid-template-columns: 1fr; } }
.claims { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
@media (max-width: 1100px) { .claims { grid-template-columns: 1fr; } }
.claim { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.claim .c-no { font-family: var(--mono) !important; font-size: 11px; color: var(--accent); letter-spacing: .1em; }
.claim .c-t { font-size: 14.5px; font-weight: 700; color: var(--ink); margin: 4px 0 8px; }
.claim .c-vs { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.claim .c-a { font-family: var(--mono) !important; font-size: 24px; font-weight: 700; color: var(--accent); }
.claim .c-b { font-family: var(--mono) !important; font-size: 15px; color: var(--ink-3); }
.claim .c-d { font-size: 12px; color: var(--ink-2); line-height: 1.6; margin-top: 6px; }
.bar-row { display: grid; grid-template-columns: 92px 1fr 56px; gap: 8px; align-items: center; font-size: 12.5px;
  margin: 7px 0; color: var(--ink-2); }
.bar-row .track { background: #eef1f3; border-radius: 4px; height: 12px; overflow: hidden; }
.bar-row .fill { height: 100%; border-radius: 4px; }
.bar-row .v { font-family: var(--mono) !important; text-align: right; color: var(--ink); }
.cmp { width: 100%; border-collapse: collapse; font-size: 13px; }
.cmp th { text-align: right; font-weight: 600; color: var(--ink-3); font-size: 12px; padding: 6px 8px;
  border-bottom: 1px solid var(--line); }
.cmp th:first-child, .cmp td:first-child { text-align: left; }
.cmp td { padding: 7px 8px; border-bottom: 1px solid #f1f3f4; text-align: right; color: var(--ink-2);
  font-family: var(--mono) !important; font-variant-numeric: tabular-nums; }
.cmp td:first-child { font-family: inherit !important; color: var(--ink); }
.cmp td.win { color: var(--good); font-weight: 700; }
.cmp td.lose { color: #b91c1c; }
.cmp tr.base td { background: #f7f8f9; }
.chart-legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--ink-2); margin-top: 6px; }
.chart-legend i { display: inline-block; width: 14px; height: 3px; margin-right: 5px; vertical-align: 3px; }
.scope { font-size: 12.5px; line-height: 1.75; color: var(--ink-2); }
.scope b { color: var(--ink); }
.scope ul { margin: 4px 0; padding-left: 20px; }
.kpi.ap { border-color: #bfe0db; background: linear-gradient(135deg, #e6f2f0, #ffffff 75%); }
.kpi.ap .k-label::before { background: var(--accent); }
.kpi.ap .k-value { color: var(--accent); }
.tabs-main button[role="tab"] { font-weight: 600 !important; font-size: 14px !important; }
.adv-card svg { width: 100%; height: auto; display: block; }
.adv-card .src { font-family: var(--mono) !important; font-size: 10.5px; color: var(--ink-3); margin-top: 8px; }
.strip-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin: 14px 2px -4px; }
.strip-head b { font-size: 14px; color: var(--ink); }
.strip-head span { font-size: 12px; color: var(--ink-3); }
.scope h4 { margin: 12px 0 4px; font-size: 14px; color: var(--ink); }
.scope .lim { border-left: 3px solid #e3b26b; background: var(--warn-bg); padding: 8px 12px; border-radius: 0 8px 8px 0;
  margin: 8px 0; }
.scope .fit { border-left: 3px solid var(--good); background: var(--good-bg); padding: 8px 12px;
  border-radius: 0 8px 8px 0; margin: 8px 0; }
"""

# 点示例图时 Gradio 只传像素数组，按内容反查文件名，才能去真值里查这张图的标注
EXAMPLE_ARRAYS: dict = {}
_IMG_NAME_BY_ID: dict = {}

EMPTY_KPI = ('<div class="empty-hint"><b>还没有结果</b> · 上传一张航拍图，或在左侧点一张示例图，'
             '三种方法会依次跑完，并在这里给出对比。</div>')

# 验证集上的离线精度（AP，COCO 口径），单图 Demo 算不出可信的 AP，必须看这一组。
# 来源：docs/RESULTS-A-visdrone（COCO 零训练）、RESULTS-ft-visdrone（微调）、RESULTS-B-dota。
BENCH = {
    "visdrone_ft": {
        "title": "VisDrone val 548 张 · 微调检测器（yolo11s-visdrone-ft）",
        "src": "docs/RESULTS-ft-visdrone-2026-09-26.md",
        "rows": [
            ["整图 full@1920（不算切片）", "—", 50.38, 42.88, "+3.30", "0.26", ""],
            ["SAHI 稠密@512（基线）", "100%", 47.37, 39.33, "0", "1.00", "base"],
            ["手工稀疏门 θ=0.99", "86.1%", 47.41, None, "+0.04", "0.94", "ok"],
            ["同数量随机跳过", "86.1%", 46.60, None, "−0.68", "0.87", "bad"],
        ],
    },
    "visdrone": {
        "title": "VisDrone val 548 张 · COCO 零训练检测器（yolo11s）",
        "src": "docs/RESULTS-A-visdrone-2026-09-26.md",
        "rows": [
            ["整图 full@1920（不算切片）", "—", 35.13, 25.52, "+6.61", "0.31", ""],
            ["SAHI 稠密@512（基线）", "100%", 28.52, 17.89, "0", "1.00", "base"],
            ["手工稀疏门 θ=0.9", "80%", 28.36, 17.74, "−0.15", "0.89", "ok"],
            ["手工稀疏门 θ=0.99", "65%", 27.97, 17.46, "−0.54", "0.76", "warn"],
        ],
    },
}


# ---------- 优势与 AP 证据：全部来自 VisDrone-ft 留出集（274 张）离线结果，不是 Demo 单图 ----------
# 来源：docs/RESULTS-router-ft-2026-09-26.md、results/visdrone_ft/router_holdout_ft.csv / router_metrics_ft.csv
FT_DIR = ROOT / "results" / "visdrone_ft"
# CSV 缺失时的兜底值 = router_holdout_ft.csv 按预算取均值（切片比例, AP×100, ms/图）；random 为 3 种子均值
_CURVES_FALLBACK = {
    "router": [(.293, 46.28, 106.4), (.404, 47.17, 133.2), (.511, 47.37, 158.6), (.612, 47.48, 182.7),
               (.716, 47.58, 207.8), (.814, 47.51, 231.1), (.908, 47.45, 252.2)],
    "hand": [(.265, 45.57, 99.1), (.370, 46.32, 124.0), (.502, 46.60, 154.1), (.630, 47.05, 186.0),
             (.730, 47.10, 209.5), (.768, 47.13, 218.2), (.872, 47.38, 243.0)],
    "random": [(.265, 41.23, 97.3), (.370, 42.61, 122.3), (.502, 43.77, 153.5), (.630, 44.66, 185.2),
               (.730, 45.53, 208.5), (.768, 45.71, 217.8), (.872, 46.51, 242.7)],
}
_DENSE_FALLBACK = (47.40, 273.2)
# 切片级排序与标定（留出集，router_metrics_ft.csv）
SLICE_METRICS = [("AUC（越高越好）", 0.824, 0.730, "hi"), ("PR-AUC（越高越好）", 0.684, 0.505, "hi"),
                 ("ECE 标定误差（越低越好）", 0.036, 0.611, "lo"), ("平均分（手工门饱和）", 0.360, 0.942, "")]
# 置换特征重要性（留出集 PR-AUC 下降，router_features_ft.csv 前 5）
FEATURES = [("车辆证据", 0.106), ("目标尺寸", 0.033), ("行人证据", 0.014), ("强框数量", 0.013), ("弱框数量", 0.006)]
CURVE_STYLE = {"router": ("可学习路由器", "#0f766e", ""), "hand": ("手工稀疏门（同预算）", "#d97706", ""),
               "random": ("随机选片（3 种子均值）", "#9aa3ab", "5 4")}


def load_budget_curves():
    """读 router_holdout_ft.csv：同预算下 路由器 / 手工门 / 随机 的 (切片比例, AP, ms) 曲线与稠密基线。"""
    import csv

    fam = {"router_global": "router", "fusion_budget": "hand", "random_budget": "random"}
    try:
        acc, dense = {}, None
        with open(FT_DIR / "router_holdout_ft.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["family"] == "sahi":
                    dense = (float(r["AP"]) * 100, float(r["ms_per_img"]))
                k = fam.get(r["family"])
                if k and r["budget"]:
                    acc.setdefault(k, {}).setdefault(float(r["budget"]), []).append(
                        (float(r["slice_frac"]), float(r["AP"]) * 100, float(r["ms_per_img"])))
        curves = {k: [tuple(float(x) for x in np.mean(v, axis=0)) for _, v in sorted(d.items())]
                  for k, d in acc.items()}
        if dense and set(curves) == set(fam.values()):
            return curves, dense, "results/visdrone_ft/router_holdout_ft.csv"
    except Exception:
        pass
    return _CURVES_FALLBACK, _DENSE_FALLBACK, "内置数值（CSV 未找到）"


CURVES, DENSE, CURVE_SRC = load_budget_curves()


def _near(pts, frac):
    return min(pts, key=lambda p: abs(p[0] - frac))


def _headline():
    """约一半切片处三种选片方式的对比点，以及路由器曲线上的最高点。"""
    r, h, z = (_near(CURVES[k], 0.5) for k in ("router", "hand", "random"))
    r3, z3 = _near(CURVES["router"], 0.3), _near(CURVES["random"], 0.3)
    peak = max(CURVES["router"], key=lambda p: p[1])
    return r, h, z, r3, z3, peak


def strip_html() -> str:
    """顶部亮点条：打开页面第一眼看到 AP 结论与其他优势。"""
    r, h, z, _, _, _ = _headline()
    ap_d, ms_d = DENSE
    cards = [
        ("s ap", f'{r[1]:.2f}<small>AP · 稠密 {ap_d:.2f}</small>', "一半切片，AP 几乎不掉",
         f"路由器只跑 {r[0]:.0%} 切片，ΔAP {r[1] - ap_d:+.2f}"),
        ("s", f'<span class="up">+{r[1] - z[1]:.1f}</span><small>AP</small>', "同预算大幅胜过随机选片",
         f"路由 {r[1]:.2f} · 手工门 {h[1]:.2f} · 随机 {z[1]:.2f}"),
        ("s", f'{ms_d / r[2]:.2f}×<small>提速</small>', "少跑一半高清推理",
         f"{r[2]:.0f} ms vs {ms_d:.0f} ms / 图（离线缓存口径）"),
        ("s", '0.036<small>ECE · 手工门 0.611</small>', "分数可信，强检测器下不饱和",
         "切片级 AUC 0.824 vs 0.730"),
    ]
    body = "".join(f'<div class="{c}"><div class="s-v">{v}</div><div class="s-t">{t}</div>'
                   f'<div class="s-d">{d}</div></div>' for c, v, t, d in cards)
    return (f'<div class="strip">{body}</div><div class="meta-line" style="margin:0 2px 8px">'
            f'以上均为 VisDrone val 留出集 274 张、微调检测器 yolo11s-visdrone-ft 上的离线结果 · {CURVE_SRC}</div>')


def budget_svg() -> str:
    """AP–切片预算曲线（内联 SVG，不依赖任何图表库）。"""
    W, H, L, R, T, B = 640, 300, 52, 22, 18, 46
    x0, x1, y0, y1 = 0.2, 1.0, 40.0, 48.5
    X = lambda f: L + (f - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    Y = lambda a: T + (y1 - a) / (y1 - y0) * (H - T - B)  # noqa: E731
    g = []
    for a in range(40, 49, 2):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(a):.1f}" y2="{Y(a):.1f}" stroke="#eef1f3"/>'
                 f'<text x="{L - 8}" y="{Y(a) + 4:.1f}" text-anchor="end" font-size="11" fill="#8a95a1">{a}</text>')
    for f in (0.2, 0.4, 0.6, 0.8, 1.0):
        g.append(f'<text x="{X(f):.1f}" y="{H - B + 18}" text-anchor="middle" font-size="11" '
                 f'fill="#8a95a1">{f:.0%}</text>')
    g.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle" font-size="12" '
             f'fill="#4a5663">激活切片比例（越左越省）</text>')
    g.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" text-anchor="middle" font-size="12" fill="#4a5663" '
             f'transform="rotate(-90 14 {(T + H - B) / 2:.0f})">AP</text>')
    ap_d = DENSE[0]
    g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(ap_d):.1f}" y2="{Y(ap_d):.1f}" stroke="#16202b" '
             f'stroke-width="1" stroke-dasharray="2 3"/>'
             f'<text x="{L + 6}" y="{Y(ap_d) - 6:.1f}" font-size="11" fill="#16202b">稠密 SAHI {ap_d:.2f}</text>')
    for k in ("random", "hand", "router"):
        _, color, dash = CURVE_STYLE[k]
        pts = CURVES[k]
        path = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in pts)
        sw = 2.8 if k == "router" else 2
        g.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        g += [f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{3.6 if k == "router" else 3}" fill="#fff" '
              f'stroke="{color}" stroke-width="2"><title>{CURVE_STYLE[k][0]}：{p[0]:.0%} 切片，AP {p[1]:.2f}，'
              f'{p[2]:.0f} ms/图</title></circle>' for p in pts]
    xd, yd = X(1.0), Y(ap_d)
    g.append(f'<path d="M{xd:.1f},{yd - 6:.1f} L{xd + 6:.1f},{yd:.1f} L{xd:.1f},{yd + 6:.1f} '
             f'L{xd - 6:.1f},{yd:.1f} Z" fill="#16202b"><title>稠密 SAHI：100% 切片，AP {ap_d:.2f}</title></path>')
    r, _, z, _, _, _ = _headline()
    xr, yr, yz = X(r[0]), Y(r[1]), Y(z[1])
    g.append(f'<line x1="{xr:.1f}" x2="{xr:.1f}" y1="{yr + 6:.1f}" y2="{yz - 6:.1f}" stroke="#15803d" '
             f'stroke-width="1.5" marker-end="url(#ah)"/>'
             f'<text x="{xr + 8:.1f}" y="{(yr + yz) / 2 + 4:.1f}" font-size="12" font-weight="600" fill="#15803d">'
             f'+{r[1] - z[1]:.1f} AP</text>'
             f'<text x="{xr - 6:.1f}" y="{yr - 12:.1f}" text-anchor="end" font-size="12" font-weight="600" '
             f'fill="#0f766e">{r[0]:.0%} 切片 · AP {r[1]:.2f}</text>')
    defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#15803d"/></marker></defs>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="AP 随切片预算变化曲线" '
            f'style="width:100%;height:auto;display:block">{defs}{"".join(g)}</svg>')


def _bar(label, v, color, lo=40.0, hi=48.0, text=None):
    w = max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100
    return (f'<div class="bar-row"><span>{label}</span><div class="track"><div class="fill" '
            f'style="width:{w:.1f}%;background:{color}"></div></div>'
            f'<span class="v">{text if text is not None else f"{v:.2f}"}</span></div>')


def advantages_html() -> str:
    r, h, z, r3, z3, peak = _headline()
    ap_d, ms_d = DENSE
    claims = f"""
<div class="claims">
  <div class="claim"><div class="c-no">CLAIM 01</div><div class="c-t">省一半切片，AP 不掉</div>
    <div class="c-vs"><span class="c-a">{r[1]:.2f}</span><span class="c-b">稠密 SAHI {ap_d:.2f}</span></div>
    <div class="c-d">路由器只激活 {r[0]:.0%} 的切片，AP 仅差 {r[1] - ap_d:+.2f}；预算放到 {peak[0]:.0%} 时
      AP {peak[1]:.2f}，<b>反而略高于稠密切片</b>（跳过的切片里多是误检来源）。</div></div>
  <div class="claim"><div class="c-no">CLAIM 02</div><div class="c-t">丢掉的确实是“没用的”切片</div>
    <div class="c-vs"><span class="c-a">+{r[1] - z[1]:.1f}</span><span class="c-b">AP vs 同预算随机</span></div>
    <div class="c-d">同样约一半切片，随机选片掉到 {z[1]:.2f}；预算压到约 {r3[0]:.0%} 时差距扩大到
      <b>+{r3[1] - z3[1]:.1f}</b> 个点。说明省下的算力不是靠运气。</div></div>
  <div class="claim"><div class="c-no">CLAIM 03</div><div class="c-t">强检测器下依然可靠</div>
    <div class="c-vs"><span class="c-a">0.036</span><span class="c-b">ECE · 手工门 0.611</span></div>
    <div class="c-d">检测器越强，弱检测到处都是，手工 noisy-OR 门整图饱和（平均分 0.94，只省 9% 切片）；
      可学习路由器保持标定与排序能力，这是它相对手工门最清楚的优势。</div></div>
</div>"""
    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c, _ in CURVE_STYLE.values())
    chart = f"""
<div class="adv-card">
  <h3>AP 随切片预算变化</h3>
  <p class="lead">横轴是只跑多少比例的切片，纵轴是留出集 AP。<b>路由器曲线在 50% 左右就追平稠密 SAHI</b>，
    随机选片一路掉点，手工门居中。鼠标悬停可看每个点的切片比例、AP 与耗时。</p>
  {budget_svg()}
  <div class="chart-legend">{legend}<span><i style="background:#16202b"></i>稠密 SAHI 基线</span></div>
  <div class="bench-src">来源 {CURVE_SRC} · VisDrone val 奇偶对半，274 张留出，不参与训练与调参</div>
</div>"""
    rows = ""
    for name, a, b, kind in SLICE_METRICS:
        ca = "win" if kind else ""
        cb = "lose" if kind else ""
        rows += f'<tr><td>{name}</td><td class="{ca}">{a:.3f}</td><td class="{cb}">{b:.3f}</td></tr>'
    feats = "".join(_bar(n, v, "#0f766e", 0, 0.11, f"{v:.3f}") for n, v in FEATURES)
    side = f"""
<div class="adv-card">
  <h3>同样约 50% 切片，谁的 AP 高</h3>
  <p class="lead">预算相同，只换“挑哪些切片”的方法。</p>
  {_bar("可学习路由", r[1], "#0f766e")}{_bar("手工稀疏门", h[1], "#d97706")}{_bar("随机选片", z[1], "#9aa3ab")}
  {_bar("稠密 SAHI", ap_d, "#16202b")}
  <div class="bench-src">条长按 AP 40–48 缩放 · 稠密 SAHI 跑 100% 切片、{ms_d:.0f} ms/图</div>
</div>
<div class="adv-card">
  <h3>切片级打分质量</h3>
  <table class="cmp"><thead><tr><th>指标（留出集）</th><th>可学习路由器</th><th>手工稀疏门</th></tr></thead>
  <tbody>{rows}</tbody></table>
  <div class="bench-src">来源 results/visdrone_ft/router_metrics_ft.csv</div>
</div>
<div class="adv-card">
  <h3>路由器在看什么</h3>
  <p class="lead">置换特征重要性（打乱该特征后 PR-AUC 下降量）。模型是线性的、约 23 个参数，可解释。</p>
  {feats}
</div>"""
    extra = """
<div class="adv-card">
  <h3>其他优势</h3>
  <div class="scope"><ul>
    <li><b>即插即用</b>：不改检测器、不重训检测器，只在 SAHI 前加一个“扫一眼 → 打分 → 选片”的步骤。</li>
    <li><b>极轻量</b>：路由器是 5 种子集成的线性模型（每个约 23 个参数），打分耗时相对切片推理可忽略。</li>
    <li><b>预算可控</b>：ρ 直接等于“每张图跑多少比例的切片”，部署时按算力预算一键设定；也可用训练好的全局阈值自适应。</li>
    <li><b>统计上站得住</b>：预注册假设、奇偶留出、按图配对 bootstrap 置信区间、同预算随机对照，负结果也照实报告。</li>
  </ul></div>
</div>"""
    return (f'<div class="adv">{claims}<div class="adv-grid"><div class="adv">{chart}{extra}</div>'
            f'<div class="adv">{side}</div></div></div>')


def scope_html() -> str:
    return """
<div class="adv">
<div class="adv-card">
  <h3>局限与适用范围</h3>
  <p class="lead">以下结论同样来自仓库里的实验记录。把边界说清楚，上面的优势才可信。</p>
  <div class="scope"><ul>
    <li><b>图不太大时，整图高分辨率推理更划算</b>：VisDrone 上 full@1920 AP 50.38、35 ms/图，同时胜过 SAHI（47.08）
      和 Glance-SAHI。本方法的比较对象是“必须切片”的场景。（RESULTS-ft 实测 1）</li>
    <li><b>DOTA 超大图上优势有限</b>：glance@0.5 少跑 47% 切片、AP −2.26，但“把切片调大”（sahi@1536，AP −2.65）几乎做到同样的事，
      两者置信区间重叠。（RESULTS-B 实测 1）</li>
    <li><b>可优化的上限本来就小</b>：即使用真值挑切片的 oracle，在 77% 切片处也只比稠密高 +0.14 AP——强检测器下漏检多是检测器能力上限。</li>
    <li><b>缩略图里看不见的目标选不到</b>：扫视图中 &lt; 4 px 的目标只保住约 65% 的 SAHI 增益，本方法的主场是“隐约可见、需要放大确认”的中等尺寸。</li>
    <li><b>单图 AP 波动很大</b>（标准差约 20）：实时对比页的单图 AP 只作示意，精度结论以验证集 / 留出集表格为准。</li>
    <li><b>负结果</b>：把分数做图内相对化以缓解饱和，ECE 变好了但排序崩了（AP −2.5 ~ −6.8），已证伪并保留为消融。（RESULTS-relgate）</li>
  </ul></div>
</div>
</div>"""


def resolve_dataset(weights: str, dataset: str) -> str:
    """auto：微调出的 2 类检测器（文件名含 -ft）用 visdrone_ft 的类别表，否则按 COCO 类别表。"""
    if dataset != "auto":
        return dataset
    name = Path(weights).name.lower()
    return "dota_ft" if "dota" in name and "ft" in name else "visdrone_ft" if "ft" in name else "visdrone"


def router_for(dataset: str) -> Path:
    """路由器是在某个检测器的扫视输出上训练的，换检测器必须换路由器（与 run_eval 的结果目录一致）。"""
    if dataset == "visdrone":
        return ROOT / "results" / "router.json"
    d = ROOT / "results" / dataset
    for name in (f"router_{dataset.replace('visdrone_', '')}.json", "router.json"):
        p = d / name
        if p.exists():
            return p
    return d / "router.json"


def router_meta(router: Path) -> dict:
    try:
        return json.loads(router.read_text()).get("meta", {})
    except Exception:
        return {}


# 手工门的默认工作点（REPORT 3.2 / 3.7）：COCO 检测器在 DOTA 俯视图上几乎认不出目标，要靠边缘先验
DEFAULT_GATE = {"dota": (0.5, 1.0)}


def coverage(dense: np.ndarray, sparse: np.ndarray, conf: float = 0.25, iou: float = 0.5):
    """稀疏方法保住了稠密结果里多少比例的高置信框。

    AP 常常三種方法完全一样，因为被跳过的切片里只有低分弱框，而弱框排在 PR 曲线末端不影响 AP。
    这个数字用来解释"AP 一样"到底是真没损失，还是指标不敏感。
    """
    a = np.asarray(dense).reshape(-1, 6)
    b = np.asarray(sparse).reshape(-1, 6)
    a = a[a[:, 4] >= conf]
    b = b[b[:, 4] >= conf]
    if len(a) == 0:
        return None
    if len(b) == 0:
        return 0.0
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    ar = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    br = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ious = inter / np.maximum(ar[:, None] + br[None, :] - inter, 1e-6)
    return float((ious.max(1) >= iou).mean())


def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}"


def _qcard(cards) -> str:
    body = "".join(f'<div class="qcard {cls}"><div class="q-label">{t}</div>'
                   f'<div class="q-value">{v}</div><div class="q-sub">{sub}</div></div>'
                   for t, v, sub, cls in cards)
    return f'<div class="qcards">{body}</div>'


def bench_html(dataset: str) -> str:
    """验证集上的离线 AP 表：单图 Demo 给不出可信的精度结论，必须看这一组。"""
    b = BENCH.get(dataset)
    if not b:
        return ('<div class="empty-hint">当前数据集（' + html_escape(dataset) +
                '）没有内置验证集 AP 表。精度结论见 docs/RESULTS-*.md。</div>')
    rows = "".join(
        f'<tr class="{cls}"><td>{html_escape(m)}</td><td>{html_escape(sl)}</td>'
        f'<td class="num"><b>{ap:.2f}</b></td><td class="num">{("—" if aps is None else f"{aps:.2f}")}</td>'
        f'<td class="num">{html_escape(d)}</td><td class="num">{html_escape(t)}</td></tr>'
        for m, sl, ap, aps, d, t, cls in b["rows"])
    return f"""
<div class="bench">
  <div class="sec-title"><b>精度</b>验证集 AP（这才是可信的精度结论）</div>
  <table><thead><tr><th>方法</th><th>切片</th><th>AP</th><th>AP<small>small</small></th>
    <th>ΔAP</th><th>耗时比</th></tr></thead><tbody>{rows}</tbody></table>
  <div class="bench-src">{html_escape(b["title"])} · 来源 {html_escape(b["src"])}</div>
</div>"""


def result_card(no: int, title: str, desc: str, img, metric: str = "") -> str:
    """结果图卡片：序号 + 标题 + 说明 + 实测指标，图片以 data URL 内嵌，避免挂 Gradio 的方框。"""
    import base64

    import cv2

    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""
    m = f'<span class="m">{html_escape(metric)}</span>' if metric else ""
    return (f'<div class="rcard"><img src="data:image/jpeg;base64,{b64}" alt="{html_escape(title)}">'
            f'<div class="rcap"><span class="no">{no}</span>'
            f'<span class="t">{html_escape(title)}</span>'
            f'<span class="d">{html_escape(desc)}</span>{m}</div></div>')


def html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def blank_panel(text: str):
    """没有可用结果时给一张说明图，避免空白或报错。"""
    import cv2

    img = np.full((360, 640, 3), 246, np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (28, 150 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 130, 145), 1, cv2.LINE_AA)
    return img


def kpi_cards(rows: list, have_router: bool, q: dict | None = None) -> str:
    """5 张 KPI 卡：本图 AP（路由 vs 稠密）放第一位，其后是路由器省片/提速、手工门、检测框差异。"""
    def frac(row):
        return float(row[2].strip("%")) / 100

    def speed(row):
        return float(row[4].replace("×", ""))

    q = q or {}
    gate = rows[1]
    main = rows[2] if have_router else gate          # 主角：有路由器就是路由器，否则退回手工门
    who = "路由器" if have_router else "手工门"
    saved = 1 - frac(main)
    sp = speed(main)
    dets_diff = int(main[5]) - int(rows[0][5])

    # 单图 AP 波动很大（标准差约 20），只能看个大概；真正的精度结论看验证集 / 留出集
    ap_main = q.get("ap_router") if have_router else q.get("ap_sparse")
    if ap_main is None or q.get("ap") is None:
        ap_card = ('<div class="kpi empty"><div class="k-label">本图 AP</div><div class="k-value">—</div>'
                   f'<div class="k-sub">{html_escape(q.get("why") or "这张图没有真值标注")}</div></div>')
    else:
        d = (ap_main - q["ap"]) * 100
        ap_card = (f'<div class="kpi ap"><div class="k-label">本图 AP · {who}</div>'
                   f'<div class="k-value">{ap_main * 100:.1f}</div>'
                   f'<div class="k-sub">稠密 SAHI {q["ap"] * 100:.1f} · Δ {d:+.1f}（单图，仅示意）</div></div>')

    saved_cls = "good" if saved >= 0.15 else "warn"
    sp_cls = "good" if sp >= 1.1 else "warn"
    det_cls = "good" if dets_diff >= -1 else "warn"
    hand_sub = f"{gate[1]} 片 · 提速 {speed(gate):.2f}×"
    return f"""
<div class="kpis">
  {ap_card}
  <div class="kpi {saved_cls}"><div class="k-label">{who}省下的切片</div>
    <div class="k-value">{saved:.0%}</div><div class="k-sub">{main[1]} 片被激活</div></div>
  <div class="kpi {sp_cls}"><div class="k-label">{who}相对 SAHI 提速</div>
    <div class="k-value">{sp:.2f}×</div><div class="k-sub">{main[3]} ms vs {rows[0][3]} ms</div></div>
  <div class="kpi {det_cls}"><div class="k-label">检测框差异（{who}）</div>
    <div class="k-value">{dets_diff:+d}</div><div class="k-sub">{main[5]} vs SAHI {rows[0][5]}</div></div>
  <div class="kpi info"><div class="k-label">手工稀疏门（对照）</div>
    <div class="k-value">{frac(gate):.0%}</div><div class="k-sub">{hand_sub}</div></div>
</div>"""


def make_runner(model, device, exclude_ids, router: Path, weights_name: str = "", dataset: str = "",
                gt_path: Path | None = None, max_dets: int = 500):
    ROUTER = router
    have_router = ROUTER.exists()
    EXCLUDE_COCO_IDS = exclude_ids
    GT = Path(gt_path) if gt_path else None
    C2E = datasets.get(dataset).get("coco_to_eval", {}) if dataset else {}
    MAX_DETS = max_dets
    HAVE_GT = bool(GT and GT.exists() and C2E)

    def measure_quality(img, name, theta, lam, rho_router, with_ap=True, dense_dets=None):
        """单独跑一遍只为算 AP —— 不计时。COCOeval 单次约 70 ms，夹进耗时测量会把速度对比带偏。

        返回 (dict, html)：dict 给 KPI 卡用，html 给结果区用。
        """
        out = {"ap": None, "ap50": None, "aps": None, "ap_sparse": None, "n_gt": None, "why": ""}
        if not with_ap:
            out["why"] = "未勾选「算 AP」"
            return out, _qcard([("单图 AP", "—", "勾选左栏「算 AP」才会计算", "dim")])
        if not HAVE_GT:
            out["why"] = "该数据集没有可用的真值标注"
            return out, _qcard([("单图 AP", "—", out["why"], "dim")])
        iid = gt_image_id(GT, name)
        if iid is None:
            out["why"] = "不在真值里"
            return out, _qcard([("单图 AP", "—", "这张图没有标注（自己上传的图通常如此）", "dim")])

        def ap_of(dets):
            m = coco_eval(GT, to_coco_dets(iid, dets, C2E), [iid], MAX_DETS)
            if not m or m.get("AP") is None:
                return None, None, None
            return m["AP"], m["AP50"], m["APs"]

        r_all, _, _ = sahi_uniform_prediction(img, model, GlanceConfig(), EXCLUDE_COCO_IDS)
        out["ap"], out["ap50"], out["aps"] = ap_of(to_np(r_all.object_prediction_list))

        r_sp, _ = glance_sliced_prediction(img, model, GlanceConfig(threshold=theta, img_weight=lam),
                                           EXCLUDE_COCO_IDS)
        out["ap_sparse"], ap50_sp, aps_sp = ap_of(to_np(r_sp.object_prediction_list))

        cards = [("SAHI 稠密", _pct(out["ap"]), f"AP50 {_pct(out['ap50'])} · APs {_pct(out['aps'])}", "hl"),
                 ("手工稀疏门", _pct(out["ap_sparse"]), f"AP50 {_pct(ap50_sp)} · APs {_pct(aps_sp)}", "")]
        if have_router:
            cfg = GlanceConfig(scorer="learned", router_path=str(ROUTER), img_weight=lam)
            if rho_router > 0:
                cfg.mode, cfg.budget = "budget", rho_router
            r_rt, _ = glance_sliced_prediction(img, model, cfg, EXCLUDE_COCO_IDS)
            ap_rt, ap50_rt, aps_rt = ap_of(to_np(r_rt.object_prediction_list))
            out["ap_router"] = ap_rt
            cards.append(("可学习路由器", _pct(ap_rt), f"AP50 {_pct(ap50_rt)} · APs {_pct(aps_rt)}", ""))

        out["n_gt"] = len(gt_boxes(GT, image_id=iid))
        cards.append(("本图真值框", str(out["n_gt"]), "有标注才谈得上 AP", "dim"))
        return out, _qcard(cards)

    def run(image, theta, lam, rho_router, min_score, fname="", with_ap=True):
        if image is None:
            return (EMPTY_KPI, bench_html(dataset), "", None, None, None, None, [],
                    "等待输入：上传图片或从左侧选一张示例图。")
        img = np.ascontiguousarray(image[..., :3])
        rows = []
        # 真值按文件名索引。Gradio 的 Image 给的是 (数组, 路径或文件名)，这里尽量拿到文件名；
        # 拿不到（比如纯数组）就退回按像素比对示例图。
        name = ""
        if isinstance(fname, str) and fname:
            name = Path(fname).name
        elif isinstance(image, (tuple, list)) and len(image) > 1 and isinstance(image[1], str):
            name = Path(image[1]).name
        if not name:
            for n, ref in EXAMPLE_ARRAYS.items():
                if ref.shape == img.shape and np.array_equal(ref, img):
                    name = n
                    break

        # 1) Dense：SAHI 全部切片
        res, dt, n = sahi_uniform_prediction(img, model, GlanceConfig(), EXCLUDE_COCO_IDS)
        t_dense = dt
        d_sahi = to_np(res.object_prediction_list)
        p1 = result_card(1, "SAHI 稠密 · 全部切片", "不筛选，每片都做高清推理（基线）",
                         draw_dets(img, d_sahi, min_score), f"{n}/{n} 片 · {dt * 1000:.0f} ms")
        rows.append(["SAHI（稠密，全激活）", f"{n}/{n}", "100%", f"{dt * 1000:.0f}", "1.00×", len(d_sahi), "—"])

        # 2) 手工稀疏门
        cfg = GlanceConfig(threshold=theta, img_weight=lam)
        t0 = time.perf_counter()
        res, st = glance_sliced_prediction(img, model, cfg, EXCLUDE_COCO_IDS)
        dt = time.perf_counter() - t0
        d = to_np(res.object_prediction_list)
        heat = draw_slices(overlay_heat(img, st.slices, st.slice_scores), st.slices, st.selected, st.slice_scores)
        p2 = result_card(2, "手工稀疏门 · 只跑可疑切片", "绿框=被激活，压暗=跳过",
                         draw_dets(draw_slices(img, st.slices, st.selected, show_scores=False), d, min_score),
                         f"{st.n_slices_run}/{st.n_slices_total} 片 · {dt * 1000:.0f} ms · 提速 {t_dense / dt:.2f}×")
        rows.append([f"手工稀疏门 θ={theta:.2f} λ={lam:.2f}", f"{st.n_slices_run}/{st.n_slices_total}",
                     f"{st.n_slices_run / st.n_slices_total:.0%}", f"{dt * 1000:.0f}", f"{t_dense / dt:.2f}×", len(d),
                     f"扫视 {st.t_glance * 1000:.0f} / 打分 {st.t_saliency * 1000:.0f} / "
                     f"切片 {st.t_slices * 1000:.0f} / 合并 {st.t_post * 1000:.0f}"])

        # 3) 可学习稀疏路由器
        if have_router:
            cfg = GlanceConfig(scorer="learned", router_path=str(ROUTER), img_weight=lam)
            if rho_router > 0:  # 0 = 用训练好的默认全局阈值；>0 = 每图 top-k 预算
                cfg.mode, cfg.budget = "budget", rho_router
            t0 = time.perf_counter()
            res, st2 = glance_sliced_prediction(img, model, cfg, EXCLUDE_COCO_IDS)
            dt = time.perf_counter() - t0
            d = to_np(res.object_prediction_list)
            p3 = result_card(3, "可学习稀疏路由器" + ("（top-k 预算）" if rho_router > 0 else "（全局阈值）"),
                             "小 MLP 打分，框上是每片分数",
                             draw_dets(draw_slices(img, st2.slices, st2.selected, st2.slice_scores), d, min_score),
                             f"{st2.n_slices_run}/{st2.n_slices_total} 片 · {dt * 1000:.0f} ms · "
                             f"提速 {t_dense / dt:.2f}×")
            rows.append(["可学习稀疏路由器" + (f"（top-{rho_router:.0%}）" if rho_router > 0 else "（全局阈值）"),
                         f"{st2.n_slices_run}/{st2.n_slices_total}", f"{st2.n_slices_run / st2.n_slices_total:.0%}",
                         f"{dt * 1000:.0f}", f"{t_dense / dt:.2f}×", len(d),
                         f"扫视 {st2.t_glance * 1000:.0f} / 打分 {st2.t_saliency * 1000:.0f} / "
                         f"切片 {st2.t_slices * 1000:.0f} / 合并 {st2.t_post * 1000:.0f}"])
        else:
            p3 = result_card(3, "可学习稀疏路由器", "未找到路由器权重，本栏禁用",
                             blank_panel("未找到路由器权重\n可学习路由栏已禁用"), "—")
            rows.append(["可学习稀疏路由器", "—", "—", "—", "—", "—", "未找到路由器 json"])

        meta = router_meta(ROUTER)
        badges = [f'<span class="badge">检测器 {Path(weights_name).name}</span>',
                  f'<span class="badge">{device}</span>',
                  f'<span class="badge{" off" if not have_router else ""}">'
                  f'{"路由器已加载" if have_router else "路由器缺失"}</span>']
        if have_router:
            badges.append(f'<span class="badge">路由阈值 {meta.get("default_threshold", 0):.3f}</span>')
        note = ("耗时为单次实测，同一张图多点几次会有 ±10% 左右的抖动；"
                "切片数、检测框数均来自真实推理，不是模拟。AP 单独跑一遍算，不计入耗时。")
        if not have_router:
            note += (f" 未找到 {ROUTER.relative_to(ROOT)}（先对当前检测器运行 run_eval.py cache 与 "
                     "scripts/train_router.py）。")
        q_ap, quality = measure_quality(img, name, theta, lam, rho_router, with_ap)
        kpi = kpi_cards(rows, have_router, q_ap if with_ap else {"why": "未勾选「算 AP」"})
        heat_card = result_card(4, "路由分数 · 热图 + 每片分数", "越亮=越可疑，框上是该片分数", heat, "")
        return (kpi + "".join(badges), bench_html(dataset), quality, heat_card, p1, p2, p3, rows, note)

    return run


def build_ui(run, theta0, lam0, rho0, examples, hero_note="", have_ap=False):
    import gradio as gr

    HERO = f"""
<div class="hero">
  <div>
    <div class="eyebrow">Glance-SAHI</div>
    <h1>先扫一眼，只切可疑区域</h1>
    <p>整图缩小看一眼 = <b>路由器</b>；切片上的检测器调用 = <b>专家</b>。只对可能有增量的切片做高清推理，其余跳过。</p>
  </div>
  <div class="legend">
    <div><i class="l-act"></i>激活切片（做高清推理）</div>
    <div><i class="l-skip"></i>跳过切片（压暗，省下的算力）</div>
    <div><i class="l-det"></i>检测框（按类别着色）</div>
    <div><i class="l-heat"></i>路由分数热图</div>
  </div>
  <div class="meta-line">{hero_note}</div>
</div>"""
    WAIT_NOTE = ("耗时为单次实测，同一张图多点几次会有 ±10% 左右的抖动；切片数与检测框数均来自真实推理，不是模拟。"
                 "AP 单独跑一遍算，不计入耗时。")
    GUIDE = """
<div class="guide">
  <b>怎么看这 4 张图</b>
  <ol>
    <li><b>SAHI 稠密</b>：每片都跑，是最准也最慢的基线。画的是检测框。</li>
    <li><b>手工稀疏门</b>：先扫一眼打分，只对分数 ≥ θ 的切片做高清推理。绿框=被激活，压暗=跳过。</li>
    <li><b>可学习稀疏路由器</b>：用一个小 MLP 代替手工打分，框上数字是每片的路由分数。</li>
    <li><b>路由分数热图</b>：越亮越可疑，用来判断"该省的地方是不是真的没东西"。</li>
  </ol>
  <b>精度看两组数</b>：验证集 AP（下面那张表，可信）和单图 AP（只反映这一张，波动很大，标准差约 20）。
  <b>判断好坏</b>：跳过区域里确实没目标，且 AP 没明显掉。
</div>"""

    with gr.Blocks(title="Glance-SAHI：先扫一眼，只切可疑区域", css=CSS,
                   theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate")) as demo:
        gr.HTML(HERO)
        gr.HTML(strip_html())

        with gr.Tabs(elem_classes=["tabs-main"]):
            with gr.Tab("① 实时对比 · 三种方法"):
                live_ui = _live_tab(gr, theta0, lam0, rho0, examples, have_ap, GUIDE, WAIT_NOTE)
            with gr.Tab("② 优势与 AP 证据"):
                gr.HTML(advantages_html())
            with gr.Tab("③ 局限与适用范围"):
                gr.HTML(scope_html())

        inp, fname, theta, lam, rho, ms, ap, rst, btn, outs = live_ui
        ins = [inp, theta, lam, rho, ms, ap]
        btn.click(run, ins, outs)
        inp.upload(run, ins, outs)
        rst.click(lambda: (theta0, lam0, rho0), None, [theta, lam, rho])
    return demo


def _live_tab(gr, theta0, lam0, rho0, examples, have_ap, guide, wait_note):
    """实时对比页：与原版逻辑完全相同，只调整了结果区顺序（可学习路由器排在最前）。"""
    with gr.Row():
        with gr.Column(scale=4, min_width=300, elem_classes=["panel"]):
            gr.HTML('<div class="sec-title"><b>输入</b>IMAGE</div>')
            inp = gr.Image(label="上传航拍图", type="numpy", height=250, sources=["upload", "clipboard"])
            fname = gr.Textbox(visible=False)
            inp.upload(lambda f: (Path(f).name if isinstance(f, str) and f else ""), [inp], [fname])
            if examples:
                gr.Examples(examples, [inp, fname], label="示例图（VisDrone val，带真值，可算 AP）")
            with gr.Accordion("路由参数", open=True):
                gr.HTML('<div class="sec-title"><b>可学习路由</b>ρ</div>')
                rho = gr.Slider(0.0, 1.0, rho0, step=0.05, label="预算 ρ",
                                info="0 = 用训练好的全局阈值；>0 = 每图只保留分数最高的前 ρ 比例切片。"
                                     "留出集上 ρ≈0.5 时 AP 与稠密持平")
                gr.HTML('<div class="sec-title"><b>手工稀疏门（对照）</b>θ / λ</div>')
                theta = gr.Slider(0.3, 0.999, theta0, step=0.005, label="阈值 θ",
                                  info="切片分数 ≥ θ 才做高清推理。调高更省但可能漏，调低更全但慢")
                lam = gr.Slider(0.0, 1.0, lam0, step=0.05, label="边缘先验权重 λ",
                                info="图像边缘先验占多大比重。检测器不认识该场景（如俯视图）时调高")
                gr.HTML('<div class="sec-title"><b>显示与评估</b></div>')
                ms = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="最低置信度",
                               info="只画置信度 ≥ 该值的框，不影响推理，只影响画面干净程度")
                ap = gr.Checkbox(value=have_ap, label="算 AP（多跑一遍推理，约 +0.7 s，不计入耗时）",
                                 visible=have_ap)
                rst = gr.Button("恢复默认参数", size="sm")
            btn = gr.Button("运行三种方法", variant="primary", size="lg", elem_classes=["run-btn"])

        with gr.Column(scale=8):
            kpi = gr.HTML(EMPTY_KPI)
            quality = gr.HTML()
            with gr.Accordion("怎么读这几张图", open=False):
                gr.HTML(guide)
            with gr.Row():
                with gr.Column(min_width=280):
                    o3 = gr.HTML(elem_classes=["result-img"])
                    heat = gr.HTML(elem_classes=["result-img"])
                with gr.Column(min_width=280):
                    o1 = gr.HTML(elem_classes=["result-img"])
                    o2 = gr.HTML(elem_classes=["result-img"])
            with gr.Accordion("逐项对比（激活切片 / 耗时 / 提速 / 检测数）", open=True):
                tbl = gr.Dataframe(headers=["方法", "激活切片", "比例", "耗时 ms", "相对 SAHI 提速", "检测数",
                                            "耗时拆分 ms"],
                                   label="对比", wrap=True, elem_classes=["tbl"])
            with gr.Accordion("验证集 AP（全量 548 张，可信的精度结论）", open=False):
                bench = gr.HTML()
            note = gr.Markdown(wait_note, elem_classes=["foot-note"])
    outs = [kpi, bench, quality, heat, o1, o2, o3, tbl, note]
    return inp, fname, theta, lam, rho, ms, ap, rst, btn, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--weights", default=str(FT_WEIGHTS) if FT_WEIGHTS.exists() else "yolo11s.pt",
                    help="COCO 预训练或微调权重，如 weights/yolo11s-visdrone-ft.pt")
    ap.add_argument("--dataset", default="auto", choices=["auto"] + list(datasets.DATASETS),
                    help="决定类别表与路由器；auto 按权重文件名推断")
    ap.add_argument("--router", default=None, help="路由器 json（默认 results/<dataset>/router*.json）")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    device = resolve_device(a.device)
    ds = resolve_dataset(a.weights, a.dataset)
    router = Path(a.router) if a.router else router_for(ds)
    have_router = router.exists()
    print(f"检测器 {a.weights}，类别表 {ds}，路由器 {router}（{'存在' if have_router else '缺失'}）")
    theta0, lam0 = DEFAULT_GATE.get(ds, (0.9, 0.3))
    ds_info = datasets.get(ds)
    have_ap = bool(ds_info.get("gt") and Path(ds_info["gt"]).exists())
    model = build_model(a.weights, conf=0.05, device=device)
    run = make_runner(model, device, ds_info["exclude_coco_ids"], router, a.weights, ds,
                      gt_path=ds_info.get("gt"), max_dets=ds_info.get("max_dets", 500))
    ex_dir, ex_names = (datasets.get(ds)["images"], DOTA_EXAMPLES) if ds.startswith("dota") else (EXAMPLES, EXAMPLE_NAMES)
    examples = [[str(ex_dir / n)] for n in ex_names if (ex_dir / n).exists()]
    # 缓存示例图像素：点示例时 Gradio 只给数组，靠它反查文件名 → 查真值 → 算 AP
    for n in ex_names:
        p = ex_dir / n
        if p.exists():
            try:
                from glance_sahi.imageio import imread_rgb
                EXAMPLE_ARRAYS[n] = np.ascontiguousarray(imread_rgb(str(p))[..., :3])
            except Exception:
                pass

    # 预热：CUDA 上下文、cuDNN 选算法、路由器加载都在第一次调用时发生（实测首张 SAHI 6.3 s，之后 0.35 s），
    # 不预热的话评委点的第一张图耗时对比完全失真。
    # 微调检测器下 noisy-OR 会饱和（手工门只省约 9%），默认直接站在论文的卖点上：路由 top-50% 预算
    rho0 = 0.5 if have_router and ds == "visdrone_ft" else 0.0
    t0 = time.perf_counter()
    run(np.zeros((1080, 1920, 3), np.uint8), theta0, lam0, rho0, 0.25, with_ap=False)
    print(f"预热完成 {time.perf_counter() - t0:.1f}s · AP 测量 {'可用' if have_ap else '不可用（无真值）'}")

    meta = router_meta(router)
    hero = (f"权重 {Path(a.weights).name} · 类别表 {ds} · 设备 {device} · "
            + (f"路由器 {router.relative_to(ROOT)}（训练图 {meta.get('n_train_images', '?')} 张，"
               f"默认阈值 {meta.get('default_threshold', 0):.3f}）" if have_router else "路由器缺失，仅显示前两种方法"))
    # 注意：css / theme 只能传给 Blocks()，本版本（5.50）的 launch() 不收这两个参数，
    # 尽管它打印的 DeprecationWarning 让人以为应该改传 launch()。
    build_ui(run, theta0, lam0, rho0, examples, hero, have_ap=have_ap).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
