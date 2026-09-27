# -*- coding: utf-8 -*-
"""生成 PPT 用 AP 对比表（创新点二：可学习路由器 vs 手写公式 vs 瞎选 vs 全切）。

数据来源：results/visdrone/router_holdout.csv（REPORT 3.15，274 张留出图）。
用法：& $py scripts/make_ap_table.py
输出：results/figures/table_router_ap.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# (预算, 手写公式 AP/APs, 学出来的 AP/APs, 瞎随机 AP/APs, 全切 AP/APs)
DATA = [
    ("约 27% 切片", (25.16, 15.53), (27.42, 18.20), (23.32, 13.13), (28.52, 18.11)),
    ("约 50% 切片", (27.21, 17.01), (28.54, 18.54), (25.54, 15.19), (28.52, 18.11)),
]

HEAD = ["只跑多少切片", "手写公式\nAP / APs", "学出来的\nAP / APs", "同预算瞎选\nAP / APs", "全切（不省）\nAP / APs"]

# (row, col) 高亮：学出来的列
HILITE = (1, 2)

rows = []
for budget, hand, learned, rand, full in DATA:
    rows.append([
        budget,
        f"{hand[0]:.2f} / {hand[1]:.2f}",
        f"{learned[0]:.2f} / {learned[1]:.2f}",
        f"{rand[0]:.2f} / {rand[1]:.2f}",
        f"{full[0]:.2f} / {full[1]:.2f}",
    ])

fig, ax = plt.subplots(figsize=(10.5, 2.9), dpi=200)
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=HEAD,
    cellLoc="center",
    loc="center",
    colWidths=[0.18, 0.205, 0.205, 0.205, 0.205],
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11.5)
tbl.scale(1, 1.9)

GREEN = "#1a7f37"
RED = "#c62828"
GREY = "#555555"

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    cell.set_linewidth(0.8)
    if r == 0:  # 表头
        cell.set_facecolor("#eef2f7")
        cell.set_text_props(weight="bold", color="#222222")
        continue
    if c == 0:
        cell.set_text_props(weight="bold", color="#222222")
        continue
    # AP 主数字着色：绿色=学出来的，红色=瞎选，灰色=其余
    if c == HILITE[1]:
        cell.set_text_props(weight="bold", color=GREEN)
        cell.set_facecolor("#e8f5ec")
    elif c == 3:
        cell.set_text_props(color=RED)
    else:
        cell.set_text_props(color=GREY)

# 50% 行的"学出来的"单元格再加一层强调（28.54 > 全切 28.52）
tbl[HILITE[0] + 1, HILITE[1]].set_facecolor("#c8e6c9")

ax.set_title(
    "VisDrone 274 张留出图：同预算对比（学出来的路由器只跑 50% 切片即追平全切）",
    fontsize=13, pad=14, weight="bold",
)
ax.text(
    0.5, -0.14,
    "数据：REPORT 3.15 / results/visdrone/router_holdout.csv　·　同检测器、同切片网格、同后处理",
    transform=ax.transAxes, ha="center", fontsize=9, color="#666666",
)

out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(out_dir, exist_ok=True)
out = os.path.abspath(os.path.join(out_dir, "table_router_ap.png"))
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
