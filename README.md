# Glance-SAHI：先扫一眼全图，只切可疑区域

在 [SAHI](https://github.com/obss/sahi)（切片推理）+ [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 的基础上做的一个免训练优化：
SAHI 对整张图均匀切片、每片都跑一次检测器；Glance-SAHI 先把整张图缩小看一眼，用粗检测和图像显著性给每片打分，只对可疑切片做高清推理。

设计与实验结论见 [REPORT.md](REPORT.md)。一句话结果（同一检测器，零训练）：

| 数据集 | 切片减少 | 提速 | 精度 |
|---|---|---|---|
| VisDrone2019-DET-val（548 张，密集城市航拍） | −21%（θ=0.9） / −37%（θ=0.99） | +12%（端到端） / +31% | AP −0.13 / −0.50 |
| DOTA-v1.0 val（458 张，大幅面遥感） | −42%（仅边缘先验 θ=0.5） | +35% | AP 2.53 → 2.61（COCO 模型在俯视图上绝对精度低，见报告 3.7） |

4K 受控实验（自造画布，只是机制验证、不含检测器，见报告 3.10）：目标只占 1.4% 面积时 **87.3% 的切片是空的**（SAHI 仍恒定切 60 片/图）。

另外三节是**同预算**下的增强（3.12–3.14，仍零训练、离线可复现，各带 3 张新图）：
- **3.12 标定**：把粗检置信度按表观尺度分箱、用保序回归标定成可当概率用的 p̂（留出集 ECE 0.51 → 0.03）。同预算下排序更准：50% 预算就能拿到 SAHI 全切 AP 的 98.8%。
- **3.13 漏检归因**：小目标漏检 35.0 个点里，**34.2 个点是零训练检测器本身检不出**（任何切片都没用），选片只占 0.6 个点——继续优化选片的收益上限很小。
- **3.14 主动选片**：两轮"边看边决定"在 VisDrone 上**没有可测收益**（与 3.13 的结论一致）；留下的是新旋钮 **E 停机**（每图自适应预算）。它在约 83% 预算处与一次性选片持平：+0.0013 AP 的差来自插值对照、没有区间，不作为"更好"的证据。

## 仓库里有什么 / 需要自己准备什么

Git 里只放**代码、文档、评测 CSV 与报告图**；体积大、可重建的东西一律不进仓库（见 `.gitignore`）：

| 不在仓库里 | 体积 | 为什么 | 怎么拿到 |
|---|---|---|---|
| `.venv/` | 约 4.9 GB | 其中 **PyTorch + CUDA 约 4.4 GB**，是本机运行环境；换机器重装即可 | 按下面「运行」一节重建（约几分钟，视网速） |
| `datasets/` | 558 MB | 公开数据集，可重新下载 | `scripts/prepare_data.py`（VisDrone / DOTA）；4K 画布用 `scripts/sparsity_sweep.py --save` |
| `yolo11s.pt` | 18 MB | 预训练权重 | 首次运行 Ultralytics 会自动下载 |
| `results/**/cache.pkl` | ~200 MB | 切片级评测缓存 | `scripts/run_eval.py cache` 重新生成 |

因此 **克隆后必须先装环境 + 准备数据**再跑评测；只有 `pytest tests` 不需要 GPU 和数据集。

## 目录

```
glance_sahi/          算法本体
  config.py           全部参数（切片网格、阈值、打分变体、绝对证据量 τ、融合权重）
  saliency.py         检测先验 S_det（noisy-OR / 4p(1−p) / 取最大 / 粗检热图）、图像先验 S_img、融合、证据量
  selector.py         阈值 θ / 固定预算 / 绝对证据量 τ 三种选片 + 随机对照 + 保底切片
  calibration.py      按"表观尺度"分箱的保序回归（PAVA）标定：把 c_j 变成可当概率用的 p̂_j（REPORT 3.12）
  active.py           主动式两轮选片 + 可计算的停机判据 E = Σ_{未跑} S_det（REPORT 3.14）
  rules.py            业务规则层（着地点 × 禁停多边形，纯几何可单测）
  predict.py          glance_sliced_prediction()：可直接替换 sahi.predict.get_sliced_prediction
  detector.py         SAHI 的 ultralytics 封装 + COCO→评测类别映射
  imageio.py          Windows 中文路径安全的图像读写（cv2.imread 会失败）
  data/visdrone.py    VisDrone 原始标注 → COCO json（另提供四类细分 json）
  data/dota.py        DOTA（ultralytics OBB 格式）→ 水平框 COCO json
scripts/
  prepare_data.py     下载 / 转换数据集（--dataset visdrone|dota|sparse4k）
  run_eval.py         cache / sim / e2e / buckets 四个子命令（--dataset visdrone|dota|sparse4k）
  make_figures.py     报告图 fig1–fig8（含打分变体 fig6、分桶精度 fig7）
  visualize.py        三联图：SAHI 全切 | Glance 选中（未选压暗）| 粗检热图
  sparsity_sweep.py   受控实验：4K 稀疏画布上的“稀疏度 → 可省比例” + fig8（不需要 GPU）
  per_class.py        person / car / truck / bus 四类 AP（GT 与检测同时 remap）
  attribute.py        漏检归因分解：检测器能力 vs 选片代价 vs 选中却漏 → fig10（REPORT 3.13）
  active_eval.py      主动式两轮选片 + E 停机判据的曲线对比（离线，无需 GPU）→ fig11（REPORT 3.14）
  calibrate.py        粗检测置信度的分箱标定：奇偶拆分拟合/留出，ECE + 同预算 AP 对比 → fig9（REPORT 3.12）
  illegal_parking.py  违停业务闭环示范（检测 → 规则层 → 标注图）
  stats.py            随机对照均值±方差、逐图切片比例统计
  coverage.py         与检测器无关的“目标覆盖率 vs 切片比例” + fig5
  lambda_check.py     图像先验权重 λ 的敏感性
  edge_vs_random.py   DOTA 上“仅边缘先验” vs 同数量随机选片
  pick_cases.py       挑选展示案例
tests/test_core.py    单元测试 28 项（网格与 SAHI 一致、noisy-OR 累积弱证据、热图、打分变体、τ/θ/E 选片、
                      规则层、类别映射、分箱标定 PAVA/ECE、主动选片的观测扩散与停机判据）
results/              VisDrone 的 CSV、图、可视化（含 3.12–3.14 的 calibration/calib_*/attribution/active_* 与 fig9–fig11）
results/dota/         DOTA 的 CSV、图、可视化
results/legacy_saliency_sahi/  参照实现归档的 v0→v1 诊断数据（slice_gain.csv 等，见 REPORT 3.8）
datasets/VisDrone-Sparse4K/    受控实验画布（sparsity_sweep.py --save 生成，可当 --dataset sparse4k 评测）
```

## 运行（Windows / PowerShell，RTX 3050 4GB 上验证）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$py = ".\.venv\Scripts\python.exe"

& $py -m pytest tests -q                             # 单元测试（不需要 GPU / 数据集）

# ---- VisDrone ----
& $py scripts/prepare_data.py                        # → datasets/VisDrone2019-DET-val/coco_eval.json
& $py scripts/run_eval.py cache                      # 每张图跑一次扫视 + 全部切片，缓存结果与耗时
& $py scripts/run_eval.py sim --op 0.9               # 扫阈值、消融（含 det_prior 三种变体）、随机对照 → results/sweep.csv
& $py scripts/run_eval.py e2e --op 0.9 --check-all   # 真实端到端计时 + “全选=官方SAHI”自检 → results/e2e.csv
& $py scripts/make_figures.py 0.9                    # → results/figures/（fig1–fig7）
& $py scripts/visualize.py --op 0.9                  # → results/vis/（三联图，含粗检热图）
& $py scripts/stats.py 0.9
& $py scripts/coverage.py visdrone
& $py scripts/run_eval.py buckets --op 0.9            # 按目标数分桶的精度 → results/buckets.csv
& $py scripts/per_class.py --op 0.9                   # 四类细分 AP → results/per_class.csv
& $py scripts/calibrate.py                           # 粗检测置信度分箱标定（奇偶拆分，离线，无需 GPU）→ calibration.csv / calib_*.csv / figures/fig9_calibration.png
& $py scripts/attribute.py                           # 漏检归因：选片代价 vs 检测器能力（离线，无需 GPU）→ attribution.csv / figures/fig10_attribution.png
& $py scripts/active_eval.py                         # 主动式两轮选片 + E 停机判据（离线，无需 GPU）→ active_holdout.csv / active_E.csv / figures/fig11_active.png
& $py scripts/illegal_parking.py --image datasets/VisDrone2019-DET-val/images/0000100_00504_d_0000004.jpg --auto-zone

# ---- 受控实验：4K 稀疏画布（不需要 GPU / 检测器） ----
& $py scripts/sparsity_sweep.py --save                # → results/sparsity_*.csv、figures/fig8_sparsity.png、datasets/VisDrone-Sparse4K/
# 想在画布上要真实 AP，再跑这三行（需要 GPU）：
& $py scripts/prepare_data.py --dataset sparse4k
& $py scripts/run_eval.py cache --dataset sparse4k
& $py scripts/run_eval.py sim   --dataset sparse4k

# ---- DOTA-v1.0 val ----
# 从 https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip 下载后只解出验证集：
#   tar -xf DOTAv1.zip DOTAv1/images/val DOTAv1/labels/val   （放到 datasets/ 下）
& $py scripts/prepare_data.py --dataset dota
& $py scripts/run_eval.py cache --dataset dota
& $py scripts/run_eval.py sim --dataset dota
& $py scripts/coverage.py dota                       # → results/dota/coverage.csv, figures/fig5_coverage.png
& $py scripts/edge_vs_random.py dota
& $py scripts/lambda_check.py dota
& $py scripts/visualize.py --dataset dota --prior edge --op 0.5
```

先想小规模试跑，给 `cache` / `e2e` 加 `--limit 50`；`sim` 现在跑全量要 100 多次 COCO 评测（约 1 小时以上），
只想看某个变体时用 `--only uncertain,heatmap --no-random`（`sim` 也支持 `--limit`，但它会落在缓存的前 N 张上，数字会更噪，别用来出报告）。

## 现场 Demo

```powershell
& $py -m pip install gradio
& $py app.py        # 浏览器打开 http://127.0.0.1:7860
```

上传一张图（或点示例），同屏对比 SAHI 全切和 Glance-SAHI 选片：切片数、检测框数、单图耗时，外加每片的扫视打分 S(k)。
可切换三种检测器预设（COCO 零训练 / VisDrone 微调 / DOTA 官方 OBB），θ、λ 可拖动。耗时是本机单次实测，每种图像尺寸第一次运行前会先完整预热一遍。

## 在自己的代码里使用

```python
from glance_sahi import GlanceConfig, glance_sliced_prediction
from glance_sahi.detector import build_model, EXCLUDE_COCO_IDS

model = build_model("yolo11s.pt", conf=0.05)
cfg = GlanceConfig(threshold=0.9)          # 检测器不认识的场景（如 COCO 模型跑遥感图）：GlanceConfig(threshold=0.5, img_weight=1.0)
# 打分函数消融变体（REPORT 3.8 / 3.9）：
#   det_prior="uncertain"（4p(1−p) 加权）/ "max"（v0 的取最大）/ "heatmap"（粗检热图，O(像素) 一次卷积）
# 绝对证据量 τ 是一个不需要归一化的旋钮（REPORT 3.9）：
#   cfg = GlanceConfig(mode="evidence", tau=1.0)   # 片内 Σc ≥ τ 才细看，可跨图统一标定
result, stats = glance_sliced_prediction(image_rgb, model, cfg, EXCLUDE_COCO_IDS)
print(stats.n_slices_run, "/", stats.n_slices_total, "slices")
result.export_visuals(export_dir="out/")   # 与 SAHI 的 PredictionResult 完全兼容
```

## 参考的开源项目与数据集

SAHI（obss/sahi）、Ultralytics、ClusDet、DMNet、GLSAN、QueryDet、UFPMP-Det、CZDet、YOLC、ESOD、GOIS、VisDrone、DOTA。链接与逐项对比见 REPORT.md 第 2.4 节和参考文献。
