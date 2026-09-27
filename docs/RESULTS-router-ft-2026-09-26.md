# 强检测器下的可学习路由器（VisDrone-ft）：换掉零训练检测器后重训

对应 REPORT 4.5 / 3.15 的后续："换成微调过的检测器重训路由器，看'一半切片不掉点'在检测器本身更强、可省空间更小时是否还成立"。检测器为 `weights/yolo11s-visdrone-ft.pt`（2 类，VisDrone 训练集微调，见 [RESULTS-ft](RESULTS-ft-visdrone-2026-09-26.md)）；缓存用 `run_eval.py cache --dataset visdrone_ft` 新建（`results/visdrone_ft/cache.pkl`），与 COCO 检测器的缓存分开。

## 协议

- 数据与划分：VisDrone2019-DET-val 548 张，奇偶对半（274 训练 / 274 留出），与 3.15 完全同协议（`split=oddeven`，标签 `gain`）。
- 训练切片 2133（正例率 0.313），留出切片 2142。
- 5 折按图分组 CV 在 (hidden, α) 网格上选中 **hidden=0（线性）、α=0**；最终模型为 5 种子集成（~23 参数/成员）。
- 默认工作点：全局阈值 μ=0.0600，训练集激活率 ρ=0.907（= 手工门 θ=0.9 在训练集上的切片比例）。
- 全程离线读缓存；留出集不参与训练与调参。

## 留出集端到端

来源：`results/visdrone_ft/router_holdout_ft.csv`、`figures/fig12_router_ft.png`。

| 方法 | 切片比例 | AP | APs | 小目标覆盖 |
|---|---:|---:|---:|---:|
| full_image（整图 640） | 0 | 37.70 | 25.49 | 0 |
| sahi_uniform（dense，全切） | 100% | 47.40 | 39.37 | 1.000 |
| oracle_gt_small | 77.3% | 47.54 | 39.48 | 1.000 |
| fusion_thr@0.9 | 90.4% | 47.41 | 39.39 | 0.999 |
| router_matched@0.9 | 90.4% | 47.43 | 39.39 | 1.000 |
| fusion_thr@0.99 | 84.1% | 47.43 | 39.37 | 0.997 |
| router_matched@0.99 | 84.1% | 47.45 | 39.39 | 0.997 |
| router_global@op（μ=0.060） | 91.5% | 47.45 | 39.36 | 1.000 |

预算曲线（同预算对比，来源同上）：

| 预算 ρ | router_global（实际切片 / AP） | random（AP，3 种子） | fusion_budget AP |
|---:|---|---|---|
| 0.3 | 29.3% / 46.28 | 41.26 / 41.04 / 41.39 | 45.57（26.5%） |
| 0.4 | 40.4% / 47.17 | 42.94 / 42.36 / 42.52 | — |
| 0.5 | 51.1% / 47.37 | 44.12 / 43.52 / 43.66 | 46.60（50.2%） |

**随机对照差距巨大**：省约一半切片时，路由 47.37 vs 随机 43.5–44.1（掉 3.3–3.9 点）；省 70% 时 46.28 vs 41.0–41.4（掉 5 点）。路由丢掉的确实是"没用的"切片。

## 切片级排序与标定

来源：`results/visdrone_ft/router_metrics_ft.csv`（留出集）。

| 打分器 | AUC | PR-AUC | ECE | 平均分 |
|---|---:|---:|---:|---:|
| fusion(det+edge) | 0.730 | 0.505 | **0.611** | 0.942 |
| det_noisyor | 0.730 | 0.505 | 0.605 | 0.931 |
| router | **0.824** | **0.684** | **0.036** | 0.360 |

- 强检测器让 noisy-OR 严重饱和：手工门的平均分被推到 0.94，ECE 高达 0.61（3.15 里 COCO 检测器下是 0.32），θ 阈值几乎失去分辨力——这正是 RESULTS-ft 里 F3"只省 9.4% 切片"的机理。
- 路由器不受影响：排序质量（AUC +0.09、PR-AUC +0.18）与标定（ECE 0.036）都大幅好于手工门。

## 特征重要性（置换，留出集 PR-AUC 下降）

来源：`results/visdrone_ft/router_features_ft.csv`。

`mass_other`（车辆类扫视证据量，0.106）≫ `log_app_size`（0.033）> `mass_person`（0.014）> `log_n_strong`（0.013）> 其余 ≤0.006。强检测器下路由主要靠"这片里有多少车"，边缘先验（edge，0.006）贡献退居其后。

## 对照 3.15（COCO 检测器）与结论

| | COCO 检测器（3.15） | VisDrone-ft 检测器（本节） |
|---|---|---|
| dense AP | 28.52 | 47.40 |
| router_global@0.5 | 28.54 @ 50.1% | 47.37 @ 51.1% |
| oracle（77.3% 切片） | 28.63（+0.11） | 47.54（+0.14） |
| 切片级 AUC（router / fusion） | 0.866 / 0.728 | 0.824 / 0.730 |
| 路由 ECE | 0.055 | 0.036 |

- **"一半切片不掉点"仍然成立**：51.1% 切片只掉 0.03 点，同预算随机掉 3.3–3.9 点；ρ=0.6 时甚至比 dense 高 0.08 点。
- 但可省空间的上限没变多少：oracle 在 77% 切片处也只有 +0.14 点——强检测器下"漏检"几乎全部是检测器能力上限，选片（无论手工还是学习）能优化的总量本来就小，这与 4.4 的定量分解一致。
- 手工门在强检测器下因 noisy-OR 饱和而失效（ECE 0.61、只省 9%），可学习路由器保持校准与分辨力——这是它相对手工稀疏门最清楚的一个卖点。

## 代码配套（同期提交）

- `router.py`：`FEATURE_CFG_KEYS` + `feature_cfg()`——会改变特征数值的配置（img_weight/det_margin/heat_sigma/img_map_size）存进 router.json 的 meta，推理时一律按训练时的值重建特征口径（`predict.score_slices`），旧 router.json 回退到默认值。`router_ft.json` 已带这些字段。
- `predict.py`：`make_postprocess` 对 OBB 检测器强制 NMS（与 `get_sliced_prediction` 一致）；`_predict_slices` 支持批推理（`GlanceConfig.batch_size`，默认 1 与既有结果逐位一致）；`sahi_uniform_prediction` 透传 `batch_size`。
- `run_eval.py`：`--batch-size` 参数（cache/e2e 共用）。
- `tests/test_router.py`：新增 `test_online_router_feature_cfg_comes_from_training`。

## 结果文件

`results/visdrone_ft/`：`router_ft.json`、`router_holdout_ft.csv`、`router_metrics_ft.csv`、`router_features_ft.csv`、`router_cv_ft.csv`、`router_sparsity_ft.csv`、`router_per_image_ft.csv`、`router_gates_ft.json`、`figures/fig12_router_ft.png`、`figures/fig14_router_sparsity_ft.png`。

复现：

```powershell
& .\.venv\Scripts\python.exe scripts/run_eval.py cache --dataset visdrone_ft --weights weights/yolo11s-visdrone-ft.pt
& .\.venv\Scripts\python.exe scripts/train_router.py --dataset visdrone_ft --tag _ft
```
