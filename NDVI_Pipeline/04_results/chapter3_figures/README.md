# 第三章论文图片清单

本文件夹中的图片由 `NDVI_Pipeline/generate_chapter3_figures.py` 生成，输出分辨率为 300 dpi，可直接用于论文第三章。

| 图号 | 文件名 | 建议图题 |
|:---|:---|:---|
| 图3.1 | `fig3_1_study_area_sampling_points.png` | 特罗多斯山脉研究区位置及采样点分布图 |
| 图3.2 | `fig3_2_raw_missing_anomaly_example.png` | 原始NDVI时间序列缺失与异常观测数据示例 |
| 图3.3 | `fig3_3_outlier_methods_comparison.png` | 不同异常值检测方法在同一采样点NDVI序列上的处理效果对比 |
| 图3.4 | `fig3_4_reconstruction_process_example.png` | NDVI时间序列重构过程示例 |

示例样点为 `point883`。该样点在原始序列中具有较明显的缺失和异常观测，适合展示从异常值剔除、15天窗口重采样、Kalman填补到Whittaker平滑拟合的完整处理过程。

