# NDVI Pipeline Cleanup Log

保留内容：
- `01_raw_data/` 原始数据。
- `02_reconstruction/step1_filter.py` 到 `step5_process_multivar_aligned.py` 及最终处理输出。
- `02_reconstruction/output/04_final_kalman_whittaker.csv` 最终NDVI重构序列。
- `03_training/xgboost_ewma_feature_ablation_fast.py`、`xgboost_structure_ablation_ewma.py`、`xgboost_ewma_rc_oof_split.py` 最终预测相关脚本。
- `04_results/chapter3_figures/`、`04_results/final_xgboost_prediction_report/` 论文图片和最终预测报告。
- `05_final_outputs/` 集中保存最终重构、预测报告、表格和图片。

已删除内容：
- 旧深度模型、TimesFM、旧训练总表、NDVI_Full、多变量输入、旧PCHIP/SG、不拟合等中间实验结果目录。
- 被最终EWMA-XGBoost/OOF残差实验替代的旧训练脚本。
- `__pycache__` Python缓存。
- 不再作为最终流程输入的 `03_linear.csv`、`03_pchip.csv`、`03_spline.csv`、`04_sg.csv`、旧 `04_whittaker.csv`。

最终主流程：
原始数据 -> LOF异常值检测 -> 15天重采样 -> Kalman填补 -> Whittaker平滑 -> EWMA-XGBoost预测。
