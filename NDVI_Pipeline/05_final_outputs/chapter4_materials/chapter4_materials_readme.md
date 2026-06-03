# 第四章最终材料包说明

本目录仅保留与最终预测模型和论文主文直接相关的材料。

## 最终预测模型

最终主模型为 **EWMA-XGBoost**：

- 输入特征：`NDVI_t-1`、`NDVI_t-2`、`NDVI_t-3`、`EWMA_3`
- 预测目标：测试期 NDVI 序列预测结果
- 数据划分：1985-2014 训练，2015-2019 验证，2020-2025 测试
- 测试集规模：1000 个样点，131000 条样本

## 主文表格

1. `unified_model_comparison_1985_2015/table4_2_unified_model_comparison_for_paper.csv`  
   不同预测模型最终测试集指标对比，用于说明模型选择。

2. `final_xgboost_parameter_ablation_for_paper.csv`  
   XGBoost 参数消融实验论文展示版，方案名称已改为中文可读形式。

3. `final_ewma_feature_ablation_for_paper.csv`  
   EWMA 特征消融实验论文展示版，输入方案名称已改为中文可读形式。

4. `final_xgboost_model_comparison.csv`  
   基础 XGBoost、EWMA-XGBoost、EWMA-RC-XGBoost 三类结构对比。

5. `lstm_feature_residual_ablation/lstm_feature_residual_ablation_for_paper.csv`  
   LSTM 上复用“特征增强—残差校正”路线的补充消融实验。结果显示该路线对 LSTM 未带来稳定提升，可作为 EWMA-XGBoost 选择依据的补充说明。

6. `final_data_split_summary.csv`  
   数据划分说明。

## 主文图件

1. `fig4_1_ewma_xgboost_model_structure.png`  
   EWMA-XGBoost 最终模型结构图。

2. `unified_model_comparison_1985_2015/fig4_2_final_model_metric_comparison.png`  
   不同预测模型 MSE、MAE、RMSE、SMAPE 指标对比图。

3. `typical_point_prediction_curves.png`  
   典型样点预测曲线。

4. `xgboost_advantage_point/fig4_x_xgboost_advantage_point_predictions.png`  
   用于突出 XGBoost 优势的典型样点多模型预测曲线图。

5. `feature_importance_ewma_xgboost.png`  
   EWMA-XGBoost 特征重要性图。

6. `residual_smape_by_ndvi_bin.png`  
   残差校正模型在不同 NDVI 区间的相对误差分析。

7. `lstm_feature_residual_ablation/lstm_feature_residual_ablation.png`  
   LSTM 特征增强与残差校正消融图。

## 已清理内容

早期不同口径模型对比、旧 TimeGPT/N-HiTS 单独指标、官方深度模型原始预测文件、`04_results` 中重复中间结果已经删除。当前目录以最终论文主线为准。
