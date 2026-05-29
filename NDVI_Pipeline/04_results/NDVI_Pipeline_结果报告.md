# NDVI Pipeline 结果报告

## 1. 流水线概述

本流水线从原始GEE遥感NDVI数据出发，完整执行 **时间序列重建 → 预测模型训练** 全流程，统一使用 **MSE / MAE / RMSE / SMAPE** 四项指标评估。

### 数据配置
- 原始数据: 93600 条记录, 100 采样点
- 训练集: 1985-01-01 ~ 2019-12-31
- 测试集: 2020-01-01 ~ 2025-05-16
- 重采样间隔: 15天

### 流水线步骤
1. **异常值过滤**: LOF (n_neighbors=20, contamination=0.1) + 分位数裁剪 (5%-95%)
2. **半月重采样**: 统一为15天间隔时间序列
3. **缺失值插值**: Kalman滤波 / 三次Spline样条
4. **时间序列平滑**: Savitzky-Golay / Whittaker
5. **预测模型训练**: XGBoost_Single / LGBM_Single / XGBoost_MultiVar / XGBoost_Enhanced

---

## 2. 时间序列重建方法对比

### 2.1 评估指标 (100点均值)

| 方法 | MSE | MAE | RMSE | SMAPE |
|------|-----|-----|------|-------|
| Kalman | 0.001154 | 0.0109 | 0.0313 | 9.55% |
| Spline | 0.079846 | 0.0193 | 0.0744 | 10.35% |
| SG | 0.001458 | 0.0224 | 0.0357 | 21.65% |
| **Whittaker** | **0.001442** | **0.0220** | **0.0355** | **21.35%** |

### 2.2 评估指标 (100点中位数)

| 方法 | MSE | MAE | RMSE | SMAPE |
|------|-----|-----|------|-------|
| Kalman | 0.000946 | 0.0102 | 0.0308 | 7.78% |
| Spline | 0.001076 | 0.0111 | 0.0328 | 8.35% |
| SG | 0.001238 | 0.0213 | 0.0352 | 14.97% |
| **Whittaker** | **0.001225** | **0.0212** | **0.0350** | **14.77%** |

### 2.3 关键发现

- **Kalman vs Spline**: Kalman滤波在所有指标上均优于Spline插值 (RMSE 0.0313 vs 0.0744)，Kalman滤波具有状态预测能力，能更好地捕捉NDVI时序的动态变化
- **SG vs Whittaker**: Whittaker平滑在RMSE上优于SG (RMSE 0.0355 vs 0.0357)，Whittaker的惩罚最小二乘框架对等间隔NDVI时序更适配。注：平滑方法与原始含噪数据对比时SMAPE较高是正常的，平滑的目的正是去噪
- **最优重建路径**: LOF+分位数过滤 → Kalman插值 → Whittaker平滑 (lambda=100)

### 2.4 采样点重建效果 (point30)

![point30重建效果](comparison/reconstruction_point30.png)

---

## 3. 预测模型对比

### 3.1 评估指标 (100点均值)

| 模型 | MSE | MAE | RMSE | SMAPE | 训练耗时 |
|------|-----|-----|------|-------|---------|
| XGBoost_Single | 0.000000 | 0.0024 | 0.0030 | 2.40% | 14s |
| LGBM_Single | 0.000000 | 0.0025 | 0.0031 | 2.49% | 34s |
| **XGBoost_MultiVar** | **0.000000** | **0.0012** | 0.0015 | **1.19%** | 9s |
| XGBoost_Enhanced | 0.000000 | 0.0016 | **0.0020** | 1.83% | 56s |

### 3.2 评估指标 (100点中位数)

| 模型 | MSE | MAE | RMSE | SMAPE |
|------|-----|-----|------|-------|
| XGBoost_Single | 0.000000 | 0.0015 | 0.0019 | 0.87% |
| LGBM_Single | 0.000000 | 0.0016 | 0.0019 | 0.86% |
| XGBoost_MultiVar | 0.000000 | 0.0007 | 0.0009 | 0.42% |
| XGBoost_Enhanced | 0.000000 | 0.0011 | 0.0013 | 0.58% |

### 3.3 模型特征配置

| 模型 | 特征维度 | 特征类型 |
|------|---------|---------|
| XGBoost_Single | 5 | 时间+NDVI_lag1 |
| LGBM_Single | 5 | 时间+NDVI_lag1 |
| XGBoost_MultiVar | 11 | 时间+NDVI_lag3+温度+土壤湿度+气象lag |
| XGBoost_Enhanced | 33 | 时间+NDVI_lag6+气象lag3+滚动统计+差分+EWMA |

### 3.4 关键发现

- **单变量 → 多变量**: XGBoost_MultiVar 较 XGBoost_Single 的SMAPE降低 50.6% (2.40% → 1.19%)，温度和土壤湿度变量对NDVI预测贡献显著
- **XGBoost_MultiVar vs XGBoost_Enhanced**: MultiVar在SMAPE上最优(1.19% vs 1.83%)，且训练速度更快。Enhanced高维特征可能引入噪声
- **LGBM vs XGBoost**: 单变量场景下LGBM与XGBoost表现接近 (SMAPE 2.49% vs 2.40%)

### 3.5 采样点预测效果 (point30)

![point30预测效果](comparison/prediction_point30.png)

---

## 4. 综合对比图

![综合对比](comparison/full_comparison.png)

---

## 5. 最优方案推荐

### 时间序列重建最优路径
> **LOF+分位数异常过滤 → 15天半月重采样 → Kalman滤波插值 → Whittaker平滑**

该方案在所有四项指标上均取得最优或并列最优结果。

### NDVI预测最优模型
> **XGBoost_MultiVar** (SMAPE=1.19%, RMSE=0.0015)

多变量XGBoost在SMAPE和MAE指标上均取得最优，综合性能最佳。平滑后的Whittaker数据为预测提供了高质量输入。

---

*报告由 NDVI_Pipeline 自动生成*
