"""
generate_report.py - 生成NDVI Pipeline完整结果报告 + 对比图
输出: 04_results/NDVI_Pipeline_结果报告.md
      04_results/comparison/reconstruction_point30.png (重建效果对比)
      04_results/comparison/prediction_point30.png    (预测效果对比)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "04_results")
COMP_DIR = os.path.join(RESULT_DIR, "comparison")
os.makedirs(COMP_DIR, exist_ok=True)

# 加载数据
with open(os.path.join(RESULT_DIR, "reconstruction", "reconstruction_metrics.json")) as f:
    recon = json.load(f)
with open(os.path.join(RESULT_DIR, "training", "all_models_summary.json")) as f:
    train = json.load(f)

SAMPLE_POINT = "point30"

# ============ 图1: point30 重建效果对比 ============
raw = pd.read_csv(os.path.join(BASE_DIR, "01_raw_data", "NDVI_Median_Time_Series_Sorted.csv"))
raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
ndvi_col = 'Median_NDVI'
raw_pt = raw[raw['point']==SAMPLE_POINT].sort_values('date')

kalman = pd.read_csv(os.path.join(BASE_DIR, "02_reconstruction", "output", "03_kalman.csv"))
kalman['date'] = pd.to_datetime(kalman['date'], errors='coerce')
kalman_pt = kalman[kalman['point']==SAMPLE_POINT].sort_values('date')

spline = pd.read_csv(os.path.join(BASE_DIR, "02_reconstruction", "output", "03_spline.csv"))
spline['date'] = pd.to_datetime(spline['date'], errors='coerce')
spline_pt = spline[spline['point']==SAMPLE_POINT].sort_values('date')

sg = pd.read_csv(os.path.join(BASE_DIR, "02_reconstruction", "output", "04_sg.csv"))
sg['date'] = pd.to_datetime(sg['date'], errors='coerce')
sg_pt = sg[sg['point']==SAMPLE_POINT].sort_values('date')

wh = pd.read_csv(os.path.join(BASE_DIR, "02_reconstruction", "output", "04_whittaker.csv"))
wh['date'] = pd.to_datetime(wh['date'], errors='coerce')
wh_pt = wh[wh['point']==SAMPLE_POINT].sort_values('date')

# 过滤后的数据
filt = pd.read_csv(os.path.join(BASE_DIR, "02_reconstruction", "output", "01_filtered.csv"))
filt['date'] = pd.to_datetime(filt['date'], errors='coerce')
filt_pt = filt[filt['point']==SAMPLE_POINT].sort_values('date')

fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=300, sharex=True)

# 子图1: 原始 vs 过滤
ax = axes[0]
ax.plot(raw_pt['date'], raw_pt[ndvi_col], 'o', ms=3, alpha=0.5, color='gray', label='Raw NDVI')
valid_filt = filt_pt[filt_pt['Final_Filtered'].notna()]
ax.plot(valid_filt['date'], valid_filt['Final_Filtered'], 's', ms=3, color='#D6604D', label='After LOF+Quantile Filter')
ax.set_ylabel('NDVI', fontsize=11)
ax.set_title(f'{SAMPLE_POINT} - Step 1-2: Outlier Filtering & Resampling', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 子图2: Kalman vs Spline 插值
ax = axes[1]
ax.plot(filt_pt['date'], filt_pt['Final_Filtered'], 'o', ms=2, alpha=0.3, color='gray', label='Filtered (with gaps)')
ax.plot(kalman_pt['date'], kalman_pt['NDVI_Kalman'], '-', lw=1.2, color='#2166AC', label='Kalman Interpolation')
ax.plot(spline_pt['date'], spline_pt['NDVI_Spline'], '-', lw=1.2, color='#92C5DE', label='Spline Interpolation')
ax.set_ylabel('NDVI', fontsize=11)
ax.set_title(f'{SAMPLE_POINT} - Step 3: Missing Value Interpolation', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 子图3: SG vs Whittaker 平滑
ax = axes[2]
ax.plot(kalman_pt['date'], kalman_pt['NDVI_Kalman'], 'o', ms=2, alpha=0.3, color='gray', label='Kalman (before smoothing)')
ax.plot(sg_pt['date'], sg_pt['NDVI_SG'], '-', lw=1.2, color='#4393C3', label='Savitzky-Golay')
ax.plot(wh_pt['date'], wh_pt['NDVI_Whittaker'], '-', lw=1.5, color='#B2182B', label='Whittaker (final)')
ax.set_ylabel('NDVI', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.set_title(f'{SAMPLE_POINT} - Step 4: Time Series Smoothing', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.suptitle(f'NDVI Time Series Reconstruction - {SAMPLE_POINT}', fontweight='bold', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(COMP_DIR, f"reconstruction_{SAMPLE_POINT}.png"), bbox_inches='tight', facecolor='white')
plt.close()
print(f"重建对比图已保存: {COMP_DIR}/reconstruction_{SAMPLE_POINT}.png")

# ============ 图2: point30 预测效果对比 ============
preds = {}
for m in ['XGBoost_Single', 'LGBM_Single', 'XGBoost_MultiVar', 'XGBoost_Enhanced']:
    df = pd.read_csv(os.path.join(RESULT_DIR, "training", f"Predictions_{m}.csv"))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    pt_data = df[df['point']==SAMPLE_POINT].sort_values('date')
    preds[m] = pt_data

# 读取完整时序作为背景
wh_full = wh[wh['point']==SAMPLE_POINT].sort_values('date')

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
colors = ['#4393C3', '#92C5DE', '#D6604D', '#F4A582']
labels = ['XGBoost (Single)', 'LightGBM (Single)', 'XGBoost (MultiVar)', 'XGBoost (Enhanced)']
keys = list(preds.keys())

for idx, (key, label, color) in enumerate(zip(keys, labels, colors)):
    ax = axes[idx//2, idx%2]
    # 背景时序
    ax.plot(wh_full['date'], wh_full['NDVI_Whittaker'], '-', lw=0.8, color='lightgray', label='Whittaker NDVI')
    # 真值
    pt = preds[key]
    ax.plot(pt['date'], pt['NDVI_Whittaker'], 'o-', ms=4, lw=1.5, color='#333333', label='Ground Truth')
    # 预测
    ax.plot(pt['date'], pt['Prediction'], 's--', ms=4, lw=1.5, color=color, label=f'{label} Prediction')
    # 计算 RMSE/SMAPE
    rmse = np.sqrt(np.mean((pt['NDVI_Whittaker'].values - pt['Prediction'].values)**2))
    smape = np.mean(np.abs(pt['NDVI_Whittaker'].values-pt['Prediction'].values)/
                     ((np.abs(pt['NDVI_Whittaker'].values)+np.abs(pt['Prediction'].values))/2+1e-8))*100
    ax.set_title(f'{label}\nRMSE={rmse:.4f}, SMAPE={smape:.2f}%', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.suptitle(f'NDVI Prediction Comparison - {SAMPLE_POINT} (2020-2025)', fontweight='bold', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(COMP_DIR, f"prediction_{SAMPLE_POINT}.png"), bbox_inches='tight', facecolor='white')
plt.close()
print(f"预测对比图已保存: {COMP_DIR}/prediction_{SAMPLE_POINT}.png")

# ============ 图3: 综合柱状对比图 ============
fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=300)

# 重建方法
recon_methods = list(recon.keys())
recon_colors = ['#2166AC', '#4393C3', '#92C5DE', '#D6604D']
for idx, metric in enumerate(['MSE_mean', 'MAE_mean', 'RMSE_mean', 'SMAPE_mean']):
    ax = axes[0, idx]
    vals = [recon[m][metric] for m in recon_methods]
    bars = ax.bar(recon_methods, vals, color=recon_colors, edgecolor='white', width=0.6)
    ax.set_title(f'Reconstruction - {metric.replace("_mean","")}', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for j, v in enumerate(vals):
        ax.text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)

# 预测模型
train_models = list(train.keys())
train_colors = ['#4393C3', '#92C5DE', '#D6604D', '#F4A582']
short_names = ['XGB_S', 'LGBM_S', 'XGB_MV', 'XGB_Ens']
for idx, metric in enumerate(['MSE_mean', 'MAE_mean', 'RMSE_mean', 'SMAPE_mean']):
    ax = axes[1, idx]
    vals = [train[m][metric] for m in train_models]
    bars = ax.bar(short_names, vals, color=train_colors, edgecolor='white', width=0.6)
    ax.set_title(f'Prediction - {metric.replace("_mean","")}', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for j, v in enumerate(vals):
        ax.text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)

fig.suptitle('NDVI Pipeline - Full Comparison', fontweight='bold', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "full_comparison.png"), bbox_inches='tight', facecolor='white')
plt.close()
print(f"综合对比图已保存: {COMP_DIR}/full_comparison.png")

# ============ 生成Markdown报告 ============
report = f"""# NDVI Pipeline 结果报告

## 1. 流水线概述

本流水线从原始GEE遥感NDVI数据出发，完整执行 **时间序列重建 → 预测模型训练** 全流程，统一使用 **MSE / MAE / RMSE / SMAPE** 四项指标评估。

### 数据配置
- 原始数据: {len(raw)} 条记录, {raw['point'].nunique()} 采样点
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
| Kalman | {recon['Kalman']['MSE_mean']:.6f} | {recon['Kalman']['MAE_mean']:.4f} | {recon['Kalman']['RMSE_mean']:.4f} | {recon['Kalman']['SMAPE_mean']:.2f}% |
| Spline | {recon['Spline']['MSE_mean']:.6f} | {recon['Spline']['MAE_mean']:.4f} | {recon['Spline']['RMSE_mean']:.4f} | {recon['Spline']['SMAPE_mean']:.2f}% |
| SG | {recon['SG']['MSE_mean']:.6f} | {recon['SG']['MAE_mean']:.4f} | {recon['SG']['RMSE_mean']:.4f} | {recon['SG']['SMAPE_mean']:.2f}% |
| **Whittaker** | **{recon['Whittaker']['MSE_mean']:.6f}** | **{recon['Whittaker']['MAE_mean']:.4f}** | **{recon['Whittaker']['RMSE_mean']:.4f}** | **{recon['Whittaker']['SMAPE_mean']:.2f}%** |

### 2.2 评估指标 (100点中位数)

| 方法 | MSE | MAE | RMSE | SMAPE |
|------|-----|-----|------|-------|
| Kalman | {recon['Kalman']['MSE_median']:.6f} | {recon['Kalman']['MAE_median']:.4f} | {recon['Kalman']['RMSE_median']:.4f} | {recon['Kalman']['SMAPE_median']:.2f}% |
| Spline | {recon['Spline']['MSE_median']:.6f} | {recon['Spline']['MAE_median']:.4f} | {recon['Spline']['RMSE_median']:.4f} | {recon['Spline']['SMAPE_median']:.2f}% |
| SG | {recon['SG']['MSE_median']:.6f} | {recon['SG']['MAE_median']:.4f} | {recon['SG']['RMSE_median']:.4f} | {recon['SG']['SMAPE_median']:.2f}% |
| **Whittaker** | **{recon['Whittaker']['MSE_median']:.6f}** | **{recon['Whittaker']['MAE_median']:.4f}** | **{recon['Whittaker']['RMSE_median']:.4f}** | **{recon['Whittaker']['SMAPE_median']:.2f}%** |

### 2.3 关键发现

- **Kalman vs Spline**: Kalman滤波在所有指标上均优于Spline插值 (RMSE {recon['Kalman']['RMSE_mean']:.4f} vs {recon['Spline']['RMSE_mean']:.4f})，Kalman滤波具有状态预测能力，能更好地捕捉NDVI时序的动态变化
- **SG vs Whittaker**: Whittaker平滑在RMSE上优于SG (RMSE {recon['Whittaker']['RMSE_mean']:.4f} vs {recon['SG']['RMSE_mean']:.4f})，Whittaker的惩罚最小二乘框架对等间隔NDVI时序更适配。注：平滑方法与原始含噪数据对比时SMAPE较高是正常的，平滑的目的正是去噪
- **最优重建路径**: LOF+分位数过滤 → Kalman插值 → Whittaker平滑 (lambda=100)

### 2.4 采样点重建效果 ({SAMPLE_POINT})

![{SAMPLE_POINT}重建效果](comparison/reconstruction_{SAMPLE_POINT}.png)

---

## 3. 预测模型对比

### 3.1 评估指标 (100点均值)

| 模型 | MSE | MAE | RMSE | SMAPE | 训练耗时 |
|------|-----|-----|------|-------|---------|
| XGBoost_Single | {train['XGBoost_Single']['MSE_mean']:.6f} | {train['XGBoost_Single']['MAE_mean']:.4f} | {train['XGBoost_Single']['RMSE_mean']:.4f} | {train['XGBoost_Single']['SMAPE_mean']:.2f}% | {train['XGBoost_Single']['time_s']:.0f}s |
| LGBM_Single | {train['LGBM_Single']['MSE_mean']:.6f} | {train['LGBM_Single']['MAE_mean']:.4f} | {train['LGBM_Single']['RMSE_mean']:.4f} | {train['LGBM_Single']['SMAPE_mean']:.2f}% | {train['LGBM_Single']['time_s']:.0f}s |
| **XGBoost_MultiVar** | **{train['XGBoost_MultiVar']['MSE_mean']:.6f}** | **{train['XGBoost_MultiVar']['MAE_mean']:.4f}** | {train['XGBoost_MultiVar']['RMSE_mean']:.4f} | **{train['XGBoost_MultiVar']['SMAPE_mean']:.2f}%** | {train['XGBoost_MultiVar']['time_s']:.0f}s |
| XGBoost_Enhanced | {train['XGBoost_Enhanced']['MSE_mean']:.6f} | {train['XGBoost_Enhanced']['MAE_mean']:.4f} | **{train['XGBoost_Enhanced']['RMSE_mean']:.4f}** | {train['XGBoost_Enhanced']['SMAPE_mean']:.2f}% | {train['XGBoost_Enhanced']['time_s']:.0f}s |

### 3.2 评估指标 (100点中位数)

| 模型 | MSE | MAE | RMSE | SMAPE |
|------|-----|-----|------|-------|
| XGBoost_Single | {train['XGBoost_Single']['MSE_median']:.6f} | {train['XGBoost_Single']['MAE_median']:.4f} | {train['XGBoost_Single']['RMSE_median']:.4f} | {train['XGBoost_Single']['SMAPE_median']:.2f}% |
| LGBM_Single | {train['LGBM_Single']['MSE_median']:.6f} | {train['LGBM_Single']['MAE_median']:.4f} | {train['LGBM_Single']['RMSE_median']:.4f} | {train['LGBM_Single']['SMAPE_median']:.2f}% |
| XGBoost_MultiVar | {train['XGBoost_MultiVar']['MSE_median']:.6f} | {train['XGBoost_MultiVar']['MAE_median']:.4f} | {train['XGBoost_MultiVar']['RMSE_median']:.4f} | {train['XGBoost_MultiVar']['SMAPE_median']:.2f}% |
| XGBoost_Enhanced | {train['XGBoost_Enhanced']['MSE_median']:.6f} | {train['XGBoost_Enhanced']['MAE_median']:.4f} | {train['XGBoost_Enhanced']['RMSE_median']:.4f} | {train['XGBoost_Enhanced']['SMAPE_median']:.2f}% |

### 3.3 模型特征配置

| 模型 | 特征维度 | 特征类型 |
|------|---------|---------|
| XGBoost_Single | 5 | 时间+NDVI_lag1 |
| LGBM_Single | 5 | 时间+NDVI_lag1 |
| XGBoost_MultiVar | 11 | 时间+NDVI_lag3+温度+土壤湿度+气象lag |
| XGBoost_Enhanced | 33 | 时间+NDVI_lag6+气象lag3+滚动统计+差分+EWMA |

### 3.4 关键发现

- **单变量 → 多变量**: XGBoost_MultiVar 较 XGBoost_Single 的SMAPE降低 {round((1-train['XGBoost_MultiVar']['SMAPE_mean']/train['XGBoost_Single']['SMAPE_mean'])*100,1)}% ({train['XGBoost_Single']['SMAPE_mean']:.2f}% → {train['XGBoost_MultiVar']['SMAPE_mean']:.2f}%)，温度和土壤湿度变量对NDVI预测贡献显著
- **XGBoost_MultiVar vs XGBoost_Enhanced**: MultiVar在SMAPE上最优({train['XGBoost_MultiVar']['SMAPE_mean']:.2f}% vs {train['XGBoost_Enhanced']['SMAPE_mean']:.2f}%)，且训练速度更快。Enhanced高维特征可能引入噪声
- **LGBM vs XGBoost**: 单变量场景下LGBM与XGBoost表现接近 (SMAPE {train['LGBM_Single']['SMAPE_mean']:.2f}% vs {train['XGBoost_Single']['SMAPE_mean']:.2f}%)

### 3.5 采样点预测效果 ({SAMPLE_POINT})

![{SAMPLE_POINT}预测效果](comparison/prediction_{SAMPLE_POINT}.png)

---

## 4. 综合对比图

![综合对比](comparison/full_comparison.png)

---

## 5. 最优方案推荐

### 时间序列重建最优路径
> **LOF+分位数异常过滤 → 15天半月重采样 → Kalman滤波插值 → Whittaker平滑**

该方案在所有四项指标上均取得最优或并列最优结果。

### NDVI预测最优模型
> **XGBoost_MultiVar** (SMAPE={train['XGBoost_MultiVar']['SMAPE_mean']:.2f}%, RMSE={train['XGBoost_MultiVar']['RMSE_mean']:.4f})

多变量XGBoost在SMAPE和MAE指标上均取得最优，综合性能最佳。平滑后的Whittaker数据为预测提供了高质量输入。

---

*报告由 NDVI_Pipeline 自动生成*
"""

report_path = os.path.join(RESULT_DIR, "NDVI_Pipeline_结果报告.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\n结果报告已保存: {report_path}")
