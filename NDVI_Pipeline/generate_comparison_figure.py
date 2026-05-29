"""
generate_comparison_figure.py - 生成论文级对比图 (300PPI)
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 重建结果
with open(os.path.join(BASE_DIR, "04_results", "reconstruction", "reconstruction_metrics.json")) as f:
    recon = json.load(f)

# 训练结果
with open(os.path.join(BASE_DIR, "04_results", "training", "all_models_summary.json")) as f:
    train = json.load(f)

# ========== 图1: 重建方法对比 ==========
fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
colors_recon = ['#2166AC', '#4393C3', '#92C5DE', '#D6604D', '#F4A582']
recon_methods = list(recon.keys())
for idx, metric in enumerate(['MSE_mean', 'MAE_mean', 'RMSE_mean', 'SMAPE_mean']):
    vals = [recon[m].get(metric, 0) for m in recon_methods]
    bars = axes[idx].bar(recon_methods, vals, color=colors_recon[:len(recon_methods)], edgecolor='white')
    axes[idx].set_title(metric.replace('_mean', ''), fontsize=12, fontweight='bold')
    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    axes[idx].tick_params(axis='x', labelrotation=25)
    for j, v in enumerate(vals):
        axes[idx].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
fig.suptitle('NDVI Time Series Reconstruction Methods Comparison', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "04_results", "comparison", "reconstruction_comparison.png"),
            bbox_inches='tight', facecolor='white')
plt.close()

# ========== 图2: 预测模型对比 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
colors_train = ['#4393C3', '#92C5DE', '#D6604D', '#F4A582']
train_models = list(train.keys())
metrics_display = [('MSE_mean', 'MSE'), ('MAE_mean', 'MAE'), ('RMSE_mean', 'RMSE'), ('SMAPE_mean', 'SMAPE(%)')]

for idx, (metric_key, metric_name) in enumerate(metrics_display):
    ax = axes[idx//2, idx%2]
    vals = [train[m][metric_key] for m in train_models]
    short_names = [n.replace('XGBoost_', 'XGB_').replace('_Single', '_S').replace('_MultiVar', '_MV').replace('_Enhanced', '_Ens') for n in train_models]
    bars = ax.bar(short_names, vals, color=colors_train, edgecolor='white', width=0.6)
    ax.set_title(metric_name, fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for j, v in enumerate(vals):
        ax.text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('NDVI Prediction Models Comparison', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "04_results", "comparison", "prediction_comparison.png"),
            bbox_inches='tight', facecolor='white')
plt.close()

print("论文级对比图已生成:")
print(f"  {os.path.join(BASE_DIR, '04_results/comparison/reconstruction_comparison.png')}")
print(f"  {os.path.join(BASE_DIR, '04_results/comparison/prediction_comparison.png')}")
