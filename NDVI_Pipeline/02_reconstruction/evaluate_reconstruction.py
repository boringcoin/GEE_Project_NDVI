"""
evaluate_reconstruction.py - 重建效果评估 (误差指标 + 重复平台诊断)
输入: 02_reconstruction/output/ 下的各步输出
输出: 04_results/reconstruction/ 评估结果 + 对比图
"""
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECON_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")
RESULT_DIR = os.path.join(BASE_DIR, "04_results", "reconstruction")
os.makedirs(RESULT_DIR, exist_ok=True)


def calc_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.zeros_like(denom, dtype='float64')
    valid = denom != 0
    diff[valid] = np.abs(y_true[valid] - y_pred[valid]) / denom[valid]
    return np.mean(diff) * 100


def evaluate_method(df, true_col, pred_col):
    valid = df[[true_col, pred_col]].dropna()
    if len(valid) < 5:
        return None
    y_true, y_pred = valid[true_col].values, valid[pred_col].values
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'SMAPE': calc_smape(y_true, y_pred),
    }


def repeated_stats(df, value_col):
    repeat_links = 0
    max_runs = []
    valid_counts = []
    for _, grp in df.sort_values(['point', 'date']).groupby('point'):
        s = grp[value_col]
        valid_counts.append(int(s.notna().sum()))
        repeat_links += (s.eq(s.shift()) & s.notna()).sum()
        arr = s.to_numpy()
        current = 1
        max_run = 1 if len(arr) else 0
        for i in range(1, len(arr)):
            if pd.notna(arr[i]) and pd.notna(arr[i - 1]) and arr[i] == arr[i - 1]:
                current += 1
            else:
                max_run = max(max_run, current)
                current = 1
        max_runs.append(max(max_run, current))
    denominator = max(len(df) - df['point'].nunique(), 1)
    return {
        'repeat_ratio': float(repeat_links / denominator),
        'max_repeat_run': int(max(max_runs) if max_runs else 0),
        'valid_min': int(np.min(valid_counts) if valid_counts else 0),
        'valid_median': float(np.median(valid_counts) if valid_counts else 0),
        'valid_max': int(np.max(valid_counts) if valid_counts else 0),
    }


def summarize_method(method, fname, pred_col, truth):
    fpath = os.path.join(RECON_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  跳过 {method}: 文件不存在")
        return None

    df = pd.read_csv(fpath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    merged = pd.merge(
        truth[['point', 'date', 'Final_Filtered']],
        df[['point', 'date', pred_col]],
        on=['point', 'date'],
        how='inner',
    )

    point_metrics = []
    for pt, grp in merged.groupby('point'):
        m = evaluate_method(grp, 'Final_Filtered', pred_col)
        if m:
            m['point'] = pt
            point_metrics.append(m)
    if not point_metrics:
        return None

    pm = pd.DataFrame(point_metrics)
    diag = repeated_stats(df, pred_col)
    summary = {
        'MSE_mean': pm['MSE'].mean(), 'MSE_median': pm['MSE'].median(),
        'MAE_mean': pm['MAE'].mean(), 'MAE_median': pm['MAE'].median(),
        'RMSE_mean': pm['RMSE'].mean(), 'RMSE_median': pm['RMSE'].median(),
        'SMAPE_mean': pm['SMAPE'].mean(), 'SMAPE_median': pm['SMAPE'].median(),
        **diag,
    }
    print(
        f"  {method}: RMSE={summary['RMSE_mean']:.4f}, "
        f"SMAPE={summary['SMAPE_mean']:.2f}%, "
        f"重复链接={summary['repeat_ratio']:.2%}, "
        f"最长重复段={summary['max_repeat_run']}"
    )
    return summary


def main():
    t0 = time.time()
    print("=" * 60)
    print("重建效果评估 (误差指标 + 重复平台诊断)")
    print("=" * 60)

    truth_path = os.path.join(RECON_DIR, "02_resampled.csv")
    truth = pd.read_csv(truth_path)
    truth['date'] = pd.to_datetime(truth['date'], errors='coerce')
    observed = truth['Final_Filtered'].notna().sum()
    print(f"  参考真值: 02_resampled.csv / Final_Filtered, 有效观测={observed}")

    results = {}
    for method, fname, col in [
        ('Linear', '03_linear.csv', 'NDVI_Linear'),
        ('PCHIP', '03_pchip.csv', 'NDVI_PCHIP'),
        ('Kalman', '03_kalman.csv', 'NDVI_Kalman'),
        ('SG', '04_sg.csv', 'NDVI_SG'),
        ('Whittaker', '04_whittaker.csv', 'NDVI_Whittaker'),
    ]:
        summary = summarize_method(method, fname, col, truth)
        if summary:
            results[method] = summary

    with open(os.path.join(RESULT_DIR, "reconstruction_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        methods = list(results.keys())
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
        colors = ['#2166AC', '#4393C3', '#92C5DE', '#D6604D', '#F4A582']
        for idx, metric in enumerate(['MSE_mean', 'MAE_mean', 'RMSE_mean', 'SMAPE_mean']):
            vals = [results[m].get(metric, 0) for m in methods]
            axes[idx].bar(methods, vals, color=colors[:len(methods)], edgecolor='white')
            axes[idx].set_title(metric.replace('_mean', ''), fontweight='bold')
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].tick_params(axis='x', labelrotation=25)
            for j, v in enumerate(vals):
                axes[idx].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
        fig.suptitle('Reconstruction Methods Comparison', fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR, "reconstruction_comparison.png"), bbox_inches='tight', facecolor='white')
        plt.close()
        print("  对比图已保存")

    print(f"  耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
