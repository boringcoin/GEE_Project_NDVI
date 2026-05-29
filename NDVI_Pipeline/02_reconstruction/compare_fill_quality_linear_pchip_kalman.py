"""
compare_fill_quality_linear_pchip_kalman.py

比较 Linear、PCHIP、Kalman 三种缺失值填补方法本身的重构质量。

评价维度:
1. 缺失填补准确性: 伪缺失验证 MSE/MAE/RMSE/SMAPE
2. 观测保持能力: 在已有观测点上的 RMSE/SMAPE
3. 曲线形态合理性: repeat_ratio/max_repeat_run
4. 平滑程度: 二阶差分粗糙度 roughness
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")
RESULT_DIR = os.path.join(BASE_DIR, "04_results", "fill_quality_linear_pchip_kalman")
os.makedirs(RESULT_DIR, exist_ok=True)

OBS_FILE = os.path.join(OUTPUT_DIR, "02_resampled.csv")
PSEUDO_FILE = os.path.join(BASE_DIR, "04_results", "imputation_methods", "imputation_methods_summary.csv")

METHODS = {
    "Linear": ("03_linear.csv", "NDVI_Linear"),
    "PCHIP": ("03_pchip.csv", "NDVI_PCHIP"),
    "Kalman": ("03_kalman.csv", "NDVI_Kalman"),
}


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.zeros_like(denom, dtype="float64")
    valid = denom != 0
    diff[valid] = np.abs(y_true[valid] - y_pred[valid]) / denom[valid]
    return float(np.mean(diff) * 100)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def repeat_stats(df, value_col):
    repeat_links = 0
    max_runs = []
    for _, group in df.sort_values(["point", "date"]).groupby("point"):
        s = group[value_col]
        repeat_links += int((s.eq(s.shift()) & s.notna()).sum())
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
    denominator = max(len(df) - df["point"].nunique(), 1)
    return repeat_links / denominator, int(max(max_runs) if max_runs else 0)


def roughness(df, value_col):
    values = []
    for _, group in df.sort_values(["point", "date"]).groupby("point"):
        y = group[value_col].to_numpy(dtype="float64")
        y = y[np.isfinite(y)]
        if len(y) < 3:
            continue
        second_diff = y[2:] - 2 * y[1:-1] + y[:-2]
        values.append(float(np.mean(second_diff ** 2)))
    return float(np.mean(values)) if values else np.nan


def load_pseudo_metrics():
    pseudo = pd.read_csv(PSEUDO_FILE)
    pseudo = pseudo[pseudo["method"].isin(METHODS.keys())].copy()
    pseudo = pseudo.set_index("method")
    return pseudo


def main():
    obs = pd.read_csv(OBS_FILE)
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs[["point", "date", "Final_Filtered"]]
    pseudo = load_pseudo_metrics()

    rows = []
    for method, (filename, value_col) in METHODS.items():
        path = os.path.join(OUTPUT_DIR, filename)
        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        merged = pd.merge(obs, data[["point", "date", value_col]], on=["point", "date"], how="left")
        valid_obs = merged["Final_Filtered"].notna() & merged[value_col].notna()
        y_true = merged.loc[valid_obs, "Final_Filtered"].to_numpy(dtype="float64")
        y_pred = merged.loc[valid_obs, value_col].to_numpy(dtype="float64")
        rep_ratio, max_run = repeat_stats(data, value_col)
        row = {
            "method": method,
            "pseudo_MSE": float(pseudo.loc[method, "MSE"]),
            "pseudo_MAE": float(pseudo.loc[method, "MAE"]),
            "pseudo_RMSE": float(pseudo.loc[method, "RMSE"]),
            "pseudo_SMAPE": float(pseudo.loc[method, "SMAPE"]),
            "obs_RMSE": rmse(y_true, y_pred),
            "obs_SMAPE": smape(y_true, y_pred),
            "repeat_ratio": rep_ratio,
            "max_repeat_run": max_run,
            "roughness_second_diff": roughness(data, value_col),
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    # Ranking: lower is better for all metrics.
    rank_cols = [
        "pseudo_RMSE",
        "pseudo_SMAPE",
        "obs_RMSE",
        "obs_SMAPE",
        "repeat_ratio",
        "max_repeat_run",
        "roughness_second_diff",
    ]
    for col in rank_cols:
        result[f"rank_{col}"] = result[col].rank(method="min", ascending=True).astype(int)
    result["mean_rank"] = result[[f"rank_{c}" for c in rank_cols]].mean(axis=1)
    result["overall_rank"] = result["mean_rank"].rank(method="min", ascending=True).astype(int)
    result = result.sort_values(["overall_rank", "method"])

    csv_path = os.path.join(RESULT_DIR, "fill_quality_linear_pchip_kalman_summary.csv")
    json_path = os.path.join(RESULT_DIR, "fill_quality_linear_pchip_kalman_summary.json")
    report_path = os.path.join(RESULT_DIR, "fill_quality_linear_pchip_kalman_report.md")
    result.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Linear、PCHIP、Kalman 填补质量比较\n\n")
        f.write("本表不依赖下游预测结果，仅从重构质量本身比较三种填补方法。\n\n")
        f.write("| 方法 | 伪缺失RMSE | 伪缺失MAE | 伪缺失SMAPE | 观测点RMSE | 观测点SMAPE | repeat_ratio | max_repeat_run | roughness | 综合排名 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in result.iterrows():
            f.write(
                f"| {r['method']} | {r['pseudo_RMSE']:.4f} | {r['pseudo_MAE']:.4f} | {r['pseudo_SMAPE']:.4f} | "
                f"{r['obs_RMSE']:.4f} | {r['obs_SMAPE']:.4f} | {r['repeat_ratio']:.6f} | "
                f"{int(r['max_repeat_run'])} | {r['roughness_second_diff']:.8f} | {int(r['overall_rank'])} |\n"
            )
        f.write("\n")
        best = result.iloc[0]
        f.write(f"综合各项重构质量指标，排名第一的方法为 **{best['method']}**。\n")
        f.write("需要注意，Linear 和 PCHIP 在已有观测点处保留原始值，因此观测点误差为0；该指标不能单独代表缺失区间恢复能力。\n")

    print(result[[
        "method", "pseudo_RMSE", "pseudo_MAE", "pseudo_SMAPE",
        "obs_RMSE", "obs_SMAPE", "repeat_ratio",
        "max_repeat_run", "roughness_second_diff", "overall_rank"
    ]].to_string(index=False))
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
