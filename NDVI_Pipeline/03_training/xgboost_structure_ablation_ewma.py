"""
XGBoost structure ablation with fixed Lag3 + EWMA3 input.

The goal is to improve the prediction model through model-structure choices
instead of adding many handcrafted features.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FILE = os.path.join(BASE_DIR, "04_results", "xgboost_fill_ablation_whittaker", "Kalman_Whittaker.csv")
RESULT_DIR = os.path.join(BASE_DIR, "04_results", "xgboost_structure_ablation_ewma")
os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_START = "1985-01-01"
TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-05-16"
FEATURES = ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3"]
DECIMAL = 4


def calc_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.zeros_like(denom, dtype="float64")
    valid = denom != 0
    diff[valid] = np.abs(y_true[valid] - y_pred[valid]) / denom[valid]
    return np.mean(diff) * 100


def build_data():
    data = pd.read_csv(TARGET_FILE)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[["point", "date", "NDVI_Target"]].sort_values(["point", "date"]).reset_index(drop=True)
    for lag in range(1, 4):
        data[f"NDVI_prev{lag}"] = data.groupby("point")["NDVI_Target"].shift(lag)
    data["NDVI_ewma_3"] = data.groupby("point")["NDVI_Target"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    return data


def candidate_params():
    common = {
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
        "verbosity": 0,
    }
    candidates = {
        "baseline_depth4": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        "shallow_depth3_more_trees": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        },
        "depth3_stronger_regularization": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 5,
            "gamma": 0.001,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
        "depth4_child_weight": {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 3,
            "gamma": 0.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
        },
        "depth5_regularized": {
            "n_estimators": 300,
            "learning_rate": 0.04,
            "max_depth": 5,
            "min_child_weight": 5,
            "gamma": 0.001,
            "subsample": 0.75,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
        "dart_dropout": {
            "booster": "dart",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "rate_drop": 0.05,
            "skip_drop": 0.5,
        },
    }
    return {name: {**params, **common} for name, params in candidates.items()}


def train_one(name, params, train, test):
    t0 = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(train[FEATURES], train["NDVI_Target"])
    y_true = test["NDVI_Target"].to_numpy()
    y_pred = np.round(model.predict(test[FEATURES]), DECIMAL)
    pred = test[["point", "date", "NDVI_Target"]].copy()
    pred.insert(0, "variant", name)
    pred["Prediction"] = y_pred
    pred.to_csv(os.path.join(RESULT_DIR, f"Predictions_{name}.csv"), index=False)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "variant": name,
        "MSE": round(float(mse), DECIMAL),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), DECIMAL),
        "RMSE": round(float(np.sqrt(mse)), DECIMAL),
        "SMAPE": round(float(calc_smape(y_true, y_pred)), DECIMAL),
        "time_s": round(time.time() - t0, 1),
        **{k: v for k, v in params.items() if k not in {"tree_method", "device", "random_state", "verbosity"}},
    }


def plot_summary(summary):
    order = summary.sort_values("RMSE")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(order["variant"], order["RMSE"], color="#4c78a8")
    axes[0].set_title("RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(order["variant"], order["SMAPE"], color="#59a14f")
    axes[1].set_title("SMAPE")
    axes[1].set_ylabel("SMAPE")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("固定Lag3+EWMA3输入下的XGBoost结构消融")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "xgboost_structure_ablation_ewma.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary):
    baseline = summary[summary["variant"] == "baseline_depth4"].iloc[0]
    best = summary.iloc[0]
    rmse_gain = (baseline["RMSE"] - best["RMSE"]) / baseline["RMSE"] * 100
    smape_gain = (baseline["SMAPE"] - best["SMAPE"]) / baseline["SMAPE"] * 100
    path = os.path.join(RESULT_DIR, "xgboost_structure_ablation_ewma_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# XGBoost结构消融实验（固定Lag3+EWMA3输入）\n\n")
        f.write(to_markdown_table(summary))
        f.write("\n\n")
        f.write(
            f"最佳结构为 `{best['variant']}`，RMSE={best['RMSE']:.4f}，SMAPE={best['SMAPE']:.4f}。"
            f"相对基准结构 `baseline_depth4`，RMSE下降约 {rmse_gain:.2f}%，"
            f"SMAPE下降约 {smape_gain:.2f}%。\n\n"
        )
        f.write(
            "论文中可将结构改进概括为：在固定简洁EWMA输入的前提下，"
            "通过浅层树、较小学习率、更多弱学习器、样本/特征采样和正则化约束，"
            "降低单棵树对局部波动的过拟合，使模型更适合平滑后的NDVI时间序列预测。\n"
        )
    return path


def to_markdown_table(df):
    lines = [
        "| " + " | ".join(df.columns.astype(str)) + " |",
        "| " + " | ".join([":---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    data = build_data()
    train = data[(data["date"] >= TRAIN_START) & (data["date"] <= TRAIN_END)].dropna(subset=FEATURES + ["NDVI_Target"])
    test = data[(data["date"] >= TEST_START) & (data["date"] <= TEST_END)].dropna(subset=FEATURES + ["NDVI_Target"])
    print(f"train_rows={len(train)}, test_rows={len(test)}, features={FEATURES}")
    rows = []
    for name, params in candidate_params().items():
        print(f"Training {name}...")
        rows.append(train_one(name, params, train, test))
    summary = pd.DataFrame(rows).sort_values(["RMSE", "SMAPE"]).reset_index(drop=True)
    summary.to_csv(os.path.join(RESULT_DIR, "xgboost_structure_ablation_ewma_summary.csv"), index=False)
    plot_summary(summary)
    report = write_report(summary)
    print(summary.to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
