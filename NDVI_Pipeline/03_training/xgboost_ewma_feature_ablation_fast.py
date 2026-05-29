"""
Fast global XGBoost ablation for EWMA features.

Unlike the point-wise script, this trains one global XGBoost model per feature
set. It is used to decide whether a concise EWMA-enhanced feature design is
worth using in the thesis narrative.
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
RESULT_DIR = os.path.join(BASE_DIR, "04_results", "xgboost_ewma_feature_ablation_fast")
os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_START = "1985-01-01"
TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-05-16"
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
    for lag in range(1, 7):
        data[f"NDVI_prev{lag}"] = data.groupby("point")["NDVI_Target"].shift(lag)
    for span in [3, 5, 7]:
        data[f"NDVI_ewma_{span}"] = data.groupby("point")["NDVI_Target"].transform(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
        )
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear
    data["doy_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 366.0)
    data["doy_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 366.0)
    return data


def feature_sets():
    seasonal = ["doy_sin", "doy_cos"]
    return {
        "Lag3": ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3"],
        "Lag3_EWMA3": ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3"],
        "Lag3_EWMA357": ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3", "NDVI_ewma_5", "NDVI_ewma_7"],
        "Lag3_Seasonal": ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3"] + seasonal,
        "Lag3_EWMA3_Seasonal": ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3"] + seasonal,
        "Lag6": ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_prev4", "NDVI_prev5", "NDVI_prev6"],
        "Lag6_EWMA357": [
            "NDVI_prev1",
            "NDVI_prev2",
            "NDVI_prev3",
            "NDVI_prev4",
            "NDVI_prev5",
            "NDVI_prev6",
            "NDVI_ewma_3",
            "NDVI_ewma_5",
            "NDVI_ewma_7",
        ],
    }


def make_model(max_depth=4):
    return xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=0,
    )


def train_variant(name, features, data):
    t0 = time.time()
    train = data[(data["date"] >= TRAIN_START) & (data["date"] <= TRAIN_END)].dropna(subset=features + ["NDVI_Target"])
    test = data[(data["date"] >= TEST_START) & (data["date"] <= TEST_END)].dropna(subset=features + ["NDVI_Target"])
    model = make_model()
    model.fit(train[features], train["NDVI_Target"])
    y_true = test["NDVI_Target"].to_numpy()
    y_pred = np.round(model.predict(test[features]), DECIMAL)

    pred = test[["point", "date", "NDVI_Target"]].copy()
    pred.insert(0, "variant", name)
    pred["Prediction"] = y_pred
    pred.to_csv(os.path.join(RESULT_DIR, f"Predictions_{name}.csv"), index=False)

    mse = mean_squared_error(y_true, y_pred)
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": features, "gain": [gain.get(f, 0.0) for f in features]})
    imp["gain_ratio"] = imp["gain"] / imp["gain"].sum()
    imp.sort_values("gain", ascending=False).to_csv(
        os.path.join(RESULT_DIR, f"FeatureImportance_{name}.csv"), index=False
    )
    return {
        "variant": name,
        "feature_count": len(features),
        "features": features,
        "train_rows": len(train),
        "test_rows": len(test),
        "MSE": round(float(mse), DECIMAL),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), DECIMAL),
        "RMSE": round(float(np.sqrt(mse)), DECIMAL),
        "SMAPE": round(float(calc_smape(y_true, y_pred)), DECIMAL),
        "time_s": round(time.time() - t0, 1),
    }


def plot_summary(summary):
    order = summary.sort_values("RMSE")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(order["variant"], order["RMSE"], color="#4c78a8")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(order["variant"], order["SMAPE"], color="#59a14f")
    axes[1].set_ylabel("SMAPE")
    axes[1].set_title("SMAPE")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("EWMA简化特征对XGBoost预测性能的影响")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "xgboost_ewma_feature_ablation_fast.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    data = build_data()
    rows = []
    for name, features in feature_sets().items():
        print(f"Training {name} ({len(features)} features)...")
        rows.append(train_variant(name, features, data))
    summary = pd.DataFrame(rows).sort_values(["RMSE", "SMAPE"]).reset_index(drop=True)
    summary.to_csv(os.path.join(RESULT_DIR, "xgboost_ewma_feature_ablation_fast_summary.csv"), index=False)
    plot_summary(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
