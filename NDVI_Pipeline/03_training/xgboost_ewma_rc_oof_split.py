"""
EWMA-RC-XGBoost with the user's chronological split.

Final evaluation:
  train/validation history: 1985-01-01 ~ 2019-12-31
  test:                     2020-01-01 ~ 2025-05-16

The final base model is trained with all pre-test data. The residual module is
trained from out-of-fold residuals generated inside the pre-test period, so it
does not need to discard the user's validation/test split or leak test labels.
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
TARGET_FILE = os.path.join(BASE_DIR, "05_final_outputs", "reconstruction", "final_ndvi_kalman_whittaker.csv")
RESULT_DIR = os.path.join(BASE_DIR, "04_results", "xgboost_ewma_rc_oof_split")
os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_START = "1985-01-01"
PRETEST_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-05-16"
OOF_FOLDS = [
    ("2016", "2016-01-01", "2016-12-31"),
    ("2017", "2017-01-01", "2017-12-31"),
    ("2018", "2018-01-01", "2018-12-31"),
    ("2019", "2019-01-01", "2019-12-31"),
]

LAG3_FEATURES = ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3"]
EWMA_FEATURES = ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3"]
RESIDUAL_FEATURES = [
    "base_pred",
    "NDVI_prev1",
    "NDVI_prev2",
    "NDVI_prev3",
    "NDVI_ewma_3",
    "NDVI_diff_1",
    "NDVI_diff_2",
    "base_minus_prev1",
]
DECIMAL = 4


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.zeros_like(denom, dtype="float64")
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out) * 100)


def metric_row(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": round(float(mse), DECIMAL),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), DECIMAL),
        "RMSE": round(float(np.sqrt(mse)), DECIMAL),
        "SMAPE": round(smape(y_true, y_pred), DECIMAL),
    }


def load_data():
    df = pd.read_csv(TARGET_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[["point", "date", "NDVI_Target"]].sort_values(["point", "date"]).reset_index(drop=True)
    for lag in range(1, 4):
        df[f"NDVI_prev{lag}"] = df.groupby("point")["NDVI_Target"].shift(lag)
    df["NDVI_ewma_3"] = df.groupby("point")["NDVI_Target"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    df["NDVI_diff_1"] = df["NDVI_prev1"] - df["NDVI_prev2"]
    df["NDVI_diff_2"] = df["NDVI_prev1"] - df["NDVI_prev3"]
    return df


def make_base_model():
    return xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=5,
        gamma=0.001,
        subsample=0.75,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=0,
    )


def make_residual_model():
    return xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        gamma=0.001,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        tree_method="hist",
        device="cuda",
        random_state=43,
        verbosity=0,
    )


def add_residual_columns(df, pred):
    out = df.copy()
    out["base_pred"] = pred
    out["base_minus_prev1"] = out["base_pred"] - out["NDVI_prev1"]
    out["residual"] = out["NDVI_Target"] - out["base_pred"]
    return out


def train_predict_base(name, features, trainval, test):
    model = make_base_model()
    model.fit(trainval[features], trainval["NDVI_Target"])
    pred = np.round(model.predict(test[features]), DECIMAL)
    out = test[["point", "date", "NDVI_Target"]].copy()
    out.insert(0, "variant", name)
    out["Prediction"] = pred
    out.to_csv(os.path.join(RESULT_DIR, f"Predictions_{name}.csv"), index=False)
    row = {"variant": name, "module": "base_model", "feature_count": len(features), "features": features}
    row.update(metric_row(out["NDVI_Target"].to_numpy(), pred))
    return row, out, model


def build_oof_residuals(data):
    oof_parts = []
    fold_rows = []
    for fold, start, end in OOF_FOLDS:
        train = data[(data["date"] >= TRAIN_START) & (data["date"] < pd.Timestamp(start))].dropna(
            subset=EWMA_FEATURES + ["NDVI_Target"]
        )
        valid = data[(data["date"] >= start) & (data["date"] <= end)].dropna(
            subset=EWMA_FEATURES + ["NDVI_Target"]
        )
        model = make_base_model()
        model.fit(train[EWMA_FEATURES], train["NDVI_Target"])
        pred = np.round(model.predict(valid[EWMA_FEATURES]), DECIMAL)
        part = add_residual_columns(valid, pred)
        part["fold"] = fold
        oof_parts.append(part)
        row = {"fold": fold, "train_rows": len(train), "valid_rows": len(valid)}
        row.update(metric_row(valid["NDVI_Target"].to_numpy(), pred))
        fold_rows.append(row)

    oof = pd.concat(oof_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    oof.to_csv(os.path.join(RESULT_DIR, "OOF_residual_training_data.csv"), index=False)
    fold_df.to_csv(os.path.join(RESULT_DIR, "OOF_base_fold_metrics.csv"), index=False)
    return oof, fold_df


def train_rc_oof(data, trainval, test):
    oof, _ = build_oof_residuals(data)
    residual_train = oof.dropna(subset=RESIDUAL_FEATURES + ["residual"])

    final_base = make_base_model()
    final_base.fit(trainval[EWMA_FEATURES], trainval["NDVI_Target"])
    base_test_pred = np.round(final_base.predict(test[EWMA_FEATURES]), DECIMAL)
    test_resid = add_residual_columns(test, base_test_pred).dropna(subset=RESIDUAL_FEATURES + ["NDVI_Target"])

    residual_model = make_residual_model()
    residual_model.fit(residual_train[RESIDUAL_FEATURES], residual_train["residual"])
    residual_pred = residual_model.predict(test_resid[RESIDUAL_FEATURES])

    rows = []
    pred_dfs = []
    for shrinkage in [0.25, 0.5, 0.75, 1.0]:
        final_pred = np.round(np.clip(test_resid["base_pred"].to_numpy() + shrinkage * residual_pred, 0, 1), DECIMAL)
        name = f"EWMA_RC_XGBoost_OOF_s{shrinkage:g}"
        out = test_resid[["point", "date", "NDVI_Target"]].copy()
        out.insert(0, "variant", name)
        out["BasePrediction"] = test_resid["base_pred"].to_numpy()
        out["ResidualPrediction"] = residual_pred
        out["Prediction"] = final_pred
        out.to_csv(os.path.join(RESULT_DIR, f"Predictions_{name}.csv"), index=False)
        row = {
            "variant": name,
            "module": "oof_residual_correction",
            "feature_count": len(EWMA_FEATURES) + len(RESIDUAL_FEATURES),
            "features": EWMA_FEATURES + RESIDUAL_FEATURES,
            "residual_shrinkage": shrinkage,
        }
        row.update(metric_row(out["NDVI_Target"].to_numpy(), final_pred))
        rows.append(row)
        pred_dfs.append(out)
    return rows, pred_dfs


def md_table(df):
    lines = [
        "| " + " | ".join(df.columns.astype(str)) + " |",
        "| " + " | ".join([":---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_summary(summary):
    order = summary.sort_values("RMSE")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(order["variant"], order["RMSE"], color="#4c78a8")
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(order["variant"], order["SMAPE"], color="#59a14f")
    axes[1].set_title("SMAPE")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("EWMA-RC-XGBoost OOF残差校正实验")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "xgboost_ewma_rc_oof_split_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary, fold_df):
    base = summary[summary["variant"] == "XGBoost_Lag3"].iloc[0]
    ewma = summary[summary["variant"] == "EWMA_XGBoost"].iloc[0]
    best = summary.iloc[0]
    report = f"""# EWMA-RC-XGBoost OOF残差校正实验

## 1. 划分口径

本实验严格采用测试期之前的全部历史数据进行最终预测建模：

- 训练/验证历史期：1985-01-01 至 2019-12-31
- 测试期：2020-01-01 至 2025-05-16

其中，最终基础模型使用 1985-2019 年全部数据训练。残差校正模块不直接使用测试标签，而是在 1985-2019 内部通过滚动 out-of-fold 方式生成残差训练样本。

## 2. OOF基础模型验证结果

{md_table(fold_df)}

## 3. 测试集结果

{md_table(summary[['variant', 'module', 'feature_count', 'MSE', 'MAE', 'RMSE', 'SMAPE', 'residual_shrinkage']])}

## 4. 结果说明

`XGBoost_Lag3` 使用前三期NDVI作为输入，RMSE={base['RMSE']:.4f}，SMAPE={base['SMAPE']:.4f}。加入EWMA趋势记忆后，`EWMA_XGBoost` 的RMSE={ewma['RMSE']:.4f}，SMAPE={ewma['SMAPE']:.4f}。

最佳方案为 `{best['variant']}`，RMSE={best['RMSE']:.4f}，SMAPE={best['SMAPE']:.4f}。该结果用于判断残差校正模块在严格时间划分下是否真正提升最终预测效果。
"""
    path = os.path.join(RESULT_DIR, "xgboost_ewma_rc_oof_split_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def main():
    t0 = time.time()
    data = load_data()
    trainval = data[(data["date"] >= TRAIN_START) & (data["date"] <= PRETEST_END)].dropna(
        subset=EWMA_FEATURES + ["NDVI_Target"]
    )
    test = data[(data["date"] >= TEST_START) & (data["date"] <= TEST_END)].dropna(
        subset=EWMA_FEATURES + ["NDVI_Target"]
    )
    print(f"trainval_rows={len(trainval)}, test_rows={len(test)}")
    rows = []
    row, _, _ = train_predict_base("XGBoost_Lag3", LAG3_FEATURES, trainval.dropna(subset=LAG3_FEATURES), test.dropna(subset=LAG3_FEATURES))
    rows.append(row)
    row, _, _ = train_predict_base("EWMA_XGBoost", EWMA_FEATURES, trainval, test)
    rows.append(row)
    rc_rows, _ = train_rc_oof(data, trainval, test)
    rows.extend(rc_rows)
    summary = pd.DataFrame(rows).sort_values(["RMSE", "SMAPE"]).reset_index(drop=True)
    summary["time_total_s"] = round(time.time() - t0, 1)
    summary.to_csv(os.path.join(RESULT_DIR, "xgboost_ewma_rc_oof_split_summary.csv"), index=False)
    fold_df = pd.read_csv(os.path.join(RESULT_DIR, "OOF_base_fold_metrics.csv"))
    plot_summary(summary)
    report = write_report(summary, fold_df)
    print(summary.to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
