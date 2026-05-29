"""
compare_imputation_methods.py - 缺失值填补方法遮挡验证对比

评价方法:
从 02_resampled.csv 中已有真实值的位置随机遮挡一部分，然后用不同方法填补，
只在这些被遮挡但有真实值的位置计算 MSE / MAE / RMSE / SMAPE。

对比方法:
Linear, Cubic Spline, PCHIP, Akima, Kalman, Seasonal Mean, KNN, Random Forest
"""
import json
import math
import os
import re
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "02_reconstruction", "output", "02_resampled.csv")
RESULT_DIR = os.path.join(BASE_DIR, "04_results", "imputation_methods")
os.makedirs(RESULT_DIR, exist_ok=True)

VALUE_COL = "Final_Filtered"
MASK_FRAC = 0.20
RANDOM_SEED = 42


def calc_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.zeros_like(denom, dtype="float64")
    valid = denom != 0
    diff[valid] = np.abs(y_true[valid] - y_pred[valid]) / denom[valid]
    return float(np.mean(diff) * 100)


def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "SMAPE": calc_smape(y_true, y_pred),
    }


def point_number(point):
    match = re.search(r"(\d+)", str(point))
    return int(match.group(1)) if match else 0


def seasonal_edge_fill(filled, original, dates):
    filled = pd.Series(filled, index=original.index, dtype="float64")
    original = pd.Series(original, index=original.index, dtype="float64")
    missing = filled.isna()
    if not missing.any():
        return filled

    valid = original.notna()
    if valid.sum() == 0:
        return filled.fillna(0.0)

    date_series = pd.to_datetime(pd.Series(dates, index=original.index))
    valid_doy = date_series[valid].dt.dayofyear.to_numpy()
    valid_values = original[valid].to_numpy(dtype="float64")
    global_mean = float(np.nanmean(valid_values))

    for idx in filled.index[missing]:
        doy = int(date_series.loc[idx].dayofyear)
        dist = np.abs(valid_doy - doy)
        circular_dist = np.minimum(dist, 366 - dist)
        nearby = valid_values[circular_dist <= 30]
        if len(nearby) == 0:
            nearest = np.argsort(circular_dist)[: min(6, len(valid_values))]
            nearby = valid_values[nearest]
        filled.loc[idx] = float(np.nanmean(nearby)) if len(nearby) else global_mean
    return filled


def interpolate_fill(series, dates, kind):
    valid = series.notna()
    if valid.sum() < 2:
        return seasonal_edge_fill(series.copy(), series, dates)

    x = np.arange(len(series), dtype="float64")
    x_valid = x[valid.to_numpy()]
    y_valid = series[valid].to_numpy(dtype="float64")
    filled = series.copy()
    inside = (~valid).to_numpy() & (x >= x_valid.min()) & (x <= x_valid.max())

    try:
        if kind == "linear":
            filled = series.interpolate(method="linear", limit_area="inside")
        elif kind == "cubic":
            if valid.sum() < 4:
                return interpolate_fill(series, dates, "linear")
            interp = CubicSpline(x_valid, y_valid, extrapolate=False)
            filled.iloc[np.where(inside)[0]] = interp(x[inside])
        elif kind == "pchip":
            interp = PchipInterpolator(x_valid, y_valid, extrapolate=False)
            filled.iloc[np.where(inside)[0]] = interp(x[inside])
        elif kind == "akima":
            if valid.sum() < 5:
                return interpolate_fill(series, dates, "pchip")
            interp = Akima1DInterpolator(x_valid, y_valid)
            filled.iloc[np.where(inside)[0]] = interp(x[inside])
        else:
            raise ValueError(kind)
    except Exception:
        return interpolate_fill(series, dates, "linear")
    return seasonal_edge_fill(filled, series, dates)


def seasonal_mean_fill(series, dates):
    filled = series.copy().astype("float64")
    missing = filled.isna()
    if not missing.any():
        return filled
    if series.notna().sum() == 0:
        return filled.fillna(0.0)

    date_series = pd.to_datetime(pd.Series(dates, index=series.index))
    valid = series.notna()
    valid_doy = date_series[valid].dt.dayofyear.to_numpy()
    valid_values = series[valid].to_numpy(dtype="float64")
    global_mean = float(np.nanmean(valid_values))

    for idx in filled.index[missing]:
        doy = int(date_series.loc[idx].dayofyear)
        dist = np.abs(valid_doy - doy)
        circular_dist = np.minimum(dist, 366 - dist)
        nearby = valid_values[circular_dist <= 30]
        if len(nearby) == 0:
            nearby = valid_values[np.argsort(circular_dist)[: min(8, len(valid_values))]]
        filled.loc[idx] = float(np.nanmean(nearby)) if len(nearby) else global_mean
    return filled


def kalman_fill(series, dates):
    valid = series.dropna()
    if len(valid) == 0:
        return series.fillna(0.0)

    state = np.array([[float(valid.iloc[0])], [0.0]])
    covariance = np.eye(2)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([[1.0, 0.0]])
    process_noise = np.array([[0.0005, 0.0], [0.0, 0.0005]])
    observation_noise = np.array([[0.02]])

    result = []
    for val in series:
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise
        if pd.isna(val):
            result.append(float(state[0, 0]))
        else:
            innovation = np.array([[float(val)]]) - observation @ state
            innovation_covariance = observation @ covariance @ observation.T + observation_noise
            gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
            state = state + gain @ innovation
            covariance = (np.eye(2) - gain @ observation) @ covariance
            result.append(float(state[0, 0]))
    return seasonal_edge_fill(pd.Series(result, index=series.index), series, dates)


def make_mask(df):
    rng = np.random.default_rng(RANDOM_SEED)
    mask = np.zeros(len(df), dtype=bool)
    for _, idx in df[df[VALUE_COL].notna()].groupby("point").groups.items():
        idx = np.array(list(idx))
        n_mask = max(1, int(len(idx) * MASK_FRAC))
        chosen = rng.choice(idx, size=n_mask, replace=False)
        mask[chosen] = True
    return mask


def evaluate_series_method(df, mask, method_name, fill_func):
    predictions = np.full(len(df), np.nan, dtype="float64")
    for _, group in df.groupby("point", sort=False):
        group = group.sort_values("date")
        series = group[VALUE_COL].astype("float64").reset_index(drop=True)
        dates = group["date"].reset_index(drop=True)
        local_mask = mask[group.index.to_numpy()]
        series_masked = series.copy()
        series_masked.iloc[np.where(local_mask)[0]] = np.nan
        filled = fill_func(series_masked, dates)
        predictions[group.index.to_numpy()] = filled.to_numpy()

    valid_eval = mask & np.isfinite(predictions)
    metrics = calc_metrics(df.loc[valid_eval, VALUE_COL].to_numpy(), predictions[valid_eval])
    metrics.update({
        "method": method_name,
        "eval_count": int(valid_eval.sum()),
    })
    return metrics


def knn_predict_group(group, local_mask):
    dates = pd.to_datetime(group["date"])
    t = np.arange(len(group), dtype="float64")
    doy = dates.dt.dayofyear.to_numpy(dtype="float64")
    features = np.column_stack([
        t / max(t.max(), 1.0),
        np.sin(2 * np.pi * doy / 366.0),
        np.cos(2 * np.pi * doy / 366.0),
    ])
    y = group[VALUE_COL].to_numpy(dtype="float64")
    train_mask = np.isfinite(y) & (~local_mask)
    pred_mask = local_mask
    preds = np.full(len(group), np.nan, dtype="float64")
    if train_mask.sum() < 3 or pred_mask.sum() == 0:
        return preds
    model = KNeighborsRegressor(n_neighbors=min(8, int(train_mask.sum())), weights="distance")
    model.fit(features[train_mask], y[train_mask])
    preds[pred_mask] = model.predict(features[pred_mask])
    return preds


def evaluate_knn(df, mask):
    predictions = np.full(len(df), np.nan, dtype="float64")
    for _, group in df.groupby("point", sort=False):
        group = group.sort_values("date")
        local_mask = mask[group.index.to_numpy()]
        predictions[group.index.to_numpy()] = knn_predict_group(group, local_mask)
    valid_eval = mask & np.isfinite(predictions)
    metrics = calc_metrics(df.loc[valid_eval, VALUE_COL].to_numpy(), predictions[valid_eval])
    metrics.update({"method": "KNN", "eval_count": int(valid_eval.sum())})
    return metrics


def build_global_features(df):
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear.to_numpy(dtype="float64")
    point_num = df["point"].map(point_number).to_numpy(dtype="float64")
    ordinal = dates.map(pd.Timestamp.toordinal).to_numpy(dtype="float64")
    ordinal = (ordinal - ordinal.min()) / max(ordinal.max() - ordinal.min(), 1)
    point_scaled = point_num / max(point_num.max(), 1)
    features = pd.DataFrame({
        "point": point_scaled,
        "time": ordinal,
        "doy_sin": np.sin(2 * np.pi * doy / 366.0),
        "doy_cos": np.cos(2 * np.pi * doy / 366.0),
        "month": dates.dt.month.to_numpy(dtype="float64") / 12.0,
    })
    if "EVI" in df.columns:
        features["EVI"] = df["EVI"].astype("float64").fillna(df["EVI"].astype("float64").median())
    return features


def evaluate_random_forest(df, mask):
    features = build_global_features(df)
    y = df[VALUE_COL].to_numpy(dtype="float64")
    train_mask = np.isfinite(y) & (~mask)
    pred_mask = mask
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=24,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(features.loc[train_mask], y[train_mask])
    predictions = np.full(len(df), np.nan, dtype="float64")
    predictions[pred_mask] = model.predict(features.loc[pred_mask])
    valid_eval = pred_mask & np.isfinite(predictions)
    metrics = calc_metrics(df.loc[valid_eval, VALUE_COL].to_numpy(), predictions[valid_eval])
    metrics.update({"method": "RandomForest", "eval_count": int(valid_eval.sum())})
    return metrics


def plot_results(summary):
    metrics = [("MSE", "MSE"), ("MAE", "MAE"), ("RMSE", "RMSE"), ("SMAPE", "SMAPE(%)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=240)
    colors = ["#5B8DB8", "#73A857", "#8E72B8", "#D18F32", "#C75D5D", "#7B7B7B", "#4C9A8A", "#B86F9D"]
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.bar(summary["method"], summary[metric], color=colors[:len(summary)], edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelrotation=25)
        for i, value in enumerate(summary[metric]):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("NDVI Missing-Value Imputation Methods: Masked Validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "imputation_methods_comparison.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    print("=" * 70)
    print("NDVI缺失值填补方法遮挡验证")
    print("=" * 70)
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["point", "date"]).reset_index(drop=True)
    print(f"输入: {len(df)} 条, {df['point'].nunique()} 点, 真实有效值: {df[VALUE_COL].notna().sum()}")

    mask = make_mask(df)
    print(f"遮挡验证点: {mask.sum()} ({MASK_FRAC:.0%} of observed values)")

    methods = [
        ("Linear", lambda s, d: interpolate_fill(s, d, "linear")),
        ("CubicSpline", lambda s, d: interpolate_fill(s, d, "cubic")),
        ("PCHIP", lambda s, d: interpolate_fill(s, d, "pchip")),
        ("Akima", lambda s, d: interpolate_fill(s, d, "akima")),
        ("Kalman", kalman_fill),
        ("SeasonalMean", seasonal_mean_fill),
    ]

    results = []
    for method_name, fill_func in methods:
        mt0 = time.time()
        print(f"\n{method_name}...")
        metrics = evaluate_series_method(df, mask, method_name, fill_func)
        metrics["time_s"] = round(time.time() - mt0, 1)
        results.append(metrics)
        print(
            f"  MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, SMAPE={metrics['SMAPE']:.2f}%, "
            f"n={metrics['eval_count']}, time={metrics['time_s']}s"
        )

    for method_name, eval_func in [("KNN", evaluate_knn), ("RandomForest", evaluate_random_forest)]:
        mt0 = time.time()
        print(f"\n{method_name}...")
        metrics = eval_func(df, mask)
        metrics["time_s"] = round(time.time() - mt0, 1)
        results.append(metrics)
        print(
            f"  MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, SMAPE={metrics['SMAPE']:.2f}%, "
            f"n={metrics['eval_count']}, time={metrics['time_s']}s"
        )

    summary = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    summary.to_csv(os.path.join(RESULT_DIR, "imputation_methods_summary.csv"), index=False)
    with open(os.path.join(RESULT_DIR, "imputation_methods_summary.json"), "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    plot_results(summary)

    print("\n" + "=" * 70)
    print("缺失值填补方法对比汇总 (按RMSE升序)")
    print("=" * 70)
    print(summary[["method", "MSE", "MAE", "RMSE", "SMAPE", "eval_count", "time_s"]].to_string(index=False))
    print(f"\n输出目录: {RESULT_DIR}")
    print(f"总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
