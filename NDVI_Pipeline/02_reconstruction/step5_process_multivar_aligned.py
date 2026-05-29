"""
step5_process_multivar_aligned.py - 多变量对齐处理

输入:
  01_raw_data/NDVI_MultiVar_1000pts.csv
  04_results/xgboost_fill_ablation_whittaker/Kalman_Whittaker.csv

输出:
  02_reconstruction/output/05_multivar_aligned_processed.csv

目标:
  将 EVI、temperature_C、soil_moisture 按与 NDVI 相同的 15 天时间轴处理，
  经异常筛选、重采样、Kalman 填补和 Whittaker 平滑后，与最终 NDVI 目标序列对齐。
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import LocalOutlierFactor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(BASE_DIR, "01_raw_data", "NDVI_MultiVar_1000pts.csv")
TARGET_FILE = os.path.join(BASE_DIR, "04_results", "xgboost_fill_ablation_whittaker", "Kalman_Whittaker.csv")
FALLBACK_TARGET_FILE = os.path.join(BASE_DIR, "02_reconstruction", "output", "04_whittaker.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")
OUTPUT = os.path.join(OUTPUT_DIR, "05_multivar_aligned_processed.csv")

START_DATE = pd.Timestamp("1985-01-01")
FREQ = "15D"
VALUE_COLS = ["EVI", "temperature_C", "soil_moisture"]


def seasonal_lof_filter(group: pd.DataFrame, col: str, n_neighbors: int = 20, contamination: float = 0.05) -> pd.Series:
    """用变量值 + 年内周期特征做 LOF，避免把正常季节高低值当成异常。"""
    result = group[col].copy()
    mask = result.notna()
    if mask.sum() <= n_neighbors:
        return result

    dates = pd.to_datetime(group.loc[mask, "date"])
    doy = dates.dt.dayofyear.to_numpy(dtype="float64")
    values = result.loc[mask].to_numpy(dtype="float64")
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return result
    z = (values - np.nanmean(values)) / std
    features = np.column_stack([
        z,
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ])
    labels = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    ).fit_predict(features)
    result.loc[mask] = np.where(labels == -1, np.nan, result.loc[mask])
    return result


def seasonal_edge_fill(filled: pd.Series, original: pd.Series, dates: pd.Series) -> pd.Series:
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


def kalman_fill(series: pd.Series, dates: pd.Series) -> pd.Series:
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


def whittaker_smooth(y: np.ndarray, lambda_: float = 100, d: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    valid = ~np.isnan(y)
    if valid.sum() == 0:
        return np.zeros(n, dtype=np.float64)
    if not np.all(valid):
        x = np.arange(n)
        y = np.interp(x, x[valid], y[valid])
    d_eff = min(d, n // 3) if n >= 6 else 1
    try:
        W = diags(valid.astype(float), 0, shape=(n, n), dtype=np.float64)
        diagonals = [(-1) ** k * math.comb(d_eff, k) for k in range(d_eff + 1)]
        offsets = np.arange(d_eff + 1)
        D = diags(diagonals, offsets, shape=(n - d_eff, n), dtype=np.float64)
        A = (W + lambda_ * D.T @ D).tocsc()
        b = W @ y
        z = spsolve(A, b)
        return np.clip(z, np.nanmin(y), np.nanmax(y))
    except Exception:
        return y


def load_target() -> pd.DataFrame:
    if os.path.exists(TARGET_FILE):
        target = pd.read_csv(TARGET_FILE)
        value_col = "NDVI_Target"
    else:
        target = pd.read_csv(FALLBACK_TARGET_FILE)
        value_col = "NDVI_Whittaker"
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    return target[["point", "date", value_col]].rename(columns={value_col: "NDVI_Target"})


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Step 5: 多变量对齐处理 (LOF + 15D + Kalman + Whittaker)")
    print("=" * 70)

    raw = pd.read_csv(RAW_FILE)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values(["point", "date"]).reset_index(drop=True)
    target = load_target()
    full_index = pd.date_range(START_DATE, raw["date"].max(), freq=FREQ)
    print(f"  原始数据: {len(raw)} 条, {raw['point'].nunique()} 点")
    print(f"  目标NDVI: {len(target)} 条, {target['point'].nunique()} 点")

    processed_parts = []
    diag_rows = []
    for idx, (point, group) in enumerate(raw.groupby("point", sort=False), start=1):
        group = group.sort_values("date").copy()
        for col in VALUE_COLS:
            group[f"{col}_Filtered"] = seasonal_lof_filter(group, col)

        offsets = ((group["date"] - START_DATE).dt.days // 15) * 15
        group["date_bin"] = START_DATE + pd.to_timedelta(offsets, unit="D")
        filtered_cols = [f"{col}_Filtered" for col in VALUE_COLS]
        rs = group.groupby("date_bin")[filtered_cols].median().reindex(full_index)
        rs["point"] = point
        rs["date"] = rs.index

        for col in VALUE_COLS:
            filtered_col = f"{col}_Filtered"
            kalman_col = f"{col}_Kalman"
            wh_col = f"{col}_Whittaker"
            series = rs[filtered_col].astype("float64").reset_index(drop=True)
            dates = pd.Series(rs["date"].values).reset_index(drop=True)
            filled = kalman_fill(series, dates)
            rs[kalman_col] = filled.values
            rs[wh_col] = whittaker_smooth(filled.values, lambda_=100, d=2)
            diag_rows.append({
                "point": point,
                "variable": col,
                "valid_after_filter_resample": int(rs[filtered_col].notna().sum()),
                "nan_after_kalman": int(pd.isna(rs[kalman_col]).sum()),
                "nan_after_whittaker": int(pd.isna(rs[wh_col]).sum()),
            })

        processed_parts.append(rs.reset_index(drop=True))
        if idx % 100 == 0:
            print(f"  进度: {idx}/{raw['point'].nunique()}")

    exog = pd.concat(processed_parts, ignore_index=True)
    keep_cols = ["point", "date"]
    for col in VALUE_COLS:
        keep_cols.extend([f"{col}_Filtered", f"{col}_Kalman", f"{col}_Whittaker"])
    exog = exog[keep_cols]

    aligned = pd.merge(target, exog, on=["point", "date"], how="left")
    for col in VALUE_COLS:
        wh_col = f"{col}_Whittaker"
        if aligned[wh_col].isna().any():
            aligned[wh_col] = aligned.groupby("point")[wh_col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
            aligned[wh_col] = aligned[wh_col].fillna(aligned[wh_col].median())

    aligned.to_csv(OUTPUT, index=False)
    diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(OUTPUT_DIR, "05_multivar_aligned_diagnostics.csv")
    diag.to_csv(diag_path, index=False)

    print(f"  输出: {OUTPUT}")
    print(f"  诊断: {diag_path}")
    print(f"  对齐数据: {len(aligned)} 条, {aligned['point'].nunique()} 点")
    print(
        "  Whittaker外源变量缺失: "
        + ", ".join(f"{col}={aligned[f'{col}_Whittaker'].isna().sum()}" for col in VALUE_COLS)
    )
    print(f"  耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
