"""
step3_interpolation.py - 缺失值插值 (Linear + PCHIP + Kalman对照)
输入: 02_reconstruction/output/02_resampled.csv
输出: 02_reconstruction/output/03_linear.csv, 03_pchip.csv, 03_kalman.csv
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "02_reconstruction", "output", "02_resampled.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")


def seasonal_edge_fill(filled, original, dates):
    """用同一物候期的多年观测均值填补首尾缺口，避免长平台常值。"""
    filled = pd.Series(filled, index=original.index, dtype='float64')
    original = pd.Series(original, index=original.index, dtype='float64')
    missing = filled.isna()
    if not missing.any():
        return filled

    valid = original.notna()
    if valid.sum() == 0:
        return filled.fillna(0.0)

    date_series = pd.to_datetime(pd.Series(dates, index=original.index))
    valid_doy = date_series[valid].dt.dayofyear.to_numpy()
    valid_values = original[valid].to_numpy(dtype='float64')
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


def linear_fill(series, dates):
    """线性插值填充观测区间内部缺口，首尾用季节均值填补。"""
    valid = series.notna()
    if valid.sum() == 0:
        return series.fillna(0.0)
    if valid.sum() == 1:
        return seasonal_edge_fill(series.copy(), series, dates)
    filled = series.interpolate(method='linear', limit_area='inside')
    return seasonal_edge_fill(filled, series, dates)


def pchip_fill(series, dates):
    """PCHIP形状保持插值，减少三次样条过冲。"""
    valid = series.notna()
    if valid.sum() < 2:
        return linear_fill(series, dates)

    x = np.arange(len(series), dtype='float64')
    x_valid = x[valid.to_numpy()]
    y_valid = series[valid].to_numpy(dtype='float64')
    filled = series.copy()
    inside = (~valid).to_numpy() & (x >= x_valid.min()) & (x <= x_valid.max())
    try:
        interp = PchipInterpolator(x_valid, y_valid, extrapolate=False)
        filled.iloc[np.where(inside)[0]] = interp(x[inside])
    except Exception:
        return linear_fill(series, dates)
    return seasonal_edge_fill(filled, series, dates)


def kalman_fill(series, dates):
    """Kalman滤波填充NaN，保留为对照方法。"""
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


def repeated_stats(df, value_col):
    repeat_links = 0
    max_runs = []
    for _, grp in df.sort_values(['point', 'date']).groupby('point'):
        s = grp[value_col]
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
    return repeat_links / denominator, int(max(max_runs) if max_runs else 0)


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 3: 缺失值插值 (Linear + PCHIP + Kalman对照)")
    print("=" * 60)

    df = pd.read_csv(INPUT)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    value_col = 'Final_Filtered'
    print(f"  输入: {len(df)} 条, NaN: {df[value_col].isna().sum()}")

    methods = [
        ("Linear", "NDVI_Linear", "03_linear.csv", linear_fill),
        ("PCHIP", "NDVI_PCHIP", "03_pchip.csv", pchip_fill),
        ("Kalman", "NDVI_Kalman", "03_kalman.csv", kalman_fill),
    ]

    outputs = []
    for method, out_col, filename, fill_fn in methods:
        print(f"  {method}填充...")
        method_results = []
        for point, grp in df.groupby('point'):
            grp = grp.sort_values('date').copy()
            series = grp[value_col].astype('float64').reset_index(drop=True)
            dates = grp['date'].reset_index(drop=True)
            grp[out_col] = fill_fn(series, dates).values
            method_results.append(grp)
        df_method = pd.concat(method_results, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, filename)
        df_method[['point', 'date', out_col]].to_csv(out_path, index=False)
        repeat_ratio, max_run = repeated_stats(df_method, out_col)
        print(
            f"    {method}: {df_method[out_col].isna().sum()} NaN剩余, "
            f"重复链接={repeat_ratio:.2%}, 最长重复段={max_run}"
        )
        outputs.append(out_path)

    print(f"  输出: {', '.join(outputs)}")
    print(f"  耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
