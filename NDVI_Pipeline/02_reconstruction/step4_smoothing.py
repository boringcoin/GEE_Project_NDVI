"""
step4_smoothing.py - 时间序列平滑 (SG + Whittaker)
输入: 优先使用 02_reconstruction/output/03_pchip.csv
输出: 02_reconstruction/output/04_sg.csv, 04_whittaker.csv (最终训练数据)
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import os, time, math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")
INPUT_CANDIDATES = [
    (os.path.join(OUTPUT_DIR, "03_pchip.csv"), "NDVI_PCHIP"),
    (os.path.join(OUTPUT_DIR, "03_linear.csv"), "NDVI_Linear"),
    (os.path.join(OUTPUT_DIR, "03_kalman.csv"), "NDVI_Kalman"),
]


def resolve_input():
    for path, value_col in INPUT_CANDIDATES:
        if os.path.exists(path):
            return path, value_col
    raise FileNotFoundError("未找到可用于平滑的插值文件: 03_pchip.csv/03_linear.csv/03_kalman.csv")

# ---- SG滤波 ----
def sg_smooth(y, window_length=31, polyorder=3):
    valid = ~np.isnan(y)
    if not np.all(valid):
        x = np.arange(len(y))
        interp_fn = interp1d(x[valid], y[valid], kind='linear', fill_value='extrapolate')
        y_interp = interp_fn(x)
    else:
        y_interp = y
    if window_length % 2 == 0:
        window_length += 1
    wl = min(window_length, len(y_interp) - 1 if len(y_interp) % 2 == 0 else len(y_interp))
    return savgol_filter(y_interp, wl, polyorder)

# ---- Whittaker平滑 ----
def whittaker_smooth(y, lambda_=10, d=2):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    valid = ~np.isnan(y)
    if not np.all(valid):
        x = np.arange(n)
        y = np.interp(x, x[valid], y[valid])
    d_eff = min(d, n // 3) if n >= 6 else 1
    try:
        W = diags(valid.astype(float), 0, shape=(n, n), dtype=np.float64)
        diagonals = [(-1)**k * math.comb(d_eff, k) for k in range(d_eff + 1)]
        offsets = np.arange(d_eff + 1)
        D = diags(diagonals, offsets, shape=(n - d_eff, n), dtype=np.float64)
        A = (W + lambda_ * D.T @ D).tocsc()
        b = W @ y
        z = spsolve(A, b)
        return np.clip(z, np.nanmin(y), np.nanmax(y))
    except Exception:
        return y

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
        'repeat_ratio': repeat_links / denominator,
        'max_repeat_run': int(max(max_runs) if max_runs else 0),
        'valid_min': int(np.min(valid_counts) if valid_counts else 0),
        'valid_median': float(np.median(valid_counts) if valid_counts else 0),
        'valid_max': int(np.max(valid_counts) if valid_counts else 0),
    }

def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 4: 时间序列平滑 (SG + Whittaker)")
    print("=" * 60)

    input_path, value_col = resolve_input()
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"  输入: {input_path}")
    print(f"  值列: {value_col}, {len(df)} 条, {df['point'].nunique()} 点")

    # SG平滑
    print("  SG平滑...")
    sg_results = []
    for point, grp in df.groupby('point'):
        grp = grp.sort_values('date').copy()
        if len(grp) < 15:
            grp['NDVI_SG'] = grp[value_col]
        else:
            grp['NDVI_SG'] = sg_smooth(grp[value_col].values)
        sg_results.append(grp)
    df_sg = pd.concat(sg_results, ignore_index=True)
    out_sg = os.path.join(OUTPUT_DIR, "04_sg.csv")
    df_sg[['point', 'date', 'NDVI_SG']].to_csv(out_sg, index=False)
    sg_diag = repeated_stats(df_sg, 'NDVI_SG')
    print(
        f"    SG重复链接={sg_diag['repeat_ratio']:.2%}, "
        f"最长重复段={sg_diag['max_repeat_run']}, "
        f"每点有效数中位={sg_diag['valid_median']:.0f}"
    )

    # Whittaker平滑 (固定lambda=100, 适配NDVI植被时序)
    print("  Whittaker平滑 (lambda=100)...")
    wh_results = []
    for point, grp in df.groupby('point'):
        grp = grp.sort_values('date').copy()
        y = grp[value_col].values
        grp['NDVI_Whittaker'] = whittaker_smooth(y, lambda_=100, d=2)
        wh_results.append(grp)
    df_wh = pd.concat(wh_results, ignore_index=True)
    out_wh = os.path.join(OUTPUT_DIR, "04_whittaker.csv")
    df_wh[['point', 'date', 'NDVI_Whittaker']].to_csv(out_wh, index=False)
    wh_diag = repeated_stats(df_wh, 'NDVI_Whittaker')
    print(
        f"    Whittaker重复链接={wh_diag['repeat_ratio']:.2%}, "
        f"最长重复段={wh_diag['max_repeat_run']}, "
        f"每点有效数中位={wh_diag['valid_median']:.0f}"
    )

    print(f"  SG输出: {out_sg}")
    print(f"  Whittaker输出: {out_wh}")
    print(f"  耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
