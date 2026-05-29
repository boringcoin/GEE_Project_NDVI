"""
step1_filter.py - NDVI异常值过滤 (LOF)
输入: 01_raw_data/NDVI_MultiVar_1000pts.csv
输出: 02_reconstruction/output/01_filtered.csv
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import os, sys, time

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "01_raw_data", "NDVI_MultiVar_1000pts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT = os.path.join(OUTPUT_DIR, "01_filtered.csv")

# 识别NDVI值列
NDVI_COL = None  # 自动检测

def detect_ndvi_column(df):
    for col in ['Median_NDVI', 'NDVI', 'NDVI_Whittaker']:
        if col in df.columns:
            return col
    # 取第一个数值列（排除point/date）
    for col in df.columns:
        if col not in ['point', 'date', '.geo'] and df[col].dtype in ['float64', 'int64']:
            return col
    raise ValueError("无法识别NDVI值列")

def filter_lof(point_data, col, n_neighbors=20, contamination=0.1):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    mask = point_data[col].notna()
    result = point_data[col].copy()
    if mask.sum() > n_neighbors:
        scores = lof.fit_predict(point_data.loc[mask, [col]])
        result[mask] = np.where(scores == -1, np.nan, point_data.loc[mask, col])
    return result

def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 1: NDVI异常值过滤 (LOF)")
    print("=" * 60)

    df = pd.read_csv(INPUT)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    global NDVI_COL
    NDVI_COL = detect_ndvi_column(df)
    print(f"  输入: {len(df)} 条, {df['point'].nunique()} 点")
    print(f"  NDVI列: {NDVI_COL}")

    # 过滤
    filtered = []
    outliers_count = 0
    for point, grp in df.groupby('point'):
        grp = grp.sort_values('date').copy()
        grp['LOF_Filtered'] = filter_lof(grp, NDVI_COL)
        # Final_Filtered is kept for downstream compatibility.
        grp['Final_Filtered'] = grp['LOF_Filtered']
        outliers = grp['Final_Filtered'].isna().sum() - grp[NDVI_COL].isna().sum()
        outliers_count += max(0, outliers)
        filtered.append(grp)

    result = pd.concat(filtered, ignore_index=True)
    result.to_csv(OUTPUT, index=False)

    total = len(result)
    removed = result['Final_Filtered'].isna().sum() - result[NDVI_COL].isna().sum()
    print(f"  异常值移除: {max(0, removed)} 条")
    print(f"  输出: {OUTPUT}")
    print(f"  耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
