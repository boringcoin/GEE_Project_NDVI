"""
step2_resample.py - 半月重采样 (15天窗口聚合)
输入: 02_reconstruction/output/01_filtered.csv
输出: 02_reconstruction/output/02_resampled.csv
"""
import pandas as pd
import os, re, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "02_reconstruction", "output", "01_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "02_reconstruction", "output")
OUTPUT = os.path.join(OUTPUT_DIR, "02_resampled.csv")

def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 2: 半月重采样 (15天窗口聚合)")
    print("=" * 60)

    df = pd.read_csv(INPUT)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # 使用Final_Filtered列作为值列
    value_col = 'Final_Filtered'
    print(f"  输入: {len(df)} 条, {df['point'].nunique()} 点")

    resample_freq = '15D'
    start_date = pd.Timestamp("1985-01-01")
    end_date = df['date'].max()
    full_index = pd.date_range(start=start_date, end=end_date, freq=resample_freq)
    value_cols = [
        col for col in ['Median_NDVI', 'NDVI', 'EVI', 'LOF_Filtered', 'Final_Filtered']
        if col in df.columns
    ]

    resampled = []
    for point, group in df.groupby('point'):
        group = group.sort_values('date').copy()
        # 去重
        group = group[~group['date'].duplicated(keep='last')]
        # 显式计算15天窗口左边界，避免不同pandas版本对resample origin的处理差异。
        offsets = ((group['date'] - start_date).dt.days // 15) * 15
        group['date_bin'] = start_date + pd.to_timedelta(offsets, unit='D')
        rs = group.groupby('date_bin')[value_cols].median()
        rs = rs.reindex(full_index)  # 空窗口留NaN给插值
        rs['point'] = point
        rs['date'] = rs.index
        resampled.append(rs.reset_index(drop=True))

    result = pd.concat(resampled, ignore_index=True)
    # 排序
    result['point_num'] = result['point'].apply(
        lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 0
    )
    result = result.sort_values(['point_num', 'date']).drop(columns='point_num')

    result.to_csv(OUTPUT, index=False)
    valid = result[value_col].notna().sum()
    print(f"  输出: {len(result)} 条, {value_col}有效: {valid} ({valid/len(result):.1%})")
    print(f"  耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
