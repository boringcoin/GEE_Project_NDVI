"""
Compare NDVI outlier detection methods used in the thesis discussion.

The raw NDVI series does not include manual outlier labels, so this script
reports diagnostic evidence rather than a supervised accuracy score:
  - how many observed values each method removes
  - whether removed values deviate more from the point-wise seasonal baseline
  - whether filtering reduces short-term temporal jumps

Outputs:
  04_results/outlier_methods/outlier_methods_summary.csv
  04_results/outlier_methods/outlier_methods_report.md
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "01_raw_data", "NDVI_MultiVar_1000pts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "04_results", "outlier_methods")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass(frozen=True)
class MethodResult:
    method: str
    filtered: pd.Series


def detect_ndvi_column(df: pd.DataFrame) -> str:
    for col in ["Median_NDVI", "NDVI", "NDVI_Whittaker"]:
        if col in df.columns:
            return col
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric:
        if col not in {"point"}:
            return col
    raise ValueError("Cannot detect NDVI column.")


def robust_z(values: pd.Series) -> pd.Series:
    med = values.median()
    mad = (values - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        std = values.std(ddof=0)
        return (values - values.mean()) / std if std and np.isfinite(std) else values * 0
    return 0.6745 * (values - med) / mad


def three_sigma(grp: pd.DataFrame, col: str) -> pd.Series:
    out = grp[col].copy()
    mask = out.notna()
    if mask.sum() < 3:
        return out
    mu = out[mask].mean()
    sigma = out[mask].std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return out
    out.loc[mask & ((out - mu).abs() > 3 * sigma)] = np.nan
    return out


def iqr_filter(grp: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    out = grp[col].copy()
    mask = out.notna()
    if mask.sum() < 4:
        return out
    q1 = out[mask].quantile(0.25)
    q3 = out[mask].quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return out
    out.loc[mask & ((out < q1 - k * iqr) | (out > q3 + k * iqr))] = np.nan
    return out


def quantile_filter(grp: pd.DataFrame, col: str, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    out = grp[col].copy()
    mask = out.notna()
    if mask.sum() == 0:
        return out
    lb = out[mask].quantile(lower)
    ub = out[mask].quantile(upper)
    out.loc[mask & ((out < lb) | (out > ub))] = np.nan
    return out


def lof_filter(grp: pd.DataFrame, col: str, n_neighbors: int = 20, contamination: float = 0.1) -> pd.Series:
    out = grp[col].copy()
    mask = out.notna()
    if mask.sum() <= n_neighbors:
        return out
    x = out.loc[mask].to_numpy(dtype=float).reshape(-1, 1)
    labels = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(x)
    out.loc[mask] = np.where(labels == -1, np.nan, out.loc[mask])
    return out


def lof_preprocessed_missing_filter(
    grp: pd.DataFrame,
    col: str,
    n_neighbors: int = 20,
    contamination: float = 0.1,
) -> pd.Series:
    """LOF after missing-value preprocessing and seasonal feature construction.

    Missing NDVI values are interpolated only to build local-neighborhood
    features. The method still only removes originally observed NDVI values.
    """
    out = grp[col].copy()
    observed = out.notna()
    if observed.sum() <= n_neighbors:
        return out

    tmp = grp[["date", col]].copy().sort_values("date")
    tmp[col] = tmp[col].interpolate(method="linear", limit_direction="both")
    tmp[col] = tmp[col].fillna(tmp[col].median())
    doy = tmp["date"].dt.dayofyear.to_numpy(dtype=float)
    features = np.column_stack(
        [
            robust_z(tmp[col]).to_numpy(dtype=float),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]
    )
    labels = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(features)
    observed_arr = observed.to_numpy()
    remove = (labels == -1) & observed_arr
    out.iloc[np.where(remove)[0]] = np.nan
    return out


def dbscan_filter(grp: pd.DataFrame, col: str, eps: float = 0.35, min_samples: int = 10) -> pd.Series:
    out = grp[col].copy()
    mask = out.notna()
    if mask.sum() < min_samples:
        return out
    valid = grp.loc[mask, ["date", col]].copy()
    doy = valid["date"].dt.dayofyear.to_numpy(dtype=float)
    features = np.column_stack(
        [
            valid[col].to_numpy(dtype=float),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]
    )
    features = StandardScaler().fit_transform(features)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    out.loc[mask] = np.where(labels == -1, np.nan, out.loc[mask])
    return out


def lof_then_quantile(grp: pd.DataFrame, col: str) -> pd.Series:
    first = lof_filter(grp, col)
    tmp = grp.copy()
    tmp["_lof"] = first
    return quantile_filter(tmp, "_lof")


def seasonal_baseline(df: pd.DataFrame, col: str) -> pd.Series:
    month_day = df["date"].dt.strftime("%m-%d")
    by_point_day = df.assign(_md=month_day).groupby(["point", "_md"])[col].transform("median")
    by_point_month = df.groupby(["point", df["date"].dt.month])[col].transform("median")
    by_point = df.groupby("point")[col].transform("median")
    return by_point_day.fillna(by_point_month).fillna(by_point)


def median_jump(values: pd.Series) -> float:
    valid = values.dropna()
    if len(valid) < 2:
        return np.nan
    return float(valid.diff().abs().dropna().median())


def summarize_method(df: pd.DataFrame, col: str, method: str, filtered: pd.Series, baseline: pd.Series) -> dict:
    original_valid = df[col].notna()
    kept = original_valid & filtered.notna()
    removed = original_valid & filtered.isna()

    residual = (df[col] - baseline).abs()
    removed_resid = residual[removed].median()
    kept_resid = residual[kept].median()
    targeting_ratio = removed_resid / kept_resid if kept_resid and np.isfinite(kept_resid) else np.nan

    before_jump = df.groupby("point")[col].apply(median_jump).median()
    tmp = df[["point"]].copy()
    tmp["_filtered"] = filtered
    after_jump = tmp.groupby("point")["_filtered"].apply(median_jump).median()

    return {
        "method": method,
        "original_valid": int(original_valid.sum()),
        "kept_valid": int(kept.sum()),
        "removed_count": int(removed.sum()),
        "removed_rate": round(float(removed.sum() / original_valid.sum()), 6),
        "valid_after_rate": round(float(kept.sum() / len(df)), 6),
        "removed_residual_median": round(float(removed_resid), 6) if np.isfinite(removed_resid) else np.nan,
        "kept_residual_median": round(float(kept_resid), 6) if np.isfinite(kept_resid) else np.nan,
        "targeting_ratio": round(float(targeting_ratio), 4) if np.isfinite(targeting_ratio) else np.nan,
        "median_jump_before": round(float(before_jump), 6) if np.isfinite(before_jump) else np.nan,
        "median_jump_after": round(float(after_jump), 6) if np.isfinite(after_jump) else np.nan,
        "jump_reduction_rate": round(float((before_jump - after_jump) / before_jump), 6)
        if before_jump and np.isfinite(before_jump) and np.isfinite(after_jump)
        else np.nan,
    }


def rank_methods(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    # Reward targeted removal and temporal smoothing, penalize over-removal.
    ranked["_score"] = (
        ranked["targeting_ratio"].rank(ascending=False, pct=True)
        + ranked["jump_reduction_rate"].rank(ascending=False, pct=True)
        + (1 - ranked["removed_rate"]).rank(ascending=False, pct=True) * 0.5
    )
    ranked["rank"] = ranked["_score"].rank(ascending=False, method="min").astype(int)
    return ranked.sort_values(["rank", "method"]).drop(columns=["_score"])


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Compare NDVI outlier detection methods")
    print("=" * 70)

    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["point", "date"]).reset_index(drop=True)
    col = detect_ndvi_column(df)
    print(f"Input: {INPUT}")
    print(f"Rows: {len(df):,}, points: {df['point'].nunique():,}, NDVI column: {col}")

    baseline = seasonal_baseline(df, col)
    methods = {
        "3Sigma": three_sigma,
        "IQR": iqr_filter,
        "DBSCAN": dbscan_filter,
        "LOF": lof_filter,
        "LOF_PreprocessedMissing": lof_preprocessed_missing_filter,
        "Quantile_5_95": quantile_filter,
        "LOF_Then_Quantile": lof_then_quantile,
    }

    summaries = []
    filtered_outputs = {}
    for name, func in methods.items():
        print(f"Running {name} ...")
        parts = []
        for _, grp in df.groupby("point", sort=False):
            parts.append(func(grp, col))
        filtered = pd.concat(parts).sort_index()
        filtered_outputs[name] = filtered
        summaries.append(summarize_method(df, col, name, filtered, baseline))

    summary = rank_methods(pd.DataFrame(summaries))
    summary_path = os.path.join(OUTPUT_DIR, "outlier_methods_summary.csv")
    summary.to_csv(summary_path, index=False)

    dropna_summary = summary.copy()
    dropna_summary["valid_after_rate"] = (
        dropna_summary["kept_valid"] / dropna_summary["original_valid"]
    ).round(6)
    dropna_summary = dropna_summary.rename(
        columns={
            "original_valid": "drop_missing_input_count",
            "valid_after_rate": "kept_rate_after_drop_missing",
        }
    )
    dropna_summary_path = os.path.join(OUTPUT_DIR, "outlier_methods_drop_missing_summary.csv")
    dropna_summary.to_csv(dropna_summary_path, index=False)

    report_path = os.path.join(OUTPUT_DIR, "outlier_methods_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NDVI 异常值检测方法对比结果\n\n")
        f.write("## 实验说明\n\n")
        f.write(
            "本实验对论文 2.3.3 中提到的 3-Sigma、IQR、DBSCAN、LOF、"
            "LOF(预处理缺失点)、分位数法，以及当前流水线使用的 LOF+分位数组合方法进行比较。"
            "由于原始数据没有人工异常标签，本实验不使用监督分类准确率，而使用以下诊断指标：\n\n"
        )
        f.write("- removed_rate：在原始有效观测中被判定为异常的比例。\n")
        f.write("- targeting_ratio：被剔除点相对季节基准的中位偏离 / 保留点相对季节基准的中位偏离，越大说明越集中剔除异常波动。\n")
        f.write("- jump_reduction_rate：过滤后相邻有效观测跳变幅度降低比例，越大说明时序更平滑。\n\n")
        f.write("## 总体对比表\n\n")
        f.write(dataframe_to_markdown(summary))
        f.write("\n\n")
        f.write("## 先去除缺失值后的对比表\n\n")
        f.write(
            "下表把评价分母限定为原始非缺失 NDVI 观测值。由于各方法本身已经只在非缺失值上计算，"
            "异常值数量与 removed_rate 不会改变；变化的是 kept_rate_after_drop_missing，"
            "它表示在去除缺失值后仍被保留下来的真实观测比例。\n\n"
        )
        keep_cols = [
            "method",
            "drop_missing_input_count",
            "kept_valid",
            "removed_count",
            "removed_rate",
            "kept_rate_after_drop_missing",
            "targeting_ratio",
            "jump_reduction_rate",
            "rank",
        ]
        f.write(dataframe_to_markdown(dropna_summary[keep_cols]))
        f.write("\n\n")
        best = summary.sort_values("rank").iloc[0]
        f.write("## 结论\n\n")
        f.write(
            f"综合剔除比例、异常 targeting_ratio 和时序跳变降低效果，排名第一的方法为 "
            f"**{best['method']}**。"
        )
        f.write(
            "当前正式流水线使用的是 **LOF_Then_Quantile**，该方法结合了局部密度异常识别和全局极端值裁剪，"
            "适合作为 NDVI 预处理中的稳健异常值过滤方案。\n"
        )

    print(f"Saved: {summary_path}")
    print(f"Saved: {dropna_summary_path}")
    print(f"Saved: {report_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
