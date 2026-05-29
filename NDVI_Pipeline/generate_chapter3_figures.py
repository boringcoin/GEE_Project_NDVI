from __future__ import annotations

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUT_DIR = BASE_DIR / "04_results" / "chapter3_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = BASE_DIR / "01_raw_data" / "NDVI_MultiVar_1000pts.csv"
FILTERED_PATH = BASE_DIR / "02_reconstruction" / "output" / "01_filtered.csv"
RESAMPLED_PATH = BASE_DIR / "02_reconstruction" / "output" / "02_resampled.csv"
KALMAN_PATH = BASE_DIR / "02_reconstruction" / "output" / "03_kalman.csv"
FINAL_WHITTAKER_PATH = (
    BASE_DIR / "04_results" / "xgboost_smoothing_ablation_kalman" / "Kalman_Whittaker.csv"
)
POINTS_PATH = ROOT_DIR / "NDVI_multi_var" / "FixedRandomPoints1000.csv"


def setup_style() -> None:
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


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
    remove = (labels == -1) & observed.to_numpy()
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
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(StandardScaler().fit_transform(features))
    out.loc[mask] = np.where(labels == -1, np.nan, out.loc[mask])
    return out


def select_example_point(raw: pd.DataFrame, filtered: pd.DataFrame) -> str:
    merged = raw[["point", "date", "NDVI"]].merge(
        filtered[["point", "date", "Final_Filtered"]],
        on=["point", "date"],
        how="left",
    )
    merged["_removed"] = merged["NDVI"].notna() & merged["Final_Filtered"].isna()
    summary = merged.groupby("point").agg(valid=("NDVI", "count"), removed=("_removed", "sum"))
    summary = summary[(summary["valid"] > 400) & (summary["removed"] > 30)]
    if summary.empty:
        return str(merged["point"].iloc[0])
    return str(summary.sort_values(["removed", "valid"], ascending=False).index[0])


def figure_study_area() -> Path:
    points = pd.read_csv(POINTS_PATH)
    points["point_num"] = points["point_id"].str.extract(r"(\d+)").astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(
        points["lon"],
        points["lat"],
        c=points["point_num"],
        s=16,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_title("特罗多斯山脉研究区1000个采样点分布", fontsize=14, pad=12)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect("equal", adjustable="box")
    pad_x = (points["lon"].max() - points["lon"].min()) * 0.08
    pad_y = (points["lat"].max() - points["lat"].min()) * 0.08
    ax.set_xlim(points["lon"].min() - pad_x, points["lon"].max() + pad_x)
    ax.set_ylim(points["lat"].min() - pad_y, points["lat"].max() + pad_y)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("样点编号")
    ax.text(
        0.02,
        0.02,
        "数据来源：FixedRandomPoints1000.csv",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    return savefig(fig, "fig3_1_study_area_sampling_points.png")


def figure_raw_missing_anomaly(raw: pd.DataFrame, filtered: pd.DataFrame, point: str) -> Path:
    r = raw[raw["point"] == point].copy().sort_values("date")
    f = filtered[filtered["point"] == point][["date", "Final_Filtered"]].copy()
    r = r.merge(f, on="date", how="left")
    r["_removed"] = r["NDVI"].notna() & r["Final_Filtered"].isna()
    r["_missing"] = r["NDVI"].isna()

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 0.8], "hspace": 0.06},
    )
    obs = r[r["NDVI"].notna()]
    removed = r[r["_removed"]]
    ax.plot(obs["date"], obs["NDVI"], color="#8aa5c1", linewidth=0.7, alpha=0.55, label="原始有效观测")
    ax.scatter(obs["date"], obs["NDVI"], s=7, color="#3b6ea8", alpha=0.55)
    ax.scatter(
        removed["date"],
        removed["NDVI"],
        s=30,
        marker="x",
        color="#d1495b",
        linewidths=1.2,
        label="LOF识别异常值",
        zorder=5,
    )
    ax.set_title(f"原始NDVI时间序列缺失与异常观测示例（{point}）", fontsize=14, pad=10)
    ax.set_ylabel("NDVI")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    ax2.scatter(r.loc[r["_missing"], "date"], np.zeros(r["_missing"].sum()), s=6, color="#999999", alpha=0.35)
    ax2.set_yticks([])
    ax2.set_ylabel("缺失")
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel("年份")
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return savefig(fig, "fig3_2_raw_missing_anomaly_example.png")


def figure_outlier_methods(raw: pd.DataFrame, point: str) -> Path:
    grp = raw[raw["point"] == point].copy().sort_values("date").reset_index(drop=True)
    methods = {
        "3-Sigma": three_sigma(grp, "NDVI"),
        "IQR": iqr_filter(grp, "NDVI"),
        "DBSCAN": dbscan_filter(grp, "NDVI"),
        "LOF": lof_filter(grp, "NDVI"),
        "LOF(预处理缺失点)": lof_preprocessed_missing_filter(grp, "NDVI"),
        "分位数法": quantile_filter(grp, "NDVI"),
    }

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    valid = grp["NDVI"].notna()
    for ax, (name, filtered) in zip(axes, methods.items()):
        removed = valid & filtered.isna()
        kept = valid & filtered.notna()
        ax.plot(grp.loc[kept, "date"], grp.loc[kept, "NDVI"], color="#4c78a8", linewidth=0.7, alpha=0.55)
        ax.scatter(grp.loc[kept, "date"], grp.loc[kept, "NDVI"], s=6, color="#4c78a8", alpha=0.45)
        ax.scatter(
            grp.loc[removed, "date"],
            grp.loc[removed, "NDVI"],
            s=24,
            marker="x",
            color="#d1495b",
            linewidths=1.0,
        )
        ax.set_title(f"{name}（剔除{int(removed.sum())}个）", fontsize=11)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[::2]:
        ax.set_ylabel("NDVI")
    for ax in axes[-2:]:
        ax.set_xlabel("年份")
    fig.suptitle(f"不同异常值检测方法处理效果对比（{point}）", fontsize=15, y=1.02)
    return savefig(fig, "fig3_3_outlier_methods_comparison.png")


def figure_reconstruction_process(point: str) -> Path:
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    filtered = pd.read_csv(FILTERED_PATH, parse_dates=["date"])
    resampled = pd.read_csv(RESAMPLED_PATH, parse_dates=["date"])
    kalman = pd.read_csv(KALMAN_PATH, parse_dates=["date"])
    whittaker = pd.read_csv(FINAL_WHITTAKER_PATH, parse_dates=["date"])

    raw_p = raw[raw["point"] == point].sort_values("date")
    filt_p = filtered[filtered["point"] == point].sort_values("date")
    res_p = resampled[resampled["point"] == point].sort_values("date")
    kal_p = kalman[kalman["point"] == point].sort_values("date")
    wh_p = whittaker[whittaker["point"] == point].sort_values("date")

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].scatter(raw_p["date"], raw_p["NDVI"], s=7, color="#4c78a8", alpha=0.45, label="原始观测")
    removed = filt_p["NDVI"].notna() & filt_p["Final_Filtered"].isna()
    axes[0].scatter(
        filt_p.loc[removed, "date"],
        filt_p.loc[removed, "NDVI"],
        s=26,
        marker="x",
        color="#d1495b",
        label="LOF剔除",
    )
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("A 原始观测与异常值剔除", loc="left", fontsize=11)

    axes[1].plot(res_p["date"], res_p["Final_Filtered"], color="#59a14f", linewidth=1.0)
    axes[1].scatter(res_p["date"], res_p["Final_Filtered"], s=8, color="#59a14f", alpha=0.7)
    axes[1].set_title("B 15天窗口重采样", loc="left", fontsize=11)

    axes[2].plot(kal_p["date"], kal_p["NDVI_Kalman"], color="#f28e2b", linewidth=1.1)
    axes[2].set_title("C Kalman缺失值填补", loc="left", fontsize=11)

    axes[3].plot(wh_p["date"], wh_p["NDVI_Target"], color="#6f4e9b", linewidth=1.25)
    axes[3].set_title("D Whittaker平滑拟合后的最终序列", loc="left", fontsize=11)
    axes[3].set_xlabel("年份")
    axes[3].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        ax.set_ylabel("NDVI")
        ax.set_ylim(0, 1)
    fig.suptitle(f"NDVI时间序列重构过程示例（{point}）", fontsize=15, y=1.01)
    return savefig(fig, "fig3_4_reconstruction_process_example.png")


def main() -> None:
    setup_style()
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    filtered = pd.read_csv(FILTERED_PATH, parse_dates=["date"])
    point = select_example_point(raw, filtered)

    outputs = [
        figure_study_area(),
        figure_raw_missing_anomaly(raw, filtered, point),
        figure_outlier_methods(raw, point),
        figure_reconstruction_process(point),
    ]
    print(f"Example point: {point}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
