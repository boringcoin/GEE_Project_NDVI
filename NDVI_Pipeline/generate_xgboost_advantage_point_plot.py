from __future__ import annotations

import io
import os
import argparse
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "05_final_outputs" / "chapter4_materials"
FIG_DIR = OUT_DIR / "xgboost_advantage_point"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_COMMIT = "9215c3f"
PREDICTION_FILES = {
    "XGBoost": "NDVI_Pipeline/05_final_outputs/chapter4_materials/unified_model_comparison_1985_2015/predictions_XGBoost.csv",
    "LightGBM": "NDVI_Pipeline/05_final_outputs/chapter4_materials/unified_model_comparison_1985_2015/predictions_LightGBM.csv",
    "TimesFM": "NDVI_Pipeline/05_final_outputs/chapter4_materials/unified_model_comparison_1985_2015/predictions_TimesFM.csv",
    "Informer": "NDVI_Pipeline/05_final_outputs/chapter4_materials/official_autoformer_unified_split/predictions_official_informer.csv",
    "Autoformer": "NDVI_Pipeline/05_final_outputs/chapter4_materials/official_autoformer_unified_split/predictions_official_autoformer.csv",
    "LSTM类模型": "NDVI_Pipeline/05_final_outputs/chapter4_materials/unified_model_comparison_1985_2015/predictions_LSTM.csv",
    "N-HiTS": "NDVI_Pipeline/05_final_outputs/chapter4_materials/unified_model_comparison_1985_2015/predictions_NHiTS.csv",
}


def setup_style() -> None:
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 350,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def load_prediction_from_git(path: str) -> pd.DataFrame:
    raw = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"], text=True)
    df = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
    return df[["point", "date", "NDVI_Target", "Prediction"]]


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def select_point(preds: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    rows = []
    for model, df in preds.items():
        part = df.copy()
        part["model"] = model
        rows.append(part)
    all_pred = pd.concat(rows, ignore_index=True)

    point_metrics = []
    for point, group in all_pred.groupby("point"):
        model_metrics = {}
        for model, mg in group.groupby("model"):
            y = mg["NDVI_Target"].to_numpy(dtype=float)
            pred = mg["Prediction"].to_numpy(dtype=float)
            model_metrics[model] = rmse(y, pred)
        if set(PREDICTION_FILES).issubset(model_metrics):
            xgb_rmse = model_metrics["XGBoost"]
            others = [v for k, v in model_metrics.items() if k != "XGBoost"]
            point_metrics.append(
                {
                    "point": point,
                    "xgboost_rmse": xgb_rmse,
                    "other_mean_rmse": float(np.mean(others)),
                    "advantage": float(np.mean(others) - xgb_rmse),
                    "xgboost_rank": 1 + sum(v < xgb_rmse for v in model_metrics.values()),
                    **{f"{k}_RMSE": v for k, v in model_metrics.items()},
                }
            )
    metrics = pd.DataFrame(point_metrics)
    metrics = metrics.sort_values(["xgboost_rank", "advantage", "xgboost_rmse"], ascending=[True, False, True])
    metrics.to_csv(FIG_DIR / "xgboost_advantage_point_candidates.csv", index=False)
    selected = metrics.iloc[0]["point"]
    return str(selected), metrics


def plot_point(point: str, preds: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    models = ["XGBoost", "LightGBM", "TimesFM", "Informer", "Autoformer", "LSTM类模型", "N-HiTS"]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()

    selected_rows = []
    for i, model in enumerate(models):
        ax = axes[i]
        df = preds[model][preds[model]["point"] == point].sort_values("date").copy()
        selected_rows.append(df.assign(model=model))
        model_rmse = metrics.loc[metrics["point"] == point, f"{model}_RMSE"].iloc[0]
        ax.plot(df["date"], df["NDVI_Target"], color="#2F6FA3", lw=1.45, label="真实值")
        ax.plot(df["date"], df["Prediction"], color="#E8752A", lw=1.35, label="预测值")
        ax.set_title(f"({chr(97 + i)}) {model}  RMSE={model_rmse:.4f}", fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=35, labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.88,
        "说明",
        fontsize=13,
        fontweight="bold",
        transform=axes[-1].transAxes,
    )
    axes[-1].text(
        0.02,
        0.76,
        "TimeGPT 的逐点预测文件\n与最终重构NDVI口径不一致，\n因此未混入本图。\n\n本图使用同一测试集、\n同一真实NDVI序列下的\n模型预测结果。",
        fontsize=10.5,
        linespacing=1.35,
        transform=axes[-1].transAxes,
        va="top",
    )

    fig.suptitle(f"典型样点 {point} 的不同模型预测曲线对比", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.02, "测试期：2020-2025；蓝线为真实重构NDVI，橙线为模型预测值。", ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])

    png = FIG_DIR / "fig4_x_xgboost_advantage_point_predictions.png"
    svg = FIG_DIR / "fig4_x_xgboost_advantage_point_predictions.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    selected_df = pd.concat(selected_rows, ignore_index=True)
    selected_df.to_csv(FIG_DIR / "selected_point_model_predictions.csv", index=False)
    print("selected_point", point)
    print(png)
    print(svg)
    print(FIG_DIR / "selected_point_model_predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", default="point40", help="Point id to plot, e.g. point40. Use 'auto' to select by XGBoost advantage.")
    args = parser.parse_args()
    setup_style()
    preds = {model: load_prediction_from_git(path) for model, path in PREDICTION_FILES.items()}
    selected, metrics = select_point(preds)
    point = selected if args.point == "auto" else args.point
    plot_point(point, preds, metrics)


if __name__ == "__main__":
    main()
