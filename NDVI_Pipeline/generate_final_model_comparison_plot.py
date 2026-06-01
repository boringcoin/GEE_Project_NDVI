from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "05_final_outputs" / "chapter4_materials" / "unified_model_comparison_1985_2015"
TABLE_FILE = OUT_DIR / "table4_2_unified_model_comparison_for_paper.csv"


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
            "font.size": 10.5,
        }
    )


def main() -> None:
    setup_style()
    df = pd.read_csv(TABLE_FILE)
    df["SMAPE"] = df["SMAPE/%"].astype(float)
    for col in ["MSE", "MAE", "RMSE"]:
        df[col] = df[col].astype(float)

    bar_color = "#3F7CAC"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    metrics = [
        ("RMSE", "RMSE"),
        ("SMAPE", "SMAPE / %"),
        ("MAE", "MAE"),
        ("MSE", "MSE"),
    ]

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        values = df[metric].to_numpy()
        x = np.arange(len(df))
        bars = ax.bar(x, values, color=bar_color, width=0.68, edgecolor="#2E3740", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["模型"], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} 对比", fontweight="bold")
        ymax = max(values) * 1.18 if max(values) > 0 else 1
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, values):
            label = f"{value:.4f}" if metric != "MSE" else f"{value:.5f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.018,
                label,
                ha="center",
                va="bottom",
                fontsize=8.2,
                rotation=0,
            )

    fig.suptitle("不同预测模型测试集指标对比", fontsize=17, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.005,
        "注：测试集为2020-2025年1000个样点共131000条样本；数值越低表示预测误差越小。",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4B5560",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])

    png = OUT_DIR / "fig4_2_final_model_metric_comparison.png"
    svg = OUT_DIR / "fig4_2_final_model_metric_comparison.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
