from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "05_final_outputs" / "chapter4_materials"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
            "figure.dpi": 140,
            "savefig.dpi": 400,
        }
    )


def add_box(ax, xy, wh, title, body, fc, ec="#2F3A45", lw=1.6, title_size=13, body_size=10.5, ls="-"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        linestyle=ls,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.12,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#1F2933",
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.05,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#263238",
        linespacing=1.35,
    )


def add_arrow(ax, start, end, color="#53606A", rad=0.0, lw=1.8):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def add_sequence(ax, x, y, n=12, target_index=None, pred_indices=None):
    for i in range(n):
        color = "#D4D8DD"
        ec = "#B6BDC5"
        if target_index is not None and i == target_index:
            color = "#F6A0A0"
            ec = "#C75D5D"
        if pred_indices and i in pred_indices:
            color = "#8BB8EA"
            ec = "#4D83C6"
        rect = Rectangle((x + i * 0.18, y), 0.12, 0.12, facecolor=color, edgecolor=ec, linewidth=0.6)
        ax.add_patch(rect)


def add_tree_icon(ax, x, y, color="#5E8CC8"):
    ax.plot([x, x], [y, y + 0.42], color=color, lw=2)
    ax.plot([x, x - 0.18], [y + 0.25, y + 0.5], color=color, lw=2)
    ax.plot([x, x + 0.18], [y + 0.25, y + 0.5], color=color, lw=2)
    ax.plot([x - 0.18, x - 0.3], [y + 0.5, y + 0.68], color=color, lw=2)
    ax.plot([x - 0.18, x - 0.06], [y + 0.5, y + 0.68], color=color, lw=2)
    ax.plot([x + 0.18, x + 0.06], [y + 0.5, y + 0.68], color=color, lw=2)
    ax.plot([x + 0.18, x + 0.3], [y + 0.5, y + 0.68], color=color, lw=2)


def main() -> None:
    setup_style()

    fig, ax = plt.subplots(figsize=(14.2, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    colors = {
        "panel": "#F6F7F9",
        "input": "#EAF3EE",
        "embed": "#FF69CE",
        "feature": "#FFF0C6",
        "target": "#F2A09B",
        "model": "#DDE9F8",
        "residual": "#F4DFDF",
        "output": "#A8E4CF",
        "blue": "#75A9E6",
        "note": "#F8F8FA",
    }

    # Two architecture lanes.
    for y, label in [(4.05, "主预测分支"), (1.65, "残差校正分支")]:
        panel = FancyBboxPatch(
            (0.25, y),
            12.6,
            1.85,
            boxstyle="round,pad=0.015,rounding_size=0.035",
            linewidth=1.3,
            linestyle="--",
            edgecolor="#777F88",
            facecolor=colors["panel"],
        )
        ax.add_patch(panel)
        ax.text(0.45, y + 1.57, label, fontsize=11.5, fontweight="bold", color="#39434D")

    # Main lane input sequence.
    ax.plot([0.75, 1.15, 1.55, 1.95], [4.78, 4.95, 4.9, 5.12], color="#3E8E4C", lw=2.1)
    ax.plot([0.75, 1.15, 1.55, 1.95], [4.55, 4.68, 4.61, 4.74], color="#4A87D4", lw=2.1)
    ax.text(0.72, 4.28, "重构NDVI\n时间序列", ha="center", va="center", fontsize=9.5)

    ax.add_patch(Rectangle((2.2, 4.28), 0.18, 1.18, facecolor=colors["embed"], edgecolor="#AC268A", lw=1.0))
    ax.text(2.29, 4.08, "特征\n嵌入", ha="center", va="top", fontsize=9.2)

    add_box(
        ax,
        (2.92, 4.48),
        (1.45, 0.82),
        "历史窗口",
        "NDVI$_{t-1:t-3}$",
        colors["target"],
        ec="#C35B56",
        title_size=10.8,
        body_size=10,
    )

    add_box(
        ax,
        (4.72, 4.35),
        (2.5, 1.12),
        "EWMA-Lag 特征构造",
        "Lag3 + EWMA$_3$\n仅由预测时刻之前NDVI计算",
        colors["feature"],
        ec="#A27B2C",
        title_size=11.2,
        body_size=9.4,
    )
    add_sequence(ax, 4.95, 4.55, n=11, target_index=7, pred_indices=[8, 9, 10])
    ax.text(5.04, 4.36, "历史", fontsize=8.6, color="#6B5B2A")
    ax.text(6.32, 4.36, "预测目标", fontsize=8.6, color="#6B5B2A")

    add_box(
        ax,
        (7.78, 4.32),
        (2.0, 1.15),
        "XGBoost Ensemble",
        "多棵正则化回归树\n逐轮拟合残差",
        colors["model"],
        ec="#3F6A9A",
        title_size=11.2,
        body_size=9.5,
    )
    for i, x in enumerate([8.15, 8.65, 9.15]):
        add_tree_icon(ax, x, 4.44, color=["#477DB9", "#5D97C8", "#6EA6D5"][i])

    add_box(
        ax,
        (10.42, 4.48),
        (1.45, 0.82),
        "Prediction head",
        "$\\hat{NDVI}_{t}$",
        colors["output"],
        ec="#3C8C76",
        title_size=10.5,
        body_size=12,
    )

    # Residual lane.
    ax.plot([0.75, 1.15, 1.55, 1.95], [2.25, 2.3, 2.21, 2.36], color="#5AA85C", lw=2.1)
    ax.text(0.82, 1.98, "训练期\nOOF预测", ha="center", va="center", fontsize=9.5)
    ax.add_patch(Rectangle((2.2, 1.88), 0.18, 1.18, facecolor=colors["embed"], edgecolor="#AC268A", lw=1.0))
    ax.text(2.29, 1.68, "同一\n特征", ha="center", va="top", fontsize=9.2)

    add_box(
        ax,
        (3.0, 2.05),
        (1.5, 0.8),
        "OOF残差",
        "$r_t = y_t - \\hat{y}_t$\n2016-2019滚动验证",
        colors["target"],
        ec="#C35B56",
        title_size=10.5,
        body_size=8.8,
    )
    add_box(
        ax,
        (5.0, 1.9),
        (2.3, 1.1),
        "Residual learner",
        "只使用历史折残差训练\n不接触测试集标签",
        colors["residual"],
        ec="#8B4B4B",
        title_size=11,
        body_size=9.3,
    )
    add_sequence(ax, 5.25, 2.12, n=10, target_index=6, pred_indices=[7, 8, 9])
    add_box(
        ax,
        (8.0, 2.0),
        (1.65, 0.88),
        "Residual head",
        "$\\hat{r}_{t}$",
        colors["residual"],
        ec="#8B4B4B",
        title_size=10.5,
        body_size=12,
    )
    add_box(
        ax,
        (10.45, 2.0),
        (1.55, 0.88),
        "Corrected output",
        "$\\hat{y}_{t}^{RC}=\\hat{y}_{t}+\\hat{r}_{t}$",
        colors["output"],
        ec="#3C8C76",
        title_size=10.2,
        body_size=10.8,
    )

    add_arrow(ax, (1.98, 4.9), (2.2, 4.9))
    add_arrow(ax, (2.38, 4.9), (2.92, 4.9))
    add_arrow(ax, (4.37, 4.9), (4.72, 4.9))
    add_arrow(ax, (7.22, 4.9), (7.78, 4.9))
    add_arrow(ax, (9.78, 4.9), (10.42, 4.9))

    add_arrow(ax, (1.98, 2.42), (2.2, 2.42), color="#8B4B4B")
    add_arrow(ax, (2.38, 2.42), (3.0, 2.42), color="#8B4B4B")
    add_arrow(ax, (4.5, 2.42), (5.0, 2.42), color="#8B4B4B")
    add_arrow(ax, (7.3, 2.42), (8.0, 2.42), color="#8B4B4B")
    add_arrow(ax, (9.65, 2.42), (10.45, 2.42), color="#8B4B4B")
    add_arrow(ax, (11.16, 4.48), (11.16, 2.88), color="#3C8C76", rad=0.0)

    ax.text(
        7,
        6.55,
        "EWMA-XGBoost NDVI时间序列预测模型结构",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#1D2733",
    )
    ax.text(
        7,
        6.2,
        "主模型使用历史Lag与EWMA特征预测下一期15天NDVI，残差校正分支作为扩展方案",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#51606D",
    )

    # Legend.
    legend_y = 0.62
    legend_items = [
        ("线性/特征嵌入", colors["embed"]),
        ("历史窗口/目标", colors["target"]),
        ("EWMA-Lag特征", colors["feature"]),
        ("XGBoost树集成", colors["model"]),
        ("残差校正", colors["residual"]),
        ("预测输出", colors["output"]),
    ]
    x = 0.65
    for label, color in legend_items:
        ax.add_patch(Rectangle((x, legend_y), 0.18, 0.18, facecolor=color, edgecolor="#5B646B", lw=0.6))
        ax.text(x + 0.24, legend_y + 0.09, label, ha="left", va="center", fontsize=9.2, color="#404A54")
        x += 1.85
    ax.text(
        0.65,
        0.28,
        "时间划分：1985-2019训练/验证，2020-2025测试；所有Lag与EWMA特征均由预测时刻之前的信息计算。",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#53606A",
    )

    png_path = OUT_DIR / "fig4_1_ewma_xgboost_model_structure.png"
    svg_path = OUT_DIR / "fig4_1_ewma_xgboost_model_structure.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
