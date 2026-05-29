"""
Generate Chapter 4 prediction materials under one evaluation protocol.

Protocol:
  - Target: final LOF + Kalman + Whittaker NDVI sequence.
  - Train/history period: 1985-01-01 ~ 2019-12-31.
  - Test period: 2020-01-01 ~ 2025-05-16.
  - Features for model comparison: NDVI_prev1, NDVI_prev2, NDVI_prev3, NDVI_ewma_3.

Outputs are written to:
  NDVI_Pipeline/05_final_outputs/chapter4_materials
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "05_final_outputs"
PRED_DIR = FINAL_DIR / "prediction"
OUT_DIR = FINAL_DIR / "chapter4_materials"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILE = FINAL_DIR / "reconstruction" / "final_ndvi_kalman_whittaker.csv"

TRAIN_START = "1985-01-01"
PRETEST_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-05-16"

FEATURES = ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3"]
DECIMAL = 4


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
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def smape(y_true, y_pred) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.zeros_like(denom, dtype="float64")
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out) * 100)


def metric_dict(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "SMAPE": smape(y_true, y_pred),
    }


def load_data() -> pd.DataFrame:
    data = pd.read_csv(TARGET_FILE, parse_dates=["date"])
    data = data[["point", "date", "NDVI_Target"]].sort_values(["point", "date"]).reset_index(drop=True)
    for lag in [1, 2, 3]:
        data[f"NDVI_prev{lag}"] = data.groupby("point")["NDVI_Target"].shift(lag)
    data["NDVI_ewma_3"] = data.groupby("point")["NDVI_Target"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[(data["date"] >= TRAIN_START) & (data["date"] <= PRETEST_END)].dropna(
        subset=FEATURES + ["NDVI_Target"]
    )
    test = data[(data["date"] >= TEST_START) & (data["date"] <= TEST_END)].dropna(
        subset=FEATURES + ["NDVI_Target"]
    )
    return train, test


def make_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=5,
        gamma=0.001,
        subsample=0.75,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=0,
    )


def make_lgbm_model():
    if lgb is None:
        raise RuntimeError("LightGBM is not installed in this environment.")
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.75,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )


def train_predict_model(model_name: str, model, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame, object]:
    t0 = time.time()
    model.fit(train[FEATURES], train["NDVI_Target"])
    pred = np.round(model.predict(test[FEATURES]), DECIMAL)
    out = test[["point", "date", "NDVI_Target"]].copy()
    out["Prediction"] = pred
    out.to_csv(OUT_DIR / f"predictions_{model_name}.csv", index=False)
    row = {
        "模型": model_name,
        "实验口径": "最新流水线",
        "样点数": int(test["point"].nunique()),
        "测试样本数": int(len(test)),
        "备注": "同一重构序列；同一EWMA输入；1985-2019训练，2020-2025测试",
        "time_s": round(time.time() - t0, 2),
    }
    row.update(metric_dict(out["NDVI_Target"].to_numpy(), pred))
    return row, out, model


def copy_existing_xgb_tables() -> None:
    for name in [
        "final_data_split_summary.csv",
        "final_xgboost_model_comparison.csv",
        "final_xgboost_parameter_ablation.csv",
        "final_ewma_feature_ablation.csv",
        "final_oof_fold_metrics.csv",
    ]:
        src = PRED_DIR / name
        if src.exists():
            (OUT_DIR / name).write_bytes(src.read_bytes())


def load_final_xgb_predictions() -> dict[str, pd.DataFrame]:
    # Recreate these three CSVs in the final folder when available through the
    # compact model-comparison outputs. If unavailable, use current run XGBoost
    # for the first two downstream analyses.
    rc_dir = BASE_DIR / "04_results" / "xgboost_ewma_rc_oof_split"
    mapping = {
        "XGBoost": rc_dir / "Predictions_XGBoost_Lag3.csv",
        "EWMA-XGBoost": rc_dir / "Predictions_EWMA_XGBoost.csv",
        "EWMA-RC-XGBoost": rc_dir / "Predictions_EWMA_RC_XGBoost_OOF_s1.csv",
    }
    preds = {}
    for key, path in mapping.items():
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            cols = {"NDVI_Target": "true_NDVI", "Prediction": f"pred_{key}"}
            preds[key] = df.rename(columns=cols)
    return preds


def build_typical_point_curve_data(preds: dict[str, pd.DataFrame]) -> None:
    if len(preds) < 3:
        return
    base = preds["EWMA-XGBoost"][["point", "date", "true_NDVI", "pred_EWMA-XGBoost"]].copy()
    merged = base.merge(
        preds["XGBoost"][["point", "date", "pred_XGBoost"]],
        on=["point", "date"],
        how="left",
    ).merge(
        preds["EWMA-RC-XGBoost"][["point", "date", "pred_EWMA-RC-XGBoost"]],
        on=["point", "date"],
        how="left",
    )
    metrics = []
    for point, g in merged.groupby("point"):
        y = g["true_NDVI"].to_numpy()
        p = g["pred_EWMA-XGBoost"].to_numpy()
        m = metric_dict(y, p)
        metrics.append({"point": point, **m})
    point_metrics = pd.DataFrame(metrics)
    point_metrics.to_csv(OUT_DIR / "typical_point_candidate_metrics.csv", index=False)
    q_good = point_metrics["RMSE"].quantile(0.1)
    q_mid = point_metrics["RMSE"].quantile(0.5)
    q_bad = point_metrics["RMSE"].quantile(0.9)
    selected = [
        ("good", point_metrics.iloc[(point_metrics["RMSE"] - q_good).abs().argmin()]["point"]),
        ("medium", point_metrics.iloc[(point_metrics["RMSE"] - q_mid).abs().argmin()]["point"]),
        ("large_error", point_metrics.iloc[(point_metrics["RMSE"] - q_bad).abs().argmin()]["point"]),
    ]
    selected_rows = []
    for label, point in selected:
        part = merged[merged["point"] == point].copy()
        part.insert(0, "case_type", label)
        selected_rows.append(part)
    typical = pd.concat(selected_rows, ignore_index=True)
    typical = typical.rename(
        columns={
            "point": "point_id",
            "pred_EWMA-XGBoost": "pred_EWMA_XGBoost",
            "pred_EWMA-RC-XGBoost": "pred_EWMA_RC_XGBoost",
        }
    )
    typical.to_csv(OUT_DIR / "typical_point_prediction_curves.csv", index=False)
    plot_typical_points(typical)


def plot_typical_points(typical: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (case_type, g) in zip(axes, typical.groupby("case_type", sort=False)):
        point = g["point_id"].iloc[0]
        ax.plot(g["date"], g["true_NDVI"], label="真实NDVI", color="#2f6f9f", lw=1.8)
        ax.plot(g["date"], g["pred_XGBoost"], label="XGBoost", color="#888888", lw=1.0, alpha=0.8)
        ax.plot(g["date"], g["pred_EWMA_XGBoost"], label="EWMA-XGBoost", color="#d95f02", lw=1.4, ls="--")
        ax.plot(g["date"], g["pred_EWMA_RC_XGBoost"], label="EWMA-RC-XGBoost", color="#59a14f", lw=1.1, ls=":")
        ax.set_ylabel("NDVI")
        ax.set_ylim(0, 1)
        ax.set_title(f"{case_type}: {point}", loc="left", fontsize=10)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("日期")
    fig.suptitle("典型样点真实值与预测值对比")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "typical_point_prediction_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def residual_analysis(preds: dict[str, pd.DataFrame]) -> None:
    rows = []
    for model_name, df in preds.items():
        pred_col = f"pred_{model_name}"
        tmp = pd.DataFrame(
            {
                "model": model_name,
                "point": df["point"],
                "date": df["date"],
                "true_NDVI": df["true_NDVI"],
                "pred_NDVI": df[pred_col],
            }
        )
        tmp["residual"] = tmp["true_NDVI"] - tmp["pred_NDVI"]
        tmp["abs_error"] = tmp["residual"].abs()
        denom = (tmp["true_NDVI"].abs() + tmp["pred_NDVI"].abs()) / 2.0
        tmp["smape_point"] = np.where(denom != 0, tmp["abs_error"] / denom * 100, 0.0)
        rows.append(tmp)
    all_resid = pd.concat(rows, ignore_index=True)
    all_resid.to_csv(OUT_DIR / "residual_analysis_samples.csv", index=False)

    bins = [-np.inf, 0.3, 0.6, np.inf]
    labels = ["低值区间(<0.3)", "中值区间(0.3-0.6)", "高值区间(>=0.6)"]
    all_resid["NDVI区间"] = pd.cut(all_resid["true_NDVI"], bins=bins, labels=labels)
    grouped = []
    for (bin_name, model), g in all_resid.groupby(["NDVI区间", "model"], observed=True):
        grouped.append(
            {
                "NDVI区间": bin_name,
                "模型": model,
                "样本数": len(g),
                "MAE": float(g["abs_error"].mean()),
                "RMSE": float(np.sqrt(np.mean(g["residual"] ** 2))),
                "SMAPE": float(g["smape_point"].mean()),
                "MBE": float(g["residual"].mean()),
            }
        )
    group_df = pd.DataFrame(grouped)
    group_df.to_csv(OUT_DIR / "residual_analysis_by_ndvi_bin.csv", index=False)
    plot_residual_bins(group_df)


def plot_residual_bins(group_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pivot = group_df.pivot(index="NDVI区间", columns="模型", values="SMAPE")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("不同NDVI区间SMAPE对比")
    ax.set_ylabel("SMAPE")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residual_smape_by_ndvi_bin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def feature_importance(model) -> None:
    if hasattr(model, "get_booster"):
        gain = model.get_booster().get_score(importance_type="gain")
        imp = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": [float(gain.get(f, 0.0)) for f in FEATURES],
            }
        )
    else:
        imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp["importance_ratio"] = imp["importance"] / imp["importance"].sum()
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT_DIR / "feature_importance_ewma_xgboost.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(imp["feature"], imp["importance_ratio"], color="#4c78a8")
    ax.set_title("EWMA-XGBoost特征重要性")
    ax.set_ylabel("重要性占比")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_ewma_xgboost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_confirmation(model_compare: pd.DataFrame) -> None:
    text = f"""# 第四章材料包说明

## 已生成内容

1. 同一口径模型对比：`model_comparison_same_protocol.csv`
2. 三类XGBoost最终结果：`final_xgboost_model_comparison.csv`
3. 数据划分：`final_data_split_summary.csv`
4. 参数消融：`final_xgboost_parameter_ablation.csv`
5. EWMA特征消融：`final_ewma_feature_ablation.csv`
6. OOF残差验证折：`final_oof_fold_metrics.csv`
7. 典型样点预测曲线数据：`typical_point_prediction_curves.csv`
8. 残差样本数据：`residual_analysis_samples.csv`
9. NDVI分区残差分析：`residual_analysis_by_ndvi_bin.csv`
10. 特征重要性：`feature_importance_ewma_xgboost.csv`

## 关键口径确认

- 预测目标：当前15天窗口的重构NDVI，即用预测时刻之前的信息预测该时间步NDVI。若按序列滚动预测解释，也可表述为“基于历史15天窗口信息预测下一15天窗口NDVI”。
- 输入信息：每个样本只使用预测时刻之前的信息，包括 `NDVI_prev1`、`NDVI_prev2`、`NDVI_prev3` 和 `NDVI_ewma_3`。EWMA由历史NDVI递推得到，不包含当前或未来NDVI。
- 训练/测试：1985-2019为训练/验证历史期，2020-2025为最终测试期。

## 同口径模型对比

{to_markdown_table(model_compare)}
"""
    (OUT_DIR / "chapter4_materials_readme.md").write_text(text, encoding="utf-8")


def to_markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns.astype(str)) + " |",
        "| " + " | ".join([":---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    setup_style()
    copy_existing_xgb_tables()
    data = load_data()
    train, test = split_data(data)
    print(f"train_rows={len(train)}, test_rows={len(test)}, points={test['point'].nunique()}", flush=True)

    rows = []
    preds = {}

    xgb_row, xgb_pred, xgb_model = train_predict_model("XGBoost", make_xgb_model(), train, test)
    rows.append(xgb_row)
    preds["XGBoost_same_protocol"] = xgb_pred

    if lgb is not None:
        lgbm_row, lgbm_pred, lgbm_model = train_predict_model("LightGBM", make_lgbm_model(), train, test)
        rows.append(lgbm_row)
        preds["LightGBM"] = lgbm_pred
    else:
        print("LightGBM is not installed; skipped.", flush=True)

    # Add TimesFM only if a same-protocol/retained zero-shot metric file exists.
    # The cleaned final package intentionally does not mix older different-protocol files.
    model_compare = pd.DataFrame(rows)
    model_compare = model_compare[
        ["模型", "实验口径", "样点数", "测试样本数", "MSE", "MAE", "RMSE", "SMAPE", "备注", "time_s"]
    ]
    model_compare.to_csv(OUT_DIR / "model_comparison_same_protocol.csv", index=False)

    # Use retained final XGBoost family predictions for curve/residual analysis.
    final_preds = load_final_xgb_predictions()
    if len(final_preds) == 3:
        build_typical_point_curve_data(final_preds)
        residual_analysis(final_preds)
    feature_importance(xgb_model)
    plot_model_compare(model_compare)
    write_confirmation(model_compare)
    print(model_compare.to_string(index=False), flush=True)
    print(OUT_DIR, flush=True)


def plot_model_compare(model_compare: pd.DataFrame) -> None:
    order = model_compare.sort_values("RMSE")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].bar(order["模型"], order["RMSE"], color="#4c78a8")
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(order["模型"], order["SMAPE"], color="#59a14f")
    axes[1].set_title("SMAPE")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("同一口径预测模型对比")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison_same_protocol.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
