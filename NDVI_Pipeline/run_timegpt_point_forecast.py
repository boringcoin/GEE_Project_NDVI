from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nixtla import NixtlaClient
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", default="point977")
    parser.add_argument("--wide-csv", required=True)
    parser.add_argument("--reconstruction-csv", default="NDVI_Pipeline/05_final_outputs/reconstruction/final_ndvi_kalman_whittaker.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-depth", type=int, default=1)
    parser.add_argument("--finetune-loss", default="default")
    parser.add_argument("--model", default="timegpt-1")
    args = parser.parse_args()

    point = args.point
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(args.wide_csv, parse_dates=["date"])
    test = test[test["point"] == point].sort_values("date").copy()
    if test.empty:
        raise ValueError(f"No rows found for {point} in {args.wide_csv}")

    full = pd.read_csv(args.reconstruction_csv, parse_dates=["date"])
    series = full[full["point"] == point].sort_values("date").copy()
    train = series[series["date"] < test["date"].min()].copy()
    if train.empty:
        raise ValueError(f"No historical rows found for {point}")

    api_df = train.rename(
        columns={"point": "unique_id", "date": "ds", "NDVI_Target": "y"}
    )[["unique_id", "ds", "y"]]

    print(f"point={point}")
    print(f"train={train['date'].min().date()} to {train['date'].max().date()}, n={len(train)}")
    print(f"test={test['date'].min().date()} to {test['date'].max().date()}, h={len(test)}")

    client = NixtlaClient()
    fcst = client.forecast(
        df=api_df,
        h=len(test),
        freq="15D",
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        model=args.model,
        finetune_steps=args.finetune_steps,
        finetune_depth=args.finetune_depth,
        finetune_loss=args.finetune_loss,
        validate_api_key=False,
    )

    pred_cols = [c for c in fcst.columns if c not in ("unique_id", "ds")]
    if not pred_cols:
        raise RuntimeError(f"No forecast column returned: {fcst.columns.tolist()}")
    pred_col = "TimeGPT" if "TimeGPT" in pred_cols else pred_cols[0]

    fcst = fcst.rename(
        columns={"unique_id": "point", "ds": "date", pred_col: "pred_TimeGPT"}
    )[["point", "date", "pred_TimeGPT"]]
    fcst["date"] = pd.to_datetime(fcst["date"])

    if not fcst["date"].reset_index(drop=True).equals(test["date"].reset_index(drop=True)):
        print("TimeGPT dates differ from final test grid; aligning by horizon order.")
        fcst["date_original_timegpt"] = fcst["date"]
        fcst["date"] = test["date"].to_numpy()

    merged = test.drop(columns=["pred_TimeGPT"], errors="ignore").merge(
        fcst[["point", "date", "pred_TimeGPT"]], on=["point", "date"], how="left"
    )
    if merged["pred_TimeGPT"].isna().any():
        raise RuntimeError("TimeGPT merge produced missing predictions")

    suffix = (
        f"finetune{args.finetune_steps}"
        if args.finetune_steps > 0
        else "zeroshot"
    )
    forecast_path = out_dir / f"{point}_timegpt_{suffix}_forecast.csv"
    wide_path = out_dir / f"{point}_predictions_with_timegpt_{suffix}.csv"
    metrics_path = out_dir / f"{point}_model_metrics_with_timegpt_{suffix}.csv"

    fcst.to_csv(forecast_path, index=False)
    merged.to_csv(wide_path, index=False)

    metric_rows = []
    y = merged["true_NDVI"].to_numpy(dtype=float)
    for col in [c for c in merged.columns if c.startswith("pred_")]:
        pred = merged[col].to_numpy(dtype=float)
        mse = float(mean_squared_error(y, pred))
        metric_rows.append(
            {
                "model": col.replace("pred_", ""),
                "MSE": mse,
                "MAE": float(mean_absolute_error(y, pred)),
                "RMSE": float(np.sqrt(mse)),
                "SMAPE": smape(y, pred),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("RMSE")
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))
    print(f"saved {forecast_path}")
    print(f"saved {wide_path}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
