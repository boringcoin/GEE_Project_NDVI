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


def make_batches(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-csv", default="NDVI_Pipeline/05_final_outputs/reconstruction/final_ndvi_kalman_whittaker.csv")
    parser.add_argument("--output-dir", default="NDVI_Pipeline/05_final_outputs/chapter4_materials/timegpt_all_points")
    parser.add_argument("--test-start", default="2020-01-13")
    parser.add_argument("--test-end", default="2025-05-16")
    parser.add_argument("--horizon", type=int, default=131)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default="timegpt-1")
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-depth", type=int, default=1)
    parser.add_argument("--finetune-loss", default="default")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.finetune_steps > 0 and out_dir.name == "timegpt_all_points":
        loss_suffix = args.finetune_loss.replace("/", "_")
        out_dir = out_dir.with_name(
            f"timegpt_all_points_finetune{args.finetune_steps}_depth{args.finetune_depth}_{loss_suffix}"
        )
    batch_dir = out_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.reconstruction_csv, parse_dates=["date"])
    df = df.sort_values(["point", "date"]).copy()
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)

    history = df[df["date"] < test_start].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    points = sorted(test["point"].unique(), key=lambda x: int(str(x).replace("point", "")))
    batches = make_batches(points, args.batch_size)

    print(f"points={len(points)}, test_rows={len(test)}, batches={len(batches)}, batch_size={args.batch_size}")
    print(f"history={history['date'].min().date()} to {history['date'].max().date()}")
    print(f"test={test['date'].min().date()} to {test['date'].max().date()}, h={args.horizon}")

    client = NixtlaClient()
    batch_paths: list[Path] = []

    for idx, batch_points in enumerate(batches, start=1):
        batch_path = batch_dir / f"timegpt_batch_{idx:03d}.csv"
        batch_paths.append(batch_path)
        if batch_path.exists():
            print(f"[{idx}/{len(batches)}] skip existing {batch_path.name}")
            continue

        train_batch = history[history["point"].isin(batch_points)].rename(
            columns={"point": "unique_id", "date": "ds", "NDVI_Target": "y"}
        )[["unique_id", "ds", "y"]]
        print(f"[{idx}/{len(batches)}] forecasting {batch_points[0]}..{batch_points[-1]}, train_rows={len(train_batch)}")

        fcst = client.forecast(
            df=train_batch,
            h=args.horizon,
            freq="15D",
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            model=args.model,
            finetune_steps=args.finetune_steps,
            finetune_depth=args.finetune_depth,
            finetune_loss=args.finetune_loss,
            validate_api_key=False,
            num_partitions=1,
        )

        pred_cols = [c for c in fcst.columns if c not in ("unique_id", "ds")]
        if not pred_cols:
            raise RuntimeError(f"No forecast column returned: {fcst.columns.tolist()}")
        pred_col = "TimeGPT" if "TimeGPT" in pred_cols else pred_cols[0]
        fcst = fcst.rename(
            columns={"unique_id": "point", "ds": "date", pred_col: "pred_TimeGPT"}
        )[["point", "date", "pred_TimeGPT"]]
        fcst["date"] = pd.to_datetime(fcst["date"])

        # The final reconstruction grid starts at 2020-01-13. If TimeGPT's generated
        # dates drift from that grid, keep predictions by horizon order per point.
        aligned_parts = []
        for point, group in fcst.groupby("point", sort=False):
            target_dates = test[test["point"] == point]["date"].sort_values().reset_index(drop=True)
            part = group.sort_values("date").reset_index(drop=True).copy()
            if len(part) != len(target_dates):
                raise RuntimeError(f"{point}: forecast length {len(part)} != target length {len(target_dates)}")
            part["date_original_timegpt"] = part["date"]
            part["date"] = target_dates.to_numpy()
            aligned_parts.append(part)
        fcst = pd.concat(aligned_parts, ignore_index=True)
        fcst.to_csv(batch_path, index=False)

    predictions = pd.concat([pd.read_csv(path, parse_dates=["date"]) for path in batch_paths], ignore_index=True)
    predictions = predictions.sort_values(["point", "date"]).copy()
    all_pred_path = out_dir / "timegpt_all_points_predictions.csv"
    predictions.to_csv(all_pred_path, index=False)

    merged = test[["point", "date", "NDVI_Target"]].merge(
        predictions[["point", "date", "pred_TimeGPT"]], on=["point", "date"], how="left"
    )
    missing = int(merged["pred_TimeGPT"].isna().sum())
    if missing:
        raise RuntimeError(f"Missing TimeGPT predictions after merge: {missing}")

    y = merged["NDVI_Target"].to_numpy(dtype=float)
    pred = merged["pred_TimeGPT"].to_numpy(dtype=float)
    mse = float(mean_squared_error(y, pred))
    metrics = pd.DataFrame(
        [
            {
                "model": "TimeGPT",
                "points": merged["point"].nunique(),
                "test_samples": len(merged),
                "MSE": mse,
                "MAE": float(mean_absolute_error(y, pred)),
                "RMSE": float(np.sqrt(mse)),
                "SMAPE": smape(y, pred),
            }
        ]
    )
    metrics_path = out_dir / "timegpt_all_points_metrics.csv"
    merged_path = out_dir / "timegpt_all_points_predictions_with_truth.csv"
    merged.to_csv(merged_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(metrics.to_string(index=False))
    print(f"saved {all_pred_path}")
    print(f"saved {merged_path}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
