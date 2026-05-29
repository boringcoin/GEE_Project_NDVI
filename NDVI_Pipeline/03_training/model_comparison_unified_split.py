from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


BASE_DIR = Path(__file__).resolve().parents[1]
TARGET_FILE = BASE_DIR / "05_final_outputs" / "reconstruction" / "final_ndvi_kalman_whittaker.csv"
OUT_DIR = BASE_DIR / "05_final_outputs" / "chapter4_materials" / "unified_model_comparison_1985_2015"

TRAIN_START = "1985-01-01"
TRAIN_END = "2014-12-31"
VAL_START = "2015-01-01"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-05-16"


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.zeros_like(denom, dtype=np.float64)
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(out.mean() * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "SMAPE": smape(y_true, y_pred),
    }


def load_series() -> pd.DataFrame:
    df = pd.read_csv(TARGET_FILE, parse_dates=["date"])
    df = df[["point", "date", "NDVI_Target"]].sort_values(["point", "date"]).reset_index(drop=True)
    return df


def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in [1, 2, 3]:
        out[f"NDVI_prev{lag}"] = out.groupby("point")["NDVI_Target"].shift(lag)
    out["NDVI_ewma_3"] = out.groupby("point")["NDVI_Target"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    return out


def split_tabular(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = features + ["NDVI_Target"]
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].dropna(subset=keep)
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].dropna(subset=keep)
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].dropna(subset=keep)
    return train, val, test


def build_sequence_arrays(df: pd.DataFrame, seq_len: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    xs, ys, dates, points, splits = [], [], [], [], []
    for point, g in df.groupby("point", sort=False):
        values = g["NDVI_Target"].to_numpy(dtype=np.float32)
        gd = g["date"].to_numpy()
        for i in range(seq_len, len(g)):
            date = pd.Timestamp(gd[i])
            if TRAIN_START <= str(date.date()) <= TRAIN_END:
                split = "train"
            elif VAL_START <= str(date.date()) <= VAL_END:
                split = "val"
            elif TEST_START <= str(date.date()) <= TEST_END:
                split = "test"
            else:
                continue
            xs.append(values[i - seq_len : i])
            ys.append(values[i])
            dates.append(date)
            points.append(point)
            splits.append(split)
    arr = {
        "X": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "split": np.asarray(splits),
    }
    meta = pd.DataFrame({"point": points, "date": dates, "split": splits})
    return arr, meta


def train_tree_model(name: str, model, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    model.fit(train[features], train["NDVI_Target"])
    pred = np.clip(model.predict(test[features]), 0, 1)
    pred = np.round(pred, 4)
    y = test["NDVI_Target"].to_numpy()
    row = {
        "模型": name,
        "模型类型": "集成学习模型",
        "样点数": int(test["point"].nunique()),
        "测试样本数": int(len(test)),
        "实验口径": "1985-2014训练，2015-2019验证，2020-2025测试",
        "time_s": round(time.time() - t0, 2),
    }
    row.update(metrics(y, pred))
    pred_df = test[["point", "date", "NDVI_Target"]].copy()
    pred_df["Prediction"] = pred
    return row, pred_df


class LSTMRegressor(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        z = self.proj(x.unsqueeze(-1)) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1]).squeeze(-1)


class AutoformerStyleRegressor(nn.Module):
    def __init__(self, seq_len: int, kernel: int = 5):
        super().__init__()
        self.kernel = kernel
        self.trend_net = nn.Sequential(nn.Linear(seq_len, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 1))
        self.resid_net = nn.Sequential(nn.Linear(seq_len, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 1))

    def moving_average(self, x):
        pad = self.kernel // 2
        z = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        return torch.nn.functional.avg_pool1d(z, self.kernel, stride=1).squeeze(1)

    def forward(self, x):
        trend = self.moving_average(x)
        resid = x - trend
        return (self.trend_net(trend) + self.resid_net(resid)).squeeze(-1)


class NHiTSStyleRegressor(nn.Module):
    def __init__(self, seq_len: int, hidden: int = 128, stacks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_len, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(stacks)
            ]
        )

    def forward(self, x):
        preds = [block(x) for block in self.blocks]
        return torch.stack(preds, dim=0).sum(dim=0).squeeze(-1) / len(self.blocks)


def train_torch_model(
    name: str,
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    meta: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    patience: int,
) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device_obj)
    X, y, split = arrays["X"], arrays["y"], arrays["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[train_mask]), torch.from_numpy(y[train_mask])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device_obj.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_mask]), torch.from_numpy(y[val_mask])),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=device_obj.type == "cuda",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_loss, bad_epochs = None, float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device_obj, non_blocking=True)
            yb = yb.to(device_obj, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_obj, non_blocking=True)
                yb = yb.to(device_obj, non_blocking=True)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
            val_loss = float(np.mean(val_losses))
        print(f"{name} epoch={epoch:02d} train_mse={np.mean(losses):.8f} val_mse={val_loss:.8f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    preds = []
    test_loader = DataLoader(torch.from_numpy(X[test_mask]), batch_size=batch_size * 2, shuffle=False)
    with torch.no_grad():
        for xb in test_loader:
            pred = model(xb.to(device_obj)).detach().cpu().numpy()
            preds.append(pred)
    pred = np.clip(np.concatenate(preds), 0, 1)
    pred = np.round(pred, 4)
    y_test = y[test_mask]
    test_meta = meta.loc[test_mask].copy()
    test_meta["NDVI_Target"] = y_test
    test_meta["Prediction"] = pred

    row = {
        "模型": name,
        "模型类型": "深度时序模型",
        "样点数": int(test_meta["point"].nunique()),
        "测试样本数": int(len(test_meta)),
        "实验口径": "1985-2014训练，2015-2019验证，2020-2025测试",
        "time_s": round(time.time() - t0, 2),
    }
    row.update(metrics(y_test, pred))
    return row, test_meta


def run_timesfm_zero_shot(df: pd.DataFrame, context_len: int, device: str) -> tuple[dict, pd.DataFrame]:
    import timesfm

    t0 = time.time()
    test_rows, preds = [], []
    test_dates_ref = None
    grouped = list(df.groupby("point", sort=False))
    horizon = None
    for point, g in grouped:
        test_g = g[(g["date"] >= TEST_START) & (g["date"] <= TEST_END)]
        horizon = len(test_g)
        test_dates_ref = test_g["date"].to_list()
        break
    backend = "gpu" if device == "cuda" else "cpu"
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            context_len=context_len,
            horizon_len=horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            num_heads=16,
            model_dims=1280,
            backend=backend,
            per_core_batch_size=32,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    contexts, points = [], []
    for point, g in grouped:
        hist = g[g["date"] <= VAL_END]["NDVI_Target"].to_numpy(dtype=np.float32)
        test_g = g[(g["date"] >= TEST_START) & (g["date"] <= TEST_END)]
        if len(test_g) != horizon:
            continue
        contexts.append(hist[-context_len:])
        points.append(point)
        test_rows.append(test_g[["point", "date", "NDVI_Target"]])
    forecast, _ = model.forecast(contexts, freq=[0] * len(contexts))
    test_df = pd.concat(test_rows, ignore_index=True)
    for arr in forecast:
        preds.extend(arr[:horizon])
    pred = np.clip(np.asarray(preds, dtype=np.float32), 0, 1)
    pred = np.round(pred, 4)
    y = test_df["NDVI_Target"].to_numpy()
    test_df["Prediction"] = pred
    row = {
        "模型": "TimesFM",
        "模型类型": "时序基础模型",
        "样点数": int(test_df["point"].nunique()),
        "测试样本数": int(len(test_df)),
        "实验口径": "zero-shot；1985-2019作为上下文，2020-2025测试",
        "time_s": round(time.time() - t0, 2),
    }
    row.update(metrics(y, pred))
    return row, test_df


def make_xgb():
    if xgb is None:
        raise RuntimeError("xgboost is not installed.")
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


def make_lgbm():
    if lgb is None:
        raise RuntimeError("lightgbm is not installed.")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="xgboost,lightgbm,lstm,informer,autoformer,nhits",
        help="Comma-separated: xgboost,lightgbm,lstm,informer,autoformer,nhits,timesfm",
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesfm-context-len", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requested = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    print("Requested models:", requested, flush=True)
    print("Split: train 1985-01-01..2014-12-31, val 2015-01-01..2019-12-31, test 2020-01-01..2025-05-16", flush=True)
    print("Target: NDVI value at each evaluated time step; inputs use only previous observations.", flush=True)

    df = load_series()
    features = ["NDVI_prev1", "NDVI_prev2", "NDVI_prev3", "NDVI_ewma_3"]
    tab = add_tabular_features(df)
    train, val, test = split_tabular(tab, features)
    split_summary = pd.DataFrame(
        [
            {"数据部分": "训练集", "时间范围": f"{TRAIN_START} 至 {TRAIN_END}", "样本数": len(train), "用途": "模型训练"},
            {"数据部分": "验证集", "时间范围": f"{VAL_START} 至 {VAL_END}", "样本数": len(val), "用途": "调参/早停"},
            {"数据部分": "测试集", "时间范围": f"{TEST_START} 至 {TEST_END}", "样本数": len(test), "用途": "最终评价"},
        ]
    )
    split_summary.to_csv(OUT_DIR / "unified_data_split_summary.csv", index=False)

    rows = []
    if "xgboost" in requested:
        row, pred = train_tree_model("XGBoost", make_xgb(), pd.concat([train, val]), test, features)
        rows.append(row)
        pred.to_csv(OUT_DIR / "predictions_XGBoost.csv", index=False)
        print(row, flush=True)
    if "lightgbm" in requested:
        row, pred = train_tree_model("LightGBM", make_lgbm(), pd.concat([train, val]), test, features)
        rows.append(row)
        pred.to_csv(OUT_DIR / "predictions_LightGBM.csv", index=False)
        print(row, flush=True)

    deep_models = [m for m in requested if m in {"lstm", "informer", "autoformer", "nhits"}]
    if deep_models:
        arrays, meta = build_sequence_arrays(df, args.seq_len)
        if "lstm" in deep_models:
            row, pred = train_torch_model(
                "LSTM类模型",
                LSTMRegressor(),
                arrays,
                meta,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                patience=args.patience,
            )
            rows.append(row)
            pred.to_csv(OUT_DIR / "predictions_LSTM.csv", index=False)
        if "informer" in deep_models:
            row, pred = train_torch_model(
                "Informer",
                TransformerRegressor(args.seq_len),
                arrays,
                meta,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                patience=args.patience,
            )
            rows.append(row)
            pred.to_csv(OUT_DIR / "predictions_Informer.csv", index=False)
        if "autoformer" in deep_models:
            row, pred = train_torch_model(
                "Autoformer",
                AutoformerStyleRegressor(args.seq_len),
                arrays,
                meta,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                patience=args.patience,
            )
            rows.append(row)
            pred.to_csv(OUT_DIR / "predictions_Autoformer.csv", index=False)
        if "nhits" in deep_models:
            row, pred = train_torch_model(
                "N-HiTS",
                NHiTSStyleRegressor(args.seq_len),
                arrays,
                meta,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                patience=args.patience,
            )
            rows.append(row)
            pred.to_csv(OUT_DIR / "predictions_NHiTS.csv", index=False)

    if "timesfm" in requested:
        row, pred = run_timesfm_zero_shot(df, args.timesfm_context_len, args.device)
        rows.append(row)
        pred.to_csv(OUT_DIR / "predictions_TimesFM.csv", index=False)
        print(row, flush=True)

    result_path = OUT_DIR / "unified_model_comparison_metrics.csv"
    if result_path.exists():
        old = pd.read_csv(result_path)
        result = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        result = result.drop_duplicates(subset=["模型"], keep="last")
    else:
        result = pd.DataFrame(rows)
    result = result[
        ["模型", "模型类型", "样点数", "测试样本数", "MSE", "MAE", "RMSE", "SMAPE", "实验口径", "time_s"]
    ]
    result.to_csv(result_path, index=False)
    print("\nSaved:", result_path, flush=True)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
