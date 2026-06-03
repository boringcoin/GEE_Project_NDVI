from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parents[1]
TARGET_FILE = BASE_DIR / "05_final_outputs" / "reconstruction" / "final_ndvi_kalman_whittaker.csv"
OUT_DIR = BASE_DIR / "05_final_outputs" / "chapter4_materials" / "lstm_feature_residual_ablation"

TRAIN_START = "1985-01-01"
TRAIN_END = "2014-12-31"
VAL_START = "2015-01-01"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-05-16"


def setup_style() -> None:
    for path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 350})


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    out = np.zeros_like(denom, dtype=float)
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(out.mean() * 100)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "SMAPE": smape(y_true, y_pred),
    }


class SeqLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TARGET_FILE, parse_dates=["date"])
    df = df[["point", "date", "NDVI_Target"]].sort_values(["point", "date"]).reset_index(drop=True)
    df["ewma_3"] = df.groupby("point")["NDVI_Target"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    df["diff_1"] = df.groupby("point")["NDVI_Target"].diff(1).shift(1)
    return df


def split_name(date: pd.Timestamp) -> str | None:
    d = str(date.date())
    if TRAIN_START <= d <= TRAIN_END:
        return "train"
    if VAL_START <= d <= VAL_END:
        return "val"
    if TEST_START <= d <= TEST_END:
        return "test"
    return None


def build_arrays(df: pd.DataFrame, seq_len: int, feature_cols: list[str]):
    xs, ys, rows = [], [], []
    for point, g in df.groupby("point", sort=False):
        g = g.reset_index(drop=True)
        feature_values = g[feature_cols].to_numpy(dtype=np.float32)
        target_values = g["NDVI_Target"].to_numpy(dtype=np.float32)
        for i in range(seq_len, len(g)):
            split = split_name(pd.Timestamp(g.loc[i, "date"]))
            if split is None:
                continue
            window = feature_values[i - seq_len : i]
            if np.isnan(window).any():
                continue
            xs.append(window)
            ys.append(target_values[i])
            rows.append(
                {
                    "point": point,
                    "date": g.loc[i, "date"],
                    "split": split,
                    "last_ndvi": target_values[i - 1],
                    "last_ewma_3": g.loc[i, "ewma_3"],
                    "last_diff_1": g.loc[i, "diff_1"],
                }
            )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(rows)


def train_lstm_variant(name: str, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, device: str, epochs: int = 8):
    t0 = time.time()
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    masks = {split: meta["split"].to_numpy() == split for split in ["train", "val", "test"]}
    model = SeqLSTM(X.shape[-1]).to(device_obj)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[masks["train"]]), torch.from_numpy(y[masks["train"]])),
        batch_size=2048,
        shuffle=True,
        num_workers=2,
        pin_memory=device_obj.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[masks["val"]]), torch.from_numpy(y[masks["val"]])),
        batch_size=4096,
        shuffle=False,
        num_workers=2,
        pin_memory=device_obj.type == "cuda",
    )
    best_state, best_val, bad_epochs = None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device_obj, non_blocking=True)
            yb = yb.to(device_obj, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_obj, non_blocking=True)
                yb = yb.to(device_obj, non_blocking=True)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))
        print(f"{name} epoch={epoch:02d} train_mse={np.mean(train_losses):.8f} val_mse={val_loss:.8f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
    pred = predict_lstm(model, X, device_obj)
    pred = np.round(np.clip(pred, 0, 1), 4)
    rows = []
    pred_df = meta[["point", "date", "split", "last_ndvi", "last_ewma_3", "last_diff_1"]].copy()
    pred_df["NDVI_Target"] = y
    pred_df["Prediction"] = pred
    for split in ["val", "test"]:
        mask = masks[split]
        row = {"模型": name, "数据部分": split, "time_s": round(time.time() - t0, 2)}
        row.update(metric_dict(y[mask], pred[mask]))
        rows.append(row)
    return model, pred_df, rows


def predict_lstm(model: nn.Module, X: np.ndarray, device_obj: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(X), batch_size=4096, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(device_obj)).detach().cpu().numpy())
    return np.concatenate(preds)


def residual_correction(base_pred_df: pd.DataFrame):
    train_resid = base_pred_df[base_pred_df["split"] == "val"].copy()
    test = base_pred_df[base_pred_df["split"] == "test"].copy()
    feature_cols = ["Prediction", "last_ndvi", "last_ewma_3", "last_diff_1"]
    train_resid = train_resid.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)
    residual_y = train_resid["NDVI_Target"] - train_resid["Prediction"]
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=0,
    )
    model.fit(train_resid[feature_cols], residual_y)
    corrected = test["Prediction"].to_numpy() + model.predict(test[feature_cols])
    corrected = np.round(np.clip(corrected, 0, 1), 4)
    out = test.copy()
    out["Prediction"] = corrected
    row = {"模型": "EWMA-LSTM + 残差校正", "数据部分": "test", "time_s": np.nan}
    row.update(metric_dict(out["NDVI_Target"].to_numpy(), corrected))
    return out, row


def plot_summary(summary: pd.DataFrame) -> None:
    setup_style()
    test = summary[summary["数据部分"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["RMSE", "SMAPE"]):
        ax.bar(test["模型"], test[metric], color="#3F7CAC", edgecolor="#2E3740", linewidth=0.5)
        ax.set_title(f"LSTM改造消融：{metric}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.22)
        for i, value in enumerate(test[metric]):
            ax.text(i, value + max(test[metric]) * 0.02, f"{value:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lstm_feature_residual_ablation.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "lstm_feature_residual_ablation.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    rows = []

    X_base, y_base, meta_base = build_arrays(df, seq_len=24, feature_cols=["NDVI_Target"])
    _, pred_base, base_rows = train_lstm_variant("基础LSTM", X_base, y_base, meta_base, device="cuda")
    rows.extend(base_rows)
    pred_base[pred_base["split"] == "test"].to_csv(OUT_DIR / "predictions_lstm_base.csv", index=False)

    X_ewma, y_ewma, meta_ewma = build_arrays(df, seq_len=24, feature_cols=["NDVI_Target", "ewma_3", "diff_1"])
    _, pred_ewma, ewma_rows = train_lstm_variant("EWMA特征增强LSTM", X_ewma, y_ewma, meta_ewma, device="cuda")
    rows.extend(ewma_rows)
    pred_ewma[pred_ewma["split"] == "test"].to_csv(OUT_DIR / "predictions_lstm_ewma.csv", index=False)

    pred_rc, rc_row = residual_correction(pred_ewma)
    rows.append(rc_row)
    pred_rc.to_csv(OUT_DIR / "predictions_lstm_ewma_residual_corrected.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "lstm_feature_residual_ablation_metrics.csv", index=False)
    paper = summary[summary["数据部分"] == "test"].copy()
    paper.insert(0, "排名", paper["RMSE"].rank(method="first").astype(int))
    paper = paper.sort_values("排名")
    paper.to_csv(OUT_DIR / "lstm_feature_residual_ablation_for_paper.csv", index=False)
    plot_summary(summary)
    print(summary.to_string(index=False))
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
