from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
AUTOFORMER_DIR = PROJECT_DIR / "external" / "Autoformer"
sys.path.insert(0, str(AUTOFORMER_DIR))

from models.Autoformer import Model as OfficialAutoformer  # noqa: E402
from models.Informer import Model as OfficialInformer  # noqa: E402
from models.Transformer import Model as OfficialTransformer  # noqa: E402


TARGET_FILE = BASE_DIR / "05_final_outputs" / "reconstruction" / "final_ndvi_kalman_whittaker.csv"
OUT_DIR = BASE_DIR / "05_final_outputs" / "chapter4_materials" / "official_autoformer_unified_split"

TRAIN_START = pd.Timestamp("1985-01-01")
TRAIN_END = pd.Timestamp("2014-12-31")
VAL_START = pd.Timestamp("2015-01-01")
VAL_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2025-05-16")


def make_time_mark(dates: np.ndarray) -> np.ndarray:
    ds = pd.to_datetime(dates)
    return np.stack(
        [
            ds.month.to_numpy(),
            ds.day.to_numpy(),
            ds.weekday.to_numpy(),
            np.zeros(len(ds), dtype=np.int64),
        ],
        axis=1,
    ).astype(np.float32)


class PointWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, seq_len: int, label_len: int):
        self.seq_len = seq_len
        self.label_len = label_len
        bounds = {
            "train": (TRAIN_START, TRAIN_END),
            "val": (VAL_START, VAL_END),
            "test": (TEST_START, TEST_END),
        }
        start, end = bounds[split]
        items = []
        self.series = {}
        for point, g in df.groupby("point", sort=False):
            values = g["NDVI_Target"].to_numpy(dtype=np.float32)
            dates = g["date"].to_numpy()
            marks = make_time_mark(dates)
            date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d").to_numpy()
            self.series[point] = (dates, date_strings, values, marks)
            for i in range(seq_len, len(g)):
                d = pd.Timestamp(dates[i])
                if start <= d <= end:
                    items.append((point, i))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        point, i = self.items[idx]
        _, date_strings, values, marks = self.series[point]
        s0, s1 = i - self.seq_len, i
        l0 = i - self.label_len
        enc = values[s0:s1, None]
        dec_zeros = np.zeros((self.label_len + 1, 1), dtype=np.float32)
        enc_mark = marks[s0:s1]
        dec_mark = marks[l0 : i + 1]
        y = np.asarray(values[i], dtype=np.float32)
        return (
            torch.from_numpy(enc),
            torch.from_numpy(enc_mark),
            torch.from_numpy(dec_zeros),
            torch.from_numpy(dec_mark),
            torch.tensor(y),
            point,
            date_strings[i],
        )


def collate_batch(batch):
    x_enc, x_mark, x_dec, x_dec_mark, y, point, date = zip(*batch)
    return (
        torch.stack(x_enc),
        torch.stack(x_mark),
        torch.stack(x_dec),
        torch.stack(x_dec_mark),
        torch.stack(y),
        list(point),
        list(date),
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.zeros_like(denom, dtype=np.float64)
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(out.mean() * 100)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "SMAPE": smape(y_true, y_pred),
    }


def build_model(args) -> nn.Module:
    config = SimpleNamespace(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=1,
        output_attention=False,
        moving_avg=args.moving_avg,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        dropout=args.dropout,
        embed="fixed",
        freq="h",
        activation="gelu",
        distil=True,
    )
    if args.model == "Autoformer":
        return OfficialAutoformer(config)
    if args.model == "Informer":
        return OfficialInformer(config)
    if args.model == "Transformer":
        return OfficialTransformer(config)
    raise ValueError(f"Unsupported model: {args.model}")


def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses, points, dates, ys, preds = [], [], [], [], []
    with torch.no_grad():
        for x_enc, x_mark, x_dec, x_dec_mark, y, point, date in loader:
            x_enc = x_enc.to(device)
            x_mark = x_mark.to(device)
            x_dec = x_dec.to(device)
            x_dec_mark = x_dec_mark.to(device)
            y = y.to(device)
            pred = model(x_enc, x_mark, x_dec, x_dec_mark)[:, -1, 0]
            losses.append(float(loss_fn(pred, y).detach().cpu()))
            points.extend(point)
            dates.extend(date)
            ys.extend(y.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())
    return float(np.mean(losses)), pd.DataFrame({"point": points, "date": dates, "NDVI_Target": ys, "Prediction": preds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["Autoformer", "Informer", "Transformer"], default="Autoformer")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--label-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--moving-avg", type=int, default=5)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    df = pd.read_csv(TARGET_FILE, parse_dates=["date"])
    df = df[["point", "date", "NDVI_Target"]].sort_values(["point", "date"]).reset_index(drop=True)

    train_ds = PointWindowDataset(df, "train", args.seq_len, args.label_len)
    val_ds = PointWindowDataset(df, "val", args.seq_len, args.label_len)
    test_ds = PointWindowDataset(df, "test", args.seq_len, args.label_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, collate_fn=collate_batch)

    print(f"Official {args.model} source: {AUTOFORMER_DIR}", flush=True)
    print(f"Samples train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    print(f"Device: {device}", flush=True)

    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_val, bad_epochs = None, float("inf"), 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_enc, x_mark, x_dec, x_dec_mark, y, _, _ in train_loader:
            x_enc = x_enc.to(device)
            x_mark = x_mark.to(device)
            x_dec = x_dec.to(device)
            x_dec_mark = x_dec_mark.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x_enc, x_mark, x_dec, x_dec_mark)[:, -1, 0]
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, _ = evaluate(model, val_loader, device, loss_fn)
        train_loss = float(np.mean(train_losses))
        print(f"epoch={epoch:02d} train_mse={train_loss:.8f} val_mse={val_loss:.8f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    _, pred_df = evaluate(model, test_loader, device, loss_fn)
    pred_df["Prediction"] = np.clip(pred_df["Prediction"].to_numpy(), 0, 1)
    pred_df["Prediction"] = np.round(pred_df["Prediction"], 4)
    model_key = args.model.lower()
    pred_df.to_csv(OUT_DIR / f"predictions_official_{model_key}.csv", index=False)

    y = pred_df["NDVI_Target"].to_numpy(dtype=float)
    pred = pred_df["Prediction"].to_numpy(dtype=float)
    row = {
        "模型": f"{args.model}官方实现",
        "模型类型": "深度时序模型",
        "样点数": int(pred_df["point"].nunique()),
        "测试样本数": int(len(pred_df)),
        "实验口径": f"THUML官方{args.model}模块；1985-2014训练，2015-2019验证，2020-2025测试",
        "time_s": round(time.time() - t0, 2),
        **metric_row(y, pred),
    }
    pd.DataFrame([row]).to_csv(OUT_DIR / f"official_{model_key}_metrics.csv", index=False)
    print(pd.DataFrame([row]).to_string(index=False), flush=True)
    print("Saved:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
