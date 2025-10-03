from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch

from .data.time_series import (
    load_series_from_csv,
    load_series_and_time,
    split_series,
    standardize_train_val_test,
    make_dataloaders,
    inverse_transform,
)
from .models.factory import build_model
from .train import get_device, set_seed, generate_walk_forward_folds
from .utils.metrics import compute_all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit best HPO params and evaluate on final test with fixed epochs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--study-summary", type=str, required=True, help="path to results/hpo/study-summary-*.json")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    summary_path = Path(args.study_summary).expanduser().resolve()
    with open(summary_path, "r", encoding="utf-8") as f:
        study = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(str(cfg.get("device", "auto")))

    dataset = cfg.get("dataset", {})
    data_path = Path(dataset.get("path", "")).expanduser().resolve()
    value_column = str(dataset.get("value_column"))
    date_column = dataset.get("date_column")
    seq_len = int(dataset.get("seq_len", 24))
    horizon = int(dataset.get("horizon", 1))
    train_ratio = float(dataset.get("train_ratio", 0.7))
    val_ratio = float(dataset.get("val_ratio", 0.15))
    scaler_method = str(dataset.get("scaler", "standard"))

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Load full series and optional timestamps
    if date_column:
        values, dt_index = load_series_and_time(str(data_path), value_column=value_column, date_column=str(date_column))
    else:
        values = load_series_from_csv(str(data_path), value_column=value_column, date_column=date_column)
        dt_index = None
    train, val, test = split_series(values, train_ratio=train_ratio, val_ratio=val_ratio)
    trainval = np.concatenate([train, val]) if val.size > 0 else train

    # One-line data range log for transparency
    tv_start_idx = 0
    tv_end_idx = train.shape[0] + val.shape[0] - 1
    test_start_idx = tv_end_idx + 1
    test_end_idx = values.shape[0] - 1
    def fmt_idx(i: int) -> str:
        if dt_index is not None and 0 <= i < len(dt_index):
            return str(dt_index[i].date())
        return str(i)
    print(
        f"[INFO] data ranges | trainval: {fmt_idx(tv_start_idx)} -> {fmt_idx(tv_end_idx)} (N={trainval.shape[0]}) | "
        f"test: {fmt_idx(test_start_idx)} -> {fmt_idx(test_end_idx)} (N={test.shape[0]})"
    )

    # Determine fixed epochs from study
    best_params = study.get("best_params", {})
    avg_best_epoch = study.get("avg_best_epoch")
    if avg_best_epoch is None:
        raise ValueError("avg_best_epoch not found in study summary; re-run HPO to capture per-fold best epochs.")
    avg_best_epoch = float(avg_best_epoch)

    # Build model config from best params
    model_type = study.get("model_type", cfg.get("models", [{}])[0].get("type", "elman"))
    hidden_size = int(best_params.get("hidden_size", cfg.get("models", [{}])[0].get("hidden_size", 32)))
    dropout = float(best_params.get("dropout", cfg.get("models", [{}])[0].get("dropout", 0.0)))
    lr = float(best_params.get("lr", cfg.get("training", {}).get("optimizer", {}).get("lr", 1e-3)))

    model_cfg = {"type": model_type, "hidden_size": hidden_size, "dropout": dropout, "activation": cfg.get("models", [{}])[0].get("activation", "tanh")}

    # Derive refit epochs by matching total optimizer updates to a typical CV fold
    cv_cfg = cfg.get("cv", {})
    k_folds = int(cv_cfg.get("k_folds", 5))
    folds_for_lengths = generate_walk_forward_folds(trainval, k_folds=k_folds, seq_len=seq_len, horizon=horizon)
    if not folds_for_lengths:
        raise ValueError("Unable to construct CV folds to estimate update budget; check seq_len/horizon/k_folds.")
    fold_train_lengths = [tr.shape[0] for tr, _ in folds_for_lengths]
    avg_fold_train_len = float(np.mean(fold_train_lengths))
    batch_size = int(cfg.get("training", {}).get("batch_size", 64))
    batches_per_epoch_cv = int(np.ceil(avg_fold_train_len / batch_size))
    steps_cv = avg_best_epoch * batches_per_epoch_cv
    batches_per_epoch_refit = int(np.ceil(trainval.shape[0] / batch_size))
    fixed_epochs = max(1, int(round(steps_cv / max(1, batches_per_epoch_refit))))

    # Scale on trainval only, evaluate on test
    train_s, _val_unused, test_s, scaler = standardize_train_val_test(trainval, trainval[-horizon - seq_len - 1 :], test, method=scaler_method)
    train_loader, _unused, test_loader = make_dataloaders(
        train_s, train_s[:1], test_s, seq_len=seq_len, horizon=horizon, batch_size=int(cfg.get("training", {}).get("batch_size", 64)), num_workers=int(cfg.get("num_workers", 0))
    )

    # Build and train for fixed epochs without early stopping
    from torch import nn

    model = build_model(model_cfg, input_size=1, output_size=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(cfg.get("training", {}).get("optimizer", {}).get("weight_decay", 0.0)))
    loss_name = str(cfg.get("training", {}).get("loss", "mse")).lower()
    if loss_name in {"huber", "smooth_l1"}:
        criterion = nn.SmoothL1Loss(beta=float(cfg.get("training", {}).get("huber_beta", 1.0)))
    elif loss_name in {"mae", "l1", "l1_loss"}:
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    train_losses: list[float] = []
    for epoch in range(1, fixed_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            y_seq, _ = model(xb)
            y_pred = y_seq[:, -1, :]
            loss = criterion(y_pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss_sum += float(loss.detach().cpu()) * xb.size(0)
            epoch_count += xb.size(0)
        train_losses.append(epoch_loss_sum / max(1, epoch_count))

    # Evaluate on original scale (guard when no windows)
    model.eval()
    overlay_time: list = []
    overlay_true: list = []
    overlay_pred: list = []
    if len(test_loader) > 0:
        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                y_seq, _ = model(xb.to(device))
                y_pred = y_seq[:, -1, :].detach().cpu().numpy()
                y_true_list.append(yb.numpy())
                y_pred_list.append(y_pred)
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true_inv = inverse_transform(y_true, scaler)
        y_pred_inv = inverse_transform(y_pred, scaler)

        # Metrics (original scale)
        metrics = compute_all_metrics(y_true_inv, y_pred_inv, train_series_for_mase=trainval, mase_seasonality=int(cfg.get("metrics", {}).get("mase_seasonality", 1)))
        metrics["mse"] = float(np.mean((y_pred_inv - y_true_inv) ** 2))

        # Overlay data (absolute index or timestamp)
        N, H = y_true_inv.shape
        start_abs = train.shape[0] + val.shape[0] + seq_len
        for i in range(N):
            for h in range(H):
                abs_idx = start_abs + i + h
                t_val = str(dt_index[abs_idx]) if dt_index is not None and abs_idx < len(dt_index) else abs_idx
                overlay_time.append(t_val)
                overlay_true.append(float(y_true_inv[i, h]))
                overlay_pred.append(float(y_pred_inv[i, h]))
    else:
        print("[WARN] No test windows available in refit; writing NaN metrics and empty overlay.")
        metrics = {"rmse": float("nan"), "mae": float("nan"), "smape": float("nan"), "mase": float("nan"), "mse": float("nan")}

    # Save
    out_dir = Path(cfg.get("results_dir", cfg_path.parent / "results")) / "refit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model_type": model_type,
        "params": {"hidden_size": hidden_size, "dropout": dropout, "lr": lr, "seq_len": seq_len, "horizon": horizon},
        "fixed_epochs": fixed_epochs,
        "refit_epoch_strategy": {
            "mode": "steps_matched",
            "avg_best_epoch": avg_best_epoch,
            "avg_fold_train_len": avg_fold_train_len,
            "batches_per_epoch_cv": batches_per_epoch_cv,
            "batches_per_epoch_refit": batches_per_epoch_refit,
            "estimated_steps_cv": steps_cv
        },
        "metrics_original": metrics,
        "loss_curve": {"train": train_losses},
        "overlay": {"time": overlay_time, "y_true": overlay_true, "y_pred": overlay_pred},
    }
    out_path = out_dir / f"refit-{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved refit summary to: {out_path}")


if __name__ == "__main__":
    main()


