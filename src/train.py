from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data.time_series import (
    inverse_transform,
    load_series_from_csv,
    load_series_and_time,
    make_dataloaders,
    make_dataloaders_with_features,
    split_series,
    standardize_train_val_test,
    standardize_features_train_val_test,
    build_time_features,
)
from .models.factory import build_model
from .utils.metrics import compute_all_metrics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str) -> torch.device:
    alias = pref.lower()
    if alias == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(alias)


def create_optimizer(parameters, cfg: dict) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adam")).lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def create_criterion(name: str, huber_beta: float = 1.0) -> nn.Module:
    alias = name.lower()
    if alias in {"mse", "mse_loss"}:
        return nn.MSELoss()
    if alias in {"mae", "l1", "l1_loss"}:
        return nn.L1Loss()
    if alias in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss(beta=huber_beta)
    raise ValueError(f"Unknown loss: {name}")


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    patience: int
    min_delta: float
    optimizer: Dict[str, Any]
    loss: str
    early_stopping_metric: str
    debug: bool
    debug_batches: int
    huber_beta: float


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    train_cfg: TrainConfig,
) -> Tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = create_optimizer(model.parameters(), train_cfg.optimizer)
    criterion = create_criterion(train_cfg.loss, huber_beta=train_cfg.huber_beta)

    best_state: Dict[str, Tensor] | None = None
    best_val = float("inf")
    no_improve_epochs = 0

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_mae": [], "best_epoch": None}

    if train_cfg.debug:
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"[DEBUG] Starting training | params={total_params} | "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
        )

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            y_seq, _ = model(xb)
            y_pred = y_seq[:, -1, :]
            loss = criterion(y_pred, yb)
            loss.backward()
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * xb.size(0)
            train_count += xb.size(0)

            if train_cfg.debug and batch_idx < train_cfg.debug_batches:
                lr = optimizer.param_groups[0].get("lr", None)
                pred_mean = float(y_pred.detach().mean().cpu())
                true_mean = float(yb.detach().mean().cpu())
                print(
                    f"[DEBUG][train] epoch={epoch} batch={batch_idx+1}/{len(train_loader)} "
                    f"xb={tuple(xb.shape)} yb={tuple(yb.shape)} loss={float(loss.detach().cpu()):.6f} "
                    f"grad_norm={float(pre_clip_norm):.4f} lr={lr} pred_mean={pred_mean:.4f} true_mean={true_mean:.4f}"
                )

        train_epoch_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_mse_sum = 0.0
        val_mae_sum = 0.0
        with torch.no_grad():
            for vbatch_idx, (xb, yb) in enumerate(val_loader):
                xb = xb.to(device)
                yb = yb.to(device)
                y_seq, _ = model(xb)
                y_pred = y_seq[:, -1, :]
                loss = criterion(y_pred, yb)
                val_loss_sum += float(loss.detach().cpu()) * xb.size(0)
                val_count += xb.size(0)
                # compute standard metrics irrespective of training loss
                val_mse_sum += float(torch.mean((y_pred - yb) ** 2).detach().cpu()) * xb.size(0)
                val_mae_sum += float(torch.mean(torch.abs(y_pred - yb)).detach().cpu()) * xb.size(0)
                if train_cfg.debug and vbatch_idx < train_cfg.debug_batches:
                    pred_mean = float(y_pred.detach().mean().cpu())
                    true_mean = float(yb.detach().mean().cpu())
                    print(
                        f"[DEBUG][val]   epoch={epoch} batch={vbatch_idx+1}/{len(val_loader)} "
                        f"xb={tuple(xb.shape)} yb={tuple(yb.shape)} loss={float(loss.detach().cpu()):.6f} "
                        f"pred_mean={pred_mean:.4f} true_mean={true_mean:.4f}"
                    )

        val_epoch_loss = val_loss_sum / max(1, val_count)
        val_epoch_mse = val_mse_sum / max(1, val_count)
        val_epoch_mae = val_mae_sum / max(1, val_count)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["val_mse"].append(val_epoch_mse)
        history["val_mae"].append(val_epoch_mae)

        # select monitored metric
        metric_name = train_cfg.early_stopping_metric.lower()
        if metric_name == "rmse":
            metric_name = "mse"  # RMSE is monotonic with MSE
        monitored_val = (
            val_epoch_mse if metric_name == "mse" else val_epoch_mae if metric_name == "mae" else val_epoch_loss
        )
        improved = monitored_val < (best_val - train_cfg.min_delta)
        if improved:
            best_val = monitored_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_epoch_loss:.6f} val_loss={val_epoch_loss:.6f} "
            f"val_mse={val_epoch_mse:.6f} val_mae={val_epoch_mae:.6f} "
            f"monitor={metric_name}:{monitored_val:.6f} best={best_val:.6f} patience={no_improve_epochs}/{train_cfg.patience}"
        )

        if no_improve_epochs >= train_cfg.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            if history["best_epoch"] is None:
                history["best_epoch"] = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        # best_epoch: first index where val equals best_val
        try:
            best_idx = int(np.argmin(np.array(history["val_loss"])) + 1)
            history["best_epoch"] = best_idx
        except Exception:
            if history["best_epoch"] is None:
                history["best_epoch"] = None
    return model, history


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_seq, _ = model(xb)
            y_pred = y_seq[:, -1, :]
            mse_sum += float(torch.mean((y_pred - yb) ** 2).detach().cpu()) * xb.size(0)
            mae_sum += float(torch.mean(torch.abs(y_pred - yb)).detach().cpu()) * xb.size(0)
            count += xb.size(0)
    return {
        "mse": mse_sum / max(1, count),
        "mae": mae_sum / max(1, count),
    }


def generate_walk_forward_folds(
    values: np.ndarray,
    k_folds: int,
    *,
    seq_len: int,
    horizon: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k walk-forward (expanding origin) folds.

    For fold i, training uses values[:train_end_i] and validation uses the
    subsequent block values[train_end_i:val_end_i]. Ensures each side has
    at least (seq_len + horizon) points to form windows.
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    n = int(values.shape[0])
    min_block = seq_len + horizon
    if n < 2 * min_block:
        return folds

    # Choose a validation block size and step to distribute k folds
    # Keep validation block reasonably sized while ensuring feasibility
    val_size = max(min_block, (n - min_block) // (k_folds + 1))
    remaining_for_steps = max(0, n - (min_block + val_size))
    step = max(1, remaining_for_steps // max(1, k_folds - 1)) if k_folds > 1 else 1

    for i in range(k_folds):
        train_end = min_block + i * step
        val_start = train_end
        val_end = min(n, val_start + val_size)

        train_values = values[:train_end]
        val_values = values[val_start:val_end]

        if train_values.shape[0] >= min_block and val_values.shape[0] >= min_block:
            folds.append((train_values, val_values))

        if val_end >= n:
            break

    return folds


def generate_sliding_blocked_folds(
    values: np.ndarray,
    k_folds: int,
    *,
    seq_len: int,
    horizon: int,
    train_window: int,
    val_window: int,
    step: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k fixed-size (pure blocked, sliding) folds.

    For each fold i, use a fixed training window of length train_window
    immediately followed by a fixed validation window of length val_window.
    The window start slides by `step` (default chosen to fit ~k folds).
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    n = int(values.shape[0])
    min_block = seq_len + horizon
    if train_window < min_block or val_window < min_block:
        return folds
    total_needed = train_window + val_window
    if n < total_needed:
        return folds

    if step is None:
        step = max(1, (n - total_needed) // max(1, k_folds - 1)) if k_folds > 1 else 1

    start = 0
    while len(folds) < k_folds and (start + total_needed) <= n:
        tr = values[start : start + train_window]
        va = values[start + train_window : start + total_needed]
        if tr.shape[0] >= min_block and va.shape[0] >= min_block:
            folds.append((tr, va))
        start += step

    return folds


def generate_cv_folds(
    values: np.ndarray,
    cv_cfg: dict,
    *,
    seq_len: int,
    horizon: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Expanding-origin blocked CV (original behavior)
    k_folds = int(cv_cfg.get("k_folds", 0))
    return generate_walk_forward_folds(values, k_folds=k_folds, seq_len=seq_len, horizon=horizon)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNNs for time series forecasting from config")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--model", type=str, default="", help="filter by model type or name (elman|jordan|multi or a model name)")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(str(cfg.get("device", "auto")))
    print(f"Using device: {device}")

    dataset_cfg = cfg.get("dataset", {})
    data_path = Path(dataset_cfg.get("path", "")).expanduser().resolve()
    value_column = str(dataset_cfg.get("value_column"))
    date_column = dataset_cfg.get("date_column")
    seq_len = int(dataset_cfg.get("seq_len", 24))
    horizon = int(dataset_cfg.get("horizon", 1))
    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    scaler_method = str(dataset_cfg.get("scaler", "standard"))

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Optional cyclic features
    features_cfg = cfg.get("features", {})
    feat_enabled = bool(features_cfg.get("enabled", False))
    feat_scaler_method = str(features_cfg.get("scaler", "standard"))
    cyclic_cfg = dict(features_cfg.get("cyclic", {}))

    if feat_enabled and date_column:
        values, dt_index = load_series_and_time(str(data_path), value_column=value_column, date_column=str(date_column))
        all_features = build_time_features(dt_index, cyclic=cyclic_cfg)
    else:
        values = load_series_from_csv(str(data_path), value_column=value_column, date_column=date_column)
        all_features = None
    train, val, test = split_series(values, train_ratio=train_ratio, val_ratio=val_ratio)
    if all_features is not None:
        n = values.shape[0]
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        feat_train = all_features[:n_train]
        feat_val = all_features[n_train : n_train + n_val]
        feat_test = all_features[n_train + n_val :]
    else:
        feat_train = feat_val = feat_test = None

    # Ensure each split has enough points to create at least one window
    # (len >= seq_len + horizon). If not, rebalance splits conservatively.
    min_block = seq_len + horizon
    n = values.shape[0]
    if any(arr.shape[0] < min_block for arr in (val, test)):
        n_train = train.shape[0]
        n_val = val.shape[0]
        n_test = test.shape[0]

        # Helper to pull points from another segment while keeping its minimum
        def pull(from_size: int, to_size: int, need: int, keep_min: int) -> tuple[int, int, int]:
            if need <= 0:
                return from_size, to_size, 0
            available = max(0, from_size - keep_min)
            take = min(need, available)
            return from_size - take, to_size + take, need - take

        # First ensure validation has at least min_block by taking from train, then test
        deficit = max(0, min_block - n_val)
        n_train, n_val, deficit = pull(n_train, n_val, deficit, keep_min=min_block)
        n_test, n_val, deficit = pull(n_test, n_val, deficit, keep_min=max(1, min_block))

        # Then ensure test has at least min_block by taking from train, then val
        deficit = max(0, min_block - n_test)
        n_train, n_test, deficit = pull(n_train, n_test, deficit, keep_min=min_block)
        n_val, n_test, deficit = pull(n_val, n_test, deficit, keep_min=min_block)

        # Final sanity: clamp to totals and non-negative
        total = max(0, n)
        n_train = max(0, min(n_train, total))
        n_val = max(0, min(n_val, total - n_train))
        n_test = max(0, total - n_train - n_val)

        # Re-slice using adjusted lengths
        train = values[:n_train]
        val = values[n_train : n_train + n_val]
        test = values[n_train + n_val : n_train + n_val + n_test]

        print(
            f"[DEBUG] adjusted splits for windows | train={train.shape[0]} val={val.shape[0]} test={test.shape[0]} "
            f"(min_block={min_block})"
        )

    training_cfg = cfg.get("training", {})
    debug_flag = bool(cfg.get("debug", False))
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    train_cfg = TrainConfig(
        batch_size=batch_size,
        epochs=int(training_cfg.get("epochs", 50)),
        patience=int(training_cfg.get("patience", 5)),
        min_delta=float(training_cfg.get("min_delta", 1e-4)),
        optimizer=dict(training_cfg.get("optimizer", {"name": "adam", "lr": 1e-3})),
        loss=str(training_cfg.get("loss", "mse")),
        early_stopping_metric=str(training_cfg.get("early_stopping_metric", "mse")),
        debug=debug_flag,
        debug_batches=int(training_cfg.get("debug_batches", 1)),
        huber_beta=float(training_cfg.get("huber_beta", 1.0)),
    )

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("No models specified in config under 'models'")

    # Optional filtering by model type or name
    if args.model:
        selector = args.model.strip().lower()
        before = len(models_cfg)
        models_cfg = [m for m in models_cfg if str(m.get("type", "")).lower() == selector or str(m.get("name", "")).lower() == selector]
        print(f"Filtering models by '{selector}': {before} -> {len(models_cfg)}")
        if not models_cfg:
            raise ValueError(f"No models match selector '{selector}'. Available: {[m.get('name') for m in cfg.get('models', [])]}")

    input_size = 1
    output_size = horizon

    results_dir = Path(cfg.get("results_dir", Path(__file__).resolve().parents[1] / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    cv_cfg = cfg.get("cv", {})
    cv_enabled = bool(cv_cfg.get("enabled", False))
    k_folds = int(cv_cfg.get("k_folds", 0))

    all_results = []
    if cv_enabled and k_folds > 0:
        # Combine base train+val for CV; keep test separate
        trainval = np.concatenate([train, val]) if val.size > 0 else train

        folds = generate_cv_folds(trainval, cv_cfg, seq_len=seq_len, horizon=horizon)
        if not folds:
            raise ValueError("Insufficient data to create CV folds. Check cv.mode/windows or reduce k_folds / seq_len / horizon.")

        for model_cfg in models_cfg:
            name = str(model_cfg.get("name", f"{model_cfg.get('type','model')}-{int(time.time())}"))
            print(f"\n=== CV Training model: {name} ({len(folds)} folds) ===")

            fold_summaries = []
            for i, (train_fold, val_fold) in enumerate(folds, start=1):
                # Scale per fold using only train_fold statistics
                train_s, val_s, _unused, scaler = standardize_train_val_test(
                    train_fold, val_fold, val_fold, method=scaler_method
                )
                train_loader, val_loader, _ignore = make_dataloaders(
                    train_s, val_s, val_s, seq_len=seq_len, horizon=horizon, batch_size=batch_size, num_workers=num_workers
                )

                if train_cfg.debug:
                    n_tr = train_loader.dataset.tensors[0].shape[0]
                    n_va = val_loader.dataset.tensors[0].shape[0]
                    print(
                        f"[DEBUG] fold={i} windows: train={n_tr} val={n_va} "
                        f"seq_len={seq_len} horizon={horizon}"
                    )

                model = build_model(model_cfg, input_size=input_size, output_size=output_size)
                model, history = train_one_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    train_cfg=train_cfg,
                )

                # Metrics on validation (scaled)
                val_scaled = evaluate_model(model, val_loader, device=device)

                # Metrics on validation (original scale)
                model.eval()
                v_true_list, v_pred_list = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        y_seq, _ = model(xb.to(device))
                        y_pred = y_seq[:, -1, :].detach().cpu().numpy()
                        v_true_list.append(yb.numpy())
                        v_pred_list.append(y_pred)
                v_true = np.concatenate(v_true_list, axis=0)
                v_pred = np.concatenate(v_pred_list, axis=0)
                v_true_inv = inverse_transform(v_true, scaler)
                v_pred_inv = inverse_transform(v_pred, scaler)
                val_orig = {
                    "mse": float(np.mean((v_pred_inv - v_true_inv) ** 2)),
                    "mae": float(np.mean(np.abs(v_pred_inv - v_true_inv))),
                }

                fold_summary = {
                    "fold": i,
                    "val_scaled": val_scaled,
                    "val_original": val_orig,
                    "history": history,
                }
                print(
                    f"Fold {i}/{len(folds)} | Val(orig): MSE={val_orig['mse']:.6f} MAE={val_orig['mae']:.6f}"
                )
                fold_summaries.append(fold_summary)

            # Aggregate metrics across folds (original scale)
            def agg(key_path: List[str]) -> Tuple[float, float]:
                vals = []
                for fs in fold_summaries:
                    d = fs
                    for k in key_path:
                        d = d[k]
                    vals.append(float(d))
                arr = np.array(vals, dtype=np.float64)
                return float(arr.mean()), float(arr.std(ddof=0))

            val_mse_mean, val_mse_std = agg(["val_original", "mse"])
            val_mae_mean, val_mae_std = agg(["val_original", "mae"])

            summary = {
                "name": name,
                "type": model_cfg.get("type"),
                "hidden_size": model_cfg.get("hidden_size"),
                "dropout": model_cfg.get("dropout", 0.0),
                "activation": model_cfg.get("activation", "tanh"),
                "cv": {
                    "k_folds": len(fold_summaries),
                    "folds": fold_summaries,
                    "aggregate_original": {
                        "val": {"mse_mean": val_mse_mean, "mse_std": val_mse_std, "mae_mean": val_mae_mean, "mae_std": val_mae_std}
                    }
                }
            }

            print(
                f"Aggregate (orig) | Val: MSE={val_mse_mean:.6f}±{val_mse_std:.6f} "
                f"MAE={val_mae_mean:.6f}±{val_mae_std:.6f}"
            )

            all_results.append(summary)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = results_dir / f"{name}-cv-{timestamp}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
    else:
        # Single split training (original behavior)
        train_s, val_s, test_s, scaler = standardize_train_val_test(train, val, test, method=scaler_method)
        if feat_enabled and feat_train is not None:
            ftr_train_s, ftr_val_s, ftr_test_s, _f_scaler = standardize_features_train_val_test(
                feat_train, feat_val, feat_test, method=feat_scaler_method
            )
            train_loader, val_loader, test_loader = make_dataloaders_with_features(
                train_s, val_s, test_s, ftr_train_s, ftr_val_s, ftr_test_s, seq_len=seq_len, horizon=horizon, batch_size=batch_size, num_workers=num_workers
            )
        else:
            train_loader, val_loader, test_loader = make_dataloaders(
                train_s, val_s, test_s, seq_len=seq_len, horizon=horizon, batch_size=batch_size, num_workers=num_workers
            )

        for model_cfg in models_cfg:
            name = str(model_cfg.get("name", f"{model_cfg.get('type','model')}-{int(time.time())}"))
            print(f"\n=== Training model: {name} ===")

            model = build_model(model_cfg, input_size=input_size, output_size=output_size)
            if train_cfg.debug:
                n_tr = train_loader.dataset.tensors[0].shape[0]
                n_va = val_loader.dataset.tensors[0].shape[0]
                n_te = test_loader.dataset.tensors[0].shape[0]
                print(
                    f"[DEBUG] windows: train={n_tr} val={n_va} test={n_te} "
                    f"seq_len={seq_len} horizon={horizon}"
                )
            model, history = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                train_cfg=train_cfg,
            )

            # Evaluate on scaled values (test). If no test windows, skip with NaNs.
            if len(test_loader) > 0:
                scaled_metrics = evaluate_model(model, test_loader, device=device)
                # Evaluate on original scale (test)
                model.eval()
                y_true_list = []
                y_pred_list = []
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
                mse_orig = float(np.mean((y_pred_inv - y_true_inv) ** 2))
                mae_orig = float(np.mean(np.abs(y_pred_inv - y_true_inv)))
            else:
                print("[WARN] No test windows available; skipping test evaluation.")
                scaled_metrics = {"mse": float("nan"), "mae": float("nan")}
                mse_orig = float("nan")
                mae_orig = float("nan")

            summary = {
                "name": name,
                "type": model_cfg.get("type"),
                "hidden_size": model_cfg.get("hidden_size"),
                "dropout": model_cfg.get("dropout", 0.0),
                "activation": model_cfg.get("activation", "tanh"),
                "metrics_scaled": scaled_metrics,
                "metrics_original": {"mse": mse_orig, "mae": mae_orig},
                "history": history,
            }

            print(
                f"Model {name} | Test (orig scale): MSE={mse_orig:.6f} MAE={mae_orig:.6f} | "
                f"Scaled: MSE={scaled_metrics['mse']:.6f} MAE={scaled_metrics['mae']:.6f}"
            )

            all_results.append(summary)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = results_dir / f"{name}-{timestamp}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    aggregate_path = results_dir / f"all_results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to: {aggregate_path}")


if __name__ == "__main__":
    main()


