from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import optuna
import torch

from .data.time_series import (
    load_series_from_csv,
    load_series_and_time,
    split_series,
    standardize_train_val_test,
    make_dataloaders,
    make_dataloaders_with_features,
    standardize_features_train_val_test,
    build_time_features,
    inverse_transform,
)
from .models.factory import build_model
from .train import (
    generate_cv_folds,
    train_one_model,
    evaluate_model,
    TrainConfig,
    get_device,
    set_seed,
)
from .utils.metrics import compute_all_metrics


def objective(
    trial: optuna.Trial,
    *,
    cfg: dict,
    values: np.ndarray,
    device: torch.device,
    model_template: dict,
    results_dir: Path,
) -> float:
    dataset_cfg = cfg.get("dataset", {})
    horizon = int(dataset_cfg.get("horizon", 1))
    scaler_method = str(dataset_cfg.get("scaler", "standard"))
    cv_cfg = cfg.get("cv", {})
    k_folds = int(cv_cfg.get("k_folds", 5))

    training_cfg = cfg.get("training", {})
    base_batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 50))
    patience = int(training_cfg.get("patience", 5))
    min_delta = float(training_cfg.get("min_delta", 1e-4))
    loss_name = str(training_cfg.get("loss", "mse"))
    huber_beta = float(training_cfg.get("huber_beta", 1.0))

    # Search space
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Clamp seq_len to be reasonable versus dataset length
    max_seq = max(6, min(48, values.shape[0] // 4))
    seq_len = trial.suggest_int("seq_len", 6, max_seq)
    hidden_size = trial.suggest_int("hidden_size", 8, 128, step=8)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Prepare folds using train+val (walk-forward)
    # We create a fresh split to keep test unseen; HPO objective uses validation.
    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    train, val, test = split_series(values, train_ratio=train_ratio, val_ratio=val_ratio)
    trainval = np.concatenate([train, val]) if val.size > 0 else train
    if 'all_features' in locals() and all_features is not None:
        n_train = train.shape[0]
        n_val = val.shape[0]
        feat_train = all_features[:n_train]
        feat_val = all_features[n_train : n_train + n_val]
        trainval_features = np.concatenate([feat_train, feat_val], axis=0)
    else:
        trainval_features = None

    folds = generate_cv_folds(trainval, cv_cfg={**cv_cfg, "k_folds": k_folds}, seq_len=seq_len, horizon=horizon)
    if not folds:
        # Infeasible choice (e.g., seq_len too large) → prune
        raise optuna.TrialPruned("No feasible folds for chosen seq_len/horizon.")

    # Training config for this trial (override lr, keep others)
    train_cfg = TrainConfig(
        batch_size=base_batch_size,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        optimizer={"name": training_cfg.get("optimizer", {}).get("name", "adam"), "lr": lr, "weight_decay": training_cfg.get("optimizer", {}).get("weight_decay", 0.0)},
        loss=loss_name,
        early_stopping_metric=str(training_cfg.get("early_stopping_metric", "mse")),
        debug=False,
        debug_batches=int(training_cfg.get("debug_batches", 1)),
        huber_beta=huber_beta,
    )

    # Build per-fold, evaluate on validation (original scale)
    fold_summaries: List[dict] = []
    input_size = 1
    output_size = horizon

    for i, (train_fold, val_fold) in enumerate(folds, start=1):
        train_s, val_s, _test_unused, scaler = standardize_train_val_test(train_fold, val_fold, val_fold, method=scaler_method)
        if trainval_features is not None:
            tr_len = train_fold.shape[0]
            va_len = val_fold.shape[0]
            f_tr = trainval_features[:tr_len]
            f_va = trainval_features[tr_len : tr_len + va_len]
            f_tr_s, f_va_s, _f_unused, _f_scaler = standardize_features_train_val_test(
                f_tr, f_va, f_va, method=feat_scaler_method
            )
            train_loader, val_loader, _ = make_dataloaders_with_features(
                train_s, val_s, val_s, f_tr_s, f_va_s, f_va_s, seq_len=seq_len, horizon=horizon, batch_size=base_batch_size, num_workers=int(cfg.get("num_workers", 0))
            )
            input_size_fold = 1 + f_tr_s.shape[1]
        else:
            train_loader, val_loader, _ = make_dataloaders(
                train_s, val_s, val_s, seq_len=seq_len, horizon=horizon, batch_size=base_batch_size, num_workers=int(cfg.get("num_workers", 0))
            )
            input_size_fold = 1

        # Build model from template with updated hidden_size and dropout
        model_cfg = dict(model_template)
        model_cfg["hidden_size"] = hidden_size
        model_cfg["dropout"] = dropout
        model = build_model(model_cfg, input_size=input_size_fold, output_size=output_size)

        # Train and validate
        model, history = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            train_cfg=train_cfg,
        )

        # Validation metrics (scaled)
        val_scaled = evaluate_model(model, val_loader, device=device)

        # Validation metrics (original scale)
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
        # Additional metrics
        val_orig = {
            "mse": float(np.mean((v_pred_inv - v_true_inv) ** 2)),
            "mae": float(np.mean(np.abs(v_pred_inv - v_true_inv))),
        }
        # RMSE, sMAPE, MASE using train_fold as baseline for MASE
        extra = compute_all_metrics(v_true_inv, v_pred_inv, train_series_for_mase=train_fold, mase_seasonality=int(cfg.get("metrics", {}).get("mase_seasonality", 1)))
        val_orig.update(extra)

        # Training metrics (original scale at best checkpoint)
        tr_true_list, tr_pred_list = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                y_seq, _ = model(xb.to(device))
                y_pred = y_seq[:, -1, :].detach().cpu().numpy()
                tr_true_list.append(yb.numpy())
                tr_pred_list.append(y_pred)
        tr_true = np.concatenate(tr_true_list, axis=0)
        tr_pred = np.concatenate(tr_pred_list, axis=0)
        tr_true_inv = inverse_transform(tr_true, scaler)
        tr_pred_inv = inverse_transform(tr_pred, scaler)
        train_orig = {
            "mse": float(np.mean((tr_pred_inv - tr_true_inv) ** 2)),
            "mae": float(np.mean(np.abs(tr_pred_inv - tr_true_inv))),
        }
        train_orig.update(
            compute_all_metrics(tr_true_inv, tr_pred_inv, train_series_for_mase=train_fold, mase_seasonality=int(cfg.get("metrics", {}).get("mase_seasonality", 1)))
        )

        fold_summaries.append({
            "fold": i,
            "val_scaled": val_scaled,
            "val_original": val_orig,
            "train_original": train_orig,
            "history": history,
        })

    # Aggregate objective: mean validation MSE (original scale)
    val_mse_mean = float(np.mean([fs["val_original"]["mse"] for fs in fold_summaries]))
    val_mse_std = float(np.std([fs["val_original"]["mse"] for fs in fold_summaries], ddof=0))
    val_mae_list = [fs["val_original"]["mae"] for fs in fold_summaries]
    val_mae_mean = float(np.mean(val_mae_list))
    val_mae_std = float(np.std(val_mae_list, ddof=0))
    val_rmse_list = [fs["val_original"].get("rmse", np.sqrt(fs["val_original"]["mse"])) for fs in fold_summaries]
    val_rmse_mean = float(np.mean(val_rmse_list))
    val_rmse_std = float(np.std(val_rmse_list, ddof=0))
    train_rmse_mean = float(np.mean([fs["train_original"].get("rmse", np.sqrt(fs["train_original"]["mse"])) for fs in fold_summaries]))
    val_smape_mean = float(np.mean([fs["val_original"].get("smape", float("nan")) for fs in fold_summaries]))
    val_mase_mean = float(np.mean([fs["val_original"].get("mase", float("nan")) for fs in fold_summaries]))
    best_epochs = [fs["history"].get("best_epoch") for fs in fold_summaries if fs["history"].get("best_epoch") is not None]
    avg_best_epoch = float(np.mean(best_epochs)) if best_epochs else None

    # Persist trial summary
    trial_dir = results_dir / "hpo"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "trial_number": int(trial.number),
        "params": {
            "lr": lr,
            "seq_len": int(seq_len),
            "hidden_size": int(hidden_size),
            "dropout": float(dropout),
            "model_type": model_template.get("type"),
        },
        "objective": {"val_mse_mean": val_mse_mean, "val_rmse_mean": val_rmse_mean, "val_rmse_std": val_rmse_std, "val_mae_mean": val_mae_mean, "val_mae_std": val_mae_std, "val_smape_mean": val_smape_mean, "val_mase_mean": val_mase_mean, "train_rmse_mean": train_rmse_mean},
        "avg_best_epoch": avg_best_epoch,
        "folds": fold_summaries,
    }
    with open(trial_dir / f"trial-{trial.number}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Attach aggregates to trial for study summary
    trial.set_user_attr("val_rmse_mean", val_rmse_mean)
    trial.set_user_attr("val_rmse_std", val_rmse_std)
    trial.set_user_attr("train_rmse_mean", train_rmse_mean)
    trial.set_user_attr("avg_best_epoch", avg_best_epoch)
    trial.set_user_attr("val_mse_mean", val_mse_mean)
    trial.set_user_attr("val_mse_std", val_mse_std)
    trial.set_user_attr("val_mae_mean", val_mae_mean)
    trial.set_user_attr("val_mae_std", val_mae_std)

    # Optimize mean validation RMSE (original units)
    return val_rmse_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for RNNs with walk-forward CV")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--study", type=str, default="rnn_hpo")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--model", type=str, default="", help="override model type: elman|jordan|multi")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(str(cfg.get("device", "auto")))

    dataset_cfg = cfg.get("dataset", {})
    data_path = Path(dataset_cfg.get("path", "")).expanduser().resolve()
    value_column = str(dataset_cfg.get("value_column"))
    date_column = dataset_cfg.get("date_column")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Build base series and optional features
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

    # Choose a model template (type/activation) from config or CLI
    model_template = None
    override_sel = args.model.strip().lower()
    if override_sel:
        # Accept either a type (elman|jordan|multi) or a model name present in config
        if override_sel in {"elman", "jordan", "multi"}:
            model_template = {"type": override_sel, "hidden_size": 32, "activation": "tanh", "dropout": 0.0}
        else:
            models_cfg = cfg.get("models", [])
            matched = next((m for m in models_cfg if str(m.get("name", "")).lower() == override_sel), None)
            if matched is None:
                avail = {
                    "types": ["elman", "jordan", "multi"],
                    "names": [m.get("name") for m in models_cfg],
                }
                raise ValueError(f"Unknown model selector '{override_sel}'. Provide one of types {avail['types']} or a name from {avail['names']}")
            model_template = {
                "type": str(matched.get("type", "elman")),
                "hidden_size": int(matched.get("hidden_size", 32)),
                "activation": str(matched.get("activation", "tanh")),
                "dropout": float(matched.get("dropout", 0.0)),
            }
    else:
        models_cfg = cfg.get("models", [])
        if not models_cfg:
            model_template = {"type": "elman", "hidden_size": 32, "activation": "tanh", "dropout": 0.0}
        else:
            # Use the first model as template for type/activation defaults
            base = dict(models_cfg[0])
            model_template = {
                "type": base.get("type", "elman"),
                "hidden_size": int(base.get("hidden_size", 32)),
                "activation": base.get("activation", "tanh"),
                "dropout": float(base.get("dropout", 0.0)),
            }

    results_dir = Path(cfg.get("results_dir", config_path.parent / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up study
    if args.storage:
        study = optuna.create_study(direction="minimize", study_name=args.study, storage=args.storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", study_name=args.study)

    study.optimize(
        lambda tr: objective(tr, cfg=cfg, values=values, device=device, model_template=model_template, results_dir=results_dir),
        n_trials=int(args.trials),
        gc_after_trial=True,
    )

    # Save study best and trials
    best = study.best_trial
    # Pull user attrs for the best trial
    best_val_rmse = best.user_attrs.get("val_rmse_mean", float(best.value))
    best_val_rmse_std = best.user_attrs.get("val_rmse_std", None)
    best_train_rmse = best.user_attrs.get("train_rmse_mean", None)
    best_avg_best_epoch = best.user_attrs.get("avg_best_epoch", None)
    best_val_mse_mean = best.user_attrs.get("val_mse_mean", None)
    best_val_mse_std = best.user_attrs.get("val_mse_std", None)
    best_val_mae_mean = best.user_attrs.get("val_mae_mean", None)
    best_val_mae_std = best.user_attrs.get("val_mae_std", None)

    summary = {
        "study": args.study,
        "n_trials": len(study.trials),
        "best_mean_RMSE": float(best_val_rmse),
        "best_RMSE_std": float(best_val_rmse_std) if best_val_rmse_std is not None else None,
        "best_mean_RMSE_train": float(best_train_rmse) if best_train_rmse is not None else None,
        "best_params": best.params,
        "best_number": int(best.number),
        "model_type": model_template.get("type"),
        "avg_best_epoch": float(best_avg_best_epoch) if best_avg_best_epoch is not None else None,
        "validation_mse": {
            "mean": float(best_val_mse_mean) if best_val_mse_mean is not None else None,
            "std": float(best_val_mse_std) if best_val_mse_std is not None else None
        },
        "validation_mae": {
            "mean": float(best_val_mae_mean) if best_val_mae_mean is not None else None,
            "std": float(best_val_mae_std) if best_val_mae_std is not None else None
        },
        "dataset": {
            "path": str(data_path),
            "value_column": value_column,
            "date_column": date_column,
            "horizon": int(dataset_cfg.get("horizon", 1))
        }
    }
    out_path = results_dir / "hpo" / f"study-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved HPO summary to: {out_path}")


if __name__ == "__main__":
    main()


