#%% Imports
import os
import json
import math
from glob import glob

import pandas as pd


# Discover HPO study summaries under results_For_4_Sets
root = os.path.dirname(os.path.abspath(__file__))
results_root = os.path.join(root, "results_For_4_Sets")

summary_files = sorted(glob(os.path.join(results_root, "**", "hpo", "study-summary-*.json"), recursive=True))

# Parse CV summaries and related best-trial folds for train stats
cv_rows = []
for sp in summary_files:
    summary = None
    try:
        with open(sp, "r") as f:
            summary = json.load(f)
    except Exception:
        summary = None
    if not summary:
        continue

    # Dataset and model labels
    dataset_dir = os.path.basename(os.path.dirname(os.path.dirname(sp)))
    dataset_label = dataset_dir
    model_label = str(summary.get("model_type") or summary.get("model") or "unknown")

    # Best hyperparameters
    best_params = summary.get("best_params", {}) or {}
    bp_lr = best_params.get("lr")
    bp_seq_len = best_params.get("seq_len")
    bp_hidden = best_params.get("hidden_size")
    bp_dropout = best_params.get("dropout")

    # Validation metrics (means and stds)
    val_rmse_mean = summary.get("best_mean_RMSE")
    val_rmse_std = summary.get("best_RMSE_std")
    if val_rmse_mean is None:
        vm = (summary.get("validation_mse") or {}).get("mean")
        if isinstance(vm, (int, float)):
            try:
                val_rmse_mean = math.sqrt(float(vm))
            except Exception:
                val_rmse_mean = None
    if val_rmse_std is None:
        vms = (summary.get("validation_mse") or {}).get("std")
        if isinstance(vms, (int, float)):
            try:
                val_rmse_std = math.sqrt(float(vms))
            except Exception:
                val_rmse_std = None

    validation_mae = summary.get("validation_mae") or {}
    val_mae_mean = validation_mae.get("mean")
    val_mae_std = validation_mae.get("std")

    # Training metrics (mean may be provided; std usually not) — compute from best trial folds
    train_rmse_mean = summary.get("best_mean_RMSE_train")
    train_rmse_std = None
    train_mae_mean = None
    train_mae_std = None

    best_number = summary.get("best_number")
    trial_path = os.path.join(os.path.dirname(sp), f"trial-{best_number}.json") if best_number is not None else None
    if trial_path and os.path.exists(trial_path):
        trial = None
        try:
            with open(trial_path, "r") as f:
                trial = json.load(f)
        except Exception:
            trial = None
        if trial:
            folds = trial.get("folds") or []
            train_rmses = []
            train_maes = []
            for fold in folds:
                tr = (fold or {}).get("train_original") or {}
                rmse = tr.get("rmse")
                if rmse is None:
                    mse = tr.get("mse")
                    if isinstance(mse, (int, float)):
                        try:
                            rmse = math.sqrt(float(mse))
                        except Exception:
                            rmse = None
                if isinstance(rmse, (int, float)):
                    train_rmses.append(float(rmse))
                mae = tr.get("mae")
                if isinstance(mae, (int, float)):
                    train_maes.append(float(mae))
            if train_rmses:
                m = float(sum(train_rmses) / len(train_rmses))
                var = float(sum((x - m) ** 2 for x in train_rmses) / len(train_rmses))
                train_rmse_std = math.sqrt(var)
                if train_rmse_mean is None:
                    train_rmse_mean = m
            if train_maes:
                m = float(sum(train_maes) / len(train_maes))
                var = float(sum((x - m) ** 2 for x in train_maes) / len(train_maes))
                train_mae_mean = m
                train_mae_std = math.sqrt(var)

    cv_rows.append({
        "summary_path": sp,
        "dataset": dataset_label,
        "model": model_label,
        "best_params": {
            "lr": bp_lr,
            "seq_len": bp_seq_len,
            "hidden_size": bp_hidden,
            "dropout": bp_dropout,
        },
        "train_rmse_mean": train_rmse_mean,
        "train_rmse_std": train_rmse_std,
        "train_mae_mean": train_mae_mean,
        "train_mae_std": train_mae_std,
        "val_rmse_mean": val_rmse_mean,
        "val_rmse_std": val_rmse_std,
        "val_mae_mean": val_mae_mean,
        "val_mae_std": val_mae_std,
    })


#%% Table 0: Best hyperparameters
table0_records = []
for r in cv_rows:
    bp = r.get("best_params") or {}
    table0_records.append({
        "Dataset": r.get("dataset"),
        "Model": r.get("model"),
        "lr": bp.get("lr"),
        "seq_len": bp.get("seq_len"),
        "hidden_size": bp.get("hidden_size"),
        "dropout": bp.get("dropout"),
    })

table0 = pd.DataFrame.from_records(table0_records)
if not table0.empty:
    table0 = table0.sort_values(["Dataset", "Model"]).reset_index(drop=True)

print("\n=== Table 0: Best hyperparameters ===")
if table0.empty:
    print("No CV summaries found.")
else:
    print(table0.to_string(index=False))


#%% Table 1: Cross-validation Train/Val RMSE & MAE (mean ± std)
table1_rows = []
for r in cv_rows:
    tm_r = r.get("train_rmse_mean")
    ts_r = r.get("train_rmse_std")
    tm_a = r.get("train_mae_mean")
    ts_a = r.get("train_mae_std")
    vm_r = r.get("val_rmse_mean")
    vs_r = r.get("val_rmse_std")
    vm_a = r.get("val_mae_mean")
    vs_a = r.get("val_mae_std")

    if tm_r is None and ts_r is None:
        train_rmse_str = "–"
    else:
        mean_str = "–" if tm_r is None else f"{float(tm_r):.3f}"
        std_str = "–" if ts_r is None else f"{float(ts_r):.3f}"
        train_rmse_str = f"{mean_str} ± {std_str}"

    if tm_a is None and ts_a is None:
        train_mae_str = "–"
    else:
        mean_str = "–" if tm_a is None else f"{float(tm_a):.3f}"
        std_str = "–" if ts_a is None else f"{float(ts_a):.3f}"
        train_mae_str = f"{mean_str} ± {std_str}"

    if vm_r is None and vs_r is None:
        val_rmse_str = "–"
    else:
        mean_str = "–" if vm_r is None else f"{float(vm_r):.3f}"
        std_str = "–" if vs_r is None else f"{float(vs_r):.3f}"
        val_rmse_str = f"{mean_str} ± {std_str}"

    if vm_a is None and vs_a is None:
        val_mae_str = "–"
    else:
        mean_str = "–" if vm_a is None else f"{float(vm_a):.3f}"
        std_str = "–" if vs_a is None else f"{float(vs_a):.3f}"
        val_mae_str = f"{mean_str} ± {std_str}"

    table1_rows.append({
        "Dataset": r.get("dataset"),
        "Model": r.get("model"),
        "Train RMSE (mean ± std)": train_rmse_str,
        "Train MAE (mean ± std)": train_mae_str,
        "Validation RMSE (mean ± std)": val_rmse_str,
        "Validation MAE (mean ± std)": val_mae_str,
    })

table1 = pd.DataFrame.from_records(table1_rows)
if not table1.empty:
    table1 = table1.sort_values(["Dataset", "Model"]).reset_index(drop=True)

print("\n=== Table 1: Cross-validation Train/Val RMSE & MAE (mean ± std) ===")
if table1.empty:
    print("No CV summaries found.")
else:
    print(table1.to_string(index=False))


#%% Table 2: Final holdout Test metrics (RMSE, MAE)
table2_rows = []
for r in cv_rows:
    sp = r.get("summary_path")
    refit_path = None
    if sp:
        hpo_dir = os.path.dirname(sp)
        dataset_dir = os.path.dirname(hpo_dir)
        candidates = sorted(glob(os.path.join(dataset_dir, "refit", "refit-*.json")))
        if candidates:
            refit_path = candidates[-1]

    rmse_val = None
    mae_val = None
    if refit_path and os.path.exists(refit_path):
        refit = None
        try:
            with open(refit_path, "r") as f:
                refit = json.load(f)
        except Exception:
            refit = None
        if refit:
            metrics = refit.get("metrics_original") or {}
            rmse_val = metrics.get("rmse")
            if rmse_val is None:
                mse = metrics.get("mse")
                if isinstance(mse, (int, float)):
                    try:
                        rmse_val = math.sqrt(float(mse))
                    except Exception:
                        rmse_val = None
            mae_val = metrics.get("mae")

    table2_rows.append({
        "Dataset": r.get("dataset"),
        "Model": r.get("model"),
        "RMSE": rmse_val,
        "MAE": mae_val,
    })

table2 = pd.DataFrame.from_records(table2_rows)
if not table2.empty:
    table2 = table2.sort_values(["Dataset", "Model"]).reset_index(drop=True)

print("\n=== Table 2: Final holdout Test metrics ===")
if table2.empty:
    print("No refit results found.")
else:
    print(table2.to_string(index=False))



# %%
