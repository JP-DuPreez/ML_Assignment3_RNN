#%% Imports and setup
import os
import json
import math
from glob import glob

import pandas as pd


# Discover CV summary files
root = os.path.dirname(os.path.abspath(__file__))

patterns = [
    os.path.join(root, "results_For_4_Sets", "**", "hpo", "study-summary-*.json"),
    os.path.join(root, "results", "**", "hpo", "study-summary-*.json"),
    os.path.join(root, "Other", "hpo", "study-summary-*.json"),
]
summary_files = []
for p in patterns:
    summary_files.extend(glob(p, recursive=True))
# Deduplicate and sort for stable order
summary_files = sorted(list(dict.fromkeys(summary_files)))


#%% Parse CV summaries into rows
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

    dataset = summary.get("dataset", {}) or {}
    dataset_path = dataset.get("path") or dataset.get("name") or "unknown_dataset"
    value_col = dataset.get("value_column")
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    dataset_label = f"{base}[{value_col}]" if value_col else base

    model = str(summary.get("model_type") or summary.get("model") or "unknown")

    best_params = summary.get("best_params", {}) or {}
    bp_lr = best_params.get("lr")
    bp_seq_len = best_params.get("seq_len")
    bp_hidden = best_params.get("hidden_size")
    bp_dropout = best_params.get("dropout")

    val_rmse_mean = summary.get("best_mean_RMSE")
    val_rmse_std = summary.get("best_RMSE_std")
    if val_rmse_mean is None:
        vm = (summary.get("validation_mse") or {}).get("mean")
        if vm is not None:
            try:
                val_rmse_mean = math.sqrt(vm)
            except Exception:
                val_rmse_mean = None
    if val_rmse_std is None:
        vms = (summary.get("validation_mse") or {}).get("std")
        if vms is not None:
            try:
                val_rmse_std = math.sqrt(vms)
            except Exception:
                val_rmse_std = None

    train_rmse_mean = summary.get("best_mean_RMSE_train")
    train_rmse_std = None

    # Try to compute train std from best trial folds
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
            for fold in folds:
                tr = (fold or {}).get("train_original") or {}
                rmse = tr.get("rmse")
                if rmse is None:
                    mse = tr.get("mse")
                    if mse is not None:
                        try:
                            rmse = math.sqrt(mse)
                        except Exception:
                            rmse = None
                if isinstance(rmse, (int, float)):
                    train_rmses.append(float(rmse))
            if train_rmses:
                m = float(sum(train_rmses) / len(train_rmses))
                var = float(sum((x - m) ** 2 for x in train_rmses) / len(train_rmses))
                std = math.sqrt(var)
                train_rmse_std = std
                if train_rmse_mean is None:
                    train_rmse_mean = m

    cv_rows.append({
        "summary_path": sp,
        "dataset": dataset_label,
        "model": model,
        "best_params": {
            "lr": bp_lr,
            "seq_len": bp_seq_len,
            "hidden_size": bp_hidden,
            "dropout": bp_dropout,
        },
        "val_rmse_mean": val_rmse_mean,
        "val_rmse_std": val_rmse_std,
        "train_rmse_mean": train_rmse_mean,
        "train_rmse_std": train_rmse_std,
    })


#%% Build Table 0 (Best hyperparameters)
table0_records = []
for r in cv_rows:
    bp = r.get("best_params") or {}
    table0_records.append({
        "Dataset": r.get("dataset"),
        "Model": r.get("model"),
        "lr": None if bp.get("lr") is None else float(bp.get("lr")),
        "seq_len": None if bp.get("seq_len") is None else int(bp.get("seq_len")),
        "hidden_size": None if bp.get("hidden_size") is None else int(bp.get("hidden_size")),
        "dropout": None if bp.get("dropout") is None else float(bp.get("dropout")),
    })

table0 = pd.DataFrame.from_records(table0_records)
if not table0.empty:
    table0 = table0.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    table0_disp = table0.copy()
    if "lr" in table0_disp.columns:
        table0_disp_lr = []
        for x in table0_disp["lr"].tolist():
            table0_disp_lr.append(None if pd.isna(x) else round(float(x), 4))
        table0_disp["lr"] = table0_disp_lr
    if "dropout" in table0_disp.columns:
        table0_disp_dropout = []
        for x in table0_disp["dropout"].tolist():
            table0_disp_dropout.append(None if pd.isna(x) else round(float(x), 3))
        table0_disp["dropout"] = table0_disp_dropout
else:
    table0_disp = table0


#%% Build Table 1 (CV Train/Val RMSE mean ± std)
table1_records = []
for r in cv_rows:
    tm = r.get("train_rmse_mean")
    ts = r.get("train_rmse_std")
    vm = r.get("val_rmse_mean")
    vs = r.get("val_rmse_std")

    if tm is None and ts is None:
        train_str = "–"
    elif tm is not None and ts is None:
        train_str = f"{tm:.2f} $\\pm$ –"
    elif tm is None and ts is not None:
        train_str = f"– $\\pm$ {ts:.2f}"
    else:
        train_str = f"{tm:.2f} $\\pm$ {ts:.2f}"

    if vm is None and vs is None:
        val_str = "–"
    elif vm is not None and vs is None:
        val_str = f"{vm:.2f} $\\pm$ –"
    elif vm is None and vs is not None:
        val_str = f"– $\\pm$ {vs:.2f}"
    else:
        val_str = f"{vm:.2f} $\\pm$ {vs:.2f}"

    table1_records.append({
        "Dataset": r.get("dataset"),
        "Model": r.get("model"),
        "Train RMSE (mean ± std)": train_str,
        "Validation RMSE (mean ± std)": val_str,
    })

table1 = pd.DataFrame.from_records(table1_records)
if not table1.empty:
    table1 = table1.sort_values(["Dataset", "Model"]).reset_index(drop=True)


#%% Build Table 2 (Final holdout Test RMSE)
table2_records = []
for r in cv_rows:
    sp = r.get("summary_path")
    model = r.get("model")
    refit_path = None
    if sp:
        hpo_dir = os.path.dirname(sp)
        dataset_dir = os.path.dirname(hpo_dir)
        candidate_glob = os.path.join(dataset_dir, "refit", "refit-*.json")
        candidates = sorted(glob(candidate_glob))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p))
            refit_path = candidates[-1]
        else:
            upper = os.path.dirname(dataset_dir)
            up_candidates = sorted(glob(os.path.join(upper, "**", "refit", "refit-*.json"), recursive=True))
            if up_candidates:
                up_candidates.sort(key=lambda p: os.path.getmtime(p))
                refit_path = up_candidates[-1]

    rmse_val = None
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
                if mse is not None:
                    try:
                        rmse_val = math.sqrt(mse)
                    except Exception:
                        rmse_val = None
            if isinstance(rmse_val, (int, float)):
                rmse_val = float(rmse_val)

    table2_records.append({
        "Dataset": r.get("dataset"),
        "Model": model,
        "Test RMSE": rmse_val if rmse_val is not None else None,
    })

table2 = pd.DataFrame.from_records(table2_records)
if not table2.empty:
    table2 = table2.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    table2_disp = table2.copy()
    if "Test RMSE" in table2_disp.columns:
        test_rmse_col = []
        for x in table2_disp["Test RMSE"].tolist():
            test_rmse_col.append(None if pd.isna(x) else round(float(x), 2))
        table2_disp["Test RMSE"] = test_rmse_col
else:
    table2_disp = table2


#%% Print tables to stdout
print("\n=== Table 0: Best hyperparameters ===")
if table0_disp.empty:
    print("No CV summaries found.")
else:
    print(table0_disp.to_string(index=False))

print("\n=== Table 1: Cross-validation Train/Val RMSE (mean ± std) ===")
if table1.empty:
    print("No CV summaries found.")
else:
    print(table1.to_string(index=False))

print("\n=== Table 2: Final holdout Test RMSE ===")
if table2_disp.empty:
    print("No refit results found.")
else:
    print(table2_disp.to_string(index=False))


#%% Save LaTeX tables
tables_dir = os.path.join(root, "results", "tables")
os.makedirs(tables_dir, exist_ok=True)

# Save minimal LaTeX (escape=False to preserve ± and symbols)
(table0_disp if isinstance(table0_disp, pd.DataFrame) else pd.DataFrame()).to_latex(
    os.path.join(tables_dir, "table0_best_hparams.tex"), index=False, escape=False
)
(table1 if isinstance(table1, pd.DataFrame) else pd.DataFrame()).to_latex(
    os.path.join(tables_dir, "table1_cv_trainval.tex"), index=False, escape=False
)
(table2_disp if isinstance(table2_disp, pd.DataFrame) else pd.DataFrame()).to_latex(
    os.path.join(tables_dir, "table2_test_rmse.tex"), index=False, escape=False
)


