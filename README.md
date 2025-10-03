### Option 4 — Time Series Forecasting with Simple RNNs

This repository implements three classic recurrent neural networks for univariate time-series forecasting and a config-driven training pipeline:

- **Elman RNN**: feedback from previous hidden state
- **Jordan RNN**: feedback from previous output
- **Multi‑recurrent RNN**: feedback from both previous hidden state and previous output

See `Conext.md` for architectural details and `data/context.md` for dataset notes.

### Requirements

- Python 3.9+ and an activated virtual environment
- Packages: PyTorch, NumPy, pandas

Install (example):
```bash
pip3 install torch numpy pandas
```

### Repository structure (key files)

- `config.json`: central configuration for dataset, training, and models
- `src/train.py`: main entrypoint (loads config, trains/evaluates, saves results)
- `src/models/`: implementations (`elman.py`, `jordan.py`, `multi_recurrent.py`, `factory.py`)
- `src/data/time_series.py`: CSV loading, splitting, scaling, windowing, dataloaders
- `src/utils/activations.py`: activation mapping
- `results/`: per-model JSON summaries (created at runtime)

### Quickstart

1) Adjust `config.json` if needed (e.g., dataset path/column).

2) Run training (optionally filter by model type or name):
```bash
# all models from config
python3 -m src.train --config /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/config.json

# single model by type
python3 -m src.train --config /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/config.json --model elman

# single model by name
python3 -m src.train --config /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/config.json --model elman_32
```

3) Inspect results in `results/` and logs printed to the console.

### Configuration

`config.json` controls the full experiment. Minimal shape:

```json
{
  "seed": 42,
  "device": "auto",
  "results_dir": "results",
  "dataset": {
    "path": "data/AirPassengers.csv",
    "value_column": "Passengers",
    "date_column": "Month",
    "seq_len": 24,
    "horizon": 1,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "scaler": "standard"
  },
  "training": {
    "batch_size": 64,
    "epochs": 50,
    "patience": 5,
    "min_delta": 0.0001,
    "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
    "loss": "mse"
  },
  "cv": { "enabled": false, "k_folds": 5 },
  "models": [
    {"name": "elman_32", "type": "elman", "hidden_size": 32, "dropout": 0.1, "activation": "tanh"},
    {"name": "jordan_32", "type": "jordan", "hidden_size": 32, "dropout": 0.1, "activation": "tanh"},
    {"name": "multi_32", "type": "multi", "hidden_size": 32, "dropout": 0.1, "activation": "tanh"}
  ]
}
```

Notes:
- **dataset**: `seq_len` is the lookback; `horizon` is the forecast length (models output a vector of length `horizon` each step; training uses the last step).
- **training**: early stopping monitors validation loss with `patience` and `min_delta`; gradient clipping at 1.0 each step.
- **optimizer**: choose `adam` (default), `adamw`, or `sgd` (`momentum` supported for `sgd`).
- **loss**: `mse` (default), `mae`, or `huber` (`huber_beta`).
- **cv (blocked sliding)**: set `enabled=true` and `k_folds`. Train/val windows are auto-sized from the data unless you explicitly add them.
- **early_stopping_metric**: choose which validation metric to monitor for early stopping (`mse` or `mae`).

### Models and shapes

- Inputs: `(batch, seq_len, 1)` for univariate series.
- Outputs: `(batch, seq_len, horizon)`; we compute loss on `y_pred = y_seq[:, -1, :]`.
- No output activation in the models; choose target scaling appropriately.

### Metrics and outputs

- MSE and MAE reported on both the scaled space and the original units (via inverse transform).
- Optional: HPO/CV and refit flows save RMSE, sMAPE, and MASE as well.
- Per‑model JSON summaries saved under `results/`, including history and metrics. A consolidated `all_results-<timestamp>.json` is also written.

### Cross‑validation (expanding‑origin, blocked)

- Train grows each fold from the start; validation is the next contiguous block (no leakage, no shuffling).
- Only `cv.enabled` and `cv.k_folds` are required; fold sizes are derived from data, ensuring each block ≥ `seq_len + horizon`.
- Early stopping with `patience` and `min_delta` per fold; metrics aggregated as mean±std on original scale.

### Hyperparameter optimisation (Optuna)

- Install Optuna:
```bash
pip3 install optuna
```

- Run an HPO study (blocked sliding CV with auto-sized windows; objective = mean validation RMSE in original units):
```bash
# optimise a specific architecture by type
python3 -m src.hpo_optuna --config /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/config.json \
  --trials 50 --study rnn_hpo --model elman

# or by model name from config["models"]
python3 -m src.hpo_optuna --config /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/config.json \
  --trials 50 --study rnn_hpo --model elman_32
```

- What it does/produces:
  - Uses blocked sliding CV with `cv.enabled=true` and `k_folds`; train/val windows are derived from data.
  - Tunes `lr`, `seq_len`, `hidden_size`, `dropout` for the selected model type.
  - Per-trial JSON: `results/hpo/trial-<n>.json` with params, per-fold metrics (RMSE/MAE/sMAPE/MASE in original units), training histories, and `avg_best_epoch`.
  - Study summary JSON: `results/hpo/study-summary-<timestamp>.json` with `best_value` (mean RMSE), `best_params`, `model_type`, and `avg_best_epoch`.
  - Study summary JSON: `results/hpo/study-summary-<timestamp>.json` with `best_mean_RMSE`, `best_RMSE_std`, `best_params`, `model_type`, `dataset`, and `avg_best_epoch`.
  - Early stopping per fold monitors `training.early_stopping_metric` (`mse`, `mae`, or `rmse` which is treated as `mse`).

### Refit and final test

After HPO, retrain on train+val with fixed epochs equal to the average of best epochs across CV, then evaluate once on test:
```bash
python3 -m src.refit --config /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/config.json \
  --study-summary /Users/jpdupreez/Downloads/CompSci/4.MachineLearning/Assignment_3/results/hpo/study-summary-<timestamp>.json
```

### Tips

- Ensure the `dataset.value_column` matches your CSV header.
- If you get “insufficient windows”, reduce `seq_len` or `horizon`, or increase data.
- For larger datasets, consider increasing `batch_size` and enabling `cuda`/Apple `mps` via `device: "auto"`.


