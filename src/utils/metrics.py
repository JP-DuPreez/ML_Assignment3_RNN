from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * num / den))


def mase_denominator(train_series: np.ndarray, seasonality: int = 1) -> float:
    s = max(1, int(seasonality))
    if train_series.shape[0] <= s:
        return float("nan")
    diffs = np.abs(train_series[s:] - train_series[:-s])
    denom = float(np.mean(diffs))
    return denom if denom > 0 else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray, seasonality: int = 1) -> float:
    denom = mase_denominator(train_series, seasonality=seasonality)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return mae / denom


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_series_for_mase: np.ndarray,
    mase_seasonality: int = 1,
) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, train_series_for_mase, mase_seasonality),
    }


