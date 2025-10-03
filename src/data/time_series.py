from __future__ import annotations

from typing import Literal, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_series_from_csv(
    path: str,
    value_column: str,
    date_column: str | None = None,
) -> np.ndarray:
    df = pd.read_csv(path)
    # Resolve date column case-insensitively if provided
    if date_column:
        date_col = date_column
        if date_col not in df.columns:
            lower_map = {c.lower(): c for c in df.columns}
            if date_column.lower() in lower_map:
                date_col = lower_map[date_column.lower()]
            else:
                date_col = None
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            except Exception:
                pass

    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in {path}")

    values = pd.to_numeric(df[value_column], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def load_series_and_time(
    path: str,
    value_column: str,
    date_column: str,
):
    """
    Load a time series and its timestamps as a pandas.DatetimeIndex, sorted by date.

    Returns
    -------
    values: np.ndarray shape (T,)
    dt_index: pandas.DatetimeIndex length T
    """
    df = pd.read_csv(path)
    # Resolve date column case-insensitively and with common fallbacks
    date_col = date_column
    if date_col not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if date_column.lower() in lower_map:
            date_col = lower_map[date_column.lower()]
        else:
            for guess in ("date", "datetime", "timestamp", "time", "month"):
                if guess in lower_map:
                    date_col = lower_map[guess]
                    break
    if date_col not in df.columns:
        raise ValueError(f"date_column '{date_column}' not found in {path}; available columns: {list(df.columns)}")
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    except Exception:
        pass
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in {path}")
    series = pd.to_numeric(df[value_column], errors="coerce")
    mask = series.notna()
    values = series[mask].to_numpy(dtype=np.float32)
    dt_index = pd.DatetimeIndex(df.loc[mask, date_col])
    if len(dt_index) != len(values):
        # Align lengths defensively
        min_len = min(len(dt_index), len(values))
        values = values[:min_len]
        dt_index = dt_index[:min_len]
    return values, dt_index


def build_time_features(
    dt_index: pd.DatetimeIndex,
    cyclic: Optional[Dict[str, bool]] = None,
) -> np.ndarray:
    """Construct cyclic time encodings (sin/cos) from a DatetimeIndex.

    Supported keys in `cyclic`: month, day_of_week, day_of_year, hour
    """
    if cyclic is None:
        cyclic = {}
    feats: list[np.ndarray] = []

    if cyclic.get("month", False):
        m = (dt_index.month.values.astype(np.int32) - 1).astype(np.float32)
        theta = 2.0 * np.pi * (m / 12.0)
        feats += [np.sin(theta), np.cos(theta)]

    if cyclic.get("day_of_week", False):
        d = dt_index.dayofweek.values.astype(np.float32)
        theta = 2.0 * np.pi * (d / 7.0)
        feats += [np.sin(theta), np.cos(theta)]

    if cyclic.get("day_of_year", False):
        doy = dt_index.dayofyear.values.astype(np.float32)
        theta = 2.0 * np.pi * (doy / 365.0)
        feats += [np.sin(theta), np.cos(theta)]

    if cyclic.get("hour", False):
        if hasattr(dt_index, "hour"):
            h = dt_index.hour.values.astype(np.float32)
            theta = 2.0 * np.pi * (h / 24.0)
            feats += [np.sin(theta), np.cos(theta)]

    if not feats:
        return np.zeros((len(dt_index), 0), dtype=np.float32)
    F = np.stack(feats, axis=1).astype(np.float32)
    return F


def standardize_features_train_val_test(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    method: Literal["standard", "minmax"] = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if train.ndim != 2:
        raise ValueError("Features must be 2D (N, F)")
    if method not in {"standard", "minmax"}:
        raise ValueError("method must be 'standard' or 'minmax'")
    if train.shape[1] == 0:
        scaler = {"method": method, "mean": np.array([]), "std": np.array([]), "min": np.array([]), "max": np.array([])}
        return train, val, test, scaler

    if method == "standard":
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True) + 1e-8
        scaler = {"method": method, "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        f = lambda x: (x - mean) / std
        return f(train).astype(np.float32), f(val).astype(np.float32), f(test).astype(np.float32), scaler

    fmin = train.min(axis=0, keepdims=True)
    fmax = train.max(axis=0, keepdims=True)
    scale = np.maximum(fmax - fmin, 1e-8)
    scaler = {"method": method, "min": fmin.astype(np.float32), "max": fmax.astype(np.float32)}
    f = lambda x: (x - fmin) / scale
    return f(train).astype(np.float32), f(val).astype(np.float32), f(test).astype(np.float32), scaler


def create_windows_with_features(
    values: np.ndarray,
    features: Optional[np.ndarray],
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if seq_len <= 0 or horizon <= 0:
        raise ValueError("seq_len and horizon must be positive")
    if features is not None and features.shape[0] != values.shape[0]:
        raise ValueError("features and values must have same length")
    x_list = []
    y_list = []
    limit = values.shape[0] - seq_len - horizon + 1
    F = 0 if features is None else int(features.shape[1])
    for start in range(max(0, limit)):
        end = start + seq_len
        x_val = values[start:end].reshape(seq_len, 1).astype(np.float32)
        x_feat = np.zeros((seq_len, 0), dtype=np.float32) if F == 0 else features[start:end, :].astype(np.float32)
        x_window = np.concatenate([x_val, x_feat], axis=1)
        y_future = values[end : end + horizon]
        if y_future.shape[0] < horizon:
            break
        x_list.append(x_window)
        y_list.append(y_future.astype(np.float32))
    X = np.stack(x_list, axis=0) if x_list else np.zeros((0, seq_len, 1 + F), dtype=np.float32)
    y = np.stack(y_list, axis=0) if y_list else np.zeros((0, horizon), dtype=np.float32)
    return X, y


def make_dataloaders_with_features(
    train_values: np.ndarray,
    val_values: np.ndarray,
    test_values: np.ndarray,
    train_features: Optional[np.ndarray],
    val_features: Optional[np.ndarray],
    test_features: Optional[np.ndarray],
    seq_len: int,
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_train, y_train = create_windows_with_features(train_values, train_features, seq_len, horizon)
    X_val, y_val = create_windows_with_features(val_values, val_features, seq_len, horizon)
    X_test, y_test = create_windows_with_features(test_values, test_features, seq_len, horizon)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def split_series(
    values: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be in (0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = values.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = values[:n_train]
    val = values[n_train : n_train + n_val]
    test = values[n_train + n_val :]
    return train, val, test


def standardize_train_val_test(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    method: Literal["standard", "minmax"] = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if method not in {"standard", "minmax"}:
        raise ValueError("method must be 'standard' or 'minmax'")

    if method == "standard":
        mean = float(train.mean())
        std = float(train.std() + 1e-8)
        scaler = {"method": method, "mean": mean, "std": std}
        f = lambda x: (x - mean) / std
        return f(train), f(val), f(test), scaler

    tmin = float(train.min())
    tmax = float(train.max())
    scale = max(tmax - tmin, 1e-8)
    scaler = {"method": method, "min": tmin, "max": tmax}
    f = lambda x: (x - tmin) / scale
    return f(train), f(val), f(test), scaler


def inverse_transform(x: np.ndarray | torch.Tensor, scaler: dict) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    method = scaler.get("method")
    if method == "standard":
        mean = scaler["mean"]
        std = scaler["std"]
        return x * std + mean
    if method == "minmax":
        tmin = scaler["min"]
        tmax = scaler["max"]
        return x * (tmax - tmin) + tmin
    raise ValueError("Unknown scaler method")


def create_windows(values: np.ndarray, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    if seq_len <= 0 or horizon <= 0:
        raise ValueError("seq_len and horizon must be positive")
    x_list = []
    y_list = []
    limit = values.shape[0] - seq_len - horizon + 1
    for start in range(max(0, limit)):
        end = start + seq_len
        x_window = values[start:end]
        y_future = values[end : end + horizon]
        if y_future.shape[0] < horizon:
            break
        x_list.append(x_window)
        y_list.append(y_future)
    X = np.array(x_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y_list, dtype=np.float32).reshape(-1, horizon)
    return X, y


def make_dataloaders(
    train_values: np.ndarray,
    val_values: np.ndarray,
    test_values: np.ndarray,
    seq_len: int,
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_train, y_train = create_windows(train_values, seq_len, horizon)
    X_val, y_val = create_windows(val_values, seq_len, horizon)
    X_test, y_test = create_windows(test_values, seq_len, horizon)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


