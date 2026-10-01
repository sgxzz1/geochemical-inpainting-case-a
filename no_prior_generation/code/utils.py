from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_grid_csv(path: str | Path, dtype=float) -> pd.DataFrame:
    grid = pd.read_csv(path, index_col=0)
    grid.index = pd.to_numeric(grid.index, errors="raise")
    grid.columns = pd.to_numeric(grid.columns, errors="raise")
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid.astype(dtype)


def save_grid_csv(array: np.ndarray, template: pd.DataFrame, path: str | Path) -> None:
    out = pd.DataFrame(array, index=template.index, columns=template.columns)
    out.index.name = "CY"
    out.columns.name = "CX"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, encoding="utf-8-sig")


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_coord_channels(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, cols, dtype=np.float32)
    y = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    x_coord = np.tile(x[None, :], (rows, 1))
    y_coord = np.tile(y[:, None], (1, cols))
    return x_coord, y_coord


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - target)[mask > 0]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


def masked_mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs((pred - target)[mask > 0])
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff))


def masked_r2_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    y = target[mask > 0]
    yhat = pred[mask > 0]
    if y.size < 2:
        return float("nan")
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def normalize_01(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(finite):
        return out
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi - lo < 1e-8:
        return out
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return out.astype(np.float32)
