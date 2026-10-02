from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

from priors import build_prior_channels
from utils import make_coord_channels, read_grid_csv


@dataclass
class DataPaths:
    full_grid: str = "/mnt/storatge/ljj/生成模型/真实数据/ag_ok_back_500m.csv"
    block_mask: str = "/mnt/storatge/ljj/生成模型/处理后数据/ag_block_mask_15pct.csv"
    hard_mask: str = "/mnt/storatge/ljj/生成模型/处理后数据/ag_hard_mask_block15_hard5.csv"
    target_mask: str = "/mnt/storatge/ljj/生成模型/处理后数据/ag_target_mask_block15_hard5.csv"


@dataclass
class Normalization:
    mean: float
    std: float

    def encode(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)

    def decode(self, array: np.ndarray) -> np.ndarray:
        return (array * self.std + self.mean).astype(np.float32)


class GeochemDataBundle:
    def __init__(self, paths: DataPaths):
        self.paths = paths
        self.full_df = read_grid_csv(paths.full_grid, dtype=float)
        self.block_mask_df = read_grid_csv(paths.block_mask, dtype=np.uint8)
        self.hard_mask_df = read_grid_csv(paths.hard_mask, dtype=np.uint8)
        self.target_mask_df = read_grid_csv(paths.target_mask, dtype=np.uint8)

        self.full_ag = self.full_df.to_numpy(dtype=np.float32)
        self.full_log = np.log1p(self.full_ag).astype(np.float32)
        self.block_mask = (self.block_mask_df.to_numpy(dtype=np.uint8) > 0).astype(np.uint8)
        self.hard_mask = (self.hard_mask_df.to_numpy(dtype=np.uint8) > 0).astype(np.uint8)
        self.target_mask = (self.target_mask_df.to_numpy(dtype=np.uint8) > 0).astype(np.uint8)

        if self.full_log.shape != self.block_mask.shape:
            raise ValueError("Full grid and masks have different shapes.")

        self.rows, self.cols = self.full_log.shape
        self.train_available_mask = (1 - self.block_mask).astype(np.uint8)
        train_values = self.full_log[self.train_available_mask > 0]
        self.norm = Normalization(mean=float(np.mean(train_values)), std=float(np.std(train_values) + 1e-6))
        self.x_coord, self.y_coord = make_coord_channels(self.rows, self.cols)
        self.x_coords_1d = self.full_df.columns.to_numpy(dtype=np.float64)
        self.y_coords_1d = self.full_df.index.to_numpy(dtype=np.float64)

    def _make_input_stack(
        self,
        condition_norm: np.ndarray,
        known_mask: np.ndarray,
        target_mask: np.ndarray,
        hard_mask: np.ndarray,
        seed: int,
        ds_level: int,
        ds_ensemble: int,
        ds_method: str,
        kriging_backend: str,
        variogram_model: str,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
        priors = build_prior_channels(
            condition_values=condition_norm,
            known_mask=known_mask.astype(np.uint8),
            target_mask=target_mask.astype(np.uint8),
            x_coord=self.x_coord,
            y_coord=self.y_coord,
            x_coords_1d=self.x_coords_1d,
            y_coords_1d=self.y_coords_1d,
            ds_level=ds_level,
            ds_ensemble=ds_ensemble,
            ds_method=ds_method,
            seed=seed,
            kriging_backend=kriging_backend,
            variogram_model=variogram_model,
        )
        inputs = np.stack(
            [
                condition_norm,
                known_mask.astype(np.float32),
                target_mask.astype(np.float32),
                hard_mask.astype(np.float32),
                priors["ds_mean"],
                priors["ds_p75"],
                priors["ds_std"],
                priors["kriging_prior"],
                priors["kriging_variance"],
                self.x_coord.astype(np.float32),
                self.y_coord.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        return inputs, priors

    def validation_sample(
        self,
        ds_level: int = 3,
        ds_ensemble: int = 16,
        ds_method: str = "cascade",
        seed: int = 123,
        kriging_backend: str = "auto",
        variogram_model: str = "spherical",
    ) -> dict[str, np.ndarray | str]:
        known_mask = (1 - self.target_mask).astype(np.uint8)
        condition_log = self.full_log.copy()
        condition_log[self.target_mask > 0] = self.norm.mean
        condition_norm = self.norm.encode(condition_log)

        inputs, priors = self._make_input_stack(
            condition_norm=condition_norm,
            known_mask=known_mask,
            target_mask=self.target_mask,
            hard_mask=self.hard_mask,
            seed=seed,
            ds_level=ds_level,
            ds_ensemble=ds_ensemble,
            ds_method=ds_method,
            kriging_backend=kriging_backend,
            variogram_model=variogram_model,
        )
        return {
            "inputs": inputs,
            "truth_norm": self.norm.encode(self.full_log),
            "condition_norm": condition_norm,
            "known_mask": known_mask.astype(np.float32),
            "target_mask": self.target_mask.astype(np.float32),
            "hard_mask": self.hard_mask.astype(np.float32),
            "priors": priors,
        }


class RandomBlockDataset(IterableDataset):
    def __init__(
        self,
        bundle: GeochemDataBundle,
        steps_per_epoch: int,
        min_block_fraction: float = 0.04,
        max_block_fraction: float = 0.12,
        min_hard_fraction: float = 0.01,
        max_hard_fraction: float = 0.10,
        ds_level: int = 3,
        ds_ensemble: int = 8,
        ds_method: str = "cascade",
        kriging_backend: str = "auto",
        variogram_model: str = "spherical",
        seed: int = 20260918,
    ):
        self.bundle = bundle
        self.steps_per_epoch = steps_per_epoch
        self.min_block_fraction = min_block_fraction
        self.max_block_fraction = max_block_fraction
        self.min_hard_fraction = min_hard_fraction
        self.max_hard_fraction = max_hard_fraction
        self.ds_level = ds_level
        self.ds_ensemble = ds_ensemble
        self.ds_method = ds_method
        self.kriging_backend = kriging_backend
        self.variogram_model = variogram_model
        self.seed = seed

    def _sample_rect(self, rng: np.random.Generator) -> tuple[int, int, int, int]:
        rows, cols = self.bundle.rows, self.bundle.cols
        total = rows * cols
        for _ in range(500):
            frac = float(rng.uniform(self.min_block_fraction, self.max_block_fraction))
            area = max(4, int(round(total * frac)))
            aspect = float(rng.uniform(0.6, 2.2))
            h = int(round(np.sqrt(area / aspect)))
            w = int(round(h * aspect))
            h = int(np.clip(h, 2, rows))
            w = int(np.clip(w, 2, cols))
            r0 = int(rng.integers(0, rows - h + 1))
            c0 = int(rng.integers(0, cols - w + 1))
            rect = np.zeros((rows, cols), dtype=np.uint8)
            rect[r0:r0 + h, c0:c0 + w] = 1
            if np.all(self.bundle.train_available_mask[rect > 0] > 0):
                return r0, r0 + h, c0, c0 + w
        raise RuntimeError("Failed to sample a pseudo block outside the fixed validation block.")

    def _make_sample(self, rng: np.random.Generator, sample_seed: int) -> dict[str, torch.Tensor]:
        b = self.bundle
        r0, r1, c0, c1 = self._sample_rect(rng)
        pseudo_block = np.zeros((b.rows, b.cols), dtype=np.uint8)
        pseudo_block[r0:r1, c0:c1] = 1

        block_indices = np.argwhere(pseudo_block > 0)
        hard_fraction = float(rng.uniform(self.min_hard_fraction, self.max_hard_fraction))
        hard_count = max(1, int(round(len(block_indices) * hard_fraction)))
        chosen = rng.choice(len(block_indices), size=hard_count, replace=False)
        pseudo_hard = np.zeros_like(pseudo_block)
        selected = block_indices[chosen]
        pseudo_hard[selected[:, 0], selected[:, 1]] = 1
        pseudo_target = ((pseudo_block == 1) & (pseudo_hard == 0)).astype(np.uint8)

        hidden_mask = ((b.block_mask == 1) | (pseudo_target == 1)).astype(np.uint8)
        known_mask = (1 - hidden_mask).astype(np.uint8)

        condition_log = b.full_log.copy()
        condition_log[hidden_mask > 0] = b.norm.mean
        condition_norm = b.norm.encode(condition_log)
        truth_norm = b.norm.encode(b.full_log)

        inputs, _ = b._make_input_stack(
            condition_norm=condition_norm,
            known_mask=known_mask,
            target_mask=pseudo_target,
            hard_mask=pseudo_hard,
            seed=sample_seed,
            ds_level=self.ds_level,
            ds_ensemble=self.ds_ensemble,
            ds_method=self.ds_method,
            kriging_backend=self.kriging_backend,
            variogram_model=self.variogram_model,
        )

        return {
            "inputs": torch.from_numpy(inputs),
            "truth_norm": torch.from_numpy(truth_norm[None, ...].astype(np.float32)),
            "condition_norm": torch.from_numpy(condition_norm[None, ...].astype(np.float32)),
            "known_mask": torch.from_numpy(known_mask[None, ...].astype(np.float32)),
            "target_mask": torch.from_numpy(pseudo_target[None, ...].astype(np.float32)),
        }

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        offset = 0 if worker is None else worker.id * 100000
        rng = np.random.default_rng(self.seed + offset)
        for _ in range(self.steps_per_epoch):
            sample_seed = int(rng.integers(0, 2**31 - 1))
            yield self._make_sample(rng, sample_seed=sample_seed)
