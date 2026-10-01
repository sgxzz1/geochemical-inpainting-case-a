from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedGeochemDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Cached split directory not found: {self.split_dir}")
        self.files = sorted(self.split_dir.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No cached samples found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.files[idx]
        with np.load(path) as data:
            return {
                "inputs": torch.from_numpy(data["inputs"].astype(np.float32)),
                "truth_norm": torch.from_numpy(data["truth_norm"].astype(np.float32)),
                "condition_norm": torch.from_numpy(data["condition_norm"].astype(np.float32)),
                "known_mask": torch.from_numpy(data["known_mask"].astype(np.float32)),
                "target_mask": torch.from_numpy(data["target_mask"].astype(np.float32)),
            }
