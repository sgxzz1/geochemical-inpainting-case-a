from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "no_prior_generation" / "code"
sys.path.insert(0, str(CODE_DIR))

from fractal_cascade import fractal_cascade_prior_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild cached samples with no hard data and DS-only priors."
    )
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--ds-ensemble", type=int, default=4)
    parser.add_argument("--ds-level", type=int, default=3)
    return parser.parse_args()


def build_one(task: tuple[str, int, str, str, int, int]) -> dict[str, int | str]:
    split, index, source_name, output_name, ds_ensemble, ds_level = task
    source_path = Path(source_name)
    output_path = Path(output_name)
    if output_path.exists():
        return {"split": split, "index": index, "status": "existing"}

    with np.load(source_path) as data:
        source_inputs = data["inputs"].astype(np.float32)
        truth = data["truth_norm"].astype(np.float32)
        condition = data["condition_norm"].astype(np.float32)
        known_mask = data["known_mask"].astype(np.float32)
        target_mask = data["target_mask"].astype(np.float32)

    hard_mask = source_inputs[3:4].copy()
    target_mask = np.clip(target_mask + hard_mask, 0.0, 1.0).astype(np.float32)
    known_mask = (known_mask * (1.0 - hard_mask)).astype(np.float32)
    condition = (condition * (1.0 - hard_mask)).astype(np.float32)

    cascade_condition = condition[0].copy()
    cascade_condition[known_mask[0] <= 0] = np.nan
    base_seed = 20260918 if split == "train" else 21260918
    ds_seed = int(base_seed + index * 10007)
    ds = fractal_cascade_prior_stats(
        condition=cascade_condition,
        original_known_values=condition[0],
        original_known_mask=known_mask[0].astype(np.uint8),
        n_realizations=ds_ensemble,
        seed=ds_seed,
        max_downsample_levels=max(ds_level, 1),
        downsample_fallback="random_known",
        distance_threshold=0.25,
        max_scan=256,
        max_radius=2,
        max_neighbors=12,
    )

    inputs = source_inputs.copy()
    inputs[0] = condition[0]
    inputs[1] = known_mask[0]
    inputs[2] = target_mask[0]
    inputs[3] = 0.0
    inputs[4] = ds.mean
    inputs[5] = ds.p75
    inputs[6] = ds.std
    inputs[7:9] = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs.astype(np.float32),
        truth_norm=truth,
        condition_norm=condition,
        known_mask=known_mask,
        target_mask=target_mask,
    )
    return {
        "split": split,
        "index": index,
        "status": "created",
        "target_pixels": int(target_mask.sum()),
    }


def main() -> None:
    args = parse_args()
    source_cache = args.source_cache.resolve()
    output_cache = args.output_cache.resolve()
    tasks = []
    split_counts = {}
    for split in ("train", "val"):
        files = sorted((source_cache / split).glob("sample_*.npz"))
        if not files:
            raise FileNotFoundError(f"No samples in {source_cache / split}")
        split_counts[split] = len(files)
        for index, source_path in enumerate(files):
            output_path = output_cache / split / source_path.name
            tasks.append(
                (
                    split,
                    index,
                    str(source_path),
                    str(output_path),
                    args.ds_ensemble,
                    args.ds_level,
                )
            )

    created = 0
    existing = 0
    target_pixels = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "created":
                created += 1
                target_pixels.append(int(result["target_pixels"]))
            else:
                existing += 1
            if completed % 24 == 0 or completed == len(tasks):
                print(
                    f"completed={completed}/{len(tasks)} created={created} existing={existing}",
                    flush=True,
                )

    source_metadata_path = source_cache / "cache_metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    metadata = {
        "purpose": "DS-only cosine diffusion cache with all pseudo hard data removed",
        "source_cache": str(source_cache),
        "train_samples": split_counts["train"],
        "val_samples": split_counts["val"],
        "remove_hard_data": True,
        "hard_channel_nonzero": False,
        "ds_method": "cascade",
        "ds_level": args.ds_level,
        "ds_ensemble": args.ds_ensemble,
        "kriging_enabled": False,
        "input_channels": [
            "condition_ag_norm",
            "known_mask",
            "target_mask_including_former_hard",
            "zero_hard_mask",
            "ds_prior_mean_norm_without_hard",
            "ds_prior_p75_norm_without_hard",
            "ds_prior_std_norm_without_hard",
            "zero_kriging",
            "zero_kriging_variance",
            "x_coord",
            "y_coord",
        ],
        "created_in_this_run": created,
        "existing_before_this_run": existing,
        "target_pixels_mean": float(np.mean(target_pixels)) if target_pixels else None,
        "workers": args.workers,
        "cpu_count": os.cpu_count(),
        "source_metadata": source_metadata,
    }
    output_cache.mkdir(parents=True, exist_ok=True)
    (output_cache / "cache_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
