from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_diffusion_from_config import (
    PROJECT_ROOT,
    ConditionalResidualDenoiser,
    CosineDiffusionSchedule,
    DataPaths,
    GeochemDataBundle,
    evaluate_case_a,
    save_grid_csv,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained cosine diffusion model with randomly sampled hard data."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hard-fraction", type=float, default=0.05)
    parser.add_argument("--mask-repeats", type=int, default=5)
    parser.add_argument("--samples-per-mask", type=int, default=50)
    parser.add_argument("--sample-batch-size", type=int, default=10)
    parser.add_argument("--hard-seed", type=int, default=20261002)
    parser.add_argument("--diffusion-seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_random_hard_sample(
    bundle: GeochemDataBundle,
    hard_fraction: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], int]:
    block_indices = np.argwhere(bundle.block_mask > 0)
    hard_count = int(round(len(block_indices) * hard_fraction))
    rng = np.random.default_rng(seed)
    hard_mask = np.zeros_like(bundle.block_mask, dtype=np.float32)
    if hard_count > 0:
        selected_indices = rng.choice(len(block_indices), size=hard_count, replace=False)
        selected = block_indices[selected_indices]
        hard_mask[selected[:, 0], selected[:, 1]] = 1.0
    target_mask = ((bundle.block_mask > 0) & (hard_mask == 0)).astype(np.float32)
    known_mask = (1.0 - target_mask).astype(np.float32)

    condition_log = bundle.full_log.copy()
    condition_log[target_mask > 0] = bundle.norm.mean
    condition_norm = bundle.norm.encode(condition_log).astype(np.float32)
    inputs = np.zeros((11, bundle.rows, bundle.cols), dtype=np.float32)
    inputs[0] = condition_norm
    inputs[1] = known_mask
    inputs[2] = target_mask
    inputs[3] = hard_mask
    inputs[9] = bundle.x_coord
    inputs[10] = bundle.y_coord
    sample = {
        "inputs": inputs,
        "truth_norm": bundle.norm.encode(bundle.full_log).astype(np.float32),
        "condition_norm": condition_norm,
        "known_mask": known_mask,
        "target_mask": target_mask,
        "hard_mask": hard_mask,
    }
    return sample, hard_count


def save_hard_locations(
    path: Path,
    hard_mask: np.ndarray,
    bundle: GeochemDataBundle,
) -> None:
    rows = []
    for row, col in np.argwhere(hard_mask > 0):
        rows.append(
            {
                "row": int(row),
                "col": int(col),
                "x": float(bundle.x_coords_1d[col]),
                "y": float(bundle.y_coords_1d[row]),
                "truth_ag": float(bundle.full_ag[row, col]),
            }
        )
    fieldnames = ["row", "col", "x", "y", "truth_ag"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    result: dict[str, object] = {"mask_runs": rows}
    metric_names = [
        "ensemble_mean_r2_ag",
        "ensemble_mean_mae_ag",
        "ensemble_mean_rmse_ag",
        "single_sample_r2_best",
        "single_sample_r2_mean",
    ]
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    return result


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.hard_fraction < 1.0):
        raise ValueError("hard-fraction must be in [0, 1)")
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = dict(checkpoint["config"])
    config["device"] = args.device
    config["final_generation_samples"] = args.samples_per_mask
    config["final_sample_batch_size"] = args.sample_batch_size
    config["seed"] = args.diffusion_seed

    model = ConditionalResidualDenoiser(
        in_channels=11,
        channels=int(config["channels"]),
        foundation_blocks=int(config["foundation_blocks"]),
        heads=int(config["attention_heads"]),
        freeze_foundation=not bool(config["train_foundation"]),
        foundation_type=str(config["foundation_type"]),
        timm_model=str(config["timm_model"]),
        token_grid=tuple(int(value) for value in config["token_grid"]),
        pretrained_foundation=False,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    schedule = CosineDiffusionSchedule(
        timesteps=int(config["timesteps"]),
        cosine_s=float(config["cosine_s"]),
        max_beta=float(config["max_beta"]),
    ).to(args.device)
    bundle = GeochemDataBundle(
        DataPaths(
            full_grid=str(project_path(str(config["full_grid"]))),
            block_mask=str(project_path(str(config["block_mask"]))),
            hard_mask=str(project_path(str(config["hard_mask"]))),
            target_mask=str(project_path(str(config["target_mask"]))),
        )
    )

    original_hard = bundle.hard_mask.astype(bool)
    rows = []
    for repeat in range(args.mask_repeats):
        mask_seed = args.hard_seed + repeat
        sample, hard_count = make_random_hard_sample(bundle, args.hard_fraction, mask_seed)
        run_dir = output_dir / f"mask_{repeat:02d}_seed_{mask_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_grid_csv(sample["hard_mask"], bundle.full_df, run_dir / "random_hard_mask.csv")
        save_hard_locations(run_dir / "random_hard_locations.csv", sample["hard_mask"], bundle)
        summary = evaluate_case_a(model, schedule, bundle, sample, config, run_dir)
        summary.update(
            {
                "mask_id": repeat,
                "hard_seed": mask_seed,
                "hard_fraction_requested": args.hard_fraction,
                "hard_pixels": hard_count,
                "hard_fraction_actual": hard_count / float(bundle.block_mask.sum()),
                "target_pixels": int(sample["target_mask"].sum()),
                "overlap_with_original_hard_pixels": int(
                    (sample["hard_mask"].astype(bool) & original_hard).sum()
                ),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint["best_epoch"]),
            }
        )
        save_json(summary, run_dir / "eval_50_samples" / "metrics_summary.json")
        rows.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    aggregate_summary = aggregate(rows)
    aggregate_summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["best_epoch"]),
            "hard_fraction_requested": args.hard_fraction,
            "mask_repeats": args.mask_repeats,
            "samples_per_mask": args.samples_per_mask,
            "hard_seed": args.hard_seed,
            "diffusion_seed": args.diffusion_seed,
        }
    )
    save_json(aggregate_summary, output_dir / "aggregate_summary.json")
    with (output_dir / "mask_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "mask_id",
            "hard_seed",
            "hard_pixels",
            "target_pixels",
            "overlap_with_original_hard_pixels",
            "ensemble_mean_r2_ag",
            "ensemble_mean_mae_ag",
            "ensemble_mean_rmse_ag",
            "single_sample_r2_best",
            "single_sample_r2_mean",
            "single_sample_r2_worst",
            "best_sample_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
