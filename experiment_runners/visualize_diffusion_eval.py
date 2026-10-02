from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a target-block comparison from saved diffusion samples."
    )
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument(
        "--hard-mask",
        type=Path,
        default=None,
        help="Optional grid CSV; marked points are included in the block crop and overlaid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    archive = np.load(eval_dir / "generated_samples_ag.npz")
    samples = archive["samples"]
    mean = archive["mean"]
    std = archive["std"]
    truth = archive["truth"]
    target_mask = archive["target_mask"].astype(bool)
    if args.hard_mask is not None:
        hard_mask = pd.read_csv(args.hard_mask.resolve(), index_col=0).to_numpy() > 0
    else:
        hard_mask = np.zeros_like(target_mask, dtype=bool)
    full_block_mask = target_mask | hard_mask

    with (eval_dir / "sample_metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    r2_values = np.asarray([float(row["target_r2_ag"]) for row in rows])
    best_index = int(np.nanargmax(r2_values))
    best = samples[best_index]

    target_rows, target_cols = np.where(full_block_mask)
    row_slice = slice(int(target_rows.min()), int(target_rows.max()) + 1)
    col_slice = slice(int(target_cols.min()), int(target_cols.max()) + 1)
    block_mask = full_block_mask[row_slice, col_slice]
    evaluation_mask = target_mask[row_slice, col_slice]
    hard_block = hard_mask[row_slice, col_slice]

    def target_block(array: np.ndarray, mask: np.ndarray = block_mask) -> np.ndarray:
        return np.where(mask, array[row_slice, col_slice], np.nan)

    truth_block = target_block(truth)
    mean_block = target_block(mean)
    best_block = target_block(best)
    error_block = target_block(np.abs(mean - truth), evaluation_mask)
    std_block = target_block(std, evaluation_mask)
    data_min = float(np.nanmin(truth_block))
    data_max = float(np.nanmax(truth_block))

    panels = [
        (
            f"Truth ({int(full_block_mask.sum())} block pixels; {int(hard_mask.sum())} hard)",
            truth_block,
            "viridis",
            data_min,
            data_max,
            "Ag",
        ),
        ("50-sample ensemble mean", mean_block, "viridis", data_min, data_max, "Ag"),
        (
            f"Best sample {best_index} | R²={r2_values[best_index]:.3f}",
            best_block,
            "viridis",
            data_min,
            data_max,
            "Ag",
        ),
        ("Ensemble absolute error", error_block, "magma", 0.0, None, "|error|"),
        ("Generation uncertainty", std_block, "magma", 0.0, None, "std"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(22, 5.2), constrained_layout=True)
    for axis, (title, values, cmap, vmin, vmax, label) in zip(axes, panels):
        image = axis.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Block column")
        axis.set_ylabel("Block row")
        if hard_block.any():
            hard_rows, hard_cols = np.where(hard_block)
            axis.scatter(
                hard_cols,
                hard_rows,
                s=28,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )
        fig.colorbar(image, ax=axis, shrink=0.82, label=label)
    if hard_block.any():
        title = "Case A inference with randomly sampled hard-data locations"
    else:
        title = "Case A inference without hard-data locations"
    fig.suptitle(title, fontsize=15)
    output_path = eval_dir / "target_block_zoom_comparison.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
