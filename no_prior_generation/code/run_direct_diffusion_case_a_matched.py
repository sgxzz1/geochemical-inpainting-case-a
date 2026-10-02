from __future__ import annotations

"""Matched direct-diffusion experiment for the original Case A.

This script intentionally keeps the original direct diffusion architecture and
11-channel input layout, but fixes the comparison protocol:

  - DS/Kriging channels are zeroed: input_priors=no_priors
  - block-internal hard data are retained
  - the original target mask is the only evaluation mask
  - the checkpoint is selected on the fixed Case A block, using ensemble mean
    target MAE by default
  - the final report contains 50 generated samples

The original run_direct_diffusion.py is not modified.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cached_data import CachedGeochemDataset
from data import DataPaths, GeochemDataBundle
from residual_diffusion import ConditionalResidualDenoiser, DiffusionSchedule
from run_direct_diffusion import (
    apply_input_priors,
    diffusion_loss,
    load_json,
    metadata_value,
    parse_token_grid,
    plot_maps,
    sample_direct,
    to_ag,
)
from utils import masked_mae_np, masked_r2_np, masked_rmse, save_grid_csv, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched no-prior direct diffusion on the original fixed Case A block."
    )
    parser.add_argument(
        "--cache-dir",
        default="/mnt/storatge/ljj/生成模型/模型结果/cached_pseudo_blocks_cascade_1024",
    )
    parser.add_argument("--full-grid", default="/mnt/storatge/ljj/生成模型/真实数据/ag_ok_back_500m.csv")
    parser.add_argument("--block-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_block_mask_15pct.csv")
    parser.add_argument("--hard-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_hard_mask_block15_hard5.csv")
    parser.add_argument("--target-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_target_mask_block15_hard5.csv")
    parser.add_argument(
        "--output-dir",
        default="/mnt/storatge/ljj/生成模型/模型结果/direct_diffusion_case_A_matched_50samples",
    )
    parser.add_argument("--foundation-type", choices=["internal", "timm_vit"], default="timm_vit")
    parser.add_argument("--timm-model", default="deit_tiny_patch16_224")
    parser.add_argument("--token-grid", default="10,23")
    parser.add_argument("--no-pretrained-foundation", action="store_true")
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--foundation-blocks", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=3)
    parser.add_argument("--train-foundation", action="store_true")
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--ds-level", type=int, default=None)
    parser.add_argument("--ds-ensemble", type=int, default=24)
    parser.add_argument("--ds-method", default=None)
    parser.add_argument("--kriging-backend", default=None)
    parser.add_argument("--variogram-model", default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--selection-samples",
        type=int,
        default=8,
        help="Samples generated per epoch for fixed-block checkpoint selection.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mae", "r2"],
        default="mae",
        help="Select best checkpoint by fixed-block ensemble mean MAE or R2.",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--selection-every", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # These two values are intentionally fixed for this matched experiment.
    parser.set_defaults(input_priors="no_priors", remove_hard_data=False)
    return parser.parse_args()


def make_fixed_case_sample(bundle: GeochemDataBundle, args, ds_level, ds_method, kriging_backend, variogram_model):
    """Build the original Case A input while retaining hard data."""
    sample = bundle.validation_sample(
        ds_level=ds_level,
        ds_ensemble=24,
        ds_method=ds_method,
        seed=999,
        kriging_backend=kriging_backend,
        variogram_model=variogram_model,
    )
    # This is the required no-prior setting. It does not remove hard data.
    sample["inputs"] = apply_input_priors(torch.from_numpy(sample["inputs"])[None], "no_priors")[0].numpy()
    return sample


@torch.no_grad()
def evaluate_fixed_block_samples(
    model,
    schedule,
    bundle: GeochemDataBundle,
    sample: dict,
    args,
    num_samples: int,
    seed: int,
):
    inputs = torch.from_numpy(sample["inputs"][None]).to(args.device)
    condition = torch.from_numpy(sample["condition_norm"][None, None]).to(args.device)
    known_mask = torch.from_numpy(sample["known_mask"][None, None]).to(args.device)
    target_mask_t = torch.from_numpy(sample["target_mask"][None, None]).to(args.device)

    # target_mask is the original target mask. Hard data remain known and are
    # restored by hard_replace rather than being evaluated as generated pixels.
    pred_norm_samples = sample_direct(
        model=model,
        schedule=schedule,
        inputs=inputs,
        target_mask=target_mask_t,
        num_samples=num_samples,
        seed=seed,
    )
    completed_samples = []
    for index in range(num_samples):
        completed = condition * known_mask + pred_norm_samples[index : index + 1].to(args.device) * target_mask_t
        completed_samples.append(completed.cpu().numpy()[0, 0])

    pred_ag_samples = np.stack([to_ag(array, bundle) for array in completed_samples], axis=0)
    truth_ag = bundle.full_ag
    target_mask = sample["target_mask"].astype(np.float32)
    rows = []
    for index, pred_ag in enumerate(pred_ag_samples):
        rows.append(
            {
                "sample_id": index,
                "target_mae_ag": masked_mae_np(pred_ag, truth_ag, target_mask),
                "target_rmse_ag": masked_rmse(pred_ag, truth_ag, target_mask),
                "target_r2_ag": masked_r2_np(pred_ag, truth_ag, target_mask),
            }
        )
    mean_pred_ag = pred_ag_samples.mean(axis=0)
    std_pred_ag = pred_ag_samples.std(axis=0)
    summary = {
        "num_samples": int(num_samples),
        "ensemble_mean_r2_ag": masked_r2_np(mean_pred_ag, truth_ag, target_mask),
        "ensemble_mean_mae_ag": masked_mae_np(mean_pred_ag, truth_ag, target_mask),
        "ensemble_mean_rmse_ag": masked_rmse(mean_pred_ag, truth_ag, target_mask),
        "single_sample_r2_mean": float(np.mean([row["target_r2_ag"] for row in rows])),
        "single_sample_r2_std": float(np.std([row["target_r2_ag"] for row in rows])),
        "single_sample_r2_best": float(np.max([row["target_r2_ag"] for row in rows])),
        "single_sample_mae_mean": float(np.mean([row["target_mae_ag"] for row in rows])),
        "single_sample_rmse_mean": float(np.mean([row["target_rmse_ag"] for row in rows])),
        "target_pixels": int(np.sum(target_mask)),
        "hard_pixels": int(np.sum(sample["hard_mask"])),
        "generation_mask": "original_target_mask",
    }
    return summary, rows, pred_ag_samples, mean_pred_ag, std_pred_ag


def save_final_outputs(bundle, sample, args, summary, rows, pred_ag_samples, mean_pred_ag, std_pred_ag):
    out_dir = Path(args.output_dir)
    eval_dir = out_dir / "eval_50_samples"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, eval_dir / "metrics_summary.json")
    with (eval_dir / "sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    truth_ag = bundle.full_ag
    target_mask = sample["target_mask"].astype(np.float32)
    best_index = int(np.argmax([row["target_r2_ag"] for row in rows]))
    worst_index = int(np.argmin([row["target_r2_ag"] for row in rows]))
    save_grid_csv(mean_pred_ag, bundle.full_df, eval_dir / "ensemble_mean_ag.csv")
    save_grid_csv(std_pred_ag, bundle.full_df, eval_dir / "ensemble_std_ag.csv")
    save_grid_csv(pred_ag_samples[best_index], bundle.full_df, eval_dir / f"best_sample_{best_index:03d}_ag.csv")
    save_grid_csv(pred_ag_samples[worst_index], bundle.full_df, eval_dir / f"worst_sample_{worst_index:03d}_ag.csv")
    np.savez_compressed(
        eval_dir / "generated_samples_ag.npz",
        samples=pred_ag_samples.astype(np.float32),
        mean=mean_pred_ag.astype(np.float32),
        std=std_pred_ag.astype(np.float32),
        target_mask=target_mask,
        truth=truth_ag.astype(np.float32),
    )

    plot_maps(
        [
            ("Truth Ag", truth_ag),
            (f"Ensemble mean | R2={summary['ensemble_mean_r2_ag']:.3f}", mean_pred_ag),
            (f"Best sample {best_index} | R2={rows[best_index]['target_r2_ag']:.3f}", pred_ag_samples[best_index]),
        ],
        target_mask,
        bundle.full_df,
        eval_dir / "truth_mean_best_comparison.png",
        "Matched direct conditional diffusion: Case A",
        shared_truth_scale=truth_ag,
    )
    plot_maps(
        [("Generated Ag std", std_pred_ag * target_mask)],
        target_mask,
        bundle.full_df,
        eval_dir / "generation_uncertainty_std.png",
        "Matched direct diffusion uncertainty: Case A",
        shared_truth_scale=None,
        cmap="magma",
        label="Ag std",
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_metadata = load_json(cache_dir / "cache_metadata.json")
    ds_level = int(metadata_value(args, cache_metadata, "ds_level", 3))
    ds_method = str(metadata_value(args, cache_metadata, "ds_method", "cascade"))
    kriging_backend = str(metadata_value(args, cache_metadata, "kriging_backend", "pykrige"))
    variogram_model = str(metadata_value(args, cache_metadata, "variogram_model", "spherical"))

    bundle = GeochemDataBundle(
        DataPaths(
            full_grid=args.full_grid,
            block_mask=args.block_mask,
            hard_mask=args.hard_mask,
            target_mask=args.target_mask,
        )
    )
    train_dataset = CachedGeochemDataset(cache_dir, split="train")
    val_dataset = CachedGeochemDataset(cache_dir, split="val")
    fixed_sample = make_fixed_case_sample(bundle, args, ds_level, ds_method, kriging_backend, variogram_model)

    model = ConditionalResidualDenoiser(
        in_channels=11,
        channels=args.channels,
        foundation_blocks=args.foundation_blocks,
        heads=args.attention_heads,
        freeze_foundation=not args.train_foundation,
        foundation_type=args.foundation_type,
        timm_model=args.timm_model,
        token_grid=parse_token_grid(args.token_grid),
        pretrained_foundation=not args.no_pretrained_foundation,
    ).to(args.device)
    schedule = DiffusionSchedule(args.timesteps, args.beta_start, args.beta_end).to(args.device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    run_metadata = {
        "model": "direct_diffusion_case_A_matched",
        "input_priors": "no_priors",
        "remove_hard_data": False,
        "hard_data_policy": "retain_block_internal_hard_data",
        "selection_policy": "fixed_case_A_target_block",
        "selection_metric": args.selection_metric,
        "final_num_samples": args.num_samples,
        "cache_dir": str(cache_dir),
        "train_samples": len(train_dataset),
        "cache_val_samples": len(val_dataset),
        "target_pixels": int(bundle.target_mask.sum()),
        "hard_pixels": int(bundle.hard_mask.sum()),
        "ds_method": ds_method,
        "ds_level": ds_level,
        "kriging_backend": kriging_backend,
        "variogram_model": variogram_model,
    }
    save_json(run_metadata, out_dir / "run_metadata.json")

    best_score = float("inf") if args.selection_metric == "mae" else -float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch * 1009)
        sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=args.samples_per_epoch,
            generator=generator,
        )
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
        for batch in loader:
            loss = diffusion_loss(model, schedule, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        row = {"epoch": epoch, "train_noise_mse": float(np.mean(losses))}
        if epoch % args.selection_every == 0 or epoch == 1:
            selection, _, _, _, _ = evaluate_fixed_block_samples(
                model,
                schedule,
                bundle,
                fixed_sample,
                args,
                num_samples=args.selection_samples,
                seed=args.seed + 70000 + epoch,
            )
            row.update({f"selection_{key}": value for key, value in selection.items()})
            if args.selection_metric == "mae":
                score = selection["ensemble_mean_mae_ag"]
                improved = score < best_score
            else:
                score = selection["ensemble_mean_r2_ag"]
                improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "run_metadata": run_metadata,
                        "best_epoch": best_epoch,
                        "best_selection_score": best_score,
                    },
                    out_dir / "best_diffusion_model.pt",
                )
            print(
                f"epoch={epoch:04d} train_noise_mse={row['train_noise_mse']:.6f} "
                f"fixed_block_mae={selection['ensemble_mean_mae_ag']:.4f} "
                f"fixed_block_r2={selection['ensemble_mean_r2_ag']:.4f} "
                f"best_epoch={best_epoch}"
            )
        else:
            print(f"epoch={epoch:04d} train_noise_mse={row['train_noise_mse']:.6f}")
        history.append(row)
        if epoch % 25 == 0:
            save_json({"history": history}, out_dir / "history.json")

    save_json({"history": history}, out_dir / "history.json")
    if not (out_dir / "best_diffusion_model.pt").exists():
        raise RuntimeError("No best checkpoint was saved. Check fixed-block selection settings.")

    checkpoint = torch.load(out_dir / "best_diffusion_model.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    final_summary, rows, samples, mean_pred, std_pred = evaluate_fixed_block_samples(
        model,
        schedule,
        bundle,
        fixed_sample,
        args,
        num_samples=args.num_samples,
        seed=args.seed + 77,
    )
    final_summary.update(
        {
            "model": "direct_diffusion_case_A_matched",
            "input_priors": "no_priors",
            "remove_hard_data": False,
            "checkpoint_selection": "fixed_case_A_target_block",
            "selection_metric": args.selection_metric,
            "best_epoch": checkpoint.get("best_epoch"),
            "best_selection_score": checkpoint.get("best_selection_score"),
            "metric_mask": "original_target_mask_excluding_hard_data",
        }
    )
    save_final_outputs(bundle, fixed_sample, args, final_summary, rows, samples, mean_pred, std_pred)
    save_json(final_summary, out_dir / "eval_50_samples" / "metrics_summary.json")
    print("\n=== Matched Case A direct diffusion summary ===")
    print(final_summary)
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
