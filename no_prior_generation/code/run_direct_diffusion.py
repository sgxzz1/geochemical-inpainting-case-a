from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from cached_data import CachedGeochemDataset
from data import DataPaths, GeochemDataBundle
from model import hard_replace
from residual_diffusion import ConditionalResidualDenoiser, DiffusionSchedule
from utils import masked_mae_np, masked_r2_np, masked_rmse, save_grid_csv, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train direct conditional diffusion for geochemical inpainting. "
            "Unlike residual diffusion, this model directly denoises target Ag "
            "in normalized log space and does not add a trend prior back."
        )
    )
    parser.add_argument("--cache-dir", default="/mnt/storatge/ljj/生成模型/模型结果/cached_pseudo_blocks_cascade_1024")
    parser.add_argument("--full-grid", default="/mnt/storatge/ljj/生成模型/真实数据/ag_ok_back_500m.csv")
    parser.add_argument("--block-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_block_mask_15pct.csv")
    parser.add_argument("--hard-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_hard_mask_block15_hard5.csv")
    parser.add_argument("--target-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_target_mask_block15_hard5.csv")
    parser.add_argument("--output-dir", default="/mnt/storatge/ljj/生成模型/模型结果/direct_diffusion_50samples")
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
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--ds-level", type=int, default=None)
    parser.add_argument("--ds-ensemble", type=int, default=24)
    parser.add_argument("--ds-method", choices=["cascade", "legacy"], default=None)
    parser.add_argument("--kriging-backend", choices=["auto", "pykrige", "idw"], default=None)
    parser.add_argument("--variogram-model", default=None)
    parser.add_argument("--input-priors", choices=["full", "no_ds", "no_kriging", "no_priors"], default="full")
    parser.add_argument(
        "--remove-hard-data",
        action="store_true",
        help=(
            "Remove block-internal hard data from condition_ag and hard_mask. "
            "Those pixels are moved into target_mask, so the whole block must be generated."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_token_grid(value: str) -> tuple[int, int]:
    left, right = value.split(",")
    return int(left), int(right)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_value(args, metadata: dict, key: str, default):
    value = getattr(args, key)
    if value is not None:
        return value
    return metadata.get(key, default)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def apply_input_priors(inputs: torch.Tensor, mode: str) -> torch.Tensor:
    out = inputs.clone()
    if mode == "full":
        return out
    if mode in {"no_ds", "no_priors"}:
        out[:, 4:5] = 0.0
        out[:, 5:6] = 0.0
        out[:, 6:7] = 0.0
    if mode in {"no_kriging", "no_priors"}:
        out[:, 7:8] = 0.0
        out[:, 8:9] = 0.0
    return out


def remove_hard_data_from_tensors(
    inputs: torch.Tensor,
    condition: torch.Tensor,
    known_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn block-internal hard data into ordinary target pixels.

    Normalized condition values use zero as the training mean, so setting hard
    pixels to 0 hides their Ag value without changing tensor shapes.
    """
    inputs = inputs.clone()
    condition = condition.clone()
    known_mask = known_mask.clone()
    target_mask = target_mask.clone()
    hard_mask = inputs[:, 3:4].clone()

    target_mask = torch.clamp(target_mask + hard_mask, 0.0, 1.0)
    known_mask = known_mask * (1.0 - hard_mask)
    condition = condition * (1.0 - hard_mask)

    inputs[:, 0:1] = condition
    inputs[:, 1:2] = known_mask
    inputs[:, 2:3] = target_mask
    inputs[:, 3:4] = 0.0
    return inputs, condition, known_mask, target_mask


def make_loader(dataset: CachedGeochemDataset, args, epoch: int, seed_offset: int = 0) -> DataLoader:
    if args.samples_per_epoch > 0:
        generator = torch.Generator()
        generator.manual_seed(args.seed + seed_offset + epoch * 1009)
        sampler = RandomSampler(dataset, replacement=True, num_samples=args.samples_per_epoch, generator=generator)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )


def diffusion_loss(model, schedule, batch, args) -> torch.Tensor:
    inputs = batch["inputs"].to(args.device)
    condition = batch["condition_norm"].to(args.device)
    known_mask = batch["known_mask"].to(args.device)
    target_mask = batch["target_mask"].to(args.device)
    if args.remove_hard_data:
        inputs, condition, known_mask, target_mask = remove_hard_data_from_tensors(
            inputs=inputs,
            condition=condition,
            known_mask=known_mask,
            target_mask=target_mask,
        )
    inputs = apply_input_priors(inputs, args.input_priors)
    truth = batch["truth_norm"].to(args.device)
    x0 = truth * target_mask
    t = torch.randint(0, schedule.timesteps, (inputs.shape[0],), device=args.device, dtype=torch.long)
    noise = torch.randn_like(x0) * target_mask
    noisy_target = schedule.q_sample(x0, t, noise) * target_mask
    pred_noise = model(inputs, noisy_target, t)
    return masked_mse(pred_noise, noise, target_mask)


@torch.no_grad()
def validate_noise_loss(model, schedule, loader, args) -> float:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= args.val_batches:
            break
        losses.append(float(diffusion_loss(model, schedule, batch, args).item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def sample_direct(model, schedule, inputs, target_mask, num_samples: int, seed: int) -> torch.Tensor:
    model.eval()
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(seed)
    samples = []
    for _ in range(num_samples):
        x = torch.randn(
            (1, 1, inputs.shape[-2], inputs.shape[-1]),
            device=inputs.device,
            generator=generator,
        ) * target_mask
        for step in reversed(range(schedule.timesteps)):
            t = torch.full((1,), step, device=inputs.device, dtype=torch.long)
            pred_noise = model(inputs, x, t) * target_mask
            noise = torch.randn(x.shape, device=inputs.device, generator=generator) if step > 0 else torch.zeros_like(x)
            x = schedule.p_step(x, t, pred_noise, noise=noise) * target_mask
        samples.append(x.detach().cpu())
    return torch.cat(samples, dim=0)


def to_ag(norm_array: np.ndarray, bundle: GeochemDataBundle) -> np.ndarray:
    log_array = bundle.norm.decode(norm_array)
    return np.maximum(np.expm1(log_array), 0).astype(np.float32)


def mask_rectangle(mask: np.ndarray, template) -> tuple[float, float, float, float]:
    rr, cc = np.where(mask > 0)
    xs = template.columns.to_numpy(dtype=float)
    ys = template.index.to_numpy(dtype=float)
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
    x0 = float(xs[cc.min()] - dx / 2.0)
    x1 = float(xs[cc.max()] + dx / 2.0)
    y0 = float(ys[rr.min()] - dy / 2.0)
    y1 = float(ys[rr.max()] + dy / 2.0)
    return x0, y0, x1 - x0, y1 - y0


def plot_maps(maps, target_mask, template, out_path: Path, title: str, shared_truth_scale=None, cmap="viridis", label="Ag"):
    extent = [
        float(template.columns.min()),
        float(template.columns.max()),
        float(template.index.min()),
        float(template.index.max()),
    ]
    scale_arrays = [shared_truth_scale] if shared_truth_scale is not None else [m[1] for m in maps]
    vmin = float(np.nanmin(scale_arrays))
    vmax = float(np.nanmax(scale_arrays))
    rect = mask_rectangle(target_mask, template)
    fig = plt.figure(figsize=(5.8 * len(maps) + 0.8, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, len(maps) + 1, width_ratios=[1.0] * len(maps) + [0.04], wspace=0.05)
    last_im = None
    for i, (subtitle, array) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        last_im = ax.imshow(array, extent=extent, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, edgecolor="red", linewidth=2.0))
        ax.set_title(subtitle)
        ax.set_xlabel("CX")
        ax.set_ylabel("CY")
    cax = fig.add_subplot(gs[0, len(maps)])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(label)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_sample_metrics(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_fixed_block(model, schedule, bundle, args, ds_level, ds_method, kriging_backend, variogram_model, out_dir: Path) -> dict:
    sample = bundle.validation_sample(
        ds_level=ds_level,
        ds_ensemble=args.ds_ensemble,
        ds_method=ds_method,
        seed=999,
        kriging_backend=kriging_backend,
        variogram_model=variogram_model,
    )
    inputs = torch.from_numpy(sample["inputs"][None, ...]).to(args.device)
    condition = torch.from_numpy(sample["condition_norm"][None, None, ...]).to(args.device)
    known_mask = torch.from_numpy(sample["known_mask"][None, None, ...]).to(args.device)
    target_mask_t = torch.from_numpy(sample["target_mask"][None, None, ...]).to(args.device)
    original_target_mask = sample["target_mask"].astype(np.float32)
    if args.remove_hard_data:
        inputs, condition, known_mask, target_mask_t = remove_hard_data_from_tensors(
            inputs=inputs,
            condition=condition,
            known_mask=known_mask,
            target_mask=target_mask_t,
        )
    inputs = apply_input_priors(inputs, args.input_priors)
    generation_mask = target_mask_t.detach().cpu().numpy()[0, 0].astype(np.float32)
    pred_norm_samples = sample_direct(model, schedule, inputs, target_mask_t, args.num_samples, args.seed + 77)
    completed_samples = []
    for i in range(args.num_samples):
        completed = hard_replace(pred_norm_samples[i : i + 1].to(args.device), condition, known_mask, target_mask_t)
        completed_samples.append(completed.cpu().numpy()[0, 0])

    pred_ag_samples = np.stack([to_ag(arr, bundle) for arr in completed_samples], axis=0)
    truth_ag = bundle.full_ag
    target_mask = original_target_mask
    rows = []
    for i, pred_ag in enumerate(pred_ag_samples):
        rows.append(
            {
                "sample_id": i,
                "target_mae_ag": masked_mae_np(pred_ag, truth_ag, target_mask),
                "target_rmse_ag": masked_rmse(pred_ag, truth_ag, target_mask),
                "target_r2_ag": masked_r2_np(pred_ag, truth_ag, target_mask),
            }
        )
    r2_values = np.array([row["target_r2_ag"] for row in rows], dtype=np.float64)
    mae_values = np.array([row["target_mae_ag"] for row in rows], dtype=np.float64)
    rmse_values = np.array([row["target_rmse_ag"] for row in rows], dtype=np.float64)
    mean_pred_ag = pred_ag_samples.mean(axis=0)
    std_pred_ag = pred_ag_samples.std(axis=0)
    generation_r2_values = np.array(
        [masked_r2_np(pred_ag, truth_ag, generation_mask) for pred_ag in pred_ag_samples],
        dtype=np.float64,
    )
    generation_mae_values = np.array(
        [masked_mae_np(pred_ag, truth_ag, generation_mask) for pred_ag in pred_ag_samples],
        dtype=np.float64,
    )
    generation_rmse_values = np.array(
        [masked_rmse(pred_ag, truth_ag, generation_mask) for pred_ag in pred_ag_samples],
        dtype=np.float64,
    )
    best_idx = int(np.nanargmax(r2_values))
    worst_idx = int(np.nanargmin(r2_values))
    eval_dir = out_dir / "eval_50_samples"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "direct_diffusion",
        "input_priors": args.input_priors,
        "remove_hard_data": bool(args.remove_hard_data),
        "num_samples": int(args.num_samples),
        "metric_mask": "original_target_mask_excluding_hard_data",
        "single_sample_r2_mean": float(np.nanmean(r2_values)),
        "single_sample_r2_std": float(np.nanstd(r2_values)),
        "single_sample_r2_best": float(r2_values[best_idx]),
        "single_sample_r2_worst": float(r2_values[worst_idx]),
        "single_sample_mae_mean": float(np.nanmean(mae_values)),
        "single_sample_rmse_mean": float(np.nanmean(rmse_values)),
        "ensemble_mean_r2_ag": masked_r2_np(mean_pred_ag, truth_ag, target_mask),
        "ensemble_mean_mae_ag": masked_mae_np(mean_pred_ag, truth_ag, target_mask),
        "ensemble_mean_rmse_ag": masked_rmse(mean_pred_ag, truth_ag, target_mask),
        "best_sample_id": best_idx,
        "worst_sample_id": worst_idx,
        "target_pixels": int(np.sum(target_mask)),
        "generation_mask": "target_mask_plus_hard_mask" if args.remove_hard_data else "original_target_mask",
        "generation_pixels": int(np.sum(generation_mask)),
        "generation_single_sample_r2_mean": float(np.nanmean(generation_r2_values)),
        "generation_single_sample_r2_std": float(np.nanstd(generation_r2_values)),
        "generation_single_sample_mae_mean": float(np.nanmean(generation_mae_values)),
        "generation_single_sample_rmse_mean": float(np.nanmean(generation_rmse_values)),
        "generation_ensemble_mean_r2_ag": masked_r2_np(mean_pred_ag, truth_ag, generation_mask),
        "generation_ensemble_mean_mae_ag": masked_mae_np(mean_pred_ag, truth_ag, generation_mask),
        "generation_ensemble_mean_rmse_ag": masked_rmse(mean_pred_ag, truth_ag, generation_mask),
        "ds_method": ds_method,
        "ds_level": ds_level,
        "ds_ensemble": args.ds_ensemble,
        "kriging_backend": str(sample["priors"].get("kriging_backend", kriging_backend)),
        "variogram_model": variogram_model,
    }
    save_json(summary, eval_dir / "metrics_summary.json")
    write_sample_metrics(rows, eval_dir / "sample_metrics.csv")
    np.savez_compressed(
        eval_dir / "generated_samples_ag.npz",
        samples=pred_ag_samples.astype(np.float32),
        mean=mean_pred_ag.astype(np.float32),
        std=std_pred_ag.astype(np.float32),
        target_mask=target_mask.astype(np.float32),
        generation_mask=generation_mask.astype(np.float32),
        truth=truth_ag.astype(np.float32),
    )
    save_grid_csv(mean_pred_ag, bundle.full_df, eval_dir / "ensemble_mean_ag.csv")
    save_grid_csv(std_pred_ag, bundle.full_df, eval_dir / "ensemble_std_ag.csv")
    save_grid_csv(pred_ag_samples[best_idx], bundle.full_df, eval_dir / f"best_sample_{best_idx:03d}_ag.csv")
    plot_maps(
        [
            ("Truth Ag", truth_ag),
            (f"Direct diffusion mean | R2={summary['ensemble_mean_r2_ag']:.3f}", mean_pred_ag),
            (f"Best {best_idx} | R2={summary['single_sample_r2_best']:.3f}", pred_ag_samples[best_idx]),
        ],
        generation_mask,
        bundle.full_df,
        eval_dir / "truth_mean_best_comparison.png",
        "Direct conditional diffusion fixed-block generation",
        shared_truth_scale=truth_ag,
    )
    plot_maps(
        [("Direct diffusion generated std", std_pred_ag * generation_mask)],
        generation_mask,
        bundle.full_df,
        eval_dir / "generation_uncertainty_std.png",
        "Direct diffusion generated uncertainty",
        shared_truth_scale=None,
        cmap="magma",
        label="Ag std",
    )
    return summary


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
    token_grid = parse_token_grid(args.token_grid)
    model = ConditionalResidualDenoiser(
        channels=args.channels,
        foundation_blocks=args.foundation_blocks,
        heads=args.attention_heads,
        freeze_foundation=not args.train_foundation,
        foundation_type=args.foundation_type,
        timm_model=args.timm_model,
        token_grid=token_grid,
        pretrained_foundation=not args.no_pretrained_foundation,
    ).to(args.device)
    schedule = DiffusionSchedule(args.timesteps, args.beta_start, args.beta_end).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    run_metadata = {
        "model": "direct_diffusion",
        "cache_dir": str(cache_dir),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "target_definition": "x0 = truth_norm, no residual trend is added back",
        "input_priors": args.input_priors,
        "remove_hard_data": bool(args.remove_hard_data),
        "effective_condition_when_no_priors_no_hard": (
            "condition_ag outside block, known_mask, target_mask, x_coord, y_coord; "
            "DS/Kriging channels are zero and block-internal hard data are hidden."
        ),
        "normalization_log_mean": bundle.norm.mean,
        "normalization_log_std": bundle.norm.std,
        "ds_method": ds_method,
        "ds_level": ds_level,
        "ds_ensemble_eval": args.ds_ensemble,
        "kriging_backend": kriging_backend,
        "variogram_model": variogram_model,
    }
    save_json(cache_metadata, out_dir / "cache_metadata.json")
    save_json(run_metadata, out_dir / "data_metadata.json")

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        loader = make_loader(train_dataset, args, epoch)
        for batch in loader:
            loss = diffusion_loss(model, schedule, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_loader = make_loader(val_dataset, args, epoch, seed_offset=900000)
        val_loss = validate_noise_loss(model, schedule, val_loader, args)
        train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_noise_mse": train_loss, "val_noise_mse": val_loss}
        history.append(row)
        print(f"epoch={epoch:04d} train_noise_mse={train_loss:.6f} val_noise_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "cache_metadata": cache_metadata,
                    "run_metadata": run_metadata,
                    "best_val_noise_mse": best_val,
                    "norm_mean": bundle.norm.mean,
                    "norm_std": bundle.norm.std,
                },
                out_dir / "best_direct_diffusion_model.pt",
            )
        if epoch % 25 == 0:
            save_json({"history": history}, out_dir / "history.json")
    save_json({"history": history}, out_dir / "history.json")

    ckpt = torch.load(out_dir / "best_direct_diffusion_model.pt", map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    summary = evaluate_fixed_block(model, schedule, bundle, args, ds_level, ds_method, kriging_backend, variogram_model, out_dir)
    print(summary)


if __name__ == "__main__":
    main()
