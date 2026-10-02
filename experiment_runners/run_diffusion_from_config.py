from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "no_prior_generation" / "code"
sys.path.insert(0, str(CODE_DIR))

from cached_data import CachedGeochemDataset  # noqa: E402
from data import DataPaths, GeochemDataBundle  # noqa: E402
from fractal_cascade import fractal_cascade_prior_stats  # noqa: E402
from residual_diffusion import ConditionalResidualDenoiser  # noqa: E402
from run_direct_diffusion import apply_input_priors, plot_maps, to_ag  # noqa: E402
from utils import masked_mae_np, masked_r2_np, masked_rmse, save_grid_csv, save_json, set_seed  # noqa: E402


class CosineDiffusionSchedule(nn.Module):
    """DDPM schedule whose terminal state is effectively pure Gaussian noise."""

    def __init__(self, timesteps: int, cosine_s: float = 0.008, max_beta: float = 0.999):
        super().__init__()
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        alpha_bar_curve = torch.cos(
            ((steps / timesteps + cosine_s) / (1.0 + cosine_s)) * math.pi / 2.0
        ).pow(2)
        alpha_bar_curve = alpha_bar_curve / alpha_bar_curve[0]
        betas = 1.0 - alpha_bar_curve[1:] / alpha_bar_curve[:-1]
        betas = betas.clamp(min=1e-8, max=max_beta).float()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_var", posterior_var.clamp_min(1e-20))

    def _gather(self, values: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, t).to(device=like.device, dtype=like.dtype)
        return out.view(-1, *([1] * (like.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._gather(self.sqrt_alpha_bars, t, x0) * x0 + self._gather(
            self.sqrt_one_minus_alpha_bars, t, x0
        ) * noise

    def p_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        beta_t = self._gather(self.betas, t, x_t)
        alpha_t = self._gather(self.alphas, t, x_t)
        alpha_bar_t = self._gather(self.alpha_bars, t, x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        )
        variance = self._gather(self.posterior_var, t, x_t)
        nonzero = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance) * noise

    def p_step_with_x0_clip(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        noise: torch.Tensor,
        clip_min: float,
        clip_max: float,
    ) -> torch.Tensor:
        """DDPM posterior step after constraining predicted clean data to its training range."""
        beta_t = self._gather(self.betas, t, x_t)
        alpha_t = self._gather(self.alphas, t, x_t)
        alpha_bar_t = self._gather(self.alpha_bars, t, x_t)
        alpha_bar_prev = self._gather(self.alpha_bars_prev, t, x_t)
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, min=clip_min, max=clip_max)
        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coef_x0 * pred_x0 + coef_xt * x_t
        variance = self._gather(self.posterior_var, t, x_t)
        nonzero = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance) * noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schedule") != "cosine":
        raise ValueError("This runner currently requires schedule=cosine")
    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_train_loader(dataset: CachedGeochemDataset, config: dict, epoch: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + epoch * 1009)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=int(config["samples_per_epoch"]),
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=str(config["device"]).startswith("cuda"),
    )


def prepare_batch(batch: dict[str, torch.Tensor], config: dict) -> tuple[torch.Tensor, ...]:
    device = str(config["device"])
    inputs = batch["inputs"].to(device)
    truth = batch["truth_norm"].to(device)
    condition = batch["condition_norm"].to(device)
    known_mask = batch["known_mask"].to(device)
    target_mask = batch["target_mask"].to(device)
    if bool(config.get("remove_hard_data", False)):
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
    inputs = apply_input_priors(inputs, str(config["input_priors"]))
    return inputs, truth, condition, known_mask, target_mask


def diffusion_loss(
    model: nn.Module,
    schedule: CosineDiffusionSchedule,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor:
    inputs, truth, _, _, target_mask = prepare_batch(batch, config)
    x0 = truth * target_mask
    t = torch.randint(
        0,
        schedule.timesteps,
        (inputs.shape[0],),
        device=inputs.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(x0) * target_mask
    noisy_target = schedule.q_sample(x0, t, noise) * target_mask
    pred_noise = model(inputs, noisy_target, t)
    denominator = target_mask.sum().clamp_min(1.0)
    return (((pred_noise - noise) ** 2) * target_mask).sum() / denominator


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def stack_items(dataset: CachedGeochemDataset, count: int) -> dict[str, torch.Tensor]:
    items = [dataset[index] for index in range(min(count, len(dataset)))]
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


@torch.no_grad()
def sample_batch(
    model: nn.Module,
    schedule: CosineDiffusionSchedule,
    inputs: torch.Tensor,
    target_mask: torch.Tensor,
    seed: int,
    x0_clip_min: float | None = None,
    x0_clip_max: float | None = None,
) -> torch.Tensor:
    model.eval()
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(seed)
    x = torch.randn(
        (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
        device=inputs.device,
        generator=generator,
    ) * target_mask
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((inputs.shape[0],), step, device=inputs.device, dtype=torch.long)
        pred_noise = model(inputs, x, t) * target_mask
        if step > 0:
            noise = torch.randn(x.shape, device=inputs.device, generator=generator)
        else:
            noise = torch.zeros_like(x)
        if x0_clip_min is not None and x0_clip_max is not None:
            x = schedule.p_step_with_x0_clip(
                x,
                t,
                pred_noise,
                noise,
                clip_min=x0_clip_min,
                clip_max=x0_clip_max,
            ) * target_mask
        else:
            x = schedule.p_step(x, t, pred_noise, noise) * target_mask
    return x


@torch.no_grad()
def generated_validation_mae(
    model: nn.Module,
    schedule: CosineDiffusionSchedule,
    fixed_batch: dict[str, torch.Tensor],
    config: dict,
) -> float:
    inputs, truth, condition, known_mask, target_mask = prepare_batch(fixed_batch, config)
    generated = sample_batch(
        model,
        schedule,
        inputs,
        target_mask,
        seed=int(config["validation_seed"]),
        x0_clip_min=config.get("x0_clip_min"),
        x0_clip_max=config.get("x0_clip_max"),
    )
    completed = known_mask * condition + target_mask * generated
    denominator = target_mask.sum().clamp_min(1.0)
    return float(((torch.abs(completed - truth) * target_mask).sum() / denominator).item())


def build_case_a_sample(
    bundle: GeochemDataBundle,
    remove_hard_data: bool = False,
    prior_config: dict | None = None,
) -> dict[str, np.ndarray]:
    if remove_hard_data:
        target_mask = bundle.block_mask.astype(np.float32)
        hard_mask = np.zeros_like(target_mask, dtype=np.float32)
    else:
        target_mask = bundle.target_mask.astype(np.float32)
        hard_mask = bundle.hard_mask.astype(np.float32)
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
    input_priors = "no_priors" if prior_config is None else str(prior_config["input_priors"])
    if input_priors in {"full", "no_kriging"}:
        cascade_condition = condition_norm.copy()
        cascade_condition[known_mask <= 0] = np.nan
        ds = fractal_cascade_prior_stats(
            condition=cascade_condition,
            original_known_values=condition_norm,
            original_known_mask=known_mask.astype(np.uint8),
            n_realizations=int(prior_config.get("ds_ensemble", 4)),
            seed=int(prior_config.get("case_a_ds_seed", 999)),
            max_downsample_levels=max(int(prior_config.get("ds_level", 3)), 1),
            downsample_fallback="random_known",
            distance_threshold=0.25,
            max_scan=256,
            max_radius=2,
            max_neighbors=12,
        )
        inputs[4] = ds.mean
        inputs[5] = ds.p75
        inputs[6] = ds.std
    return {
        "inputs": inputs,
        "truth_norm": bundle.norm.encode(bundle.full_log).astype(np.float32),
        "condition_norm": condition_norm,
        "known_mask": known_mask,
        "target_mask": target_mask,
        "hard_mask": hard_mask,
    }


@torch.no_grad()
def evaluate_case_a(
    model: nn.Module,
    schedule: CosineDiffusionSchedule,
    bundle: GeochemDataBundle,
    sample: dict[str, np.ndarray],
    config: dict,
    output_dir: Path,
) -> dict:
    device = str(config["device"])
    total_samples = int(config["final_generation_samples"])
    chunk_size = int(config["final_sample_batch_size"])
    base_inputs = torch.from_numpy(sample["inputs"][None]).to(device)
    base_target = torch.from_numpy(sample["target_mask"][None, None]).to(device)
    base_condition = torch.from_numpy(sample["condition_norm"][None, None]).to(device)
    base_known = torch.from_numpy(sample["known_mask"][None, None]).to(device)

    completed_norm = []
    for start in range(0, total_samples, chunk_size):
        count = min(chunk_size, total_samples - start)
        inputs = base_inputs.repeat(count, 1, 1, 1)
        target = base_target.repeat(count, 1, 1, 1)
        generated = sample_batch(
            model,
            schedule,
            inputs,
            target,
            seed=int(config["seed"]) + 77000 + start,
            x0_clip_min=config.get("x0_clip_min"),
            x0_clip_max=config.get("x0_clip_max"),
        )
        completed = base_known.repeat(count, 1, 1, 1) * base_condition.repeat(
            count, 1, 1, 1
        ) + target * generated
        completed_norm.append(completed.cpu().numpy()[:, 0])
    completed_norm_array = np.concatenate(completed_norm, axis=0)
    pred_ag_samples = np.stack([to_ag(array, bundle) for array in completed_norm_array], axis=0)
    truth_ag = bundle.full_ag
    target_mask = sample["target_mask"]

    rows = []
    for index, prediction in enumerate(pred_ag_samples):
        rows.append(
            {
                "sample_id": index,
                "target_mae_ag": masked_mae_np(prediction, truth_ag, target_mask),
                "target_rmse_ag": masked_rmse(prediction, truth_ag, target_mask),
                "target_r2_ag": masked_r2_np(prediction, truth_ag, target_mask),
            }
        )
    mean_prediction = pred_ag_samples.mean(axis=0)
    std_prediction = pred_ag_samples.std(axis=0)
    r2_values = np.asarray([row["target_r2_ag"] for row in rows])
    best_index = int(np.nanargmax(r2_values))
    worst_index = int(np.nanargmin(r2_values))
    summary = {
        "model": "direct_diffusion_cosine",
        "schedule": "cosine",
        "timesteps": schedule.timesteps,
        "terminal_alpha_bar": float(schedule.alpha_bars[-1].item()),
        "input_priors": str(config["input_priors"]),
        "num_samples": total_samples,
        "single_sample_r2_mean": float(np.nanmean(r2_values)),
        "single_sample_r2_std": float(np.nanstd(r2_values)),
        "single_sample_r2_best": float(r2_values[best_index]),
        "single_sample_r2_worst": float(r2_values[worst_index]),
        "single_sample_mae_mean": float(np.mean([row["target_mae_ag"] for row in rows])),
        "single_sample_rmse_mean": float(np.mean([row["target_rmse_ag"] for row in rows])),
        "ensemble_mean_r2_ag": masked_r2_np(mean_prediction, truth_ag, target_mask),
        "ensemble_mean_mae_ag": masked_mae_np(mean_prediction, truth_ag, target_mask),
        "ensemble_mean_rmse_ag": masked_rmse(mean_prediction, truth_ag, target_mask),
        "best_sample_id": best_index,
        "worst_sample_id": worst_index,
        "target_pixels": int(target_mask.sum()),
        "x0_clip_min": config.get("x0_clip_min"),
        "x0_clip_max": config.get("x0_clip_max"),
    }
    ds_mean_ag = None
    if str(config["input_priors"]) in {"full", "no_kriging"}:
        ds_mean_ag = to_ag(sample["inputs"][4], bundle)
        summary.update(
            {
                "ds_prior_mean_r2_ag": masked_r2_np(ds_mean_ag, truth_ag, target_mask),
                "ds_prior_mean_mae_ag": masked_mae_np(ds_mean_ag, truth_ag, target_mask),
                "ds_prior_mean_rmse_ag": masked_rmse(ds_mean_ag, truth_ag, target_mask),
            }
        )

    eval_dir = output_dir / "eval_50_samples"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, eval_dir / "metrics_summary.json")
    with (eval_dir / "sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        eval_dir / "generated_samples_ag.npz",
        samples=pred_ag_samples.astype(np.float32),
        mean=mean_prediction.astype(np.float32),
        std=std_prediction.astype(np.float32),
        target_mask=target_mask.astype(np.float32),
        truth=truth_ag.astype(np.float32),
    )
    save_grid_csv(mean_prediction, bundle.full_df, eval_dir / "ensemble_mean_ag.csv")
    save_grid_csv(std_prediction, bundle.full_df, eval_dir / "ensemble_std_ag.csv")
    if ds_mean_ag is not None:
        save_grid_csv(ds_mean_ag, bundle.full_df, eval_dir / "ds_prior_mean_ag.csv")
        plot_maps(
            [(f"DS mean prior | R2={summary['ds_prior_mean_r2_ag']:.3f}", ds_mean_ag)],
            target_mask,
            bundle.full_df,
            eval_dir / "ds_prior_mean_ag.png",
            "DS prior computed without block-internal hard data",
            shared_truth_scale=truth_ag,
        )
    save_grid_csv(
        pred_ag_samples[best_index], bundle.full_df, eval_dir / f"best_sample_{best_index:03d}_ag.csv"
    )
    plot_maps(
        [
            ("Truth Ag", truth_ag),
            (f"Cosine ensemble mean | R2={summary['ensemble_mean_r2_ag']:.3f}", mean_prediction),
            (f"Best sample {best_index} | R2={r2_values[best_index]:.3f}", pred_ag_samples[best_index]),
        ],
        target_mask,
        bundle.full_df,
        eval_dir / "truth_mean_best_comparison.png",
        "No-prior direct diffusion with cosine schedule",
        shared_truth_scale=truth_ag,
    )
    plot_maps(
        [("Generated Ag std", std_prediction * target_mask)],
        target_mask,
        bundle.full_df,
        eval_dir / "generation_uncertainty_std.png",
        "Cosine diffusion uncertainty",
        cmap="magma",
        label="Ag std",
    )
    return summary


def main() -> None:
    arguments = parse_args()
    config = load_config(arguments.config)
    set_seed(int(config["seed"]))
    device = str(config["device"])
    cache_dir = resolve_project_path(str(config["cache_dir"]))
    output_dir = resolve_project_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    train_dataset = CachedGeochemDataset(cache_dir, "train")
    val_dataset = CachedGeochemDataset(cache_dir, "val")
    fixed_validation_batch = stack_items(val_dataset, int(config["validation_generation_cases"]))

    model = ConditionalResidualDenoiser(
        in_channels=11,
        channels=int(config["channels"]),
        foundation_blocks=int(config["foundation_blocks"]),
        heads=int(config["attention_heads"]),
        freeze_foundation=not bool(config["train_foundation"]),
        foundation_type=str(config["foundation_type"]),
        timm_model=str(config["timm_model"]),
        token_grid=tuple(int(value) for value in config["token_grid"]),
        pretrained_foundation=bool(config["pretrained_foundation"]),
    ).to(device)
    schedule = CosineDiffusionSchedule(
        timesteps=int(config["timesteps"]),
        cosine_s=float(config["cosine_s"]),
        max_beta=float(config["max_beta"]),
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    run_metadata = {
        "config": config,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "terminal_alpha_bar": float(schedule.alpha_bars[-1].item()),
        "terminal_signal_coefficient": float(schedule.sqrt_alpha_bars[-1].item()),
        "terminal_noise_coefficient": float(schedule.sqrt_one_minus_alpha_bars[-1].item()),
        "checkpoint_selection": "fixed cached validation cases generated from pure noise",
    }
    save_json(run_metadata, output_dir / "run_metadata.json")
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2), flush=True)

    best_validation = float("inf")
    best_epoch = None
    history = []
    validation_every = int(config["validation_generation_every"])
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for batch in make_train_loader(train_dataset, config, epoch):
            loss = diffusion_loss(model, schedule, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            update_ema(ema_model, model, float(config["ema_decay"]))
            losses.append(float(loss.item()))

        row = {"epoch": epoch, "train_noise_mse": float(np.mean(losses))}
        if epoch % validation_every == 0:
            validation_mae = generated_validation_mae(
                ema_model, schedule, fixed_validation_batch, config
            )
            row["generated_val_mae_norm"] = validation_mae
            if np.isfinite(validation_mae) and validation_mae < best_validation:
                best_validation = validation_mae
                best_epoch = epoch
                torch.save(
                    {
                        "model": ema_model.state_dict(),
                        "raw_model": model.state_dict(),
                        "config": config,
                        "best_epoch": best_epoch,
                        "best_generated_val_mae_norm": best_validation,
                        "schedule": {
                            "name": "cosine",
                            "timesteps": schedule.timesteps,
                            "terminal_alpha_bar": float(schedule.alpha_bars[-1].item()),
                        },
                    },
                    output_dir / "best_cosine_diffusion_model.pt",
                )
            print(
                f"epoch={epoch:04d} train_noise_mse={row['train_noise_mse']:.6f} "
                f"generated_val_mae_norm={validation_mae:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch:04d} train_noise_mse={row['train_noise_mse']:.6f}",
                flush=True,
            )
        history.append(row)
        if epoch % 10 == 0:
            save_json({"history": history}, output_dir / "history.json")

    save_json({"history": history}, output_dir / "history.json")
    checkpoint_path = output_dir / "best_cosine_diffusion_model.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("No finite generated-validation checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ema_model.load_state_dict(checkpoint["model"], strict=True)

    bundle = GeochemDataBundle(
        DataPaths(
            full_grid=str(resolve_project_path(str(config["full_grid"]))),
            block_mask=str(resolve_project_path(str(config["block_mask"]))),
            hard_mask=str(resolve_project_path(str(config["hard_mask"]))),
            target_mask=str(resolve_project_path(str(config["target_mask"]))),
        )
    )
    case_sample = build_case_a_sample(
        bundle,
        remove_hard_data=bool(config.get("remove_hard_data", False)),
        prior_config=config,
    )
    summary = evaluate_case_a(
        ema_model,
        schedule,
        bundle,
        case_sample,
        config,
        output_dir,
    )
    summary.update(
        {
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_generated_val_mae_norm": float(checkpoint["best_generated_val_mae_norm"]),
            "ema_decay": float(config["ema_decay"]),
            "checkpoint_selection": "generated cached validation MAE",
        }
    )
    save_json(summary, output_dir / "eval_50_samples" / "metrics_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
