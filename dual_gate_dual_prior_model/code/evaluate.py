from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from data import DataPaths, GeochemDataBundle
from model import create_inpainter, hard_replace
from utils import masked_mae_np, masked_r2_np, masked_rmse, save_grid_csv, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--full-grid", default="/mnt/storatge/ljj/生成模型/真实数据/ag_ok_back_500m.csv")
    parser.add_argument("--block-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_block_mask_15pct.csv")
    parser.add_argument("--hard-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_hard_mask_block15_hard5.csv")
    parser.add_argument("--target-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_target_mask_block15_hard5.csv")
    parser.add_argument("--output-dir", default="/mnt/storatge/ljj/生成模型/模型结果/ds_kriging_prior_inpaint/eval")
    parser.add_argument("--ds-level", type=int, default=None)
    parser.add_argument("--ds-ensemble", type=int, default=24)
    parser.add_argument("--ds-method", choices=["cascade", "legacy"], default=None)
    parser.add_argument("--kriging-backend", choices=["auto", "pykrige", "idw"], default=None)
    parser.add_argument("--variogram-model", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_token_grid(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("token grid must look like 10,23")
    return int(parts[0]), int(parts[1])


def to_ag(norm_array: np.ndarray, bundle: GeochemDataBundle) -> np.ndarray:
    log_array = bundle.norm.decode(norm_array)
    return np.maximum(np.expm1(log_array), 0).astype(np.float32)


def plot_grid(array, template, title, out_path, label="Ag", log_color=False, cmap="viridis", vmin=None, vmax=None):
    data = np.array(array, dtype=float)
    if log_color:
        data = np.log1p(np.maximum(data, 0))
        label = "log1p(Ag)"
    extent = [
        float(template.columns.min()),
        float(template.columns.max()),
        float(template.index.min()),
        float(template.index.max()),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(data, extent=extent, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("CX")
    ax.set_ylabel("CY")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def mask_rectangle(mask: np.ndarray, template) -> tuple[float, float, float, float]:
    rr, cc = np.where(mask > 0)
    if rr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    xs = template.columns.to_numpy(dtype=float)
    ys = template.index.to_numpy(dtype=float)
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
    x0 = float(xs[cc.min()] - dx / 2.0)
    x1 = float(xs[cc.max()] + dx / 2.0)
    y0 = float(ys[rr.min()] - dy / 2.0)
    y1 = float(ys[rr.max()] + dy / 2.0)
    return x0, y0, x1 - x0, y1 - y0


def plot_truth_prediction_comparison(
    truth_ag: np.ndarray,
    pred_ag: np.ndarray,
    target_mask: np.ndarray,
    template,
    out_path: Path,
):
    extent = [
        float(template.columns.min()),
        float(template.columns.max()),
        float(template.index.min()),
        float(template.index.max()),
    ]
    vmin = float(np.nanmin(truth_ag))
    vmax = float(np.nanmax(truth_ag))
    rect = mask_rectangle(target_mask, template)

    fig = plt.figure(figsize=(15.5, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06)
    ax_truth = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1], sharex=ax_truth, sharey=ax_truth)
    cax = fig.add_subplot(gs[0, 2])

    for ax, data, title in [
        (ax_truth, truth_ag, "Truth Ag"),
        (ax_pred, pred_ag, "Predicted Ag"),
    ]:
        im = ax.imshow(data, extent=extent, origin="lower", aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, edgecolor="red", linewidth=2.0))
        ax.set_title(title)
        ax.set_xlabel("CX")
        ax.set_ylabel("CY")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Ag")
    fig.suptitle("Truth vs prediction with target mask")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = DataPaths(
        full_grid=args.full_grid,
        block_mask=args.block_mask,
        hard_mask=args.hard_mask,
        target_mask=args.target_mask,
    )
    bundle = GeochemDataBundle(paths)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_args = ckpt.get("args", {})

    ds_level = int(args.ds_level if args.ds_level is not None else model_args.get("ds_level", 3))
    ds_method = str(args.ds_method if args.ds_method is not None else model_args.get("ds_method", "cascade"))
    kriging_backend = str(args.kriging_backend if args.kriging_backend is not None else model_args.get("kriging_backend", "auto"))
    variogram_model = str(args.variogram_model if args.variogram_model is not None else model_args.get("variogram_model", "spherical"))
    token_grid = parse_token_grid(str(model_args.get("token_grid", "10,23")))

    model = create_inpainter(
        model_variant=str(model_args.get("model_variant", "dual_gate")),
        residual_scale=float(model_args.get("residual_scale", 0.35)),
        channels=int(model_args.get("channels", 64)),
        foundation_blocks=int(model_args.get("foundation_blocks", 2)),
        heads=int(model_args.get("attention_heads", 4)),
        freeze_foundation=False,
        foundation_type=str(model_args.get("foundation_type", "internal")),
        timm_model=str(model_args.get("timm_model", "deit_tiny_patch16_224")),
        token_grid=token_grid,
        pretrained_foundation=False,
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    sample = bundle.validation_sample(
        ds_level=ds_level,
        ds_ensemble=args.ds_ensemble,
        ds_method=ds_method,
        seed=999,
        kriging_backend=kriging_backend,
        variogram_model=variogram_model,
    )
    with torch.no_grad():
        inputs = torch.from_numpy(sample["inputs"][None, ...]).to(args.device)
        truth = torch.from_numpy(sample["truth_norm"][None, None, ...]).to(args.device)
        condition = torch.from_numpy(sample["condition_norm"][None, None, ...]).to(args.device)
        known_mask = torch.from_numpy(sample["known_mask"][None, None, ...]).to(args.device)
        target_mask = torch.from_numpy(sample["target_mask"][None, None, ...]).to(args.device)
        out = model(inputs)
        completed_norm = hard_replace(out["pred"], condition, known_mask, target_mask)

    pred_log = bundle.norm.decode(completed_norm.cpu().numpy()[0, 0])
    pred_ag = np.maximum(np.expm1(pred_log), 0).astype(np.float32)
    truth_ag = bundle.full_ag
    target = sample["target_mask"]

    metrics = {
        "target_mae_ag": masked_mae_np(pred_ag, truth_ag, target),
        "target_rmse_ag": masked_rmse(pred_ag, truth_ag, target),
        "target_r2_ag": masked_r2_np(pred_ag, truth_ag, target),
        "target_pixels": int(np.sum(target)),
        "kriging_backend": str(sample["priors"].get("kriging_backend", "")),
        "ds_method": str(sample["priors"].get("ds_method", "")),
    }
    save_json(metrics, out_dir / "metrics.json")
    save_grid_csv(pred_ag, bundle.full_df, out_dir / "completed_ag_pred.csv")
    save_grid_csv(pred_ag - truth_ag, bundle.full_df, out_dir / "completed_ag_error.csv")

    ds_mean_ag = to_ag(sample["priors"]["ds_mean"], bundle)
    kriging_ag = to_ag(sample["priors"]["kriging_prior"], bundle)
    save_grid_csv(ds_mean_ag, bundle.full_df, out_dir / "ds_prior_mean_ag.csv")
    save_grid_csv(kriging_ag, bundle.full_df, out_dir / "kriging_prior_ag.csv")
    save_grid_csv(sample["priors"]["kriging_variance"], bundle.full_df, out_dir / "kriging_variance.csv")

    plot_grid(pred_ag, bundle.full_df, "Completed Ag prediction", out_dir / "completed_ag_pred.png")
    plot_grid(pred_ag, bundle.full_df, "Completed Ag prediction, log color", out_dir / "completed_ag_pred_logcolor.png", log_color=True)
    plot_grid(np.abs(pred_ag - truth_ag) * target, bundle.full_df, "Absolute error on target mask", out_dir / "target_abs_error.png")
    plot_truth_prediction_comparison(truth_ag, pred_ag, target, bundle.full_df, out_dir / "truth_vs_prediction.png")
    plot_grid(ds_mean_ag, bundle.full_df, "DS fractal prior mean", out_dir / "ds_prior_mean_ag.png")
    plot_grid(kriging_ag, bundle.full_df, "Kriging prior", out_dir / "kriging_prior_ag.png")
    plot_grid(sample["priors"]["kriging_variance"], bundle.full_df, "Kriging variance", out_dir / "kriging_variance.png", label="variance 0-1")
    print(metrics)


if __name__ == "__main__":
    main()
