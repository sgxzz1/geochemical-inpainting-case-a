from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from cached_data import CachedGeochemDataset
from data import DataPaths, GeochemDataBundle
from model import DSKrigingPriorInpainter, create_inpainter, hard_replace
from utils import masked_mae, masked_mae_np, masked_r2_np, masked_rmse, read_grid_csv, save_grid_csv, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the deterministic DS+Kriging prior inpainter separately for multiple fixed Ag cases, "
            "then evaluate each case and draw truth-vs-prediction figures."
        )
    )
    parser.add_argument("--case-dirs", nargs="+", required=True, help="Case directories, e.g. Ag_case_B Ag_case_C Ag_case_D.")
    parser.add_argument("--full-grid", default="/mnt/storatge/ljj/生成模型/真实数据/ag_ok_back_500m.csv")
    parser.add_argument("--cache-name", default="cached_dataset", help="Cached dataset folder name inside each case directory.")
    parser.add_argument("--result-name", default="prediction_model", help="Output folder name inside each case directory.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional common output root. If set, outputs go to output_root/case_name instead of case_dir/result_name.",
    )
    parser.add_argument("--foundation-checkpoint", default=None)
    parser.add_argument("--foundation-type", choices=["internal", "timm_vit"], default="timm_vit")
    parser.add_argument("--timm-model", default="deit_tiny_patch16_224")
    parser.add_argument("--token-grid", default="10,23")
    parser.add_argument("--no-pretrained-foundation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--known-loss-weight", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--foundation-blocks", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=3)
    parser.add_argument("--freeze-foundation", action="store_true", default=True)
    parser.add_argument("--train-foundation", action="store_true")
    parser.add_argument(
        "--skip-train-if-exists",
        action="store_true",
        help="If best_model.pt already exists for a case, skip training and only evaluate/plot.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fig-width", type=float, default=16.0)
    parser.add_argument("--fig-height", type=float, default=5.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_token_grid(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("--token-grid must look like 10,23")
    return int(parts[0]), int(parts[1])


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_one(directory: Path, pattern: str, label: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {label} found by pattern {pattern!r} in {directory}")
    return candidates[0]


def find_case_paths(case_dir: Path, cache_name: str, result_name: str, output_root: str | None) -> dict[str, Path]:
    masks_dir = case_dir / "masks"
    priors_dir = case_dir / "priors"
    if output_root:
        out_dir = Path(output_root) / case_dir.name
    else:
        out_dir = case_dir / result_name
    return {
        "case_dir": case_dir,
        "cache_dir": case_dir / cache_name,
        "out_dir": out_dir,
        "block_mask": find_one(masks_dir, "*block_mask*.csv", "block mask"),
        "hard_mask": find_one(masks_dir, "*hard_mask*.csv", "hard mask"),
        "target_mask": find_one(masks_dir, "*target_mask*.csv", "target mask"),
        "cascade_mean": find_one(priors_dir, "cascade_ds_mean_ag.csv", "cascade DS mean prior"),
        "cascade_p75": find_one(priors_dir, "cascade_ds_p75_ag.csv", "cascade DS p75 prior"),
        "cascade_std": find_one(priors_dir, "cascade_ds_std_norm.csv", "cascade DS std prior"),
        "kriging_prior": find_one(priors_dir, "kriging_prior_ag.csv", "Kriging prior"),
        "kriging_variance": find_one(priors_dir, "kriging_variance.csv", "Kriging variance"),
    }


def read_mask(path: Path) -> np.ndarray:
    return (read_grid_csv(path, dtype=float).to_numpy(dtype=np.float32) > 0).astype(np.float32)


def ag_csv_to_norm(path: Path, bundle: GeochemDataBundle) -> np.ndarray:
    ag = read_grid_csv(path, dtype=float).to_numpy(dtype=np.float32)
    return bundle.norm.encode(np.log1p(np.maximum(ag, 0.0)).astype(np.float32)).astype(np.float32)


def build_fixed_case_sample(paths: dict[str, Path], bundle: GeochemDataBundle) -> dict[str, np.ndarray]:
    target_mask = bundle.target_mask.astype(np.float32)
    hard_mask = bundle.hard_mask.astype(np.float32)
    known_mask = (1.0 - target_mask).astype(np.float32)

    condition_log = bundle.full_log.copy()
    condition_log[target_mask > 0] = bundle.norm.mean
    condition_norm = bundle.norm.encode(condition_log).astype(np.float32)

    ds_mean = ag_csv_to_norm(paths["cascade_mean"], bundle)
    ds_p75 = ag_csv_to_norm(paths["cascade_p75"], bundle)
    ds_std = read_grid_csv(paths["cascade_std"], dtype=float).to_numpy(dtype=np.float32)
    kriging_prior = ag_csv_to_norm(paths["kriging_prior"], bundle)
    kriging_variance = read_grid_csv(paths["kriging_variance"], dtype=float).to_numpy(dtype=np.float32)

    inputs = np.stack(
        [
            condition_norm,
            known_mask,
            target_mask,
            hard_mask,
            ds_mean,
            ds_p75,
            ds_std,
            kriging_prior,
            kriging_variance,
            bundle.x_coord.astype(np.float32),
            bundle.y_coord.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "inputs": inputs,
        "truth_norm": bundle.norm.encode(bundle.full_log).astype(np.float32),
        "condition_norm": condition_norm,
        "known_mask": known_mask,
        "target_mask": target_mask,
        "hard_mask": hard_mask,
    }


def make_loader(dataset: CachedGeochemDataset, args, epoch: int, case_seed: int) -> DataLoader:
    if args.samples_per_epoch > 0:
        generator = torch.Generator()
        generator.manual_seed(case_seed + epoch * 1009)
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=args.samples_per_epoch,
            generator=generator,
        )
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


def create_model(args, pretrained_foundation: bool) -> DSKrigingPriorInpainter:
    token_grid = parse_token_grid(args.token_grid)
    model = DSKrigingPriorInpainter(
        channels=args.channels,
        foundation_blocks=args.foundation_blocks,
        heads=args.attention_heads,
        freeze_foundation=args.freeze_foundation and not args.train_foundation,
        foundation_type=args.foundation_type,
        timm_model=args.timm_model,
        token_grid=token_grid,
        pretrained_foundation=pretrained_foundation,
    )
    if args.foundation_checkpoint:
        model.load_foundation_checkpoint(args.foundation_checkpoint, strict=False)
        model.freeze_foundation()
    return model


@torch.no_grad()
def validate_fixed_sample(model, sample: dict[str, np.ndarray], device: str) -> float:
    model.eval()
    inputs = torch.from_numpy(sample["inputs"][None]).to(device)
    truth = torch.from_numpy(sample["truth_norm"][None, None]).to(device)
    condition = torch.from_numpy(sample["condition_norm"][None, None]).to(device)
    known_mask = torch.from_numpy(sample["known_mask"][None, None]).to(device)
    target_mask = torch.from_numpy(sample["target_mask"][None, None]).to(device)
    out = model(inputs)
    completed = hard_replace(out["pred"], condition, known_mask, target_mask)
    return float(masked_mae(completed, truth, target_mask).item())


def norm_to_ag(norm_array: np.ndarray, bundle: GeochemDataBundle) -> np.ndarray:
    return np.maximum(np.expm1(bundle.norm.decode(norm_array)), 0.0).astype(np.float32)


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


def plot_truth_prediction(
    truth_ag: np.ndarray,
    pred_ag: np.ndarray,
    block_mask: np.ndarray,
    template,
    metrics: dict,
    title_prefix: str,
    out_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    extent = [
        float(template.columns.min()),
        float(template.columns.max()),
        float(template.index.min()),
        float(template.index.max()),
    ]
    vmin = float(np.nanmin(truth_ag))
    vmax = float(np.nanmax(truth_ag))
    rect = mask_rectangle(block_mask, template)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.04)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    for ax, data, title in [
        (axes[0], truth_ag, "True Ag"),
        (axes[1], pred_ag, f"{title_prefix} prediction"),
    ]:
        im = ax.imshow(data, extent=extent, origin="lower", aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, edgecolor="red", linewidth=2.0))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("CX")
        ax.set_ylabel("CY")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Ag")
    fig.suptitle(
        f"{title_prefix}: truth vs prediction with target mask | "
        f"MAE={metrics['target_mae_ag']:.2f}, RMSE={metrics['target_rmse_ag']:.2f}, R2={metrics['target_r2_ag']:.3f}",
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def evaluate_case(model, sample: dict[str, np.ndarray], bundle: GeochemDataBundle, paths: dict[str, Path], args, out_dir: Path, case_name: str) -> dict:
    model.eval()
    inputs = torch.from_numpy(sample["inputs"][None]).to(args.device)
    condition = torch.from_numpy(sample["condition_norm"][None, None]).to(args.device)
    known_mask = torch.from_numpy(sample["known_mask"][None, None]).to(args.device)
    target_mask = torch.from_numpy(sample["target_mask"][None, None]).to(args.device)

    out = model(inputs)
    completed_norm = hard_replace(out["pred"], condition, known_mask, target_mask).cpu().numpy()[0, 0]
    pred_ag = norm_to_ag(completed_norm, bundle)
    truth_ag = bundle.full_ag
    target = sample["target_mask"]

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "case": case_name,
        "target_mae_ag": masked_mae_np(pred_ag, truth_ag, target),
        "target_rmse_ag": masked_rmse(pred_ag, truth_ag, target),
        "target_r2_ag": masked_r2_np(pred_ag, truth_ag, target),
        "target_pixels": int(np.sum(target)),
        "block_pixels": int(np.sum(bundle.block_mask)),
        "hard_pixels": int(np.sum(bundle.hard_mask)),
    }
    save_json(metrics, eval_dir / "metrics.json")
    save_grid_csv(pred_ag, bundle.full_df, eval_dir / "completed_ag_pred.csv")
    save_grid_csv(pred_ag - truth_ag, bundle.full_df, eval_dir / "completed_ag_error.csv")
    plot_truth_prediction(
        truth_ag=truth_ag,
        pred_ag=pred_ag,
        block_mask=bundle.block_mask,
        template=bundle.full_df,
        metrics=metrics,
        title_prefix=case_name,
        out_path=eval_dir / "truth_vs_prediction.png",
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        dpi=args.dpi,
    )
    return metrics


def load_best_model(checkpoint_path: Path, args) -> DSKrigingPriorInpainter:
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model_args = ckpt.get("args", {})
    token_grid = parse_token_grid(str(model_args.get("token_grid", args.token_grid)))
    model = create_inpainter(
        model_variant=str(model_args.get("model_variant", "dual_gate")),
        channels=int(model_args.get("channels", args.channels)),
        foundation_blocks=int(model_args.get("foundation_blocks", args.foundation_blocks)),
        heads=int(model_args.get("attention_heads", args.attention_heads)),
        freeze_foundation=False,
        foundation_type=str(model_args.get("foundation_type", args.foundation_type)),
        timm_model=str(model_args.get("timm_model", args.timm_model)),
        token_grid=token_grid,
        pretrained_foundation=False,
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def train_one_case(case_dir: Path, args, case_index: int) -> dict:
    paths = find_case_paths(case_dir, args.cache_name, args.result_name, args.output_root)
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_model.pt"

    bundle = GeochemDataBundle(
        DataPaths(
            full_grid=args.full_grid,
            block_mask=str(paths["block_mask"]),
            hard_mask=str(paths["hard_mask"]),
            target_mask=str(paths["target_mask"]),
        )
    )
    fixed_sample = build_fixed_case_sample(paths, bundle)
    cache_metadata = load_json(paths["cache_dir"] / "cache_metadata.json")
    case_seed = int(args.seed + case_index * 1000003)

    run_metadata = {
        "case": case_dir.name,
        "case_dir": str(case_dir),
        "cache_dir": str(paths["cache_dir"]),
        "output_dir": str(out_dir),
        "full_grid": str(args.full_grid),
        "block_mask": str(paths["block_mask"]),
        "hard_mask": str(paths["hard_mask"]),
        "target_mask": str(paths["target_mask"]),
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "known_loss_weight": args.known_loss_weight,
        "normalization_log_mean": bundle.norm.mean,
        "normalization_log_std": bundle.norm.std,
        "target_pixels": int(np.sum(bundle.target_mask)),
        "block_pixels": int(np.sum(bundle.block_mask)),
        "hard_pixels": int(np.sum(bundle.hard_mask)),
        "cache_metadata": cache_metadata,
    }
    save_json(run_metadata, out_dir / "run_metadata.json")

    if args.skip_train_if_exists and checkpoint_path.exists():
        print(f"\n=== {case_dir.name}: found existing checkpoint, skip training ===")
        model = load_best_model(checkpoint_path, args)
        metrics = evaluate_case(model, fixed_sample, bundle, paths, args, out_dir, case_dir.name)
        return {"case": case_dir.name, **metrics, "checkpoint": str(checkpoint_path)}

    print(f"\n=== Training {case_dir.name} ===")
    print(f"cache_dir={paths['cache_dir']}")
    print(f"out_dir={out_dir}")
    train_dataset = CachedGeochemDataset(paths["cache_dir"], split="train")

    model = create_model(args, pretrained_foundation=not args.no_pretrained_foundation).to(args.device)
    if args.foundation_type == "timm_vit":
        print(f"[foundation] {args.timm_model}, blocks={args.foundation_blocks}, frozen={args.freeze_foundation and not args.train_foundation}")
    elif args.foundation_type == "internal" and args.freeze_foundation and not args.train_foundation and not args.foundation_checkpoint:
        print("[warning] Frozen internal foundation has no checkpoint; this is not pretrained transfer.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    ckpt_args = vars(args).copy()
    ckpt_args.update(
        {
            "model_variant": "dual_gate",
            "case": case_dir.name,
            "token_grid": args.token_grid,
            "channels": args.channels,
            "foundation_blocks": args.foundation_blocks,
            "attention_heads": args.attention_heads,
            "foundation_type": args.foundation_type,
            "timm_model": args.timm_model,
        }
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        loader = make_loader(train_dataset, args, epoch, case_seed)
        for batch in loader:
            inputs = batch["inputs"].to(args.device)
            truth = batch["truth_norm"].to(args.device)
            condition = batch["condition_norm"].to(args.device)
            known_mask = batch["known_mask"].to(args.device)
            target_mask = batch["target_mask"].to(args.device)

            out = model(inputs)
            completed = hard_replace(out["pred"], condition, known_mask, target_mask)
            loss_target = masked_mae(completed, truth, target_mask)
            loss_known = masked_mae(out["pred"], truth, known_mask)
            loss = loss_target + args.known_loss_weight * loss_known

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss = validate_fixed_sample(model, fixed_sample, args.device)
        train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_loss": train_loss, "val_target_mae_norm": val_loss}
        history.append(row)
        print(f"[{case_dir.name}] epoch={epoch:04d} train_loss={train_loss:.6f} val_mae_norm={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": ckpt_args,
                    "cache_metadata": cache_metadata,
                    "run_metadata": run_metadata,
                    "norm_mean": bundle.norm.mean,
                    "norm_std": bundle.norm.std,
                    "best_val": best_val,
                },
                checkpoint_path,
            )
        if epoch % 25 == 0:
            save_json({"history": history}, out_dir / "history.json")

    save_json({"history": history}, out_dir / "history.json")
    best_model = load_best_model(checkpoint_path, args)
    metrics = evaluate_case(best_model, fixed_sample, bundle, paths, args, out_dir, case_dir.name)
    return {"case": case_dir.name, **metrics, "checkpoint": str(checkpoint_path)}


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = [
        "case",
        "target_mae_ag",
        "target_rmse_ag",
        "target_r2_ag",
        "target_pixels",
        "block_pixels",
        "hard_pixels",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main():
    args = parse_args()
    set_seed(args.seed)
    case_dirs = [Path(p) for p in args.case_dirs]
    summary = []
    for idx, case_dir in enumerate(case_dirs):
        summary.append(train_one_case(case_dir, args, idx))

    summary_root = Path(args.output_root) if args.output_root else case_dirs[0].parent / "multi_case_prediction_summary"
    save_json({"cases": summary}, summary_root / "summary_metrics.json")
    write_summary(summary, summary_root / "summary_metrics.csv")
    print("\n=== Multi-case summary ===")
    for row in summary:
        print(
            f"{row['case']}: MAE={row['target_mae_ag']:.3f}, "
            f"RMSE={row['target_rmse_ag']:.3f}, R2={row['target_r2_ag']:.3f}"
        )
    print(f"summary_dir={summary_root}")


if __name__ == "__main__":
    main()
