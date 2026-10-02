from __future__ import annotations

"""Run seven controlled direct/residual prior ablations.

The script deliberately reuses the existing 11-channel cached samples and the
existing DSKrigingPriorInpainter backbone. It only selects input channels and
changes the final prediction rule for each ablation.

Experiments:
  no_prior_direct:  context -> direct Ag
  ds_direct:        context + DS features -> direct Ag
  ds_residual:      context + DS features -> DS mean + residual
  kriging_direct:   context + Kriging features -> direct Ag
  kriging_residual: context + Kriging features -> Kriging + residual
  mixed_direct:     context + DS/Kriging features -> direct Ag
  mixed_residual:   context + DS/Kriging features -> fused prior + residual

The current full dual-gate model is intentionally not retrained here. It can
be compared with the outputs already produced by train_multi_case_predictor.py.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from cached_data import CachedGeochemDataset
from data import DataPaths, GeochemDataBundle
from model import DSKrigingPriorInpainter, hard_replace
from train_multi_case_predictor import (
    build_fixed_case_sample,
    find_case_paths,
    load_json,
    parse_token_grid,
    plot_truth_prediction,
)
from utils import masked_mae, masked_mae_np, masked_r2_np, masked_rmse, save_grid_csv, save_json, set_seed


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    channel_indices: tuple[int, ...]
    mode: str
    prior_kind: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "no_prior_direct": ExperimentSpec(
        name="no_prior_direct",
        channel_indices=(0, 1, 2, 3, 9, 10),
        mode="direct",
        prior_kind="none",
    ),
    "ds_direct": ExperimentSpec(
        name="ds_direct",
        channel_indices=(0, 1, 2, 3, 4, 5, 6, 9, 10),
        mode="direct",
        prior_kind="ds",
    ),
    "ds_residual": ExperimentSpec(
        name="ds_residual",
        channel_indices=(0, 1, 2, 3, 4, 5, 6, 9, 10),
        mode="residual",
        prior_kind="ds",
    ),
    "kriging_direct": ExperimentSpec(
        name="kriging_direct",
        channel_indices=(0, 1, 2, 3, 7, 8, 9, 10),
        mode="direct",
        prior_kind="kriging",
    ),
    "kriging_residual": ExperimentSpec(
        name="kriging_residual",
        channel_indices=(0, 1, 2, 3, 7, 8, 9, 10),
        mode="residual",
        prior_kind="kriging",
    ),
    "mixed_direct": ExperimentSpec(
        name="mixed_direct",
        channel_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        mode="direct",
        prior_kind="mixed",
    ),
    "mixed_residual": ExperimentSpec(
        name="mixed_residual",
        channel_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        mode="residual",
        prior_kind="mixed",
    ),
}


class AblationInpainter(DSKrigingPriorInpainter):
    """Existing backbone with an ablation-specific output rule.

    The feature extractor, pretrained foundation blocks, prediction heads, and
    normalization are inherited from the existing model. Only the input
    channel count and final direct/residual rule differ.
    """

    def __init__(self, spec: ExperimentSpec, **kwargs):
        super().__init__(in_channels=len(spec.channel_indices), **kwargs)
        self.spec_name = spec.name
        self.mode = spec.mode
        self.prior_kind = spec.prior_kind

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.extract_features(x)

        direct = self.direct_head(feat)
        zero = torch.zeros_like(direct)
        alpha = torch.zeros_like(direct)
        beta = torch.zeros_like(direct)

        if self.mode == "direct":
            pred = direct
            prior_fused = zero
            residual = zero
            prior_corrected = direct
        elif self.mode == "residual" and self.prior_kind == "ds":
            ds_prior = x[:, 4:5]
            residual = self.residual_head(feat)
            pred = ds_prior + residual
            prior_fused = ds_prior
            prior_corrected = pred
            alpha = torch.ones_like(direct)
            beta = torch.ones_like(direct)
        elif self.mode == "residual" and self.prior_kind == "kriging":
            kriging_prior = x[:, 4:5]
            residual = self.residual_head(feat)
            pred = kriging_prior + residual
            prior_fused = kriging_prior
            prior_corrected = pred
            alpha = torch.ones_like(direct)
        elif self.mode == "residual" and self.prior_kind == "mixed":
            # In the mixed input layout, channel 4 is DS mean and channel 7
            # is Kriging mean, matching the original 11-channel model.
            ds_prior = x[:, 4:5]
            kriging_prior = x[:, 7:8]
            residual = self.residual_head(feat)
            beta = self.beta_head(feat)
            prior_fused = beta * ds_prior + (1.0 - beta) * kriging_prior
            pred = prior_fused + residual
            prior_corrected = pred
            alpha = torch.ones_like(direct)
        else:
            raise ValueError(f"Unsupported experiment rule: {self.spec_name}")

        return {
            "pred": pred,
            "direct": direct,
            "residual": residual,
            "alpha": alpha,
            "beta": beta,
            "prior_fused": prior_fused,
            "prior_corrected": prior_corrected,
        }


class ChannelSubsetDataset(Dataset):
    def __init__(self, base: Dataset, channel_indices: tuple[int, ...]):
        self.base = base
        self.channel_indices = list(channel_indices)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        item["inputs"] = item["inputs"][self.channel_indices].clone()
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Run seven direct/residual DS-Kriging prior ablations.")
    parser.add_argument("--case-dirs", nargs="*", default=[], help="Case B/C/D directories.")
    parser.add_argument("--include-case-a", action="store_true", help="Also run the original fixed block as Case A.")
    parser.add_argument("--full-grid", default="/mnt/storatge/ljj/生成模型/真实数据/ag_ok_back_500m.csv")
    parser.add_argument("--cache-name", default="cached_dataset")
    parser.add_argument("--output-root", default="/mnt/storatge/ljj/生成模型/模型结果/seven_prior_ablation")
    parser.add_argument("--case-a-name", default="Ag_case_A")
    parser.add_argument("--case-a-cache-dir", default="/mnt/storatge/ljj/生成模型/模型结果/cached_pseudo_blocks_cascade_1024")
    parser.add_argument("--case-a-block-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_block_mask_15pct.csv")
    parser.add_argument("--case-a-hard-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_hard_mask_block15_hard5.csv")
    parser.add_argument("--case-a-target-mask", default="/mnt/storatge/ljj/生成模型/处理后数据/ag_target_mask_block15_hard5.csv")

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="Experiments to run. Default: all seven.",
    )
    parser.add_argument("--foundation-checkpoint", default=None)
    parser.add_argument("--foundation-type", choices=["internal", "timm_vit"], default="timm_vit")
    parser.add_argument("--timm-model", default="deit_tiny_patch16_224")
    parser.add_argument("--token-grid", default="10,23")
    parser.add_argument("--no-pretrained-foundation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--known-loss-weight", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--foundation-blocks", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=3)
    parser.add_argument("--freeze-foundation", action="store_true", default=True)
    parser.add_argument("--train-foundation", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--selection-source",
        choices=["fixed_case", "cached_val"],
        default="fixed_case",
        help=(
            "Choose checkpoints on the fixed evaluation case or on the cached "
            "validation split. Use cached_val to keep the fixed case evaluation-only."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fig-width", type=float, default=16.0)
    parser.add_argument("--fig-height", type=float, default=5.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_model(args, spec: ExperimentSpec, pretrained_foundation: bool) -> AblationInpainter:
    model = AblationInpainter(
        spec=spec,
        channels=args.channels,
        foundation_blocks=args.foundation_blocks,
        heads=args.attention_heads,
        freeze_foundation=args.freeze_foundation and not args.train_foundation,
        foundation_type=args.foundation_type,
        timm_model=args.timm_model,
        token_grid=parse_token_grid(args.token_grid),
        pretrained_foundation=pretrained_foundation,
    )
    if args.foundation_checkpoint:
        model.load_foundation_checkpoint(args.foundation_checkpoint, strict=False)
        model.freeze_foundation()
    return model


def load_model(checkpoint: Path, args, spec: ExperimentSpec) -> AblationInpainter:
    ckpt = torch.load(checkpoint, map_location=args.device)
    model_args = ckpt.get("args", {})
    proxy = argparse.Namespace(**vars(args))
    proxy.channels = int(model_args.get("channels", args.channels))
    proxy.foundation_blocks = int(model_args.get("foundation_blocks", args.foundation_blocks))
    proxy.attention_heads = int(model_args.get("attention_heads", args.attention_heads))
    proxy.foundation_type = str(model_args.get("foundation_type", args.foundation_type))
    proxy.timm_model = str(model_args.get("timm_model", args.timm_model))
    proxy.token_grid = str(model_args.get("token_grid", args.token_grid))
    proxy.foundation_checkpoint = None
    model = make_model(proxy, spec, pretrained_foundation=False).to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def make_loader(dataset: Dataset, args, epoch: int, seed: int) -> DataLoader:
    if args.samples_per_epoch > 0:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch * 1009)
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


def build_case_records(args) -> list[dict]:
    records = []
    if args.include_case_a:
        records.append(
            {
                "name": args.case_a_name,
                "case_dir": None,
                "cache_dir": Path(args.case_a_cache_dir),
                "block_mask": Path(args.case_a_block_mask),
                "hard_mask": Path(args.case_a_hard_mask),
                "target_mask": Path(args.case_a_target_mask),
                "precomputed_priors": False,
                "paths": {},
            }
        )
    for case_dir_str in args.case_dirs:
        case_dir = Path(case_dir_str)
        paths = find_case_paths(case_dir, args.cache_name, "unused_prediction_name", None)
        records.append(
            {
                "name": case_dir.name,
                "case_dir": case_dir,
                "cache_dir": paths["cache_dir"],
                "block_mask": paths["block_mask"],
                "hard_mask": paths["hard_mask"],
                "target_mask": paths["target_mask"],
                "precomputed_priors": True,
                "paths": paths,
            }
        )
    return records


def build_fixed_sample(record: dict, bundle: GeochemDataBundle) -> dict[str, np.ndarray]:
    if record["precomputed_priors"]:
        return build_fixed_case_sample(record["paths"], bundle)
    return bundle.validation_sample(
        ds_level=3,
        ds_ensemble=24,
        ds_method="cascade",
        seed=999,
        kriging_backend="pykrige",
        variogram_model="spherical",
    )


def select_sample_channels(sample: dict[str, np.ndarray], spec: ExperimentSpec) -> dict[str, np.ndarray]:
    out = dict(sample)
    out["inputs"] = sample["inputs"][list(spec.channel_indices)].astype(np.float32)
    return out


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


@torch.no_grad()
def validate_loader(model, loader: DataLoader, device: str) -> float:
    """Return pixel-weighted target MAE for a cached validation split."""
    model.eval()
    absolute_error = 0.0
    target_pixels = 0.0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        truth = batch["truth_norm"].to(device)
        condition = batch["condition_norm"].to(device)
        known_mask = batch["known_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        out = model(inputs)
        completed = hard_replace(out["pred"], condition, known_mask, target_mask)
        absolute_error += float((torch.abs(completed - truth) * target_mask).sum().item())
        target_pixels += float(target_mask.sum().item())
    if target_pixels <= 0:
        return float("nan")
    return absolute_error / target_pixels


@torch.no_grad()
def evaluate_variant(model, sample, bundle, args, out_dir: Path, case_name: str, spec: ExperimentSpec) -> dict:
    model.eval()
    inputs = torch.from_numpy(sample["inputs"][None]).to(args.device)
    condition = torch.from_numpy(sample["condition_norm"][None, None]).to(args.device)
    known_mask = torch.from_numpy(sample["known_mask"][None, None]).to(args.device)
    target_mask = torch.from_numpy(sample["target_mask"][None, None]).to(args.device)
    out = model(inputs)

    pred_norm = hard_replace(out["pred"], condition, known_mask, target_mask).cpu().numpy()[0, 0]
    pred_ag = np.maximum(np.expm1(bundle.norm.decode(pred_norm)), 0.0).astype(np.float32)
    truth_ag = bundle.full_ag
    target = sample["target_mask"]

    eval_dir = out_dir / "eval"
    diag_dir = out_dir / "diagnostics"
    eval_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "case": case_name,
        "experiment": spec.name,
        "mode": spec.mode,
        "prior_kind": spec.prior_kind,
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

    # Save the internal candidates for checking whether direct/residual rules
    # behaved as intended. residual_norm is a log-space normalized correction.
    raw_outputs = {
        "direct_pred_norm": out["direct"].cpu().numpy()[0, 0],
        "residual_norm": out["residual"].cpu().numpy()[0, 0],
        "prior_fused_norm": out["prior_fused"].cpu().numpy()[0, 0],
        "prior_corrected_norm": out["prior_corrected"].cpu().numpy()[0, 0],
        "alpha": out["alpha"].cpu().numpy()[0, 0],
        "beta": out["beta"].cpu().numpy()[0, 0],
    }
    for name, array in raw_outputs.items():
        save_grid_csv(array, bundle.full_df, diag_dir / f"{name}.csv")

    plot_truth_prediction(
        truth_ag=truth_ag,
        pred_ag=pred_ag,
        block_mask=bundle.block_mask,
        template=bundle.full_df,
        metrics=metrics,
        title_prefix=f"{case_name} {spec.name}",
        out_path=eval_dir / "truth_vs_prediction.png",
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        dpi=args.dpi,
    )
    return metrics


def train_one(record: dict, spec: ExperimentSpec, args, case_index: int) -> dict:
    case_name = record["name"]
    out_dir = Path(args.output_root) / case_name / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "best_model.pt"

    bundle = GeochemDataBundle(
        DataPaths(
            full_grid=args.full_grid,
            block_mask=str(record["block_mask"]),
            hard_mask=str(record["hard_mask"]),
            target_mask=str(record["target_mask"]),
        )
    )
    fixed_sample = select_sample_channels(build_fixed_sample(record, bundle), spec)
    cache_metadata = load_json(Path(record["cache_dir"]) / "cache_metadata.json")
    case_seed = int(args.seed + case_index * 1000003 + list(EXPERIMENTS).index(spec.name) * 10007)

    run_metadata = {
        "case": case_name,
        "experiment": spec.name,
        "mode": spec.mode,
        "prior_kind": spec.prior_kind,
        "input_channel_indices_from_original_11": list(spec.channel_indices),
        "input_channels": len(spec.channel_indices),
        "cache_dir": str(record["cache_dir"]),
        "output_dir": str(out_dir),
        "full_grid": str(args.full_grid),
        "block_mask": str(record["block_mask"]),
        "hard_mask": str(record["hard_mask"]),
        "target_mask": str(record["target_mask"]),
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "known_loss_weight": args.known_loss_weight,
        "selection_source": args.selection_source,
        "normalization_log_mean": bundle.norm.mean,
        "normalization_log_std": bundle.norm.std,
        "target_pixels": int(bundle.target_mask.sum()),
        "block_pixels": int(bundle.block_mask.sum()),
        "hard_pixels": int(bundle.hard_mask.sum()),
        "cache_metadata": cache_metadata,
    }
    save_json(run_metadata, out_dir / "run_metadata.json")

    if args.skip_existing and checkpoint.exists():
        print(f"\n=== {case_name}/{spec.name}: checkpoint exists, skip training ===")
        model = load_model(checkpoint, args, spec)
        metrics = evaluate_variant(model, fixed_sample, bundle, args, out_dir, case_name, spec)
        return {**metrics, "checkpoint": str(checkpoint)}

    print(f"\n=== Training {case_name}/{spec.name} ===")
    print(f"cache_dir={record['cache_dir']}")
    print(f"out_dir={out_dir}")
    base_dataset = CachedGeochemDataset(record["cache_dir"], split="train")
    train_dataset = ChannelSubsetDataset(base_dataset, spec.channel_indices)
    val_loader = None
    if args.selection_source == "cached_val":
        val_base_dataset = CachedGeochemDataset(record["cache_dir"], split="val")
        val_dataset = ChannelSubsetDataset(val_base_dataset, spec.channel_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )
    model = make_model(args, spec, pretrained_foundation=not args.no_pretrained_foundation).to(args.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    ckpt_args = vars(args).copy()
    ckpt_args.update(
        {
            "ablation": spec.name,
            "model_variant": "dual_gate_backbone_ablation",
            "input_channel_indices": list(spec.channel_indices),
            "input_channels": len(spec.channel_indices),
            "channels": args.channels,
            "foundation_blocks": args.foundation_blocks,
            "attention_heads": args.attention_heads,
            "foundation_type": args.foundation_type,
            "timm_model": args.timm_model,
            "token_grid": args.token_grid,
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

        if val_loader is not None:
            val_loss = validate_loader(model, val_loader, args.device)
        else:
            val_loss = validate_fixed_sample(model, fixed_sample, args.device)
        train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_loss": train_loss, "val_target_mae_norm": val_loss}
        history.append(row)
        print(f"[{case_name}/{spec.name}] epoch={epoch:04d} train_loss={train_loss:.6f} val_mae_norm={val_loss:.6f}")

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
                checkpoint,
            )
        if epoch % 25 == 0:
            save_json({"history": history}, out_dir / "history.json")

    save_json({"history": history}, out_dir / "history.json")
    best_model = load_model(checkpoint, args, spec)
    metrics = evaluate_variant(best_model, fixed_sample, bundle, args, out_dir, case_name, spec)
    return {**metrics, "checkpoint": str(checkpoint)}


def write_summary(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    save_json({"experiments": rows}, output_root / "summary_metrics.json")
    fields = [
        "case",
        "experiment",
        "mode",
        "prior_kind",
        "target_mae_ag",
        "target_rmse_ag",
        "target_r2_ag",
        "target_pixels",
        "block_pixels",
        "hard_pixels",
        "checkpoint",
    ]
    with (output_root / "summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    records = build_case_records(args)
    if not records:
        raise ValueError("No cases provided. Use --include-case-a and/or --case-dirs.")

    selected_specs = [EXPERIMENTS[name] for name in args.experiments]
    rows = []
    for case_index, record in enumerate(records):
        for spec in selected_specs:
            rows.append(train_one(record, spec, args, case_index))

    output_root = Path(args.output_root)
    write_summary(rows, output_root)
    print("\n=== Seven prior ablation summary ===")
    for row in rows:
        print(
            f"{row['case']}/{row['experiment']}: "
            f"MAE={row['target_mae_ag']:.3f}, "
            f"RMSE={row['target_rmse_ag']:.3f}, "
            f"R2={row['target_r2_ag']:.3f}"
        )
    print(f"summary_dir={output_root}")


if __name__ == "__main__":
    main()
