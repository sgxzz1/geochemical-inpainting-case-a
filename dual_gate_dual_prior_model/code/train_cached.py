from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cached_data import CachedGeochemDataset
from model import create_inpainter, hard_replace
from utils import masked_mae, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train using precomputed pseudo-block samples.")
    parser.add_argument("--cache-dir", default="/mnt/storatge/ljj/生成模型/模型结果/cached_pseudo_blocks_cascade_1024")
    parser.add_argument("--output-dir", default="/mnt/storatge/ljj/生成模型/模型结果/ds_kriging_prior_inpaint_cached_cascade_deit_tiny")
    parser.add_argument("--foundation-checkpoint", default=None)
    parser.add_argument("--model-variant", choices=["dual_gate", "multi_prior"], default="multi_prior")
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--foundation-type", choices=["internal", "timm_vit"], default="internal")
    parser.add_argument("--timm-model", default="deit_tiny_patch16_224")
    parser.add_argument("--token-grid", default="10,23")
    parser.add_argument("--no-pretrained-foundation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--std-loss-weight", type=float, default=0.05)
    parser.add_argument("--high-loss-weight", type=float, default=0.03)
    parser.add_argument("--known-loss-weight", type=float, default=0.02)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--foundation-blocks", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--freeze-foundation", action="store_true", default=True)
    parser.add_argument("--train-foundation", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_token_grid(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("--token-grid must look like 10,23")
    return int(parts[0]), int(parts[1])


def load_cache_metadata(cache_dir: Path) -> dict:
    path = cache_dir / "cache_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=(-2, -1), keepdim=True) / denom
    var = (((values - mean) ** 2) * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return torch.sqrt(var.clamp_min(1e-8)).mean()


def masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    rows = []
    b = values.shape[0]
    for i in range(b):
        selected = values[i, 0][mask[i, 0] > 0]
        if selected.numel() == 0:
            rows.append(values.new_tensor(0.0))
            continue
        k = max(1, int(round(selected.numel() * fraction)))
        rows.append(torch.topk(selected, k=k, largest=True).values.mean())
    return torch.stack(rows).mean()


@torch.no_grad()
def validate(model, loader: DataLoader, device: str, args) -> float:
    model.eval()
    losses = []
    for batch in loader:
        inputs = batch["inputs"].to(device)
        truth = batch["truth_norm"].to(device)
        condition = batch["condition_norm"].to(device)
        known_mask = batch["known_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        out = model(inputs)
        completed = hard_replace(out["pred"], condition, known_mask, target_mask)
        loss_target = masked_mae(completed, truth, target_mask)
        pred_std = masked_std(out["pred"], target_mask)
        truth_std = masked_std(truth, target_mask)
        loss_std = torch.abs(pred_std - truth_std)
        pred_high = masked_topk_mean(out["pred"], target_mask, fraction=0.10)
        truth_high = masked_topk_mean(truth, target_mask, fraction=0.10)
        loss_high = torch.abs(pred_high - truth_high)
        loss = loss_target + args.std_loss_weight * loss_std + args.high_loss_weight * loss_high
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main():
    args = parse_args()
    set_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_metadata = load_cache_metadata(cache_dir)
    save_json(cache_metadata, out_dir / "cache_metadata.json")

    train_dataset = CachedGeochemDataset(cache_dir, split="train")
    val_dataset = CachedGeochemDataset(cache_dir, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    token_grid = parse_token_grid(args.token_grid)
    model = create_inpainter(
        model_variant=args.model_variant,
        residual_scale=args.residual_scale,
        channels=args.channels,
        foundation_blocks=args.foundation_blocks,
        heads=args.attention_heads,
        freeze_foundation=args.freeze_foundation and not args.train_foundation,
        foundation_type=args.foundation_type,
        timm_model=args.timm_model,
        token_grid=token_grid,
        pretrained_foundation=not args.no_pretrained_foundation,
    ).to(args.device)

    if args.foundation_checkpoint:
        model.load_foundation_checkpoint(args.foundation_checkpoint, strict=False)
        model.freeze_foundation()
    elif args.foundation_type == "internal" and args.freeze_foundation and not args.train_foundation:
        print("[warning] Foundation blocks are frozen but no foundation checkpoint was provided.")
    elif args.foundation_type == "timm_vit":
        print(f"[foundation] Loaded timm model: {args.timm_model}")
        print(f"[foundation] Using first {args.foundation_blocks} transformer blocks.")
        print(f"[foundation] Frozen: {args.freeze_foundation and not args.train_foundation}")
        print(f"[foundation] Token grid: {token_grid}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ckpt_args = vars(args).copy()
    # Store prior settings in the checkpoint so evaluate.py can reproduce the fixed validation prior.
    for key in ["ds_method", "ds_level", "ds_ensemble", "kriging_backend", "variogram_model"]:
        if key in cache_metadata:
            ckpt_args[key] = cache_metadata[key]
    ckpt_args["model_variant"] = args.model_variant
    ckpt_args["residual_scale"] = args.residual_scale

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs = batch["inputs"].to(args.device)
            truth = batch["truth_norm"].to(args.device)
            condition = batch["condition_norm"].to(args.device)
            known_mask = batch["known_mask"].to(args.device)
            target_mask = batch["target_mask"].to(args.device)

            out = model(inputs)
            completed = hard_replace(out["pred"], condition, known_mask, target_mask)
            loss_target = masked_mae(completed, truth, target_mask)
            loss_known = masked_mae(out["pred"], truth, known_mask)
            pred_std = masked_std(out["pred"], target_mask)
            truth_std = masked_std(truth, target_mask)
            loss_std = torch.abs(pred_std - truth_std)
            pred_high = masked_topk_mean(out["pred"], target_mask, fraction=0.10)
            truth_high = masked_topk_mean(truth, target_mask, fraction=0.10)
            loss_high = torch.abs(pred_high - truth_high)
            loss = (
                loss_target
                + args.known_loss_weight * loss_known
                + args.std_loss_weight * loss_std
                + args.high_loss_weight * loss_high
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss = validate(model, val_loader, args.device, args)
        train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_loss": train_loss, "val_target_mae_norm": val_loss}
        history.append(row)
        print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_mae_norm={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": ckpt_args,
                    "cache_metadata": cache_metadata,
                    "best_val": best_val,
                },
                out_dir / "best_model.pt",
            )

        if epoch % 25 == 0:
            save_json({"history": history}, out_dir / "history.json")

    save_json({"history": history}, out_dir / "history.json")


if __name__ == "__main__":
    main()
