from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FoundationSpatialBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.local = ConvBlock(channels)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.local(x)
        tokens = x.flatten(2).transpose(1, 2)
        attn_in = self.norm(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TimmViTFoundationModule(nn.Module):
    def __init__(
        self,
        model_name: str,
        channels: int,
        num_blocks: int = 4,
        token_grid: tuple[int, int] = (10, 23),
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for --foundation-type timm_vit. Install it with: pip install timm"
            ) from exc

        vit = timm.create_model(model_name, pretrained=pretrained)
        embed_dim = getattr(vit, "embed_dim", None) or getattr(vit, "num_features", None)
        if embed_dim is None:
            raise ValueError(f"Could not infer embed_dim for timm model {model_name}.")
        if embed_dim != channels:
            raise ValueError(
                f"{model_name} uses embed_dim={embed_dim}, but channels={channels}. "
                "For deit_tiny_patch16_224 use --channels 192."
            )
        if num_blocks > len(vit.blocks):
            raise ValueError(f"Requested {num_blocks} blocks, but {model_name} has only {len(vit.blocks)}.")

        self.channels = channels
        self.token_grid = token_grid
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(num_blocks)])
        self.norm = getattr(vit, "norm", nn.Identity())
        # Position embeddings affect every transformer block output and must be
        # checkpointed, especially when the DeiT backbone is trained from a
        # random initialization. Omitting this buffer makes a reloaded model use
        # a newly randomized position embedding.
        self.register_buffer("pos_embed", self._make_pos_embed(vit, token_grid), persistent=True)
        if freeze:
            self.freeze()

    def _make_pos_embed(self, vit: nn.Module, token_grid: tuple[int, int]) -> torch.Tensor:
        pos_embed = getattr(vit, "pos_embed", None)
        if pos_embed is None:
            return torch.zeros(1, token_grid[0] * token_grid[1], self.channels)

        pe = pos_embed.detach().clone()
        patch_pe = pe[:, 1:, :] if pe.shape[1] > 1 else pe
        n = patch_pe.shape[1]
        src_size = int(round(n ** 0.5))
        if src_size * src_size != n:
            return torch.zeros(1, token_grid[0] * token_grid[1], self.channels)

        patch_pe = patch_pe.reshape(1, src_size, src_size, self.channels).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=token_grid, mode="bicubic", align_corners=False)
        return patch_pe.permute(0, 2, 3, 1).reshape(1, token_grid[0] * token_grid[1], self.channels)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, self.token_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        coarse = tokens.transpose(1, 2).reshape(b, c, self.token_grid[0], self.token_grid[1])
        up = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
        return x + up


class DSKrigingPriorInpainter(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        channels: int = 64,
        foundation_blocks: int = 2,
        heads: int = 4,
        freeze_foundation: bool = True,
        foundation_type: str = "internal",
        timm_model: str = "deit_tiny_patch16_224",
        token_grid: tuple[int, int] = (10, 23),
        pretrained_foundation: bool = True,
    ):
        super().__init__()
        self.foundation_type = foundation_type
        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.pre = nn.Sequential(ConvBlock(channels), ConvBlock(channels))
        if foundation_type == "internal":
            self.foundation = nn.ModuleList(
                [FoundationSpatialBlock(channels, heads=heads) for _ in range(foundation_blocks)]
            )
        elif foundation_type == "timm_vit":
            self.foundation = TimmViTFoundationModule(
                model_name=timm_model,
                channels=channels,
                num_blocks=foundation_blocks,
                token_grid=token_grid,
                pretrained=pretrained_foundation,
                freeze=freeze_foundation,
            )
        else:
            raise ValueError(f"Unsupported foundation_type: {foundation_type}")

        self.geo_fusion = nn.Sequential(ConvBlock(channels), ConvBlock(channels))
        self.direct_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.residual_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.alpha_head = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())
        self.beta_head = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())

        if freeze_foundation:
            self.freeze_foundation()

    def freeze_foundation(self) -> None:
        if hasattr(self.foundation, "freeze"):
            self.foundation.freeze()
        else:
            for p in self.foundation.parameters():
                p.requires_grad = False

    def unfreeze_foundation(self) -> None:
        if hasattr(self.foundation, "unfreeze"):
            self.foundation.unfreeze()
        else:
            for p in self.foundation.parameters():
                p.requires_grad = True

    def load_foundation_checkpoint(self, path: str | Path, strict: bool = False) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("foundation", ckpt)
        missing, unexpected = self.foundation.load_state_dict(state, strict=strict)
        if missing:
            print(f"[foundation] missing keys: {missing}")
        if unexpected:
            print(f"[foundation] unexpected keys: {unexpected}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_adapter(x)
        feat = self.pre(feat)
        if isinstance(self.foundation, nn.ModuleList):
            for block in self.foundation:
                feat = block(feat)
        else:
            feat = self.foundation(feat)
        return self.geo_fusion(feat)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Channel order:
        # 0 condition, 1 known_mask, 2 target_mask, 3 hard_mask,
        # 4 ds_mean, 5 ds_p75, 6 ds_std, 7 kriging_prior,
        # 8 kriging_variance, 9 x_coord, 10 y_coord.
        ds_prior = x[:, 4:5]
        kriging_prior = x[:, 7:8]

        feat = self.extract_features(x)

        direct = self.direct_head(feat)
        residual = self.residual_head(feat)
        alpha = self.alpha_head(feat)
        beta = self.beta_head(feat)
        prior_fused = beta * ds_prior + (1.0 - beta) * kriging_prior
        prior_corrected = prior_fused + residual
        pred = alpha * prior_corrected + (1.0 - alpha) * direct

        return {
            "pred": pred,
            "direct": direct,
            "residual": residual,
            "alpha": alpha,
            "beta": beta,
            "prior_fused": prior_fused,
            "prior_corrected": prior_corrected,
        }


class MultiPriorFusionInpainter(DSKrigingPriorInpainter):
    """Spatially varying fusion of DS, Kriging, and direct prediction.

    This variant treats DS/Kriging/direct as three candidate explanations instead
    of assuming one prior is globally reliable. A softmax head predicts local
    weights w_ds, w_kriging, and w_direct; a bounded residual then makes a modest
    correction in normalized log space.
    """

    def __init__(self, *args, residual_scale: float = 0.35, **kwargs):
        super().__init__(*args, **kwargs)
        channels = self.direct_head.in_channels
        self.weight_head = nn.Conv2d(channels, 3, 3, padding=1)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ds_prior = x[:, 4:5]
        kriging_prior = x[:, 7:8]

        feat = self.extract_features(x)
        direct = self.direct_head(feat)
        residual = torch.tanh(self.residual_head(feat)) * self.residual_scale
        weights = torch.softmax(self.weight_head(feat), dim=1)
        w_ds = weights[:, 0:1]
        w_kriging = weights[:, 1:2]
        w_direct = weights[:, 2:3]

        prior_fused = w_ds * ds_prior + w_kriging * kriging_prior
        fused = prior_fused + w_direct * direct
        pred = fused + residual

        # Backward-compatible diagnostic aliases.
        alpha = 1.0 - w_direct
        beta = w_ds / (w_ds + w_kriging + 1e-6)
        prior_corrected = pred

        return {
            "pred": pred,
            "direct": direct,
            "residual": residual,
            "alpha": alpha,
            "beta": beta,
            "prior_fused": prior_fused,
            "prior_corrected": prior_corrected,
            "w_ds": w_ds,
            "w_kriging": w_kriging,
            "w_direct": w_direct,
        }


def create_inpainter(
    model_variant: str = "dual_gate",
    residual_scale: float = 0.35,
    **kwargs,
) -> nn.Module:
    if model_variant == "dual_gate":
        return DSKrigingPriorInpainter(**kwargs)
    if model_variant == "multi_prior":
        return MultiPriorFusionInpainter(residual_scale=residual_scale, **kwargs)
    raise ValueError("--model-variant must be dual_gate or multi_prior")


def hard_replace(pred: torch.Tensor, condition: torch.Tensor, known_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    return known_mask * condition + target_mask * pred
