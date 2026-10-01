from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvBlock, FoundationSpatialBlock, TimmViTFoundationModule


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create standard sinusoidal embeddings for integer diffusion steps."""
    half = dim // 2
    device = timesteps.device
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)

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
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta_t = self._gather(self.betas, t, x_t)
        alpha_t = self._gather(self.alphas, t, x_t)
        alpha_bar_t = self._gather(self.alpha_bars, t, x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        )
        if noise is None:
            noise = torch.randn_like(x_t)
        var = self._gather(self.posterior_var, t, x_t)
        nonzero = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(var) * noise


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        channels: int = 192,
        foundation_blocks: int = 4,
        heads: int = 3,
        freeze_foundation: bool = True,
        foundation_type: str = "timm_vit",
        timm_model: str = "deit_tiny_patch16_224",
        token_grid: tuple[int, int] = (10, 23),
        pretrained_foundation: bool = True,
    ):
        super().__init__()
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
        self.post = nn.Sequential(ConvBlock(channels), ConvBlock(channels))
        if freeze_foundation:
            self.freeze_foundation()

    def freeze_foundation(self) -> None:
        if hasattr(self.foundation, "freeze"):
            self.foundation.freeze()
        else:
            for p in self.foundation.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_adapter(x)
        feat = self.pre(feat)
        if isinstance(self.foundation, nn.ModuleList):
            for block in self.foundation:
                feat = block(feat)
        else:
            feat = self.foundation(feat)
        return self.post(feat)


class ConditionalResidualDenoiser(nn.Module):
    """Small conditional DDPM denoiser for geochemical residual inpainting.

    The model does not generate the full Ag field directly. It receives the
    geochemical condition stack plus a noisy residual map and predicts the
    diffusion noise inside the target mask.
    """

    def __init__(
        self,
        in_channels: int = 11,
        channels: int = 192,
        foundation_blocks: int = 4,
        heads: int = 3,
        freeze_foundation: bool = True,
        foundation_type: str = "timm_vit",
        timm_model: str = "deit_tiny_patch16_224",
        token_grid: tuple[int, int] = (10, 23),
        pretrained_foundation: bool = True,
        time_dim: int = 192,
    ):
        super().__init__()
        self.channels = channels
        self.condition_encoder = ConditionEncoder(
            in_channels=in_channels,
            channels=channels,
            foundation_blocks=foundation_blocks,
            heads=heads,
            freeze_foundation=freeze_foundation,
            foundation_type=foundation_type,
            timm_model=timm_model,
            token_grid=token_grid,
            pretrained_foundation=pretrained_foundation,
        )
        self.noisy_residual_adapter = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )
        self.denoise = nn.Sequential(
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
        )
        self.out = nn.Conv2d(channels, 1, 3, padding=1)
        self.time_dim = time_dim

    def forward(self, condition_stack: torch.Tensor, noisy_residual: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond_feat = self.condition_encoder(condition_stack)
        noisy_feat = self.noisy_residual_adapter(noisy_residual)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_feat = self.time_mlp(t_emb).to(dtype=cond_feat.dtype).view(-1, self.channels, 1, 1)
        feat = cond_feat + noisy_feat + t_feat
        feat = self.denoise(feat)
        return self.out(feat)


def select_trend(inputs: torch.Tensor, mode: str = "ds_mean") -> torch.Tensor:
    ds_mean = inputs[:, 4:5]
    ds_p75 = inputs[:, 5:6]
    kriging = inputs[:, 7:8]
    if mode == "ds_mean":
        return ds_mean
    if mode == "ds_p75":
        return ds_p75
    if mode == "kriging":
        return kriging
    if mode == "ds_kriging_mean":
        return 0.5 * ds_mean + 0.5 * kriging
    raise ValueError("--trend-mode must be ds_mean, ds_p75, kriging, or ds_kriging_mean")
