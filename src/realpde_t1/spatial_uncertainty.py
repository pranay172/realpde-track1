"""Spatially-adaptive heteroskedastic uncertainty modeling for Track 1."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SpatialUncertaintyNet(nn.Module):
    """Lightweight 3D Conv network that predicts pixel-level residual uncertainty scale."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 3,
        hidden_channels: int = 32,
        n_blocks: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        blocks: list[nn.Module] = []
        channels = in_channels
        for _ in range(n_blocks):
            blocks.append(nn.Conv3d(channels, hidden_channels, kernel_size, padding=pad))
            blocks.append(nn.GELU())
            channels = hidden_channels
        self.body = nn.Sequential(*blocks)
        self.out = nn.Conv3d(channels, out_channels, kernel_size, padding=pad)
        # Initialize output layer so initial scale predictions are modest
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, inputs: torch.Tensor, point_prediction: torch.Tensor) -> torch.Tensor:
        """Predict local standard deviation / scale s > 0 for each pixel.

        Args:
            inputs: (B, T, H, W, 3) normalized input context.
            point_prediction: (B, T, H, W, 3) normalized point prediction.
        Returns:
            scale: (B, T, H, W, 3) positive predicted scale.
        """
        stacked = torch.cat((inputs, point_prediction), dim=-1).permute(0, 4, 1, 2, 3)
        raw = self.out(self.body(stacked)).permute(0, 2, 3, 4, 1)
        # Softplus ensures strictly positive uncertainty scale
        scale = F.softplus(raw) + 1e-4
        return scale


class SmoothSPSLoss(nn.Module):
    """Direct differentiable surrogate loss for the SPS score + Laplace NLL."""

    def __init__(
        self,
        sigma_global: float = 0.0563870259,
        tau: float = 0.005,
        lambda_sps: float = 1.0,
        beta_floor: float = 0.05,
    ) -> None:
        super().__init__()
        self.sigma_global = sigma_global
        self.tau = tau
        self.lambda_sps = lambda_sps
        self.beta_floor = beta_floor

    def forward(
        self,
        pred_phys: torch.Tensor,
        target_phys: torch.Tensor,
        scale_phys: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            pred_phys: (B, T, H, W, 2) physical velocity prediction.
            target_phys: (B, T, H, W, 2) physical ground truth target.
            scale_phys: (B, T, H, W, 2) physical predicted scale.
        """
        res = torch.abs(target_phys - pred_phys)
        half_width = scale_phys + self.beta_floor * self.sigma_global

        # Laplace NLL: |y - y_hat| / s + log(s)
        nll = torch.mean(res / half_width + torch.log(half_width))

        # Scored mask: non-zero targets
        scored = (target_phys != 0.0).float()
        n_scored = torch.sum(scored) + 1e-8

        # Smooth coverage surrogate: Sigmoid((h - res) / tau)
        smooth_inside = torch.sigmoid((half_width - res) / self.tau)
        nil = (2.0 * half_width) / self.sigma_global
        exp_nil = torch.exp(-nil)

        # Smooth SPS reward per pixel
        smooth_sps = torch.sum(smooth_inside * exp_nil * scored) / n_scored
        loss_sps = -smooth_sps

        total_loss = nll + self.lambda_sps * loss_sps
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "smooth_sps": float(smooth_sps.detach().cpu() * 100.0),
        }
        return total_loss, metrics


def compute_spatial_bounds(
    prediction_phys: np.ndarray,
    scale_phys: np.ndarray,
    gamma: float = 1.0,
    beta_floor: float = 0.05,
    sigma_global: float = 0.0563870259,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (lower, upper) arrays from point prediction and predicted scale."""
    half_width = gamma * scale_phys + beta_floor * sigma_global
    half_width[..., 2] = 0.0
    lower = prediction_phys - half_width
    upper = prediction_phys + half_width
    lower[..., 2] = 0.0
    upper[..., 2] = 0.0
    return lower.astype(np.float32, copy=False), upper.astype(np.float32, copy=False)
