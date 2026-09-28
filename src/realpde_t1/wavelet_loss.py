"""Differentiable Multi-Scale 2D Wavelet Decomposition and TKE Preservation Loss."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DWT2D(nn.Module):
    """Differentiable 2D Discrete Haar Wavelet Transform."""

    def __init__(self) -> None:
        super().__init__()
        # 4 Haar Wavelet 2D filters of size (2, 2)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose 2D spatial fields into (LL, LH, HL, HH).

        Args:
            x: (N, 1, H, W) tensor where H and W are even.
        Returns:
            (LL, LH, HL, HH) each of shape (N, 1, H/2, W/2).
        """
        out = F.conv2d(x, self.filters, stride=2, padding=0)  # (N, 4, H/2, W/2)
        ll = out[:, 0:1]
        lh = out[:, 1:2]
        hl = out[:, 2:3]
        hh = out[:, 3:4]
        return ll, lh, hl, hh


class MultiScaleWaveletTKELoss(nn.Module):
    """Combines Relative L2, Temporal Mean Profile, TKE relative error, and High-Frequency Wavelet losses."""

    def __init__(
        self,
        lambda_mean: float = 0.2,
        lambda_tke: float = 0.4,
        lambda_wavelet: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_tke = lambda_tke
        self.lambda_wavelet = lambda_wavelet
        self.eps = eps
        self.dwt = DWT2D()

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        mean_target: torch.Tensor,
        std_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss on (B, T, H, W, 3) normalized tensors.

        Channel 0=u, 1=v, 2=pressure (unscored).
        """
        # Convert to physical velocity fields for physical loss calculation
        pred_phys = pred_norm * std_target + mean_target
        pred_phys[..., 2] = 0.0
        tgt_phys = target_norm * std_target + mean_target
        tgt_phys[..., 2] = 0.0

        b = pred_phys.shape[0]

        # 1. Scale-Balanced Relative L2 Field Loss (on u and v)
        diff_u = pred_phys[..., 0] - tgt_phys[..., 0]
        diff_v = pred_phys[..., 1] - tgt_phys[..., 1]
        denom_u = torch.norm(tgt_phys[..., 0].reshape(b, -1), p=2, dim=1) + self.eps
        denom_v = torch.norm(tgt_phys[..., 1].reshape(b, -1), p=2, dim=1) + self.eps
        l2_u = torch.norm(diff_u.reshape(b, -1), p=2, dim=1) / denom_u
        l2_v = torch.norm(diff_v.reshape(b, -1), p=2, dim=1) / denom_v
        loss_field = 0.5 * torch.mean(l2_u + l2_v)

        # 2. Decoupled Time-Mean Velocity Profile Loss (MVPE)
        pred_mean_u = torch.mean(pred_phys[..., 0], dim=1)  # (B, H, W)
        pred_mean_v = torch.mean(pred_phys[..., 1], dim=1)
        tgt_mean_u = torch.mean(tgt_phys[..., 0], dim=1)
        tgt_mean_v = torch.mean(tgt_phys[..., 1], dim=1)

        diff_m_u = pred_mean_u - tgt_mean_u
        diff_m_v = pred_mean_v - tgt_mean_v
        mean_denom_u = torch.norm(tgt_mean_u.reshape(b, -1), p=2, dim=1) + self.eps
        mean_denom_v = torch.norm(tgt_mean_v.reshape(b, -1), p=2, dim=1) + self.eps
        loss_mean = 0.5 * torch.mean(
            torch.norm(diff_m_u.reshape(b, -1), p=2, dim=1) / mean_denom_u
            + torch.norm(diff_m_v.reshape(b, -1), p=2, dim=1) / mean_denom_v
        )

        # 3. Turbulent Kinetic Energy Relative Loss
        pred_fl_u = pred_phys[..., 0] - pred_mean_u.unsqueeze(1)
        pred_fl_v = pred_phys[..., 1] - pred_mean_v.unsqueeze(1)
        tgt_fl_u = tgt_phys[..., 0] - tgt_mean_u.unsqueeze(1)
        tgt_fl_v = tgt_phys[..., 1] - tgt_mean_v.unsqueeze(1)

        tke_pred = 0.5 * (pred_fl_u**2 + pred_fl_v**2)  # (B, T, H, W)
        tke_tgt = 0.5 * (tgt_fl_u**2 + tgt_fl_v**2)

        tke_diff = tke_pred - tke_tgt
        tke_denom = torch.norm(tke_tgt.reshape(b, -1), p=2, dim=1) + self.eps
        loss_tke = torch.mean(torch.norm(tke_diff.reshape(b, -1), p=2, dim=1) / tke_denom)

        # 4. Multi-Scale 2D Wavelet Fluctuation Loss (on u' and v')
        # Reshape (B*T, 1, H, W)
        fl_u_flat = pred_fl_u.reshape(-1, 1, 32, 64)
        fl_v_flat = pred_fl_v.reshape(-1, 1, 32, 64)
        tgt_u_flat = tgt_fl_u.reshape(-1, 1, 32, 64)
        tgt_v_flat = tgt_fl_v.reshape(-1, 1, 32, 64)

        # Level 1 DWT
        p_ll_u, p_lh_u, p_hl_u, p_hh_u = self.dwt(fl_u_flat)
        t_ll_u, t_lh_u, t_hl_u, t_hh_u = self.dwt(tgt_u_flat)

        p_ll_v, p_lh_v, p_hl_v, p_hh_v = self.dwt(fl_v_flat)
        t_ll_v, t_lh_v, t_hl_v, t_hh_v = self.dwt(tgt_v_flat)

        # High-frequency subbands penalty (L1 relative)
        loss_w_u = (
            torch.mean(torch.abs(p_lh_u - t_lh_u)) / (torch.mean(torch.abs(t_lh_u)) + self.eps)
            + torch.mean(torch.abs(p_hl_u - t_hl_u)) / (torch.mean(torch.abs(t_hl_u)) + self.eps)
            + torch.mean(torch.abs(p_hh_u - t_hh_u)) / (torch.mean(torch.abs(t_hh_u)) + self.eps)
        ) / 3.0

        loss_w_v = (
            torch.mean(torch.abs(p_lh_v - t_lh_v)) / (torch.mean(torch.abs(t_lh_v)) + self.eps)
            + torch.mean(torch.abs(p_hl_v - t_hl_v)) / (torch.mean(torch.abs(t_hl_v)) + self.eps)
            + torch.mean(torch.abs(p_hh_v - t_hh_v)) / (torch.mean(torch.abs(t_hh_v)) + self.eps)
        ) / 3.0

        loss_wavelet = 0.5 * (loss_w_u + loss_w_v)

        total_loss = (
            loss_field
            + self.lambda_mean * loss_mean
            + self.lambda_tke * loss_tke
            + self.lambda_wavelet * loss_wavelet
        )

        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "loss_field": float(loss_field.detach().cpu()),
            "loss_mean": float(loss_mean.detach().cpu()),
            "loss_tke": float(loss_tke.detach().cpu()),
            "loss_wavelet": float(loss_wavelet.detach().cpu()),
        }
        return total_loss, metrics
