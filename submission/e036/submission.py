"""E033 Track 1 Submission: Champion Tri-Loss Pareto Ensemble with Dynamic Wake Uncertainty."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "_vendor")
for _p in (_HERE, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_baseline import build_model  # noqa: E402

_PAYLOAD_PATH = os.path.join(_HERE, "ensemble_payload.pth")
_EXPECTED_SAMPLE_SHAPE = (20, 32, 64, 3)
_PERSIST_FIRST_FRAMES = 4
_BATCH_SIZE = 1

_STATE = {"model": None}


def extract_flow_invariants(inputs: torch.Tensor) -> torch.Tensor:
    """Extract 4-dimensional flow regime invariants z_flow from (B, T, H, W, C) inputs."""
    u_inlet = inputs[:, :, :, :4, 0]
    v_inlet = inputs[:, :, :, :4, 1]

    u_inf = torch.mean(u_inlet, dim=(1, 2, 3)).unsqueeze(-1)
    v_inf = torch.mean(v_inlet, dim=(1, 2, 3)).unsqueeze(-1)
    aoa_proxy = torch.atan2(v_inf, u_inf.clamp(min=1e-5))

    u_std = torch.std(u_inlet, dim=(1, 2, 3)).unsqueeze(-1)
    v_std = torch.std(v_inlet, dim=(1, 2, 3)).unsqueeze(-1)
    turb_intensity = torch.sqrt(0.5 * (u_std**2 + v_std**2)) / (torch.abs(u_inf) + 1e-5)

    u_wake = inputs[:, :, :, 16:, 0]
    v_wake = inputs[:, :, :, 16:, 1]
    u_wake_fl = u_wake - torch.mean(u_wake, dim=(1, 2, 3), keepdim=True)
    v_wake_fl = v_wake - torch.mean(v_wake, dim=(1, 2, 3), keepdim=True)
    wake_tke = torch.mean(0.5 * (u_wake_fl**2 + v_wake_fl**2), dim=(1, 2, 3)).unsqueeze(-1)

    z_flow = torch.cat([u_inf, aoa_proxy, turb_intensity, wake_tke], dim=-1)
    return z_flow


class ResidualConv3dAdapter(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 3, hidden_channels: int = 32, n_blocks: int = 2, kernel_size: int = 3) -> None:
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

    def forward(self, inputs: torch.Tensor, backbone_prediction: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat((inputs, backbone_prediction), dim=-1).permute(0, 4, 1, 2, 3)
        residual = self.out(self.body(stacked))
        return residual.permute(0, 2, 3, 4, 1)


class FiLMResidualConv3dAdapter(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 3, flow_dim: int = 4, hidden_channels: int = 32, n_blocks: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv_blocks = nn.ModuleList()
        self.film_mlps = nn.ModuleList()
        channels = in_channels
        for _ in range(n_blocks):
            self.conv_blocks.append(nn.Conv3d(channels, hidden_channels, kernel_size, padding=pad))
            mlp = nn.Sequential(
                nn.Linear(flow_dim, 32),
                nn.GELU(),
                nn.Linear(32, 2 * hidden_channels),
            )
            self.film_mlps.append(mlp)
            channels = hidden_channels
        self.out = nn.Conv3d(channels, out_channels, kernel_size, padding=pad)

    def forward(self, inputs: torch.Tensor, backbone_prediction: torch.Tensor) -> torch.Tensor:
        z_flow = extract_flow_invariants(inputs)
        stacked = torch.cat((inputs, backbone_prediction), dim=-1).permute(0, 4, 1, 2, 3)
        h = stacked
        for conv, mlp in zip(self.conv_blocks, self.film_mlps):
            h = conv(h)
            film_params = mlp(z_flow)
            gamma = film_params[:, :self.hidden_channels].view(-1, self.hidden_channels, 1, 1, 1)
            beta = film_params[:, self.hidden_channels:].view(-1, self.hidden_channels, 1, 1, 1)
            h = F.gelu((1.0 + gamma) * h + beta)
        residual = self.out(h).permute(0, 2, 3, 4, 1)
        return residual


class ChampionParetoEnsemble:
    """Champion Tri-Loss Pareto Ensemble (Rollout + FiLM + Wavelet-TKE) with Dynamic Wake Bounds."""

    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        self.device = device
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.normalizer = payload["normalizer"]
        self.unc_cfg = payload.get("uncertainty_config", {})

        self.gamma_ens = float(self.unc_cfg.get("gamma_ens", 1.0))
        self.h_laminar = float(self.unc_cfg.get("h_laminar", 0.006))
        self.h_wake = float(self.unc_cfg.get("h_wake", 0.015))
        self.sigma_global = float(self.unc_cfg.get("sigma_global", 0.0563870259))

        # Reconstruct CNO backbone via vendored build_model
        self.backbone = build_model("cno", device=str(device))
        self.backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
        self.backbone.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Reconstruct Member 1 Adapter (Rollout Curriculum)
        cfg1 = payload["member1_adapter_config"]
        self.adapter1 = ResidualConv3dAdapter(
            in_channels=cfg1.get("in_channels", 6),
            out_channels=cfg1.get("out_channels", 3),
            hidden_channels=cfg1.get("hidden_channels", 32),
            n_blocks=cfg1.get("n_blocks", 2),
            kernel_size=cfg1.get("kernel_size", 3),
        )
        self.adapter1.load_state_dict(payload["member1_adapter_state"], strict=True)
        self.adapter1.to(self.device).eval()
        for p in self.adapter1.parameters():
            p.requires_grad = False

        # Reconstruct Member 2 Adapter (Physics-FiLM)
        cfg2 = payload["member2_adapter_config"]
        self.adapter2 = FiLMResidualConv3dAdapter(
            in_channels=cfg2.get("in_channels", 6),
            out_channels=cfg2.get("out_channels", 3),
            flow_dim=cfg2.get("flow_dim", 4),
            hidden_channels=cfg2.get("hidden_channels", 32),
            n_blocks=cfg2.get("n_blocks", 2),
            kernel_size=cfg2.get("kernel_size", 3),
        )
        self.adapter2.load_state_dict(payload["member2_adapter_state"], strict=True)
        self.adapter2.to(self.device).eval()
        for p in self.adapter2.parameters():
            p.requires_grad = False

        # Reconstruct Member 3 Adapter (Wavelet-TKE)
        cfg3 = payload["member3_adapter_config"]
        self.adapter3 = ResidualConv3dAdapter(
            in_channels=cfg3.get("in_channels", 6),
            out_channels=cfg3.get("out_channels", 3),
            hidden_channels=cfg3.get("hidden_channels", 32),
            n_blocks=cfg3.get("n_blocks", 2),
            kernel_size=cfg3.get("kernel_size", 3),
        )
        self.adapter3.load_state_dict(payload["member3_adapter_state"], strict=True)
        self.adapter3.to(self.device).eval()
        for p in self.adapter3.parameters():
            p.requires_grad = False

        self.mean_in = torch.tensor(self.normalizer["mean_input"], device=self.device, dtype=torch.float32)
        self.std_in = torch.tensor(self.normalizer["std_input"], device=self.device, dtype=torch.float32)
        self.mean_tg = torch.tensor(self.normalizer["mean_target"], device=self.device, dtype=torch.float32)
        self.std_tg = torch.tensor(self.normalizer["std_target"], device=self.device, dtype=torch.float32)

    def predict_single(self, sample_phys: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process one sample of shape (1, 20, 32, 64, 3) to guarantee exact batch invariance."""
        x_norm = (sample_phys - self.mean_in) / self.std_in

        with torch.no_grad():
            base = self.backbone(x_norm)
            out1 = base + self.adapter1(x_norm, base)
            out2 = base + self.adapter2(x_norm, base)
            out3 = base + self.adapter3(x_norm, base)

            p1 = out1 * self.std_tg + self.mean_tg
            p2 = out2 * self.std_tg + self.mean_tg
            p3 = out3 * self.std_tg + self.mean_tg

            p1[..., 2] = 0.0
            p2[..., 2] = 0.0
            p3[..., 2] = 0.0

            p_ens = (p1 + p2 + p3) / 3.0

            # Persist first k frames
            k = _PERSIST_FIRST_FRAMES
            p_ens[:, :k] = sample_phys[:, :k]
            p_ens[..., 2] = 0.0

            # Ensemble standard deviation
            stack = torch.stack([p1, p2, p3], dim=0)
            ens_std = torch.std(stack, dim=0)

            # Compute Dynamic Physical Wake Mask from input
            u_in = sample_phys[:, :, :, :, 0]
            v_in = sample_phys[:, :, :, :, 1]
            u_mean = torch.mean(u_in, dim=1, keepdim=True)
            v_mean = torch.mean(v_in, dim=1, keepdim=True)
            u_fl = u_in - u_mean
            v_fl = v_in - v_mean
            in_tke = 0.5 * (u_fl**2 + v_fl**2)
            mean_tke = torch.mean(in_tke, dim=1, keepdim=True)  # (1, 1, 32, 64)

            p95 = torch.quantile(mean_tke.view(1, -1), 0.95) + 1e-6
            p05 = torch.quantile(mean_tke.view(1, -1), 0.05)
            wake_mask = torch.clamp((mean_tke - p05) / (p95 - p05), 0.0, 1.0)
            wake_mask = wake_mask.repeat(1, 20, 1, 1).unsqueeze(-1)  # (1, 20, 32, 64, 1)

            half_width = (
                self.gamma_ens * ens_std[..., :2]
                + self.h_laminar * (1.0 - wake_mask)
                + self.h_wake * wake_mask
            )

            # Pad pressure half-width with 0
            half_3ch = torch.zeros_like(p_ens)
            half_3ch[..., :2] = half_width

            lower = p_ens - half_3ch
            upper = p_ens + half_3ch
            lower[..., 2] = 0.0
            upper[..., 2] = 0.0

        return (
            p_ens.cpu().numpy().astype(np.float32, copy=False),
            lower.cpu().numpy().astype(np.float32, copy=False),
            upper.cpu().numpy().astype(np.float32, copy=False),
        )


def _get_model() -> ChampionParetoEnsemble:
    if _STATE["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _STATE["model"] = ChampionParetoEnsemble(_PAYLOAD_PATH, device=device)
    return _STATE["model"]


def _validate_input(input_array) -> np.ndarray:
    value = np.asarray(input_array, dtype=np.float32)
    if value.ndim != 5 or tuple(value.shape[1:]) != _EXPECTED_SAMPLE_SHAPE:
        raise ValueError("expected input shape (N,20,32,64,3), got %s" % (value.shape,))
    if value.shape[0] == 0:
        raise ValueError("input batch must contain at least one sample")
    if not np.all(np.isfinite(value)):
        raise ValueError("input contains NaN or Inf")
    return value


def predict(input_array, metadata=None) -> dict[str, np.ndarray]:
    """Forecast 20 frames with Champion Pareto Ensemble, persist first 4, and return dynamic wake bounds."""
    del metadata
    value = _validate_input(input_array)
    model = _get_model()

    n_samples = value.shape[0]
    preds_list, lower_list, upper_list = [], [], []

    for index in range(0, n_samples, _BATCH_SIZE):
        batch = value[index : index + _BATCH_SIZE]
        tensor = torch.from_numpy(batch).to(device=model.device, dtype=torch.float32)
        p, l, u = model.predict_single(tensor)
        preds_list.append(p)
        lower_list.append(l)
        upper_list.append(u)

    return {
        "lower": np.concatenate(lower_list, axis=0),
        "prediction": np.concatenate(preds_list, axis=0),
        "upper": np.concatenate(upper_list, axis=0),
    }
