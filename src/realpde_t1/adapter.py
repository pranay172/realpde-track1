"""Frozen-backbone residual adapters for Track 1 Sim2Real transfer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def freeze_module(module: nn.Module) -> None:
    """Stop gradients and running-stat updates on a backbone."""
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


class ResidualConv3dAdapter(nn.Module):
    """Small Conv3d residual on concat(input, backbone prediction)."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 3,
        hidden_channels: int = 16,
        n_blocks: int = 2,
        kernel_size: int = 3,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or n_blocks <= 0:
            raise ValueError("adapter hidden channels and block count must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("adapter kernel size must be a positive odd integer")
        pad = kernel_size // 2
        blocks: list[nn.Module] = []
        channels = in_channels
        for _ in range(n_blocks):
            blocks.append(nn.Conv3d(channels, hidden_channels, kernel_size, padding=pad))
            blocks.append(nn.GELU())
            channels = hidden_channels
        self.body = nn.Sequential(*blocks)
        self.out = nn.Conv3d(channels, out_channels, kernel_size, padding=pad)
        if zero_init_last:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, inputs: torch.Tensor, backbone_prediction: torch.Tensor) -> torch.Tensor:
        if inputs.shape != backbone_prediction.shape:
            raise ValueError(
                f"adapter input/backbone shape mismatch: "
                f"{inputs.shape}/{backbone_prediction.shape}"
            )
        if inputs.ndim != 5:
            raise ValueError("expected tensors with (N,T,H,W,C) layout")
        stacked = torch.cat((inputs, backbone_prediction), dim=-1).permute(0, 4, 1, 2, 3)
        residual = self.out(self.body(stacked))
        return residual.permute(0, 2, 3, 4, 1)


class ResidualCNO(nn.Module):
    """Ŷ = frozen_backbone(X) + adapter(X, backbone(X))."""

    def __init__(self, backbone: nn.Module, adapter: ResidualConv3dAdapter) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def train(self, mode: bool = True) -> ResidualCNO:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.backbone(inputs)
        return base + self.adapter(inputs, base)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def wrap_frozen_cno(backbone: nn.Module, adapter_config: dict[str, Any]) -> ResidualCNO:
    """Freeze a CNO and attach the configured residual adapter."""
    if adapter_config.get("name") != "residual_conv3d":
        raise ValueError(f"unsupported adapter: {adapter_config.get('name')}")
    if adapter_config.get("input") != "concat_input_and_backbone":
        raise ValueError("E018 adapter input must be concat_input_and_backbone")
    freeze_module(backbone)
    adapter = ResidualConv3dAdapter(
        in_channels=int(adapter_config.get("in_channels", 6)),
        out_channels=int(adapter_config.get("out_channels", 3)),
        hidden_channels=int(adapter_config["hidden_channels"]),
        n_blocks=int(adapter_config["n_blocks"]),
        kernel_size=int(adapter_config["kernel_size"]),
        zero_init_last=bool(adapter_config.get("zero_init_last", True)),
    )
    return ResidualCNO(backbone, adapter)


class HeteroskedasticResidualConv3dAdapter(nn.Module):
    """Small Conv3d residual on concat(input, backbone prediction) with dual field & uncertainty heads."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 3,
        uncertainty_channels: int = 2,
        hidden_channels: int = 32,
        n_blocks: int = 2,
        kernel_size: int = 3,
        zero_init_last: bool = True,
        initial_log_var: float = -4.0,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or n_blocks <= 0:
            raise ValueError("adapter hidden channels and block count must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("adapter kernel size must be a positive odd integer")
        pad = kernel_size // 2
        blocks: list[nn.Module] = []
        channels = in_channels
        for _ in range(n_blocks):
            blocks.append(nn.Conv3d(channels, hidden_channels, kernel_size, padding=pad))
            blocks.append(nn.GELU())
            channels = hidden_channels
        self.body = nn.Sequential(*blocks)
        self.field_head = nn.Conv3d(channels, out_channels, kernel_size, padding=pad)
        self.uncertainty_head = nn.Conv3d(channels, uncertainty_channels, kernel_size, padding=pad)
        if zero_init_last:
            nn.init.zeros_(self.field_head.weight)
            nn.init.zeros_(self.field_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, float(initial_log_var))

    def forward(
        self, inputs: torch.Tensor, backbone_prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape != backbone_prediction.shape:
            raise ValueError(
                f"adapter input/backbone shape mismatch: "
                f"{inputs.shape}/{backbone_prediction.shape}"
            )
        if inputs.ndim != 5:
            raise ValueError("expected tensors with (N,T,H,W,C) layout")
        stacked = torch.cat((inputs, backbone_prediction), dim=-1).permute(0, 4, 1, 2, 3)
        features = self.body(stacked)
        residual = self.field_head(features).permute(0, 2, 3, 4, 1)
        log_var = self.uncertainty_head(features).permute(0, 2, 3, 4, 1)
        return residual, log_var


class HeteroskedasticResidualCNO(nn.Module):
    """Ŷ = frozen_backbone(X) + residual(X, backbone(X)), with log_var uncertainty prediction."""

    def __init__(
        self, backbone: nn.Module, adapter: HeteroskedasticResidualConv3dAdapter
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def train(self, mode: bool = True) -> HeteroskedasticResidualCNO:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base = self.backbone(inputs)
        residual, log_var = self.adapter(inputs, base)
        return base + residual, log_var

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def wrap_frozen_heteroskedastic_cno(
    backbone: nn.Module, adapter_config: dict[str, Any]
) -> HeteroskedasticResidualCNO:
    """Freeze a CNO and attach the configured heteroskedastic residual adapter."""
    if adapter_config.get("name") != "heteroskedastic_residual_conv3d":
        raise ValueError(f"unsupported adapter: {adapter_config.get('name')}")
    if adapter_config.get("input") != "concat_input_and_backbone":
        raise ValueError("adapter input must be concat_input_and_backbone")
    freeze_module(backbone)
    adapter = HeteroskedasticResidualConv3dAdapter(
        in_channels=int(adapter_config.get("in_channels", 6)),
        out_channels=int(adapter_config.get("out_channels", 3)),
        uncertainty_channels=int(adapter_config.get("uncertainty_channels", 2)),
        hidden_channels=int(adapter_config.get("hidden_channels", 32)),
        n_blocks=int(adapter_config.get("n_blocks", 2)),
        kernel_size=int(adapter_config.get("kernel_size", 3)),
        zero_init_last=bool(adapter_config.get("zero_init_last", True)),
        initial_log_var=float(adapter_config.get("initial_log_var", -4.0)),
    )
    return HeteroskedasticResidualCNO(backbone, adapter)


def load_heteroskedastic_residual_cno(
    checkpoint: dict[str, Any], device: torch.device
) -> HeteroskedasticResidualCNO:
    """Rebuild a HeteroskedasticResidualCNO from a checkpoint."""
    from load_baseline import build_model

    if checkpoint.get("model_kind") != "heteroskedastic_residual_cno":
        raise ValueError("checkpoint is not a heteroskedastic_residual_cno")
    backbone = build_model("cno", device=str(device))
    model = wrap_frozen_heteroskedastic_cno(backbone, checkpoint["adapter_config"])
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(f"load mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device)


def load_residual_cno(checkpoint: dict[str, Any], device: torch.device) -> ResidualCNO:
    """Rebuild a ResidualCNO from an E018/E019 checkpoint."""
    from load_baseline import build_model

    if checkpoint.get("model_kind") != "residual_cno":
        raise ValueError("checkpoint is not a residual_cno")
    backbone = build_model("cno", device=str(device))
    model = wrap_frozen_cno(backbone, checkpoint["adapter_config"])
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(f"residual CNO load mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device)


def extract_flow_invariants(inputs: torch.Tensor) -> torch.Tensor:
    """Extract 4-dimensional flow regime invariants z_flow from (B, T, H, W, C) inputs."""
    # Inflow boundary at left: x in [0, 4]
    u_inlet = inputs[:, :, :, :4, 0]
    v_inlet = inputs[:, :, :, :4, 1]

    u_inf = torch.mean(u_inlet, dim=(1, 2, 3)).unsqueeze(-1)  # (B, 1)
    v_inf = torch.mean(v_inlet, dim=(1, 2, 3)).unsqueeze(-1)

    aoa_proxy = torch.atan2(v_inf, u_inf.clamp(min=1e-5))

    u_std = torch.std(u_inlet, dim=(1, 2, 3)).unsqueeze(-1)
    v_std = torch.std(v_inlet, dim=(1, 2, 3)).unsqueeze(-1)
    turb_intensity = torch.sqrt(0.5 * (u_std**2 + v_std**2)) / (torch.abs(u_inf) + 1e-5)

    # Downstream wake energy: x in [16, 64]
    u_wake = inputs[:, :, :, 16:, 0]
    v_wake = inputs[:, :, :, 16:, 1]
    u_wake_fl = u_wake - torch.mean(u_wake, dim=(1, 2, 3), keepdim=True)
    v_wake_fl = v_wake - torch.mean(v_wake, dim=(1, 2, 3), keepdim=True)
    wake_tke = torch.mean(0.5 * (u_wake_fl**2 + v_wake_fl**2), dim=(1, 2, 3)).unsqueeze(-1)

    z_flow = torch.cat([u_inf, aoa_proxy, turb_intensity, wake_tke], dim=-1)  # (B, 4)
    return z_flow


class FiLMResidualConv3dAdapter(nn.Module):
    """Conv3d residual adapter with Physics-Conditioned FiLM modulation."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 3,
        flow_dim: int = 4,
        hidden_channels: int = 32,
        n_blocks: int = 2,
        kernel_size: int = 3,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        self.conv_blocks = nn.ModuleList()
        self.film_mlps = nn.ModuleList()
        channels = in_channels
        for _ in range(n_blocks):
            self.conv_blocks.append(nn.Conv3d(channels, hidden_channels, kernel_size, padding=pad))
            # MLP maps flow descriptor (B, 4) -> (gamma, beta) of size 2 * hidden_channels
            mlp = nn.Sequential(
                nn.Linear(flow_dim, 32),
                nn.GELU(),
                nn.Linear(32, 2 * hidden_channels),
            )
            # Initialize FiLM MLP output weights/bias to 0 so gamma=0, beta=0 initially
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            self.film_mlps.append(mlp)
            channels = hidden_channels

        self.out = nn.Conv3d(channels, out_channels, kernel_size, padding=pad)
        if zero_init_last:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, inputs: torch.Tensor, backbone_prediction: torch.Tensor) -> torch.Tensor:
        z_flow = extract_flow_invariants(inputs) # (B, 4)
        stacked = torch.cat((inputs, backbone_prediction), dim=-1).permute(0, 4, 1, 2, 3) # (B, C, T, H, W)

        h = stacked
        for conv, mlp in zip(self.conv_blocks, self.film_mlps):
            h = conv(h)
            film_params = mlp(z_flow) # (B, 2 * hidden_channels)
            gamma = film_params[:, :self.hidden_channels].view(-1, self.hidden_channels, 1, 1, 1)
            beta = film_params[:, self.hidden_channels:].view(-1, self.hidden_channels, 1, 1, 1)
            h = F.gelu((1.0 + gamma) * h + beta)

        residual = self.out(h).permute(0, 2, 3, 4, 1)
        return residual


class FiLMResidualCNO(nn.Module):
    """Ŷ = frozen_backbone(X) + film_adapter(X, backbone(X))."""

    def __init__(self, backbone: nn.Module, adapter: FiLMResidualConv3dAdapter) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def train(self, mode: bool = True) -> FiLMResidualCNO:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.backbone(inputs)
        return base + self.adapter(inputs, base)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def wrap_frozen_film_cno(backbone: nn.Module, adapter_config: dict[str, Any]) -> FiLMResidualCNO:
    """Freeze a CNO and attach the configured FiLM residual adapter."""
    if adapter_config.get("name") != "film_residual_conv3d":
        raise ValueError(f"unsupported adapter: {adapter_config.get('name')}")
    freeze_module(backbone)
    adapter = FiLMResidualConv3dAdapter(
        in_channels=int(adapter_config.get("in_channels", 6)),
        out_channels=int(adapter_config.get("out_channels", 3)),
        flow_dim=int(adapter_config.get("flow_dim", 4)),
        hidden_channels=int(adapter_config.get("hidden_channels", 32)),
        n_blocks=int(adapter_config.get("n_blocks", 2)),
        kernel_size=int(adapter_config.get("kernel_size", 3)),
        zero_init_last=bool(adapter_config.get("zero_init_last", True)),
    )
    return FiLMResidualCNO(backbone, adapter)


def load_film_residual_cno(checkpoint: dict[str, Any], device: torch.device) -> FiLMResidualCNO:
    """Rebuild a FiLMResidualCNO from a checkpoint."""
    from load_baseline import build_model

    if checkpoint.get("model_kind") != "film_residual_cno":
        raise ValueError("checkpoint is not a film_residual_cno")
    backbone = build_model("cno", device=str(device))
    model = wrap_frozen_film_cno(backbone, checkpoint["adapter_config"])
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(f"film residual CNO load mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device)
