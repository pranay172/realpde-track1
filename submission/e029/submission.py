"""E029 Track 1 submission: All-Data Residual Rollout Curriculum CNO with persist-first-4 and calibrated bounds."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_CHECKPOINT = os.path.join(_HERE, "model.pth")
_EXPECTED_SAMPLE_SHAPE = (20, 32, 64, 3)

# All-public-data Gaussian statistics fitted over the 81 usable public real trajectories.
_MEAN_INPUT = np.array(
    [0.15398305654525757, -0.0005586966290138662, 0.0], dtype=np.float32
)
_STD_INPUT = np.array(
    [0.1145491823554039, 0.015571942552924156, 1.0], dtype=np.float32
)
_MEAN_TARGET = np.array(
    [0.15397901833057404, -0.000563189503736794, 0.0], dtype=np.float32
)
_STD_TARGET = np.array(
    [0.11454980820417404, 0.015575873665511608, 1.0], dtype=np.float32
)

# E006 calibrated interval: half_width = alpha * abs(prediction) + beta * sigma_global.
_ALPHA = np.float32(0.025)
_ADDITIVE_HALF_WIDTH = np.float32(0.15 * 0.0563870259)

# Evaluate every sample independently for strict batch invariance.
_BATCH_SIZE = 1

# Replace the first k neural frames with last-frame persistence.
_PERSIST_FIRST_FRAMES = 4

_STATE = {"model": None, "device": None}


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
        hidden_channels: int = 32,
        n_blocks: int = 2,
        kernel_size: int = 3,
        zero_init_last: bool = True,
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
        if zero_init_last:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, inputs: torch.Tensor, backbone_prediction: torch.Tensor) -> torch.Tensor:
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


def wrap_frozen_cno(backbone: nn.Module, adapter_config: dict[str, Any]) -> ResidualCNO:
    """Freeze a CNO and attach the configured residual adapter."""
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


def _load_residual_cno(checkpoint_path: str, device: torch.device) -> ResidualCNO:
    from load_baseline import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_kind") != "residual_cno":
        raise ValueError("checkpoint is not a residual_cno")
    backbone = build_model("cno", device=str(device))
    model = wrap_frozen_cno(backbone, checkpoint["adapter_config"])
    res = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if getattr(res, "missing_keys", []) or getattr(res, "unexpected_keys", []):
        raise ValueError(f"residual CNO load mismatch: {res}")
    return model.to(device)


def _get_model():
    if _STATE["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_residual_cno(_CHECKPOINT, device=device)
        model.eval()
        _STATE["model"] = model
        _STATE["device"] = device
    return _STATE["model"], _STATE["device"]


def _validate_input(input_array) -> np.ndarray:
    value = np.asarray(input_array, dtype=np.float32)
    if value.ndim != 5 or tuple(value.shape[1:]) != _EXPECTED_SAMPLE_SHAPE:
        raise ValueError(
            "expected input shape (N,20,32,64,3), got %s" % (value.shape,)
        )
    if value.shape[0] == 0:
        raise ValueError("input batch must contain at least one sample")
    if not np.all(np.isfinite(value)):
        raise ValueError("input contains NaN or Inf")
    return value


def _persist_first(prediction: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Overwrite the first k predicted frames with the last input frame."""
    output = np.array(prediction, dtype=np.float32, copy=True)
    k = int(_PERSIST_FIRST_FRAMES)
    if k:
        output[:, :k] = inputs[:, -1:, ...]
    output[..., 2] = 0.0
    return output


def _bounds(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half_width = _ALPHA * np.abs(prediction) + _ADDITIVE_HALF_WIDTH
    half_width[..., 2] = 0.0
    lower = prediction - half_width
    upper = prediction + half_width
    lower[..., 2] = 0.0
    upper[..., 2] = 0.0
    return lower.astype(np.float32, copy=False), upper.astype(np.float32, copy=False)


def predict(input_array, metadata=None):
    """Forecast 20 frames with Residual CNO, persist first 4, and return calibrated bounds."""
    del metadata
    value = _validate_input(input_array)
    model, device = _get_model()

    mean_input = torch.as_tensor(_MEAN_INPUT, device=device)
    std_input = torch.as_tensor(_STD_INPUT, device=device)
    mean_target = torch.as_tensor(_MEAN_TARGET, device=device)
    std_target = torch.as_tensor(_STD_TARGET, device=device)

    n_samples = value.shape[0]
    predictions = np.empty_like(value)
    for index in range(0, n_samples, _BATCH_SIZE):
        batch = value[index : index + _BATCH_SIZE]
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        norm_in = (tensor - mean_input) / std_input
        with torch.no_grad():
            norm_out = model(norm_in)
        pred_phys = norm_out * std_target + mean_target
        pred_phys[..., 2] = 0.0
        predictions[index : index + _BATCH_SIZE] = (
            pred_phys.detach().to(dtype=torch.float32).cpu().numpy()
        )

    hybrid = _persist_first(predictions, value)
    lower, upper = _bounds(hybrid)
    return {
        "lower": lower,
        "prediction": hybrid,
        "upper": upper,
    }
