"""E024 Track 1 submission: shared-backbone residual ensemble.

Point predictions average three residual adapters on one frozen CNO. The
heteroskedastic member's log-variance is discarded. Bounds are the frozen
E006 interval, not learned uncertainty.
"""

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

# E005/E019 Gaussian statistics fitted only on the 64 training trajectories.
_MEAN_INPUT = np.array(
    [0.15338869392871857, -0.0004647469031624496, 0.0], dtype=np.float32
)
_STD_INPUT = np.array(
    [0.11143741756677628, 0.015735402703285217, 1.0], dtype=np.float32
)
_MEAN_TARGET = np.array(
    [0.15338662266731262, -0.00046777844545431435, 0.0], dtype=np.float32
)
_STD_TARGET = np.array(
    [0.11143583804368973, 0.015741463750600815, 1.0], dtype=np.float32
)

# E006 calibrated interval: half_width = alpha * abs(prediction) + beta * sigma_global.
_ALPHA = np.float32(0.025)
_ADDITIVE_HALF_WIDTH = np.float32(0.15 * 0.0563870259)

# Evaluate every sample independently so floating-point result cannot depend
# on the evaluator's batch composition.
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
        hidden_channels: int = 16,
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
        stacked = torch.cat((inputs, backbone_prediction), dim=-1).permute(0, 4, 1, 2, 3)
        features = self.body(stacked)
        residual = self.field_head(features).permute(0, 2, 3, 4, 1)
        log_var = self.uncertainty_head(features).permute(0, 2, 3, 4, 1)
        return residual, log_var


class SharedBackboneResidualEnsemble(nn.Module):
    """Ensemble of lightweight residual adapters evaluated on a single shared CNO backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        adapters: list[nn.Module],
    ) -> None:
        super().__init__()
        freeze_module(backbone)
        self.backbone = backbone
        self.adapters = nn.ModuleList(adapters)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.backbone(inputs)

        residuals: list[torch.Tensor] = []
        for adapter in self.adapters:
            if isinstance(adapter, HeteroskedasticResidualConv3dAdapter):
                # Keep only the field residual. Learned log-variance is not used
                # at inference; _bounds applies the frozen E006 interval.
                res, _ = adapter(inputs, base)
                residuals.append(res)
            elif isinstance(adapter, ResidualConv3dAdapter):
                res = adapter(inputs, base)
                residuals.append(res)
            else:
                out = adapter(inputs, base)
                if isinstance(out, tuple):
                    residuals.append(out[0])
                else:
                    residuals.append(out)

        stacked_residuals = torch.stack(residuals, dim=0)
        mean_residual = torch.mean(stacked_residuals, dim=0)
        return base + mean_residual


def _build_adapter(cfg: dict[str, Any]) -> nn.Module:
    name = cfg.get("name")
    if name == "residual_conv3d":
        return ResidualConv3dAdapter(
            in_channels=int(cfg.get("in_channels", 6)),
            out_channels=int(cfg.get("out_channels", 3)),
            hidden_channels=int(cfg["hidden_channels"]),
            n_blocks=int(cfg["n_blocks"]),
            kernel_size=int(cfg["kernel_size"]),
            zero_init_last=bool(cfg.get("zero_init_last", True)),
        )
    elif name == "heteroskedastic_residual_conv3d":
        return HeteroskedasticResidualConv3dAdapter(
            in_channels=int(cfg.get("in_channels", 6)),
            out_channels=int(cfg.get("out_channels", 3)),
            uncertainty_channels=int(cfg.get("uncertainty_channels", 2)),
            hidden_channels=int(cfg.get("hidden_channels", 32)),
            n_blocks=int(cfg.get("n_blocks", 2)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            zero_init_last=bool(cfg.get("zero_init_last", True)),
            initial_log_var=float(cfg.get("initial_log_var", -4.0)),
        )
    raise ValueError(f"unknown adapter type: {name}")


def _load_shared_backbone_ensemble(checkpoint_path: str, device: torch.device) -> SharedBackboneResidualEnsemble:
    from load_baseline import build_model

    bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if bundle.get("model_kind") != "shared_backbone_residual_ensemble":
        raise ValueError("checkpoint is not a shared_backbone_residual_ensemble")

    backbone = build_model("cno", device=str(device))
    backbone.load_state_dict(bundle["backbone_state_dict"], strict=True)

    adapters = []
    for entry in bundle["adapters"]:
        cfg = entry["config"]
        ad = _build_adapter(cfg).to(device)
        ad.load_state_dict(entry["state_dict"], strict=True)
        adapters.append(ad)

    ensemble = SharedBackboneResidualEnsemble(backbone, adapters)
    return ensemble.to(device)


def _get_model():
    if _STATE["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_shared_backbone_ensemble(_CHECKPOINT, device=device)
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
    if not np.all(np.isfinite(value)):
        raise ValueError("input array contains non-finite values")
    return value


def _persist_first(prediction: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    output = np.array(prediction, dtype=np.float32, copy=True)
    k = int(_PERSIST_FIRST_FRAMES)
    if k:
        output[:, :k] = inputs[:, -1:, ...]
    output[..., 2] = 0.0
    return output


def _bounds(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen E006 interval; not derived from adapter log-variance."""
    half_width = _ALPHA * np.abs(prediction) + _ADDITIVE_HALF_WIDTH
    half_width[..., 2] = 0.0
    lower = prediction - half_width
    upper = prediction + half_width
    lower[..., 2] = 0.0
    upper[..., 2] = 0.0
    return lower.astype(np.float32, copy=False), upper.astype(np.float32, copy=False)


def predict(input_array, metadata=None) -> dict[str, np.ndarray]:
    """Forecast 20 frames with Shared-Backbone Residual Ensemble, persist first 4, and return calibrated bounds."""
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
