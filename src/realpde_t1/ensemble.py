"""Shared-backbone residual ensembling for Track 1."""

from __future__ import annotations

from typing import Any, Sequence
from pathlib import Path
import hashlib
import torch
from torch import nn

from realpde_t1.adapter import freeze_module, ResidualConv3dAdapter, HeteroskedasticResidualConv3dAdapter


class SharedBackboneResidualEnsemble(nn.Module):
    """Ensemble of lightweight residual adapters evaluated on a single shared CNO backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        adapters: list[nn.Module],
        adapter_configs: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        if not adapters:
            raise ValueError("ensemble must contain at least one adapter")
        freeze_module(backbone)
        self.backbone = backbone
        self.adapters = nn.ModuleList(adapters)
        self.adapter_configs = adapter_configs

    def train(self, mode: bool = True) -> SharedBackboneResidualEnsemble:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass running the backbone once and averaging adapter corrections.

        Returns:
            mean_prediction: (N, T, H, W, 3)
            variance_prediction: (N, T, H, W, 2) total variance or None
        """
        with torch.no_grad():
            base = self.backbone(inputs)

        residuals: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []

        for adapter in self.adapters:
            if isinstance(adapter, HeteroskedasticResidualConv3dAdapter):
                res, log_var = adapter(inputs, base)
                residuals.append(res)
                variances.append(torch.exp(log_var))
            elif isinstance(adapter, ResidualConv3dAdapter):
                res = adapter(inputs, base)
                residuals.append(res)
            else:
                out = adapter(inputs, base)
                if isinstance(out, tuple):
                    residuals.append(out[0])
                    variances.append(torch.exp(out[1]))
                else:
                    residuals.append(out)

        stacked_residuals = torch.stack(residuals, dim=0) # (M, N, T, H, W, 3)
        mean_residual = torch.mean(stacked_residuals, dim=0)
        mean_prediction = base + mean_residual

        # Calculate total variance = epistemic variance across adapters + mean aleatoric variance
        if len(residuals) > 1:
            epistemic_var = torch.var(stacked_residuals[..., :2], dim=0, unbiased=False)
        else:
            epistemic_var = torch.zeros_like(mean_prediction[..., :2])

        if variances:
            aleatoric_var = torch.mean(torch.stack(variances, dim=0), dim=0)
            total_var = epistemic_var + aleatoric_var
        else:
            total_var = epistemic_var

        return mean_prediction, total_var


def build_residual_adapter_from_config(config: dict[str, Any]) -> nn.Module:
    """Instantiate a residual adapter according to its configuration dict."""
    name = config.get("name")
    if name == "residual_conv3d":
        return ResidualConv3dAdapter(
            in_channels=int(config.get("in_channels", 6)),
            out_channels=int(config.get("out_channels", 3)),
            hidden_channels=int(config.get("hidden_channels", 16)),
            n_blocks=int(config.get("n_blocks", 2)),
            kernel_size=int(config.get("kernel_size", 3)),
            zero_init_last=bool(config.get("zero_init_last", True)),
        )
    elif name == "heteroskedastic_residual_conv3d":
        return HeteroskedasticResidualConv3dAdapter(
            in_channels=int(config.get("in_channels", 6)),
            out_channels=int(config.get("out_channels", 3)),
            uncertainty_channels=int(config.get("uncertainty_channels", 2)),
            hidden_channels=int(config.get("hidden_channels", 32)),
            n_blocks=int(config.get("n_blocks", 2)),
            kernel_size=int(config.get("kernel_size", 3)),
            zero_init_last=bool(config.get("zero_init_last", True)),
            initial_log_var=float(config.get("initial_log_var", -4.0)),
        )
    raise ValueError(f"unknown adapter type: {name}")


def load_shared_backbone_ensemble(
    checkpoints: list[dict[str, Any]],
    device: torch.device,
) -> SharedBackboneResidualEnsemble:
    """Build and load a SharedBackboneResidualEnsemble from multiple adapter checkpoints."""
    from load_baseline import build_model

    if not checkpoints:
        raise ValueError("must provide at least one checkpoint")

    backbone = build_model("cno", device=str(device))
    # Load backbone weights from the first checkpoint
    backbone_state = {}
    for k, v in checkpoints[0]["model_state_dict"].items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=True)

    adapters: list[nn.Module] = []
    adapter_configs: list[dict[str, Any]] = []

    for ckpt in checkpoints:
        ad_cfg = ckpt["adapter_config"]
        adapter = build_residual_adapter_from_config(ad_cfg).to(device)
        # Extract adapter weights from the checkpoint state dict
        adapter_state = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("adapter."):
                adapter_state[k[len("adapter."):]] = v
        adapter.load_state_dict(adapter_state, strict=True)
        adapters.append(adapter)
        adapter_configs.append(ad_cfg)

    ensemble = SharedBackboneResidualEnsemble(backbone, adapters, adapter_configs)
    return ensemble.to(device)


def _prefixed_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: tensor for key, tensor in state.items() if key.startswith(prefix)}


def extract_backbone_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    return _prefixed_state(checkpoint["model_state_dict"], "backbone.")


def extract_adapter_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    return _prefixed_state(checkpoint["model_state_dict"], "adapter.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_identical_backbones(states: Sequence[dict[str, torch.Tensor]]) -> None:
    if not states:
        raise ValueError("no backbone states to compare")
    reference = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state.keys() != reference.keys():
            raise AssertionError(f"backbone key mismatch at member {index}")
        for key, tensor in reference.items():
            if not torch.equal(tensor.cpu(), state[key].cpu()):
                raise AssertionError(f"backbone tensor mismatch at member {index}: {key}")


def build_ensemble_bundle(
    checkpoints: Sequence[dict[str, Any]],
    *,
    source_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pack a shared-backbone ensemble from residual-CNO member checkpoints."""
    if len(checkpoints) < 2:
        raise ValueError("ensemble bundle requires at least two member checkpoints")
    backbones = [extract_backbone_state_dict(item) for item in checkpoints]
    if any(not state for state in backbones):
        raise ValueError("a member checkpoint is missing backbone weights")
    assert_identical_backbones(backbones)
    adapters = []
    for item in checkpoints:
        adapter_state = extract_adapter_state_dict(item)
        if not adapter_state:
            raise ValueError("a member checkpoint is missing adapter weights")
        if "adapter_config" not in item:
            raise ValueError("a member checkpoint is missing adapter_config")
        adapters.append(
            {
                "config": dict(item["adapter_config"]),
                "state_dict": {key: tensor.detach().cpu().clone() for key, tensor in adapter_state.items()},
            }
        )
    bundle: dict[str, Any] = {
        "model_kind": "shared_backbone_residual_ensemble",
        "backbone_state_dict": {
            key: tensor.detach().cpu().clone() for key, tensor in backbones[0].items()
        },
        "adapters": adapters,
    }
    if source_sha256 is not None:
        bundle["source_checkpoint_sha256"] = list(source_sha256)
    return bundle


def assert_bundle_matches_sources(
    bundle: dict[str, Any],
    checkpoints: Sequence[dict[str, Any]],
) -> None:
    """Check that a saved bundle's tensors equal the source checkpoints."""
    if bundle.get("model_kind") != "shared_backbone_residual_ensemble":
        raise AssertionError("bundle is not a shared_backbone_residual_ensemble")
    expected = build_ensemble_bundle(checkpoints)
    if bundle["backbone_state_dict"].keys() != expected["backbone_state_dict"].keys():
        raise AssertionError("bundle backbone keys do not match sources")
    for key, tensor in expected["backbone_state_dict"].items():
        if not torch.equal(bundle["backbone_state_dict"][key].cpu(), tensor.cpu()):
            raise AssertionError(f"bundle backbone mismatch: {key}")
    if len(bundle["adapters"]) != len(expected["adapters"]):
        raise AssertionError("bundle adapter count does not match sources")
    for index, (got, want) in enumerate(zip(bundle["adapters"], expected["adapters"])):
        if got["config"] != want["config"]:
            raise AssertionError(f"bundle adapter config mismatch at member {index}")
        if got["state_dict"].keys() != want["state_dict"].keys():
            raise AssertionError(f"bundle adapter key mismatch at member {index}")
        for key, tensor in want["state_dict"].items():
            if not torch.equal(got["state_dict"][key].cpu(), tensor.cpu()):
                raise AssertionError(f"bundle adapter tensor mismatch at member {index}: {key}")


def load_member_checkpoints(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    checkpoints = []
    hashes = []
    for path in paths:
        checkpoints.append(torch.load(path, map_location="cpu", weights_only=False))
        hashes.append(_sha256_file(Path(path)))
    return checkpoints, hashes
