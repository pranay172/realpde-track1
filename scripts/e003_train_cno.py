#!/usr/bin/env python3
"""Train a fixed-budget CNO experiment without touching E002 validation fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
DEFAULT_CONFIG = REPO / "configs" / "e003_cno_finetune.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e003_cno_finetune"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.adapter import wrap_frozen_cno  # noqa: E402
from realpde_t1.training import (  # noqa: E402
    RealWindowDataset,
    WindowGeometry,
    compute_gaussian_stats,
    make_optimizer,
    mfdr_spectral_loss,
    scale_balanced_uv_relative_mse,
    split_mean_fluct_tke_loss,
    wake_weighted_relative_mse,
)
from realpde_t1.validation import (  # noqa: E402
    audit_split,
    files_for_reynolds,
    load_config,
)
from load_baseline import load_baseline  # noqa: E402


def parse_args(
    default_config: Path = DEFAULT_CONFIG,
    default_artifacts: Path = DEFAULT_ARTIFACTS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--artifacts", type=Path, default=default_artifacts)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="run two updates on one training file")
    parser.add_argument("--smoke-updates", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--smoke-preload", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "resume an interrupted fixed-budget run from its recovery latest.pth "
            "(written every 100 updates); configuration must be unchanged"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve_repo_path(value: str) -> Path:
    return (REPO / value).resolve()


def atomic_torch_save(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def main(
    default_config: Path = DEFAULT_CONFIG,
    default_artifacts: Path = DEFAULT_ARTIFACTS,
) -> None:
    args = parse_args(default_config, default_artifacts)
    raw_config = args.config.read_bytes()
    config = json.loads(raw_config)
    experiment = str(config["experiment"])
    config_sha256 = hashlib.sha256(raw_config).hexdigest()
    split_config_path = resolve_repo_path(config["split_config"])
    split_config = load_config(split_config_path)
    data_root = PROJECT / "RealPDE-Competition-Data" / "train_real"
    manifest = audit_split(data_root, split_config)
    seed = int(config["seed"])
    set_determinism(seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    training_files = list(split_config["training_files"])
    if "training_nominal_re" in config:
        training_files = files_for_reynolds(
            training_files, config["training_nominal_re"]
        )
    train_config = config["training"]
    data_config = config["data"]
    num_updates = int(train_config["num_updates"])
    batch_size = int(train_config["batch_size"])
    preload = bool(data_config["preload_trajectories"])
    if args.smoke:
        training_files = training_files[:1]
        num_updates = args.smoke_updates
        batch_size = args.smoke_batch_size
        preload = args.smoke_preload
        if num_updates <= 0 or batch_size <= 0:
            raise ValueError("smoke updates and batch size must be positive")

    geometry = WindowGeometry(
        input_frames=int(data_config["input_frames"]),
        output_frames=int(data_config["output_frames"]),
        temporal_stride=int(data_config["window_stride"]),
        spatial_stride=int(data_config["spatial_stride"]),
    )
    print("computing train-only Gaussian statistics", flush=True)
    stats = compute_gaussian_stats(data_root, training_files, geometry)
    dataset = RealWindowDataset(data_root, training_files, geometry, preload=preload)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(data_config["shuffle"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"] and device.type == "cuda"),
        generator=generator,
        drop_last=True,
    )
    iterator = iter(loader)

    initial_checkpoint = resolve_repo_path(config["initial_checkpoint"])
    model_config = config.get("model") or {}
    base_model_type = model_config.get("base_model_type")
    if base_model_type:
        model, initial_metadata = load_baseline(base_model_type, str(initial_checkpoint), device=str(device))
    else:
        model, initial_metadata = load_baseline(str(initial_checkpoint), device=str(device))
    model = model.to(device)
    adapter_config = None
    if model_config.get("wrapper") == "residual_cno":
        if model_config.get("trainable") != "adapter_only":
            raise ValueError("residual_cno trains adapter_only; last-block unfreeze is a later experiment")
        adapter_config = dict(model_config["adapter"])
        model = wrap_frozen_cno(model, adapter_config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    optimizer = make_optimizer(
        trainable,
        name=str(train_config.get("optimizer", "Adam")),
        learning_rate=float(train_config["learning_rate"]),
        betas=train_config["betas"],
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=float(train_config["eta_min"])
    )
    mean_input = torch.from_numpy(stats.mean_input).to(device)
    std_input = torch.from_numpy(stats.std_input).to(device)
    mean_target = torch.from_numpy(stats.mean_target).to(device)
    std_target = torch.from_numpy(stats.std_target).to(device)

    if args.smoke:
        output_root = args.artifacts
    else:
        output_root = args.artifacts
        if (output_root / "final.pth").exists():
            raise FileExistsError(
                f"refusing to replace completed {experiment} checkpoint: {output_root / 'final.pth'}"
            )
    resume_info: dict[str, Any] | None = None
    start_update = 1
    if args.resume_from is not None:
        if args.smoke:
            raise ValueError("--resume-from is not available with --smoke")
        recovery = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if recovery.get("status") != "recovery":
            raise ValueError(f"resume source is not a recovery checkpoint: {args.resume_from}")
        if recovery.get("experiment") != experiment:
            raise ValueError(
                f"recovery experiment mismatch: {recovery.get('experiment')} != {experiment}"
            )
        if recovery.get("config_sha256") != config_sha256:
            raise ValueError("recovery checkpoint was written under a different configuration")
        if recovery.get("split_config_sha256") != split_config["config_sha256"]:
            raise ValueError("recovery checkpoint was written under a different split")
        recovered_update = int(recovery["update"])
        if not 0 < recovered_update < num_updates:
            raise ValueError(
                f"recovered update {recovered_update} is outside (0, {num_updates}); "
                "the run may already be complete"
            )
        model.load_state_dict(recovery["model_state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        scheduler.load_state_dict(recovery["scheduler_state_dict"])
        start_update = recovered_update + 1
        resume_info = {
            "path": str(args.resume_from),
            "recovery_sha256": sha256_file(args.resume_from),
            "recovery_update": recovered_update,
            "note": (
                "data-shuffle RNG position is not checkpointed, so the batch order after "
                "resume differs from an uninterrupted run; update budget and schedules are "
                "restored exactly"
            ),
        }
        print(json.dumps({"resumed_from_update": recovered_update}), flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "normalizer.json").write_text(
        json.dumps(stats.as_json(), indent=2) + "\n", encoding="utf-8"
    )
    history_path = output_root / "training.jsonl"
    history_stream = history_path.open(
        "a" if resume_info is not None else "w", encoding="utf-8"
    )
    losses: list[float] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(
        json.dumps({
            "device": str(device), "files": len(training_files), "windows": len(dataset),
            "batch_size": batch_size, "updates": num_updates, "normalizer": stats.as_json(),
        }, indent=2),
        flush=True,
    )
    try:
        for update in range(start_update, num_updates + 1):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets = next(iterator)
            model.train()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs = (inputs - mean_input) / std_input
            targets = (targets - mean_target) / std_target
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss_config = train_config["loss"]
            loss_parts: dict[str, float] = {}
            if isinstance(loss_config, str):
                if loss_config != "normalized MSE over u, v, and zero-pressure channels":
                    raise ValueError(f"unsupported training loss: {loss_config}")
                loss = torch.mean((prediction - targets) ** 2)
            elif loss_config.get("name") == "per_window_physical_relative_mse":
                loss = scale_balanced_uv_relative_mse(
                    prediction,
                    targets,
                    mean_target,
                    std_target,
                    measured_channels=int(loss_config["measured_channels"]),
                    denominator_epsilon=float(loss_config["denominator_epsilon"]),
                )
            elif loss_config.get("name") == "split_mean_fluct_tke":
                if float(loss_config.get("lambda_spectrum", 0.0)) != 0.0:
                    raise ValueError("E016 does not implement a spectral term")
                loss, parts = split_mean_fluct_tke_loss(
                    prediction,
                    targets,
                    mean_target,
                    std_target,
                    measured_channels=int(loss_config["measured_channels"]),
                    denominator_epsilon=float(loss_config["denominator_epsilon"]),
                    tke_denominator_epsilon=float(loss_config["tke_denominator_epsilon"]),
                    lambda_mean=float(loss_config["lambda_mean"]),
                    lambda_fluct=float(loss_config["lambda_fluct"]),
                    lambda_tke=float(loss_config["lambda_tke"]),
                )
                loss_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}
            elif loss_config.get("name") == "wake_weighted_relative_mse":
                if loss_config.get("weight") != "target_vorticity_dt":
                    raise ValueError("E017 supports only target vorticity-change weights")
                loss, parts = wake_weighted_relative_mse(
                    prediction,
                    targets,
                    mean_target,
                    std_target,
                    measured_channels=int(loss_config["measured_channels"]),
                    denominator_epsilon=float(loss_config["denominator_epsilon"]),
                    weight_epsilon=float(loss_config["weight_epsilon"]),
                    alpha=float(loss_config["alpha"]),
                )
                loss_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}
            elif loss_config.get("name") == "mfdr_spectral_loss":
                loss, parts = mfdr_spectral_loss(
                    prediction,
                    targets,
                    mean_target,
                    std_target,
                    measured_channels=int(loss_config["measured_channels"]),
                    denominator_epsilon=float(loss_config["denominator_epsilon"]),
                    tke_denominator_epsilon=float(loss_config["tke_denominator_epsilon"]),
                    spectral_epsilon=float(loss_config.get("spectral_epsilon", 1e-6)),
                    lambda_mean=float(loss_config["lambda_mean"]),
                    lambda_fluct=float(loss_config["lambda_fluct"]),
                    lambda_tke=float(loss_config["lambda_tke"]),
                    lambda_spec=float(loss_config["lambda_spec"]),
                )
                loss_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}
            else:
                raise ValueError(f"unsupported training loss: {loss_config}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}: {loss}")
            loss.backward()
            clip = float(train_config["gradient_clip_norm"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            elapsed = time.perf_counter() - started
            row = {
                "update": update,
                "loss": value,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
                **loss_parts,
            }
            history_stream.write(json.dumps(row) + "\n")
            if update == start_update or update % 10 == 0 or update == num_updates:
                history_stream.flush()
                recent = float(np.mean(losses[-25:]))
                print(
                    f"update={update}/{num_updates} loss={value:.6f} "
                    f"mean25={recent:.6f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            if not args.smoke and update % 100 == 0:
                recovery = {
                    "experiment": experiment,
                    "status": "recovery",
                    "update": update,
                    "model_state_dict": cpu_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "normalizer": stats.as_json(),
                    "config_sha256": config_sha256,
                    "split_config_sha256": split_config["config_sha256"],
                }
                atomic_torch_save(recovery, output_root / "latest.pth")
            if (
                update < num_updates
                and elapsed > float(train_config["maximum_wall_minutes"]) * 60
            ):
                raise TimeoutError(
                    f"{experiment} exceeded its {train_config['maximum_wall_minutes']}-minute wall budget"
                )
    finally:
        history_stream.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    checkpoint = {
        "format_version": 2 if adapter_config is not None else 1,
        "experiment": experiment,
        "status": "fixed_budget_final",
        "model_kind": "residual_cno" if adapter_config is not None else "cno",
        "adapter_config": adapter_config,
        "model_state_dict": cpu_state_dict(model),
        "normalizer": stats.as_json(),
        "training": {
            "updates": num_updates,
            "batch_size": batch_size,
            "loss": train_config["loss"],
            "optimizer": str(train_config.get("optimizer", "Adam")),
            "examples_seen": num_updates * batch_size,
            "effective_epochs": num_updates * batch_size / len(dataset),
            "losses": losses,
            "final_learning_rate": float(scheduler.get_last_lr()[0]),
            "training_seconds": training_seconds,
        },
        "provenance": {
            "git_commit": git_head(),
            "config_sha256": config_sha256,
            "split_config_sha256": split_config["config_sha256"],
            "data_inventory_sha256": manifest["data_inventory_sha256"],
            "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
            "initial_checkpoint_metadata": initial_metadata,
            "training_files": training_files,
            "validation_field_values_accessed": [],
            "validation_metadata_audited": True,
            "resume": resume_info,
        },
    }
    final_path = output_root / ("smoke_final.pth" if args.smoke else "final.pth")
    atomic_torch_save(checkpoint, final_path)
    report = {
        "experiment": experiment,
        "status": "smoke_complete" if args.smoke else "training_complete_not_yet_evaluated",
        "git_commit": git_head(),
        "config_sha256": config_sha256,
        "split_config_sha256": split_config["config_sha256"],
        "data_inventory_sha256": manifest["data_inventory_sha256"],
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "training_files": len(training_files),
        "validation_field_values_accessed": [],
        "validation_metadata_audited": True,
        "training_windows": len(dataset),
        "normalizer": stats.as_json(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "model_kind": "residual_cno" if adapter_config is not None else "cno",
        "adapter_config": adapter_config,
        "updates": num_updates,
        "batch_size": batch_size,
        "loss": train_config["loss"],
        "optimizer": str(train_config.get("optimizer", "Adam")),
        "examples_seen": num_updates * batch_size,
        "effective_epochs": num_updates * batch_size / len(dataset),
        "training_seconds": training_seconds,
        "initial_mean_loss_25": float(np.mean(losses[:25])),
        "final_mean_loss_25": float(np.mean(losses[-25:])),
        "minimum_training_loss": float(np.min(losses)),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "resumed_from": resume_info,
        "final_checkpoint": str(final_path),
        "final_checkpoint_bytes": final_path.stat().st_size,
        "final_checkpoint_sha256": sha256_file(final_path),
    }
    report_name = "smoke_report.json" if args.smoke else "train_report.json"
    (output_root / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
