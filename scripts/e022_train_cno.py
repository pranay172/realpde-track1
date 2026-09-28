#!/usr/bin/env python3
"""Train the preregistered E022 Heteroskedastic Spatio-Temporal Residual CNO."""

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
DEFAULT_CONFIG = REPO / "configs" / "e022_heteroskedastic_residual.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e022_heteroskedastic_residual"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.adapter import wrap_frozen_heteroskedastic_cno  # noqa: E402
from realpde_t1.training import (  # noqa: E402
    RealWindowDataset,
    WindowGeometry,
    compute_gaussian_stats,
    heteroskedastic_residual_loss,
    make_optimizer,
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

    adapter_config = dict(model_config["adapter"])
    model = wrap_frozen_heteroskedastic_cno(model, adapter_config).to(device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

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

    output_root = args.artifacts
    if not args.smoke and (output_root / "final.pth").exists():
        raise FileExistsError(
            f"refusing to replace completed {experiment} checkpoint: {output_root / 'final.pth'}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "normalizer.json").write_text(
        json.dumps(stats.as_json(), indent=2) + "\n", encoding="utf-8"
    )
    history_path = output_root / "training.jsonl"
    history_stream = history_path.open("w", encoding="utf-8")
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

    loss_config = train_config["loss"]
    lambda_mean = float(loss_config.get("lambda_mean", 0.1))
    lambda_nll = float(loss_config.get("lambda_nll", 0.05))

    try:
        for update in range(1, num_updates + 1):
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

            prediction, log_var = model(inputs)
            loss, parts = heteroskedastic_residual_loss(
                prediction,
                log_var,
                targets,
                mean_target,
                std_target,
                measured_channels=int(loss_config.get("measured_channels", 2)),
                denominator_epsilon=float(loss_config.get("denominator_epsilon", 1e-12)),
                lambda_mean=lambda_mean,
                lambda_nll=lambda_nll,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at update {update}: {loss}")
            loss.backward()
            clip = float(train_config["gradient_clip_norm"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip)
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
            }
            row.update({k: float(v.detach().cpu()) for k, v in parts.items()})
            history_stream.write(json.dumps(row) + "\n")
            if update % 25 == 0 or update == 1 or update == num_updates:
                history_stream.flush()
                print(
                    f"update {update:04d}/{num_updates:04d} | "
                    f"loss: {value:.6f} (field: {row['field_rel']:.6f}, mean: {row['mean_wake']:.6f}, nll: {row['nll']:.4f}, std: {row['mean_predicted_std']:.4f}) | "
                    f"lr: {row['learning_rate']:.3e} | elapsed: {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        history_stream.close()

    total_time = time.perf_counter() - started
    final_state = cpu_state_dict(model)
    bundle = {
        "experiment": experiment,
        "model_kind": "heteroskedastic_residual_cno",
        "adapter_config": adapter_config,
        "config_sha256": config_sha256,
        "git_head": git_head(),
        "seed": seed,
        "model_state_dict": final_state,
        "train_losses": losses,
        "iteration": num_updates,
        "normalizer": stats.as_json(),
    }
    final_path = output_root / "final.pth"
    atomic_torch_save(bundle, final_path)
    peak_gpu_bytes = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    report = {
        "experiment": experiment,
        "config_sha256": config_sha256,
        "checkpoint_sha256": sha256_file(final_path),
        "total_wall_seconds": total_time,
        "peak_gpu_bytes": peak_gpu_bytes,
        "first_25_mean_loss": float(np.mean(losses[:25])),
        "final_25_mean_loss": float(np.mean(losses[-25:])),
        "final_loss": losses[-1],
    }
    (output_root / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("\nTraining completed successfully:")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
