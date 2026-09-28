#!/usr/bin/env python3
"""Train Experiment E027: Residual Rollout Curriculum Training on Frozen E005 CNO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
DATA = REPO.parent / "RealPDE-Competition-Data"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from load_baseline import load_baseline  # noqa: E402
from realpde_t1.adapter import wrap_frozen_cno  # noqa: E402
from realpde_t1.training import (  # noqa: E402
    RealWindowDataset,
    WindowGeometry,
    make_optimizer,
    scale_balanced_uv_relative_mse,
    temporal_mean_relative_mse,
)
from realpde_t1.validation import audit_split, load_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "e027_residual_rollout_curriculum.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e027_residual_rollout",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train(
    config_path: Path,
    artifacts_dir: Path,
    device_name: str,
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = str(config["experiment"])
    if (artifacts_dir / "final.pth").exists():
        raise FileExistsError(f"refusing to replace completed checkpoint: {artifacts_dir}")
    split = load_config(REPO / config["split_config"])
    audit_split(DATA / "train_real", split)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 50-frame geometry: 20 input frames + 30 target frames
    geom = WindowGeometry(
        input_frames=int(config["data"]["input_frames"]),
        output_frames=int(config["data"]["output_frames"]),
        temporal_stride=int(config["data"]["window_stride"]),
        spatial_stride=int(config["data"]["spatial_stride"]),
    )
    dataset = RealWindowDataset(
        data_root=DATA / "train_real",
        files=split["training_files"],
        geometry=geom,
        preload=bool(config["data"].get("preload_trajectories", True)),
    )

    batch_size = int(config["training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", True)),
    )

    # Load frozen initial backbone
    initial_ckpt_path = REPO / config["initial_checkpoint"]
    init_ckpt = torch.load(initial_ckpt_path, map_location="cpu", weights_only=False)
    if config.get("require_clean_backbone_provenance", False):
        provenance = init_ckpt.get("provenance", {})
        if set(provenance.get("training_files", [])) != set(split["training_files"]):
            raise ValueError("backbone training membership does not match adapter fold")
        if provenance.get("validation_field_values_accessed") != []:
            raise ValueError("backbone lacks clean validation-access provenance")
    normalizer = init_ckpt["normalizer"]

    backbone_model, _ = load_baseline(
        config["model"]["base_model_type"],
        str(initial_ckpt_path),
        device=str(device),
    )
    model = wrap_frozen_cno(backbone_model, config["model"]["adapter"]).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())

    train_cfg = config["training"]
    optimizer = make_optimizer(
        trainable_params,
        name=train_cfg["optimizer"],
        learning_rate=float(train_cfg["learning_rate"]),
        betas=train_cfg["betas"],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    num_updates = int(train_cfg["num_updates"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_updates, eta_min=float(train_cfg.get("eta_min", 0.0))
    )

    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    curriculum_cfg = train_cfg.get("curriculum", {})
    tf_updates = int(curriculum_cfg.get("teacher_forcing_updates", 300))
    p_rollout = float(curriculum_cfg.get("rollout_probability", 0.5))
    step2_w = float(curriculum_cfg.get("step2_weight", 0.5))
    lambda_m = float(train_cfg["loss"].get("lambda_mean", 0.1))

    print(f"Starting {experiment} training: {num_updates} updates, batch {batch_size}, {trainable_count} trainable params...")
    losses: list[float] = []
    step2_losses: list[float] = []
    update = 0
    start_time = time.perf_counter()

    data_iter = iter(loader)
    while update < num_updates:
        try:
            inputs_50, targets_50 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs_50, targets_50 = next(data_iter)

        update += 1
        model.train()
        optimizer.zero_grad()

        inputs_50 = inputs_50.to(device, non_blocking=True)
        targets_50 = targets_50.to(device, non_blocking=True)

        x1 = inputs_50[:, :20]
        y1 = targets_50[:, :20]

        x1_norm = (x1 - mean_in) / std_in
        y1_norm = (y1 - mean_tg) / std_tg

        pred1_norm = model(x1_norm)
        l1_field = scale_balanced_uv_relative_mse(
            pred1_norm, y1_norm, mean_tg, std_tg, measured_channels=2
        )
        l1_mean = temporal_mean_relative_mse(
            pred1_norm, y1_norm, mean_tg, std_tg, measured_channels=2
        )
        loss1 = l1_field + lambda_m * l1_mean

        do_rollout = (update > tf_updates) and (random.random() < p_rollout)
        if do_rollout:
            # Construct 20-frame autoregressive rollout context: x1[10:20] + pred1_phys[0:10]
            pred1_phys = pred1_norm * std_tg + mean_tg
            pred1_phys[..., 2] = 0.0

            x2 = torch.cat((x1[:, 10:20], pred1_phys[:, :10]), dim=1)
            y2 = targets_50[:, 10:30]

            x2_norm = (x2 - mean_in) / std_in
            y2_norm = (y2 - mean_tg) / std_tg

            pred2_norm = model(x2_norm)
            l2_field = scale_balanced_uv_relative_mse(
                pred2_norm, y2_norm, mean_tg, std_tg, measured_channels=2
            )
            l2_mean = temporal_mean_relative_mse(
                pred2_norm, y2_norm, mean_tg, std_tg, measured_channels=2
            )
            loss2 = l2_field + lambda_m * l2_mean

            loss = (1.0 - step2_w) * loss1 + step2_w * loss2
            step2_losses.append(float(loss2.detach().cpu()))
        else:
            loss = loss1

        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {update}")
        loss.backward()
        optimizer.step()
        scheduler.step()

        val = float(loss.detach().cpu())
        losses.append(val)
        if time.perf_counter() - start_time > float(train_cfg["maximum_wall_minutes"]) * 60:
            raise TimeoutError(f"{experiment} exceeded registered training time budget")

        if update == 1 or update % 25 == 0 or update == num_updates:
            mean25 = float(np.mean(losses[-25:]))
            elapsed = time.perf_counter() - start_time
            print(
                f"update={update}/{num_updates} loss={val:.6f} mean25={mean25:.6f} "
                f"rollout={'yes' if do_rollout else 'no '} elapsed_s={elapsed:.1f}"
            )

    elapsed_total = time.perf_counter() - start_time
    # Save final checkpoint
    final_ckpt_path = artifacts_dir / "final.pth"
    payload = {
        "experiment": str(config.get("experiment", "E027")),
        "status": "fixed_budget_final",
        "model_kind": "residual_cno",
        "adapter_config": config["model"]["adapter"],
        "normalizer": normalizer,
        "model_state_dict": model.state_dict(),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "training_seconds": elapsed_total,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "config_sha256": sha256_file(config_path),
            "split_config_sha256": sha256_file(REPO / config["split_config"]),
            "initial_checkpoint_sha256": sha256_file(initial_ckpt_path),
            "training_files": list(split["training_files"]),
            "validation_field_values_accessed": [],
        },
    }
    torch.save(payload, final_ckpt_path)

    # Save normalizer json
    (artifacts_dir / "normalizer.json").write_text(
        json.dumps(normalizer, indent=2), encoding="utf-8"
    )

    report = {
        "experiment": experiment,
        "status": "training_complete",
        "seed": seed,
        "updates": num_updates,
        "batch_size": batch_size,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "initial_mean_loss_25": float(np.mean(losses[:25])),
        "final_mean_loss_25": float(np.mean(losses[-25:])),
        "minimum_loss": float(np.min(losses)),
        "training_seconds": elapsed_total,
        "checkpoint_sha256": sha256_file(final_ckpt_path),
        "provenance": payload["provenance"],
    }
    (artifacts_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"{experiment} training finished in {elapsed_total:.1f}s. Checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.artifacts, args.device)
