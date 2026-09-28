#!/usr/bin/env python3
"""Train Experiment E033: Champion Tri-Loss Pareto Ensemble on All 81 Public Trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
DATA = REPO.parent / "RealPDE-Competition-Data"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from load_baseline import load_baseline  # noqa: E402
from realpde_t1.adapter import (  # noqa: E402
    wrap_frozen_cno,
    wrap_frozen_film_cno,
)
from realpde_t1.training import (  # noqa: E402
    RealWindowDataset,
    WindowGeometry,
    make_optimizer,
    scale_balanced_uv_relative_mse,
    temporal_mean_relative_mse,
)
from realpde_t1.validation import audit_split, load_config  # noqa: E402
from realpde_t1.wavelet_loss import MultiScaleWaveletTKELoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "e033_champion_ensemble.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e033_champion_ensemble",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_rollout_member(
    backbone_model: torch.nn.Module,
    adapter_cfg: dict,
    normalizer: dict,
    dataset: RealWindowDataset,
    device: torch.device,
    num_updates: int = 600,
) -> tuple[dict, float]:
    print("\n--- TRAINING MEMBER 1: ROLLOUT CURRICULUM CHAMPION ---", flush=True)
    model = wrap_frozen_cno(backbone_model, adapter_cfg).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(trainable_params, name="AdamW", learning_rate=1e-3, betas=[0.9, 0.999], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=0.0)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    start_time = time.perf_counter()
    data_iter = iter(loader)
    losses = []

    for update in range(1, num_updates + 1):
        try:
            inputs_50, targets_50 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs_50, targets_50 = next(data_iter)

        model.train()
        optimizer.zero_grad()
        inputs_50 = inputs_50.to(device, non_blocking=True)
        targets_50 = targets_50.to(device, non_blocking=True)

        x1 = inputs_50[:, :20]
        y1 = targets_50[:, :20]
        x1_norm = (x1 - mean_in) / std_in
        y1_norm = (y1 - mean_tg) / std_tg

        pred1_norm = model(x1_norm)
        l1_field = scale_balanced_uv_relative_mse(pred1_norm, y1_norm, mean_tg, std_tg, measured_channels=2)
        l1_mean = temporal_mean_relative_mse(pred1_norm, y1_norm, mean_tg, std_tg, measured_channels=2)
        loss1 = l1_field + 0.1 * l1_mean

        do_rollout = (update > 300) and (random.random() < 0.5)
        if do_rollout:
            pred1_phys = pred1_norm * std_tg + mean_tg
            pred1_phys[..., 2] = 0.0
            x2 = torch.cat((x1[:, 10:20], pred1_phys[:, :10]), dim=1)
            y2 = targets_50[:, 10:30]
            x2_norm = (x2 - mean_in) / std_in
            y2_norm = (y2 - mean_tg) / std_tg
            pred2_norm = model(x2_norm)
            l2_field = scale_balanced_uv_relative_mse(pred2_norm, y2_norm, mean_tg, std_tg, measured_channels=2)
            l2_mean = temporal_mean_relative_mse(pred2_norm, y2_norm, mean_tg, std_tg, measured_channels=2)
            loss = 0.5 * loss1 + 0.5 * (l2_field + 0.1 * l2_mean)
        else:
            loss = loss1

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if update == 1 or update % 100 == 0 or update == num_updates:
            print(f"Member 1 update={update:3d}/{num_updates} loss={losses[-1]:.6f} elapsed_s={time.perf_counter() - start_time:.1f}", flush=True)

    elapsed = time.perf_counter() - start_time
    return model.adapter.state_dict(), elapsed


def train_film_member(
    backbone_model: torch.nn.Module,
    adapter_cfg: dict,
    normalizer: dict,
    dataset: RealWindowDataset,
    device: torch.device,
    num_updates: int = 600,
) -> tuple[dict, float]:
    print("\n--- TRAINING MEMBER 2: PHYSICS-CONDITIONED INVARIANT FILM ADAPTER ---", flush=True)
    model = wrap_frozen_film_cno(backbone_model, adapter_cfg).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(trainable_params, name="AdamW", learning_rate=1e-3, betas=[0.9, 0.999], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=0.0)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    start_time = time.perf_counter()
    data_iter = iter(loader)
    losses = []

    for update in range(1, num_updates + 1):
        try:
            inputs_50, targets_50 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs_50, targets_50 = next(data_iter)

        model.train()
        optimizer.zero_grad()
        inputs_50 = inputs_50.to(device, non_blocking=True)
        targets_50 = targets_50.to(device, non_blocking=True)

        x1 = inputs_50[:, :20]
        y1 = targets_50[:, :20]
        x1_norm = (x1 - mean_in) / std_in
        y1_norm = (y1 - mean_tg) / std_tg

        pred1_norm = model(x1_norm)
        l1_field = scale_balanced_uv_relative_mse(pred1_norm, y1_norm, mean_tg, std_tg, measured_channels=2)
        l1_mean = temporal_mean_relative_mse(pred1_norm, y1_norm, mean_tg, std_tg, measured_channels=2)
        loss1 = l1_field + 0.1 * l1_mean

        do_rollout = (update > 300) and (random.random() < 0.5)
        if do_rollout:
            pred1_phys = pred1_norm * std_tg + mean_tg
            pred1_phys[..., 2] = 0.0
            x2 = torch.cat((x1[:, 10:20], pred1_phys[:, :10]), dim=1)
            y2 = targets_50[:, 10:30]
            x2_norm = (x2 - mean_in) / std_in
            y2_norm = (y2 - mean_tg) / std_tg
            pred2_norm = model(x2_norm)
            l2_field = scale_balanced_uv_relative_mse(pred2_norm, y2_norm, mean_tg, std_tg, measured_channels=2)
            l2_mean = temporal_mean_relative_mse(pred2_norm, y2_norm, mean_tg, std_tg, measured_channels=2)
            loss = 0.5 * loss1 + 0.5 * (l2_field + 0.1 * l2_mean)
        else:
            loss = loss1

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if update == 1 or update % 100 == 0 or update == num_updates:
            print(f"Member 2 update={update:3d}/{num_updates} loss={losses[-1]:.6f} elapsed_s={time.perf_counter() - start_time:.1f}", flush=True)

    elapsed = time.perf_counter() - start_time
    return model.adapter.state_dict(), elapsed


def train_wavelet_member(
    backbone_model: torch.nn.Module,
    adapter_cfg: dict,
    normalizer: dict,
    dataset: RealWindowDataset,
    device: torch.device,
    num_updates: int = 600,
) -> tuple[dict, float]:
    print("\n--- TRAINING MEMBER 3: MULTI-SCALE WAVELET-TKE CHAMPION ---", flush=True)
    model = wrap_frozen_cno(backbone_model, adapter_cfg).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(trainable_params, name="AdamW", learning_rate=1e-3, betas=[0.9, 0.999], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=0.0)

    criterion = MultiScaleWaveletTKELoss(lambda_mean=0.2, lambda_tke=0.4, lambda_wavelet=0.2, eps=1e-6).to(device)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    start_time = time.perf_counter()
    data_iter = iter(loader)
    losses = []

    for update in range(1, num_updates + 1):
        try:
            inputs_50, targets_50 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs_50, targets_50 = next(data_iter)

        model.train()
        optimizer.zero_grad()
        inputs = inputs_50[:, :20].to(device, non_blocking=True)
        targets = targets_50[:, :20].to(device, non_blocking=True)

        x_norm = (inputs - mean_in) / std_in
        y_norm = (targets - mean_tg) / std_tg

        pred_norm = model(x_norm)
        loss, metrics = criterion(pred_norm, y_norm, mean_tg, std_tg)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if update == 1 or update % 100 == 0 or update == num_updates:
            print(f"Member 3 update={update:3d}/{num_updates} loss={losses[-1]:.6f} l_tke={metrics['loss_tke']:.4f} elapsed_s={time.perf_counter() - start_time:.1f}", flush=True)

    elapsed = time.perf_counter() - start_time
    return model.adapter.state_dict(), elapsed


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts_dir = args.artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 20260815)))
    device = torch.device(args.device)

    # 1. Pre-audit split (all 81 trajectories)
    split = load_config(REPO / config["split_config"])
    audit_split(DATA / "train_real", split)
    print(f"Loaded all {len(split['training_files'])} public training trajectories.")

    # 2. Geometry for 50-frame rollout datasets
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

    # 3. Load frozen all-data CNO backbone
    initial_ckpt_path = REPO / config["initial_checkpoint"]
    init_ckpt = torch.load(initial_ckpt_path, map_location="cpu", weights_only=False)
    normalizer = init_ckpt["normalizer"]

    backbone_model, _ = load_baseline(
        config["models"]["base_model_type"],
        str(initial_ckpt_path),
        device=str(device),
    )
    for p in backbone_model.parameters():
        p.requires_grad = False
    backbone_model.eval()

    num_updates = int(config["training"]["num_updates_per_member"])

    # Train Member 1: Rollout Curriculum
    m1_state, t1 = train_rollout_member(
        backbone_model, config["models"]["member1_rollout"], normalizer, dataset, device, num_updates=num_updates
    )

    # Train Member 2: Physics-Conditioned Invariant FiLM Adapter
    m2_state, t2 = train_film_member(
        backbone_model, config["models"]["member2_film"], normalizer, dataset, device, num_updates=num_updates
    )

    # Train Member 3: Multi-Scale Wavelet-TKE Adapter
    m3_state, t3 = train_wavelet_member(
        backbone_model, config["models"]["member3_wavelet"], normalizer, dataset, device, num_updates=num_updates
    )

    total_training_time = t1 + t2 + t3

    # 4. Save consolidated single-backbone ensemble payload
    payload = {
        "experiment": str(config.get("experiment", "E033")),
        "status": "all_public_data_champion_ensemble",
        "base_model_type": "cno",
        "backbone_state_dict": backbone_model.state_dict(),
        "member1_adapter_config": config["models"]["member1_rollout"],
        "member1_adapter_state": m1_state,
        "member2_adapter_config": config["models"]["member2_film"],
        "member2_adapter_state": m2_state,
        "member3_adapter_config": config["models"]["member3_wavelet"],
        "member3_adapter_state": m3_state,
        "normalizer": normalizer,
        "uncertainty_config": config["uncertainty"],
        "total_training_seconds": total_training_time,
    }

    final_payload_path = artifacts_dir / "ensemble_payload.pth"
    torch.save(payload, final_payload_path)
    (artifacts_dir / "normalizer.json").write_text(json.dumps(normalizer, indent=2), encoding="utf-8")

    size_mb = final_payload_path.stat().st_size / (1024 * 1024)
    print(f"\nAll 3 ensemble members trained in {total_training_time:.1f}s.")
    print(f"Consolidated payload saved to {final_payload_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
