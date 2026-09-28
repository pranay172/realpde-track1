#!/usr/bin/env python3
"""Train Experiment E031: Physics-Conditioned Invariant FiLM Adapter with Rollout Curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
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
from realpde_t1.adapter import wrap_frozen_film_cno  # noqa: E402
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
        default=REPO / "configs" / "e031_film_adapter.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e031_film_adapter",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts_dir = args.artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 20260815)))
    device = torch.device(args.device)

    # Pre-audit training split
    split = load_config(REPO / config["split_config"])
    audit_split(DATA / "train_real", split)

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
    normalizer = init_ckpt["normalizer"]

    backbone_model, _ = load_baseline(
        config["model"]["base_model_type"],
        str(initial_ckpt_path),
        device=str(device),
    )
    model = wrap_frozen_film_cno(backbone_model, config["model"]["adapter"]).to(device)

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
    tf_updates = int(curriculum_cfg.get("phase1_updates", 300))
    p_rollout = float(curriculum_cfg.get("rollout_prob", 0.5))
    step2_w = 0.5
    lambda_m = float(train_cfg["loss"].get("lambda_mean", 0.1))

    print(f"Starting E031 training: {num_updates} updates, batch {batch_size}, {trainable_count} trainable params...")
    losses: list[float] = []
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
        else:
            loss = loss1

        loss.backward()
        if float(train_cfg.get("gradient_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, float(train_cfg["gradient_clip_norm"])
            )
        optimizer.step()
        scheduler.step()

        val = float(loss.detach().cpu())
        losses.append(val)

        if update == 1 or update % 25 == 0 or update == num_updates:
            mean25 = float(np.mean(losses[-25:]))
            elapsed = time.perf_counter() - start_time
            print(
                f"update={update}/{num_updates} loss={val:.6f} mean25={mean25:.6f} "
                f"rollout={'yes' if do_rollout else 'no '} elapsed_s={elapsed:.1f}"
            )

    elapsed_total = time.perf_counter() - start_time
    final_ckpt_path = artifacts_dir / "final.pth"
    payload = {
        "experiment": "E031",
        "status": "fixed_budget_final",
        "model_kind": "film_residual_cno",
        "adapter_config": config["model"]["adapter"],
        "normalizer": normalizer,
        "model_state_dict": model.state_dict(),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "training_seconds": elapsed_total,
    }
    torch.save(payload, final_ckpt_path)

    (artifacts_dir / "normalizer.json").write_text(
        json.dumps(normalizer, indent=2), encoding="utf-8"
    )
    print(f"\nE031 training finished in {elapsed_total:.1f}s. Saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
