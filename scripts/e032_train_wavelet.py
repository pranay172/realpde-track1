#!/usr/bin/env python3
"""Train Experiment E032: Multi-Scale Wavelet & Fluctuation-Decoupled Spectral Energy Adapter."""

from __future__ import annotations

import argparse
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
from realpde_t1.adapter import wrap_frozen_cno  # noqa: E402
from realpde_t1.training import (  # noqa: E402
    RealWindowDataset,
    WindowGeometry,
    make_optimizer,
)
from realpde_t1.validation import audit_split, load_config  # noqa: E402
from realpde_t1.wavelet_loss import MultiScaleWaveletTKELoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "e032_wavelet_tke_adapter.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e032_wavelet_adapter",
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

    # 20 input + 20 output frames
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

    loss_cfg = train_cfg["loss"]
    criterion = MultiScaleWaveletTKELoss(
        lambda_mean=float(loss_cfg.get("lambda_mean", 0.2)),
        lambda_tke=float(loss_cfg.get("lambda_tke", 0.4)),
        lambda_wavelet=float(loss_cfg.get("lambda_wavelet", 0.2)),
        eps=float(loss_cfg.get("eps", 1e-6)),
    ).to(device)

    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    print(f"Starting E032 training: {num_updates} updates, batch {batch_size}, {trainable_count} trainable params...")
    losses: list[float] = []
    update = 0
    start_time = time.perf_counter()

    data_iter = iter(loader)
    while update < num_updates:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)

        update += 1
        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        x_norm = (inputs - mean_in) / std_in
        y_norm = (targets - mean_tg) / std_tg

        pred_norm = model(x_norm)
        loss, metrics = criterion(pred_norm, y_norm, mean_tg, std_tg)

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
                f"update={update:3d}/{num_updates} loss={val:.6f} mean25={mean25:.6f} "
                f"l_field={metrics['loss_field']:.4f} l_tke={metrics['loss_tke']:.4f} "
                f"l_wavelet={metrics['loss_wavelet']:.4f} elapsed_s={elapsed:.1f}"
            )

    elapsed_total = time.perf_counter() - start_time
    final_ckpt_path = artifacts_dir / "final.pth"
    payload = {
        "experiment": "E032",
        "status": "fixed_budget_final",
        "model_kind": "residual_cno",
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
    print(f"\nE032 training finished in {elapsed_total:.1f}s. Saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
