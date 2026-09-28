#!/usr/bin/env python3
"""Train the E030 Spatially-Adaptive Heteroskedastic Uncertainty Network on Fold A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))
sys.path.insert(0, str(REPO / "scripts"))

from realpde_t1.adapter import load_residual_cno
from realpde_t1.spatial_uncertainty import SmoothSPSLoss, SpatialUncertaintyNet
from realpde_t1.training import RealWindowDataset, WindowGeometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "e030_spatial_uncertainty.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e030_spatial_uncertainty",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts_dir = args.artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 20260815)))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Load frozen point champion model
    point_ckpt_path = REPO / config["point_checkpoint"]
    point_ckpt = torch.load(point_ckpt_path, map_location="cpu", weights_only=False)
    normalizer = point_ckpt["normalizer"]

    point_model = load_residual_cno(point_ckpt, device=device)
    point_model.eval()
    for param in point_model.parameters():
        param.requires_grad = False

    # 2. Build SpatialUncertaintyNet
    unc_cfg = config["model"]
    unc_net = SpatialUncertaintyNet(
        in_channels=unc_cfg["in_channels"],
        out_channels=unc_cfg["out_channels"],
        hidden_channels=unc_cfg["hidden_channels"],
        n_blocks=unc_cfg["n_blocks"],
        kernel_size=unc_cfg["kernel_size"],
    ).to(device)

    trainable_count = sum(p.numel() for p in unc_net.parameters() if p.requires_grad)
    print(f"Instantiated SpatialUncertaintyNet with {trainable_count} trainable parameters.")

    # 3. Load dataset
    split = json.loads((REPO / config["split_config"]).read_text(encoding="utf-8"))
    data_dir = REPO.parent / "RealPDE-Competition-Data" / split.get("dataset", "train_real")

    geom = WindowGeometry(
        input_frames=int(config["data"]["input_frames"]),
        output_frames=int(config["data"]["output_frames"]),
        temporal_stride=int(config["data"]["window_stride"]),
        spatial_stride=int(config["data"]["spatial_stride"]),
    )
    dataset = RealWindowDataset(
        data_root=data_dir,
        files=split["training_files"],
        geometry=geom,
        preload=bool(config["data"].get("preload_trajectories", True)),
    )

    batch_size = int(config["training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", True)),
    )

    # 4. Optimizer and Loss
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        unc_net.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    num_updates = int(train_cfg["num_updates"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_updates,
        eta_min=float(train_cfg.get("eta_min", 0.0)),
    )

    loss_cfg = train_cfg["loss"]
    criterion = SmoothSPSLoss(
        sigma_global=float(loss_cfg["sigma_global"]),
        tau=float(loss_cfg["tau"]),
        lambda_sps=float(loss_cfg["lambda_sps"]),
        beta_floor=float(loss_cfg["beta_floor"]),
    )

    # Convert normalizer to tensors
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tgt = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tgt = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    # 5. Training Loop
    print(f"Starting E030 training on {device}: {num_updates} updates...")
    start_time = time.perf_counter()
    data_iter = iter(loader)
    losses = []

    unc_net.train()
    for update in range(1, num_updates + 1):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        # Forward frozen point model
        norm_in = (inputs - mean_in) / std_in
        with torch.no_grad():
            norm_point_pred = point_model(norm_in)
            pred_phys = norm_point_pred * std_tgt + mean_tgt
            pred_phys[..., 2] = 0.0

        # Forward uncertainty net
        optimizer.zero_grad()
        norm_scale = unc_net(norm_in, norm_point_pred)
        # Convert scale to physical units using target standard deviation
        scale_phys = norm_scale * std_tgt
        scale_phys[..., 2] = 0.0

        loss, metrics = criterion(
            pred_phys[..., :2],
            targets[..., :2],
            scale_phys[..., :2],
        )

        loss.backward()
        if float(train_cfg.get("gradient_clip_norm", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(
                unc_net.parameters(), float(train_cfg["gradient_clip_norm"])
            )
        optimizer.step()
        scheduler.step()

        val = float(loss.detach().cpu())
        losses.append(val)

        if update == 1 or update % 25 == 0 or update == num_updates:
            elapsed = time.perf_counter() - start_time
            mean25 = float(np.mean(losses[-25:]))
            print(
                f"update={update:3d}/{num_updates} loss={val:.6f} mean25={mean25:.6f} "
                f"nll={metrics['nll']:.4f} smooth_sps={metrics['smooth_sps']:.2f}% elapsed_s={elapsed:.1f}"
            )

    elapsed_total = time.perf_counter() - start_time

    # Save final checkpoint
    final_ckpt_path = artifacts_dir / "final.pth"
    payload = {
        "experiment": "E030",
        "status": "fixed_budget_final",
        "model_kind": "spatial_uncertainty_net",
        "model_config": unc_cfg,
        "point_checkpoint": str(point_ckpt_path),
        "normalizer": normalizer,
        "model_state_dict": unc_net.state_dict(),
        "trainable_parameters": trainable_count,
        "training_seconds": elapsed_total,
    }
    torch.save(payload, final_ckpt_path)
    print(f"\nE030 training finished in {elapsed_total:.1f}s. Checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
