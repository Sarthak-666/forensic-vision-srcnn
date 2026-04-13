"""
Forensic-Vision SRCNN — Training Script
========================================
Trains the SRCNN model on Y-channel patches using a hybrid MSE+SSIM loss.

Usage
-----
  # Full training (requires images in data/raw):
  python train.py

  # Quick smoke-test on synthetic data (no real images needed):
  python train.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from src.loss import HybridLoss
from src.model import SRCNN
from src.train_utils import (
    _IMG_EXTS,
    _SyntheticDataset,
    SRCNNDataset,
    train_one_epoch,
    validate,
    init_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SRCNN")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single-epoch smoke-test on synthetic data (no real images needed)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    p_cfg = cfg["paths"]

    patch_size   = int(d_cfg["patch_size"])
    scale_factor = int(d_cfg["scale_factor"])
    sigma_range  = tuple(d_cfg["sigma_range"])
    batch_size   = int(t_cfg["batch_size"])
    num_epochs   = int(t_cfg["num_epochs"])
    alpha        = float(t_cfg["alpha"])
    lr           = float(t_cfg["learning_rate"])
    step_size    = int(t_cfg["lr_step_size"])
    gamma        = float(t_cfg["lr_gamma"])
    num_workers  = int(t_cfg["num_workers"])
    val_split    = float(d_cfg["val_split"])

    ckpt_dir  = Path(p_cfg["checkpoints"])
    log_dir   = Path(p_cfg["logs"])
    best_path = Path(p_cfg["best_model"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ------------------------------------------------------------
    if args.smoke:
        print("\n[smoke] Generating synthetic data - bypassing filesystem.\n")
        num_epochs = 1
        n_train, n_val = 32, 8
        train_ds = _SyntheticDataset(n_train, patch_size, scale_factor)
        val_ds   = _SyntheticDataset(n_val,   patch_size, scale_factor)
    else:
        hr_dir = Path(d_cfg["hr_dir"])
        image_paths = sorted(
            p for p in hr_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        if not image_paths:
            sys.exit(
                f"No images found in '{hr_dir}'. "
                "Add HR images or run with --smoke for a synthetic test."
            )
        print(f"Found {len(image_paths)} images in '{hr_dir}'.")

        n_val   = max(1, int(len(image_paths) * val_split))
        n_train = len(image_paths) - n_val
        full_ds = SRCNNDataset(image_paths, patch_size, scale_factor, sigma_range)
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Split: {n_train} train / {n_val} val patches per epoch.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model / optimiser / scheduler / loss --------------------------------
    model     = SRCNN(channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = HybridLoss(alpha=alpha)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SRCNN parameters: {total_params:,}\n")

    # ---- Logging setup -------------------------------------------------------
    log_path   = log_dir / "train_log.csv"
    log_writer, log_file = init_log(log_path)
    print(f"Logging to: {log_path}")
    print(f"Best checkpoint: {best_path}\n")

    # ---- Training loop -------------------------------------------------------
    best_psnr = -float("inf")
    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'PSNR':>7}  {'SSIM':>6}  {'LR':>9}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss                    = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                        diag_first_batch=(epoch == 1))
        val_loss, val_psnr, val_ssim  = validate(model, val_loader, criterion, device)
        current_lr                    = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {train_loss:>10.5f}  {val_loss:>9.5f}  "
            f"{val_psnr:>7.3f}  {val_ssim:>6.4f}  {current_lr:>9.2e}  "
            f"({elapsed:.1f}s)"
        )

        log_writer.writerow(
            dict(
                epoch=epoch,
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
                val_psnr=f"{val_psnr:.4f}",
                val_ssim=f"{val_ssim:.4f}",
                lr=f"{current_lr:.2e}",
            )
        )
        log_file.flush()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_loss": val_loss,
                },
                best_path,
            )

    log_file.close()
    print(f"\nTraining complete. Best PSNR: {best_psnr:.3f} dB  ->  {best_path}")


if __name__ == "__main__":
    main()
