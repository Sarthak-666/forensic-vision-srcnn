"""
Forensic-Vision SRCNN — Fine-Tuning Script
===========================================
Loads ``checkpoints/pretrained.pth`` as starting weights and fine-tunes on
the SCFace dataset with a lower learning rate (default 1e-5).

Also computes the bicubic-upsampling baseline PSNR on the validation set so
that every epoch's improvement over bicubic is clearly visible in the log.

Usage
-----
  # Full fine-tuning (requires SCFace images in data/scface/):
  python finetune.py

  # Full-epoch run on synthetic data (no real images needed):
  python finetune.py --synth

  # Quick 1-epoch smoke-test on synthetic data:
  python finetune.py --smoke
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
from torch.utils.data import DataLoader

from src.loss import HybridLoss
from src.model import SRCNN
from src.train_utils import (
    _SyntheticDataset,
    compute_psnr,
    train_one_epoch,
    validate,
    init_log,
)


def _bicubic_baseline_psnr(loader: DataLoader, device: torch.device) -> float:
    """PSNR of bicubic-upsampled input vs HR target (no SRCNN)."""
    total = 0.0
    n = 0
    with torch.no_grad():
        for lr_up, hr in loader:
            lr_up, hr = lr_up.to(device), hr.to(device)
            total += compute_psnr(lr_up, hr)
            n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SRCNN on SCFace")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Single-epoch test on synthetic data (no real images needed)",
    )
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Full-epoch run on synthetic data (no real images needed)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    p_cfg = cfg["paths"]

    patch_size   = int(d_cfg.get("scface_patch_size", 33))
    stride       = int(d_cfg.get("stride", 14))
    scale_factor = int(d_cfg["scale_factor"])
    sigma_range  = tuple(d_cfg["sigma_range"])
    batch_size   = int(t_cfg["batch_size"])
    alpha        = float(t_cfg["alpha"])
    finetune_lr  = float(t_cfg.get("finetune_lr", 1e-5))
    step_size    = int(t_cfg["lr_step_size"])
    gamma        = float(t_cfg["lr_gamma"])
    num_workers  = int(t_cfg["num_workers"])
    num_epochs   = int(t_cfg.get("finetune_epochs", 50))

    ckpt_dir        = Path(p_cfg["checkpoints"])
    log_dir         = Path(p_cfg["logs"])
    pretrained_path = ckpt_dir / "pretrained.pth"
    finetuned_path  = ckpt_dir / "finetuned_scface.pth"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ------------------------------------------------------------
    if args.smoke:
        print("\n[smoke] Synthetic data - skipping filesystem.\n")
        num_epochs = 1
        train_ds = _SyntheticDataset(32, patch_size, scale_factor)
        val_ds   = _SyntheticDataset(8,  patch_size, scale_factor)
    elif args.synth:
        print(f"\n[synth] Synthetic data - {num_epochs} epochs.\n")
        train_ds = _SyntheticDataset(128, patch_size, scale_factor)
        val_ds   = _SyntheticDataset(32,  patch_size, scale_factor)
    else:
        from data.scface_dataset import SCFaceDataset
        scface_dir = Path(d_cfg["finetune_dataset_path"])
        if not scface_dir.exists():
            sys.exit(
                f"SCFace directory '{scface_dir}' not found. "
                "Populate data/scface/ with SCFace images, or run with --smoke."
            )
        train_ds = SCFaceDataset(
            scface_dir, patch_size, stride, scale_factor, sigma_range, split="train"
        )
        val_ds = SCFaceDataset(
            scface_dir, patch_size, stride, scale_factor, sigma_range, split="val"
        )
        print(f"SCFace patches: {len(train_ds):,} train / {len(val_ds):,} val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model — load pretrained weights ------------------------------------
    model = SRCNN(channels=1).to(device)
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print(f"[warn] {pretrained_path} not found — fine-tuning from random init.")

    optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = HybridLoss(alpha=alpha)

    # ---- Bicubic baseline ---------------------------------------------------
    print("\nComputing bicubic baseline PSNR on validation set...")
    baseline_psnr = _bicubic_baseline_psnr(val_loader, device)
    print(f"Bicubic baseline PSNR : {baseline_psnr:.3f} dB\n")

    # ---- Logging ------------------------------------------------------------
    log_writer, log_file = init_log(log_dir / "finetune_log.csv")

    # ---- Training loop ------------------------------------------------------
    best_psnr = -float("inf")
    header = (
        f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
        f"{'PSNR':>7}  {'SSIM':>6}  {'vs Bicubic':>10}  {'LR':>9}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss                   = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                       diag_first_batch=(epoch == 1))
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        current_lr                   = scheduler.get_last_lr()[0]
        scheduler.step()

        delta = val_psnr - baseline_psnr
        print(
            f"{epoch:>6}  {train_loss:>10.5f}  {val_loss:>9.5f}  "
            f"{val_psnr:>7.3f}  {val_ssim:>6.4f}  {delta:>+10.3f}  "
            f"{current_lr:>9.2e}  ({time.time() - t0:.1f}s)"
        )
        log_writer.writerow(dict(
            epoch=epoch, train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}", val_psnr=f"{val_psnr:.4f}",
            val_ssim=f"{val_ssim:.4f}", lr=f"{current_lr:.2e}",
        ))
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
                    "bicubic_baseline_psnr": baseline_psnr,
                },
                finetuned_path,
            )

    log_file.close()
    improvement = best_psnr - baseline_psnr
    print(
        f"\nFine-tuning complete."
        f"\n  Best PSNR  : {best_psnr:.3f} dB"
        f"\n  Bicubic    : {baseline_psnr:.3f} dB"
        f"\n  Improvement: {improvement:+.3f} dB"
        f"\n  Checkpoint : {finetuned_path}"
    )


if __name__ == "__main__":
    main()
