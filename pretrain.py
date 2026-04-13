"""
Forensic-Vision SRCNN — Pre-training Script
=============================================
Pre-trains SRCNN on a general-purpose dataset (e.g. BSDS500 or an ImageNet
subset) before fine-tuning on SCFace.  Uses the same training loop as
train.py but reads from ``pretrain_dataset_path`` and saves to
``checkpoints/pretrained.pth``.

Usage
-----
  # Full pre-training (requires images in data/pretrain/):
  python pretrain.py

  # Full-epoch run on synthetic data (no real images needed):
  python pretrain.py --synth

  # Quick 1-epoch smoke-test on synthetic data:
  python pretrain.py --smoke
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
    parser = argparse.ArgumentParser(description="Pre-train SRCNN on a general dataset")
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

    patch_size   = int(d_cfg["patch_size"])
    scale_factor = int(d_cfg["scale_factor"])
    sigma_range  = tuple(d_cfg["sigma_range"])
    batch_size   = int(t_cfg["batch_size"])
    alpha        = float(t_cfg["alpha"])
    lr           = float(t_cfg["learning_rate"])
    step_size    = int(t_cfg["lr_step_size"])
    gamma        = float(t_cfg["lr_gamma"])
    num_workers  = int(t_cfg["num_workers"])
    val_split    = float(d_cfg["val_split"])
    num_epochs   = int(t_cfg.get("pretrain_epochs", t_cfg["num_epochs"]))

    ckpt_dir        = Path(p_cfg["checkpoints"])
    log_dir         = Path(p_cfg["logs"])
    pretrained_path = ckpt_dir / "pretrained.pth"
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
        pretrain_dir = Path(d_cfg["pretrain_dataset_path"])
        img_paths = sorted(
            p for p in pretrain_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        if not img_paths:
            sys.exit(
                f"No images found in '{pretrain_dir}'. "
                "Populate data/pretrain/ with BSDS500 or ImageNet images, "
                "or run with --smoke."
            )
        print(f"Found {len(img_paths)} pre-training images in '{pretrain_dir}'.")

        n_val   = max(1, int(len(img_paths) * val_split))
        n_train = len(img_paths) - n_val
        full_ds = SRCNNDataset(img_paths, patch_size, scale_factor, sigma_range)
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Split: {n_train} train / {n_val} val images.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model / optimiser / loss -------------------------------------------
    model     = SRCNN(channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = HybridLoss(alpha=alpha)

    print(f"SRCNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Pre-training for {num_epochs} epoch(s)  ->  {pretrained_path}\n")

    # ---- Logging ------------------------------------------------------------
    log_writer, log_file = init_log(log_dir / "pretrain_log.csv")

    # ---- Training loop ------------------------------------------------------
    best_psnr = -float("inf")
    header = (
        f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
        f"{'PSNR':>7}  {'SSIM':>6}  {'LR':>9}"
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

        print(
            f"{epoch:>6}  {train_loss:>10.5f}  {val_loss:>9.5f}  "
            f"{val_psnr:>7.3f}  {val_ssim:>6.4f}  {current_lr:>9.2e}  "
            f"({time.time() - t0:.1f}s)"
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
                },
                pretrained_path,
            )

    log_file.close()
    print(f"\nPre-training complete. Best PSNR: {best_psnr:.3f} dB  ->  {pretrained_path}")


if __name__ == "__main__":
    main()
