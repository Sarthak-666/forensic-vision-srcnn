"""
Forensic-Vision SRCNN — Hyperparameter Search
===============================================
Runs two independent sweeps and logs every result to logs/hparam_search.csv.

  Sweep A — vary n1 (Layer-1 feature maps) in {32, 64, 128}, alpha fixed at 0.8
  Sweep B — vary alpha (HybridLoss weight)  in {0.6, 0.8, 1.0}, n1 fixed at 64

Duplicate experiments (n1=64, alpha=0.8 appears in both sweeps) are run only
once.  Per-experiment checkpoints are saved under checkpoints/hparam/ and are
never overwritten by subsequent runs (the file name encodes n1 and alpha).

Usage
-----
  # Full sweep (requires images in pretrain_dataset_path):
  python hparam_search.py

  # Full-epoch sweep on synthetic data (no real images needed):
  python hparam_search.py --synth

  # Quick 1-epoch synthetic smoke-test:
  python hparam_search.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

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
)

_LOG_FIELDS = ["n1", "alpha", "best_val_psnr", "best_val_ssim", "epochs"]


def _run_experiment(
    n1: int,
    alpha: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    step_size: int,
    gamma: float,
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one SRCNN variant. Returns (best_val_psnr, best_val_ssim)."""
    model     = SRCNN(channels=1, n1=n1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = HybridLoss(alpha=alpha)

    best_psnr = -float("inf")
    best_ssim = 0.0

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            torch.save(
                {
                    "n1": n1, "alpha": alpha, "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_psnr": val_psnr, "val_ssim": val_ssim,
                },
                ckpt_path,
            )

    return best_psnr, best_ssim


def main() -> None:
    parser = argparse.ArgumentParser(description="SRCNN hyperparameter search")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch run on synthetic data (no real images needed)",
    )
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Full-epoch run on synthetic data (no real images needed)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg  = cfg["data"]
    t_cfg  = cfg["training"]
    p_cfg  = cfg["paths"]
    hs_cfg = cfg.get("hparam_search", {})

    patch_size   = int(d_cfg["patch_size"])
    scale_factor = int(d_cfg["scale_factor"])
    sigma_range  = tuple(d_cfg["sigma_range"])
    batch_size   = int(t_cfg["batch_size"])
    lr           = float(t_cfg["learning_rate"])
    step_size    = int(t_cfg["lr_step_size"])
    gamma        = float(t_cfg["lr_gamma"])
    num_workers  = int(t_cfg["num_workers"])
    val_split    = float(d_cfg["val_split"])
    num_epochs   = int(t_cfg.get("pretrain_epochs", t_cfg["num_epochs"]))

    n1_values    = hs_cfg.get("n1_values",    [32, 64, 128])
    alpha_values = hs_cfg.get("alpha_values", [0.6, 0.8, 1.0])
    default_n1    = 64
    default_alpha = 0.8

    ckpt_dir = Path(p_cfg["checkpoints"]) / "hparam"
    log_dir  = Path(p_cfg["logs"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ------------------------------------------------------------
    if args.smoke:
        print("\n[smoke] Synthetic data - 1 epoch per experiment.\n")
        num_epochs = 1
        train_ds = _SyntheticDataset(32, patch_size, scale_factor)
        val_ds   = _SyntheticDataset(8,  patch_size, scale_factor)
    elif args.synth:
        print(f"\n[synth] Synthetic data - {num_epochs} epochs per experiment.\n")
        print("  NOTE: Synthetic random pixels cannot reach 28-32 dB PSNR.")
        print("  Add real images to data/pretrain/ for publication-quality results.\n")
        train_ds = _SyntheticDataset(128, patch_size, scale_factor)
        val_ds   = _SyntheticDataset(32,  patch_size, scale_factor)
    else:
        pretrain_dir = Path(d_cfg.get("pretrain_dataset_path", d_cfg["hr_dir"]))
        img_paths = sorted(
            p for p in pretrain_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        if not img_paths:
            sys.exit(
                f"No images in '{pretrain_dir}'. "
                "Run with --smoke or populate the directory."
            )
        n_val   = max(1, int(len(img_paths) * val_split))
        n_train = len(img_paths) - n_val
        full_ds = SRCNNDataset(img_paths, patch_size, scale_factor, sigma_range)
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Build experiment list (de-duplicated) ------------------------------
    # Sweep A: vary n1 with fixed alpha
    # Sweep B: vary alpha with fixed n1
    seen: set = set()
    experiments: List[Tuple[int, float]] = []
    for n1 in n1_values:
        key = (n1, default_alpha)
        if key not in seen:
            seen.add(key)
            experiments.append(key)
    for alpha in alpha_values:
        key = (default_n1, alpha)
        if key not in seen:
            seen.add(key)
            experiments.append(key)

    total = len(experiments)
    print(
        f"Experiments : {total}  ({len(n1_values)} n1 values + "
        f"{len(alpha_values)} alpha values, 1 overlap removed)"
    )
    print(f"Epochs each : {num_epochs}\n")

    # ---- CSV log (append so previous runs are preserved) --------------------
    log_path = log_dir / "hparam_search.csv"
    write_header = not log_path.exists()
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=_LOG_FIELDS)
    if write_header:
        log_writer.writeheader()

    # ---- Run ----------------------------------------------------------------
    print(
        f"{'#':>3}  {'n1':>4}  {'alpha':>5}  "
        f"{'Best PSNR':>9}  {'Best SSIM':>9}  {'Time':>7}"
    )
    print("-" * 48)

    for i, (n1, alpha) in enumerate(experiments, 1):
        ckpt_path = ckpt_dir / f"n1_{n1}_alpha_{alpha:.1f}.pth"
        t0 = time.time()

        best_psnr, best_ssim = _run_experiment(
            n1=n1, alpha=alpha,
            train_loader=train_loader, val_loader=val_loader,
            num_epochs=num_epochs, lr=lr,
            step_size=step_size, gamma=gamma,
            ckpt_path=ckpt_path, device=device,
        )
        elapsed = time.time() - t0

        print(
            f"{i:>3}  {n1:>4}  {alpha:>5.1f}  "
            f"{best_psnr:>9.3f}  {best_ssim:>9.4f}  {elapsed:>6.1f}s"
        )
        log_writer.writerow(dict(
            n1=n1, alpha=alpha,
            best_val_psnr=f"{best_psnr:.4f}",
            best_val_ssim=f"{best_ssim:.4f}",
            epochs=num_epochs,
        ))
        log_file.flush()

    log_file.close()
    print(f"\nResults saved  ->  {log_path}")
    print(f"Checkpoints    ->  {ckpt_dir}/")


if __name__ == "__main__":
    main()
