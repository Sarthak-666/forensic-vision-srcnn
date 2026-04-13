"""
Shared training utilities used by train.py, pretrain.py, finetune.py,
and hparam_search.py.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.degradation import DegradationPipeline
from src.loss import HybridLoss, ssim
from src.model import SRCNN

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

_DEFAULT_LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SRCNNDataset(Dataset):
    """Loads HR images, extracts random Y-channel patches, applies
    degradation on-the-fly to produce (LR-up, HR) pairs.

    Args:
        image_paths: list of paths to HR images.
        patch_size:  spatial size of each patch (H = W = patch_size).
        scale_factor: bicubic downscale factor.
        sigma_range: (min, max) noise std for the degradation pipeline.
        augment:     if True, apply random horizontal/vertical flips.
    """

    def __init__(
        self,
        image_paths: List[Path],
        patch_size: int,
        scale_factor: int,
        sigma_range: Tuple[float, float],
        augment: bool = True,
    ) -> None:
        self.paths = image_paths
        self.patch_size = patch_size
        self.augment = augment
        self.pipeline = DegradationPipeline(
            scale_factor=scale_factor, sigma_range=sigma_range
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("YCbCr")
        y, _, _ = img.split()

        w, h = y.size
        if w < self.patch_size or h < self.patch_size:
            y = y.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC,
            )
            w, h = y.size

        x0 = random.randint(0, w - self.patch_size)
        y0 = random.randint(0, h - self.patch_size)
        patch = y.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))

        if self.augment:
            if random.random() > 0.5:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                patch = patch.transpose(Image.FLIP_TOP_BOTTOM)

        arr = torch.from_numpy(np.array(patch, dtype=np.float32)[None] / 255.0)
        return self.pipeline(arr)


class _SyntheticDataset(Dataset):
    """Generates random (LR-up, HR) patch pairs for smoke-testing."""

    def __init__(self, n: int, patch_size: int, scale_factor: int) -> None:
        self.n = n
        self.patch_size = patch_size
        self.pipeline = DegradationPipeline(scale_factor=scale_factor, seed=0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = torch.rand(1, self.patch_size, self.patch_size)
        return self.pipeline(hr)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (data range = 1.0)."""
    mse = F.mse_loss(pred.clamp(0.0, 1.0), target).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_val(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ssim(pred.clamp(0.0, 1.0), target).item()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SRCNN,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: HybridLoss,
    device: torch.device,
    diag_first_batch: bool = False,
) -> float:
    """Run one training epoch.

    Args:
        diag_first_batch: if True, print normalization and loss diagnostics
                          for the very first batch (useful for detecting NaN
                          or unnormalized inputs).
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (lr_up, hr) in enumerate(tqdm(loader, desc="  train", leave=False)):
        lr_up, hr = lr_up.to(device), hr.to(device)
        pred = model(lr_up)
        loss = criterion(pred, hr)

        if diag_first_batch and batch_idx == 0:
            mse_val = F.mse_loss(pred.detach(), hr).item()
            print(
                f"    [diag batch-0] "
                f"lr_up [{lr_up.min():.4f}, {lr_up.max():.4f}]  "
                f"hr [{hr.min():.4f}, {hr.max():.4f}]  "
                f"loss={loss.item():.6f}  MSE={mse_val:.6f}  "
                f"NaN={torch.isnan(loss).item()}  >1000={loss.item() > 1000}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate(
    model: SRCNN,
    loader: torch.utils.data.DataLoader,
    criterion: HybridLoss,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    with torch.no_grad():
        for lr_up, hr in tqdm(loader, desc="  val  ", leave=False):
            lr_up, hr = lr_up.to(device), hr.to(device)
            pred = model(lr_up)
            total_loss += criterion(pred, hr).item()
            total_psnr += compute_psnr(pred, hr)
            total_ssim += compute_ssim_val(pred, hr)
    n = max(len(loader), 1)
    return total_loss / n, total_psnr / n, total_ssim / n


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def init_log(
    log_path: Path,
    fieldnames: List[str] | None = None,
    mode: str = "w",
) -> Tuple[csv.DictWriter, object]:
    """Open a CSV log file and return (DictWriter, file_handle).

    Args:
        log_path:   path to the CSV file.
        fieldnames: column names; defaults to epoch/loss/psnr/ssim/lr.
        mode:       ``"w"`` (default) to create a fresh log each run,
                    ``"a"`` to append to an existing file.
    """
    if fieldnames is None:
        fieldnames = _DEFAULT_LOG_FIELDS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (mode == "w") or not log_path.exists()
    f = open(log_path, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    f.flush()
    return writer, f
