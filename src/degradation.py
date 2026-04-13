"""
Data degradation pipeline for CCTV / forensic surveillance simulation.

Pipeline:
    HR image (Y-channel, float [0,1])
    → bicubic downsampling  (÷ scale_factor)
    → Gaussian noise injection
    → bicubic upsampling back to HR size   ← this is the SRCNN input
    → returned alongside the clean HR      ← this is the SRCNN target

All operations are deterministic given a fixed random seed and work on
both individual PIL Images and batched torch Tensors.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image (L or RGB) to a float32 tensor in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]          # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)        # (C, H, W)
    return torch.from_numpy(arr)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, H, W) or (3, H, W) float32 tensor back to PIL."""
    arr = tensor.clamp(0, 1).numpy()
    if arr.shape[0] == 1:
        arr = (arr[0] * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Core degradation steps
# ---------------------------------------------------------------------------

def bicubic_downsample(
    img: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """Bicubic downsampling by integer scale_factor.

    Args:
        img: float tensor (C, H, W) or (B, C, H, W), values in [0, 1].
        scale_factor: downscale factor (e.g. 2, 3, 4).

    Returns:
        Downsampled tensor of shape (C, H//scale, W//scale) or (B, ...).
    """
    squeeze = img.ndim == 3
    if squeeze:
        img = img.unsqueeze(0)

    lr = F.interpolate(
        img,
        scale_factor=1.0 / scale_factor,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    return lr.squeeze(0) if squeeze else lr


def bicubic_upsample(
    img: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """Bicubic upsampling by integer scale_factor.

    Args:
        img: float tensor (C, H, W) or (B, C, H, W).
        scale_factor: upscale factor matching the downsample step.

    Returns:
        Upsampled tensor restored to the original spatial size.
    """
    squeeze = img.ndim == 3
    if squeeze:
        img = img.unsqueeze(0)

    hr = F.interpolate(
        img,
        scale_factor=float(scale_factor),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    return hr.squeeze(0) if squeeze else hr


def add_gaussian_noise(
    img: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise to simulate CCTV sensor noise.

    Args:
        img: float tensor, values in [0, 1].
        sigma: noise standard deviation (e.g. 0.01 – 0.05).
        generator: optional torch.Generator for reproducibility.

    Returns:
        Noisy tensor, clamped to [0, 1].
    """
    noise = torch.zeros_like(img).normal_(mean=0.0, std=sigma, generator=generator)
    return (img + noise).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

class DegradationPipeline:
    """Simulate low-quality CCTV footage from a high-resolution image.

    Usage::

        pipeline = DegradationPipeline(scale_factor=3, sigma_range=(0.01, 0.04))
        lr_up, hr = pipeline(hr_tensor)   # lr_up → SRCNN input; hr → target

    Args:
        scale_factor: spatial downscaling factor.
        sigma_range: (min, max) range for Gaussian noise std; drawn uniformly
                     at each call to mimic varying sensor conditions.
        seed: optional fixed seed for reproducibility during evaluation.
    """

    def __init__(
        self,
        scale_factor: int = 3,
        sigma_range: Tuple[float, float] = (0.01, 0.04),
        seed: int | None = None,
    ) -> None:
        if scale_factor < 2:
            raise ValueError("scale_factor must be >= 2")
        if sigma_range[0] < 0 or sigma_range[0] > sigma_range[1]:
            raise ValueError("sigma_range must satisfy 0 <= min <= max")

        self.scale_factor = scale_factor
        self.sigma_range = sigma_range
        self._generator: torch.Generator | None = None

        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def __call__(
        self,
        hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the full degradation pipeline.

        Args:
            hr: clean HR tensor, shape (C, H, W) or (B, C, H, W),
                float32 in [0, 1].  H and W must be divisible by scale_factor.

        Returns:
            (lr_bicubic_up, hr_clean) — both at the original HR spatial size.
            lr_bicubic_up is the noisy bicubic-upsampled input for SRCNN.
            hr_clean is the unmodified target.
        """
        # 1. Downscale (bicubic)
        lr = bicubic_downsample(hr, self.scale_factor)

        # 2. Inject noise to simulate CCTV sensor noise
        sigma = random.uniform(*self.sigma_range)
        lr_noisy = add_gaussian_noise(lr, sigma=sigma, generator=self._generator)

        # 3. Upscale back to HR dimensions (bicubic baseline — SRCNN input)
        #    Clamp both tensors: bicubic can overshoot [0,1] on small patches.
        lr_up = bicubic_upsample(lr_noisy, self.scale_factor).clamp(0.0, 1.0)

        return lr_up, hr.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Convenience: work directly with PIL Images
    # ------------------------------------------------------------------

    def from_pil(
        self,
        image: Image.Image,
        use_y_channel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply degradation to a PIL Image.

        Args:
            image: input HR image (RGB or grayscale).
            use_y_channel: if True, convert to YCbCr and extract Y channel only.

        Returns:
            (lr_up, hr) tensors, shape (1, H, W) when use_y_channel=True.
        """
        if use_y_channel:
            if image.mode != "YCbCr":
                image = image.convert("YCbCr")
            y, _, _ = image.split()
            tensor = _to_tensor(y)    # (1, H, W)
        else:
            tensor = _to_tensor(image.convert("RGB"))   # (3, H, W)

        return self(tensor)
