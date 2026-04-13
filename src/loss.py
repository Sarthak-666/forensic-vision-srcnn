"""
Hybrid loss: L = alpha * L_MSE + (1 - alpha) * L_SSIM
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Differentiable SSIM for (B, C, H, W) tensors."""
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    kernel = _gaussian_kernel(window_size, sigma, pred.device)
    # Expand to (out_ch=1, in_ch=1, H, W) — applied per-channel
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    channels = pred.shape[1]
    kernel = kernel.expand(channels, 1, window_size, window_size)

    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=channels)

    mu_x = conv(pred)
    mu_y = conv(target)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = conv(pred * pred) - mu_x2
    sigma_y2 = conv(target * target) - mu_y2
    sigma_xy = conv(pred * target) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (numerator / denominator).mean()


class HybridLoss(nn.Module):
    """alpha * MSE + (1 - alpha) * (1 - SSIM).

    Args:
        alpha: weight of the MSE term (0 = SSIM-only, 1 = MSE-only).
    """

    def __init__(self, alpha: float = 0.8) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss  = F.mse_loss(pred, target)
        # Clamp pred to [0, 1] before SSIM: the formula is calibrated to
        # data_range=1.0 and produces corrupted gradients when pred contains
        # values outside that range (SRCNN has no output activation).
        ssim_loss = 1.0 - ssim(pred.clamp(0.0, 1.0), target)
        return self.alpha * mse_loss + (1.0 - self.alpha) * ssim_loss
