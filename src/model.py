"""
SRCNN model — 9-5-5 architecture (Dong et al. 2014).

Three convolutional layers:
  1. Patch Extraction  : 9×9, 64 maps, ReLU
  2. Non-linear Mapping: 5×5, 32 maps, ReLU
  3. Reconstruction   : 5×5, c maps, linear
"""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network (9-5-5 variant).

    Operates on the Y (luminance) channel of YCbCr images.
    Input is a bicubic-upscaled LR patch; output is the HR estimate
    at the same spatial resolution.

    Args:
        channels: number of output channels (1 for Y-only mode).
        n1: feature maps in Layer 1 (patch extraction). Default 64.
        n2: feature maps in Layer 2 (non-linear mapping). Default 32.
    """

    def __init__(self, channels: int = 1, n1: int = 64, n2: int = 32) -> None:
        super().__init__()

        # Layer 1 — Patch Extraction & Representation
        self.patch_extraction = nn.Conv2d(
            in_channels=channels,
            out_channels=n1,
            kernel_size=9,
            padding=4,   # 'same' spatial size
            padding_mode="zeros",
        )

        # Layer 2 — Non-linear Mapping
        self.nonlinear_mapping = nn.Conv2d(
            in_channels=n1,
            out_channels=n2,
            kernel_size=5,
            padding=2,
            padding_mode="zeros",
        )

        # Layer 3 — Reconstruction (linear activation, no ReLU)
        self.reconstruction = nn.Conv2d(
            in_channels=n2,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            padding_mode="zeros",
        )

        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with Kaiming normal (He init), biases to zero."""
        for layer in (self.patch_extraction, self.nonlinear_mapping, self.reconstruction):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: bicubic-upscaled LR tensor, shape (B, C, H, W), values in [0, 1].
        Returns:
            HR estimate, same shape as x.
        """
        x = self.relu(self.patch_extraction(x))
        x = self.relu(self.nonlinear_mapping(x))
        x = self.reconstruction(x)
        return x
