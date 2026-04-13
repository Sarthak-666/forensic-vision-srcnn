"""
Quick smoke-test: verifies model forward pass and degradation pipeline.
Run from the project root:  python scripts/verify_pipeline.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
from src.model import SRCNN
from src.degradation import DegradationPipeline
from src.loss import HybridLoss

SCALE   = 3
BATCH   = 2
H, W    = 96, 96   # must be divisible by SCALE

print("=== SRCNN smoke-test ===")

# ---- Degradation pipeline ------------------------------------------------
pipeline = DegradationPipeline(scale_factor=SCALE, sigma_range=(0.01, 0.04), seed=42)
hr_batch = torch.rand(BATCH, 1, H, W)           # simulate Y-channel HR patches
lr_up, hr_clean = pipeline(hr_batch)

assert lr_up.shape  == hr_batch.shape, f"Shape mismatch: {lr_up.shape}"
assert hr_clean.shape == hr_batch.shape
assert lr_up.min() >= 0.0 and lr_up.max() <= 1.0
print(f"  Degradation pipeline OK  |  input {tuple(hr_batch.shape)} -> LR-up {tuple(lr_up.shape)}")

# ---- Model forward pass --------------------------------------------------
model = SRCNN(channels=1)
out   = model(lr_up)

assert out.shape == lr_up.shape, f"Output shape mismatch: {out.shape}"
print(f"  SRCNN forward pass OK    |  output {tuple(out.shape)}")

# ---- Loss ----------------------------------------------------------------
criterion = HybridLoss(alpha=0.5)
loss = criterion(out, hr_clean)
loss.backward()

print(f"  HybridLoss OK            |  loss = {loss.item():.6f}")
print("\nAll checks passed.")
