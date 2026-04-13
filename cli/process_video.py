"""
Forensic-Vision SRCNN — CLI Video Processing Tool
===================================================
Enhances surveillance video frame-by-frame using the trained SRCNN model.

Pipeline per frame
------------------
  BGR frame  →  YCbCr  →  Y-channel
     → [optional Gaussian denoise]
     → bicubic downsample × scale  →  bicubic upsample (SRCNN input)
     → SRCNN inference
     → merge enhanced Y with original Cb/Cr
     → BGR output frame

Usage
-----
  python cli/process_video.py --input footage.mp4 --output enhanced.mp4 --scale 3
  python cli/process_video.py --input footage.mp4 --output enhanced.mp4 --scale 3 --denoise
  python cli/process_video.py --input footage.mp4 --output enhanced.mp4 \\
      --checkpoint checkpoints/finetuned_scface.pth --scale 3 --denoise
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Allow running from project root or from cli/ directly
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.degradation import DegradationPipeline
from src.model import SRCNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a uint8 BGR frame into float32 Y (in [0,1]) and uint8 Cb/Cr."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_ycbcr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y   = frame_ycbcr[:, :, 0].astype(np.float32) / 255.0
    cb  = frame_ycbcr[:, :, 1]
    cr  = frame_ycbcr[:, :, 2]
    return y, cb, cr


def _ycbcr_to_bgr(y_arr: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Merge enhanced Y (float32 [0,1]) with Cb/Cr (uint8) back to BGR uint8."""
    y_uint8 = (y_arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    ycbcr = np.stack([y_uint8, cb, cr], axis=2)
    rgb   = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _gaussian_denoise(y: np.ndarray, ksize: int = 3, sigma: float = 0.5) -> np.ndarray:
    """Lightweight Gaussian pre-filter (replaces DnCNN when not available)."""
    return cv2.GaussianBlur(y, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


@torch.no_grad()
def _srcnn_enhance(
    y: np.ndarray,
    model: SRCNN,
    pipeline: DegradationPipeline,
    device: torch.device,
) -> np.ndarray:
    """Run the full SRCNN pipeline on a float32 Y-channel array."""
    h, w = y.shape
    tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    # Bicubic downsample → noise → upsample (same as training pipeline)
    lr_up, _ = pipeline(tensor.squeeze(0))                   # (1,H,W)
    inp = lr_up.unsqueeze(0).to(device)                      # (1,1,H,W)

    out = model(inp).cpu().squeeze().clamp(0.0, 1.0).numpy()  # (H,W)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance surveillance video with SRCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True, help="Input video path")
    parser.add_argument("--output",     required=True, help="Output video path")
    parser.add_argument("--config",     default=str(_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(_ROOT / "checkpoints" / "finetuned_scface.pth"))
    parser.add_argument("--scale",      type=int, default=3,
                        help="Bicubic downscale factor used during training")
    parser.add_argument("--denoise",    action="store_true",
                        help="Apply Gaussian pre-denoising before SRCNN")
    parser.add_argument("--denoise-sigma", type=float, default=0.5,
                        help="Gaussian sigma for pre-denoising (when --denoise is set)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sigma_range = tuple(cfg["data"]["sigma_range"])

    # ── Model ────────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=device)
    n1    = ckpt.get("n1", 64)
    model = SRCNN(channels=1, n1=n1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model : SRCNN n1={n1}  from {ckpt_path.name}")
    print(f"Device: {device}  |  scale: {args.scale}  |  denoise: {args.denoise}")

    pipeline = DegradationPipeline(
        scale_factor=args.scale,
        sigma_range=sigma_range,
        seed=0,
    )

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.input}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")

    out_path   = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer     = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"Cannot create output video: {args.output}")

    print(f"Input : {args.input}  ({width}×{height} @ {fps:.1f} fps, {total} frames)")
    print(f"Output: {args.output}\n")

    # ── Process frames ───────────────────────────────────────────────────────
    with tqdm(total=total or None, unit="frame", desc="Processing") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            y, cb, cr = _bgr_to_ycbcr(frame)

            if args.denoise:
                y = _gaussian_denoise(y, sigma=args.denoise_sigma)

            y_enhanced = _srcnn_enhance(y, model, pipeline, device)
            out_frame  = _ycbcr_to_bgr(y_enhanced, cb, cr)

            writer.write(out_frame)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"\nDone. Enhanced video saved to: {args.output}")


if __name__ == "__main__":
    main()
