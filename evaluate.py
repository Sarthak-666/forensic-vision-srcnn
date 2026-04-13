"""
Forensic-Vision SRCNN — Final Evaluation
==========================================
Loads a trained checkpoint, runs inference on a validation image set,
and writes per-image PSNR / SSIM results to logs/final_evaluation.csv.

Usage
-----
  python evaluate.py                            # uses defaults from config.yaml
  python evaluate.py --checkpoint checkpoints/finetuned_scface.pth
  python evaluate.py --image-dir data/scface --split val
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from typing import List

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from src.degradation import DegradationPipeline, _to_tensor
from src.loss import ssim
from src.model import SRCNN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ssim(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)).item()


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(image_dir: pathlib.Path) -> List[pathlib.Path]:
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not paths:
        sys.exit(f"No images found in '{image_dir}'.")
    return paths


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_image(
    img_path: pathlib.Path,
    model: SRCNN,
    pipeline: DegradationPipeline,
    device: torch.device,
) -> dict:
    """Return a dict with bicubic/srcnn PSNR and SSIM for one image."""
    img = Image.open(img_path).convert("YCbCr")
    y, _, _ = img.split()

    # Crop to nearest multiple of scale_factor so bicubic round-trip is lossless
    sf = pipeline.scale_factor
    w, h = y.size
    w_crop, h_crop = (w // sf) * sf, (h // sf) * sf
    if w_crop != w or h_crop != h:
        y = y.crop((0, 0, w_crop, h_crop))

    hr = _to_tensor(y)          # (1, H, W) in [0, 1]

    # Degrade: bicubic down → noise → bicubic up (SRCNN input)
    lr_up, hr_clean = pipeline(hr)

    # Add batch dimension for model
    lr_up_b = lr_up.unsqueeze(0).to(device)
    pred    = model(lr_up_b).cpu().squeeze(0)   # (1, H, W)

    # Per-image metrics (using unsqueezed batch tensors)
    bic_psnr  = _psnr(lr_up,  hr_clean)
    src_psnr  = _psnr(pred,   hr_clean)
    bic_ssim  = _ssim(lr_up.unsqueeze(0),  hr_clean.unsqueeze(0))
    src_ssim  = _ssim(pred.unsqueeze(0),   hr_clean.unsqueeze(0))

    return {
        "image_id":    img_path.stem,
        "bicubic_psnr": round(bic_psnr, 4),
        "srcnn_psnr":   round(src_psnr, 4),
        "bicubic_ssim": round(bic_ssim, 4),
        "srcnn_ssim":   round(src_ssim, 4),
        "psnr_gain":    round(src_psnr - bic_psnr, 4),
        "ssim_gain":    round(src_ssim - bic_ssim, 4),
        # also carry tensors for downstream use
        "_lr_up":  lr_up,
        "_pred":   pred,
        "_hr":     hr_clean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SRCNN vs bicubic baseline")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/finetuned_scface.pth",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image-dir",  default=None,
                        help="Directory of evaluation images (overrides config)")
    parser.add_argument("--output-csv", default="logs/final_evaluation.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    scale_factor = int(d_cfg["scale_factor"])
    sigma_range  = tuple(d_cfg["sigma_range"])

    # ── Image directory ──────────────────────────────────────────────────────
    if args.image_dir:
        image_dir = pathlib.Path(args.image_dir)
    else:
        # Prefer SCFace; fall back to pretrain BSDS300 val split
        scface_dir = pathlib.Path(d_cfg.get("finetune_dataset_path", "data/scface"))
        pretrain_dir = pathlib.Path(d_cfg.get("pretrain_dataset_path", "data/pretrain"))
        if scface_dir.exists() and any(scface_dir.rglob("*.jpg")):
            image_dir = scface_dir
            print(f"Evaluation dataset: SCFace  ({image_dir})")
        else:
            image_dir = pretrain_dir
            print(f"SCFace not found — using BSDS300 val split from {image_dir}")

    all_paths = collect_images(image_dir)

    # Use last 20 % as validation images (same deterministic split as training)
    import random
    rng = random.Random(42)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * float(d_cfg.get("val_split", 0.2))))
    val_paths = shuffled[:n_val]
    print(f"Evaluating on {len(val_paths)} validation images.\n")

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    n1 = ckpt.get("n1", 64)
    model = SRCNN(channels=1, n1=n1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path.name}  (n1={n1}, epoch={ckpt.get('epoch','?')})\n")

    pipeline = DegradationPipeline(scale_factor=scale_factor,
                                   sigma_range=sigma_range, seed=0)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    csv_path = pathlib.Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "bicubic_psnr", "srcnn_psnr",
                  "bicubic_ssim", "srcnn_ssim", "psnr_gain", "ssim_gain"]

    rows: list = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for path in tqdm(val_paths, desc="Evaluating"):
            try:
                result = evaluate_image(path, model, pipeline, device)
            except Exception as exc:
                print(f"  [skip] {path.name}: {exc}")
                continue

            row = {k: result[k] for k in fieldnames}
            writer.writerow(row)
            f.flush()
            rows.append(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    if not rows:
        print("No images evaluated.")
        return

    n = len(rows)
    mean_bic_psnr = sum(r["bicubic_psnr"] for r in rows) / n
    mean_src_psnr = sum(r["srcnn_psnr"]   for r in rows) / n
    mean_bic_ssim = sum(r["bicubic_ssim"] for r in rows) / n
    mean_src_ssim = sum(r["srcnn_ssim"]   for r in rows) / n
    mean_psnr_gain = mean_src_psnr - mean_bic_psnr
    mean_ssim_gain = mean_src_ssim - mean_bic_ssim

    print(f"\n{'='*52}")
    print(f"  Images evaluated : {n}")
    print(f"  {'Metric':<18} {'Bicubic':>10} {'SRCNN':>10} {'Gain':>8}")
    print(f"  {'-'*48}")
    print(f"  {'PSNR (dB)':<18} {mean_bic_psnr:>10.3f} {mean_src_psnr:>10.3f} {mean_psnr_gain:>+8.3f}")
    print(f"  {'SSIM':<18} {mean_bic_ssim:>10.4f} {mean_src_ssim:>10.4f} {mean_ssim_gain:>+8.4f}")
    print(f"{'='*52}")
    print(f"\nResults saved  ->  {csv_path}")

    if mean_src_psnr > mean_bic_psnr:
        print("PASS: SRCNN PSNR exceeds bicubic baseline.")
    else:
        print(f"NOTE: SRCNN is {abs(mean_psnr_gain):.3f} dB below bicubic — "
              "more training epochs recommended (run pretrain.py with more epochs).")


if __name__ == "__main__":
    main()
