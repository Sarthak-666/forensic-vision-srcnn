"""
Forensic-Vision SRCNN — Visual Comparison Utility
===================================================
Saves side-by-side comparison images:
    LR input  |  Bicubic  |  SRCNN output  |  HR ground truth

Usage
-----
  python utils/visualize.py                          # 5 random val images
  python utils/visualize.py --n 10 --image-dir data/pretrain
  python utils/visualize.py --checkpoint checkpoints/finetuned_scface.pth
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.degradation import DegradationPipeline, _to_tensor
from src.model import SRCNN

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


_LABEL_H  = 20   # pixels reserved for the text label below each panel
_PANEL_GAP = 4   # gap between panels in pixels


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, H, W) float tensor [0,1] → PIL grayscale image."""
    arr = (t.squeeze(0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _add_label(panel: Image.Image, label: str) -> Image.Image:
    """Return a new image with a white label bar appended at the bottom."""
    bar = Image.new("L", (panel.width, _LABEL_H), color=240)
    draw_img = Image.new("L", (panel.width, panel.height + _LABEL_H), color=255)
    draw_img.paste(panel, (0, 0))
    draw_img.paste(bar,   (0, panel.height))
    # Write label text using PIL's default bitmap font (no external fonts needed)
    from PIL import ImageDraw
    d = ImageDraw.Draw(draw_img)
    d.text((4, panel.height + 3), label, fill=0)
    return draw_img


@torch.no_grad()
def save_comparison(
    img_path: pathlib.Path,
    model: SRCNN,
    pipeline: DegradationPipeline,
    device: torch.device,
    out_dir: pathlib.Path,
) -> None:
    """Generate and save a 4-panel PNG for one image using Pillow only."""
    img = Image.open(img_path).convert("YCbCr")
    y, _, _ = img.split()

    # Crop to scale_factor multiple (same fix as evaluate.py)
    sf = pipeline.scale_factor
    w, h = y.size
    y = y.crop((0, 0, (w // sf) * sf, (h // sf) * sf))

    hr_tensor = _to_tensor(y)                                   # (1, H, W) [0,1]
    lr_up, hr_clean = pipeline(hr_tensor)
    pred = model(lr_up.unsqueeze(0).to(device)).cpu().squeeze(0)

    panels = [
        ("LR Input (bicubic up)", _tensor_to_pil(lr_up)),
        ("Bicubic baseline",      _tensor_to_pil(lr_up)),
        ("SRCNN output",          _tensor_to_pil(pred)),
        ("HR ground truth",       _tensor_to_pil(hr_clean)),
    ]

    labelled = [_add_label(img_pil, lbl) for lbl, img_pil in panels]

    total_w = sum(p.width for p in labelled) + _PANEL_GAP * (len(labelled) - 1)
    max_h   = max(p.height for p in labelled)
    canvas  = Image.new("L", (total_w, max_h), color=200)

    x = 0
    for panel in labelled:
        canvas.paste(panel, (x, 0))
        x += panel.width + _PANEL_GAP

    out_path = out_dir / f"{img_path.stem}_comparison.png"
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SRCNN visual comparisons")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/finetuned_scface.pth")
    parser.add_argument("--image-dir",  default=None)
    parser.add_argument("--out-dir",    default="results/comparisons")
    parser.add_argument("--n",          type=int, default=5,
                        help="Number of sample images to visualise")
    parser.add_argument("--seed",       type=int, default=42)
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
        scface_dir   = pathlib.Path(d_cfg.get("finetune_dataset_path", "data/scface"))
        pretrain_dir = pathlib.Path(d_cfg.get("pretrain_dataset_path", "data/pretrain"))
        image_dir = scface_dir if (scface_dir.exists() and
                                   any(scface_dir.rglob("*.jpg"))) else pretrain_dir

    all_paths = sorted(p for p in image_dir.rglob("*")
                       if p.suffix.lower() in _IMG_EXTS)
    if not all_paths:
        sys.exit(f"No images found in {image_dir}")

    # Pick `n` validation images deterministically
    rng = random.Random(args.seed)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * float(d_cfg.get("val_split", 0.2))))
    val_paths = shuffled[:n_val]
    sample = val_paths[:args.n]
    print(f"Generating {len(sample)} comparisons from {image_dir}")

    # ── Model ────────────────────────────────────────────────────────────────
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    n1   = ckpt.get("n1", 64)
    model = SRCNN(channels=1, n1=n1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pipeline = DegradationPipeline(scale_factor=scale_factor,
                                   sigma_range=sigma_range, seed=0)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(sample, desc="Saving comparisons"):
        try:
            save_comparison(path, model, pipeline, device, out_dir)
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}")

    saved = sorted(out_dir.glob("*.png"))
    print(f"\nSaved {len(saved)} PNG(s) to {out_dir}/")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
