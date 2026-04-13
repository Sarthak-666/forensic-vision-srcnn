# Forensic-Vision SRCNN

Super-Resolution Convolutional Neural Network for forensic surveillance image enhancement.  
Trained on the BSDS300 dataset; designed for CCTV and SCFace surveillance footage.

---

## Project structure

```
.
├── config.yaml              # All hyperparameters (edit here, not in code)
├── train.py                 # General training loop
├── pretrain.py              # Pre-train on BSDS300 / general images
├── finetune.py              # Fine-tune on SCFace surveillance images
├── hparam_search.py         # Sweep n1 ∈ {32,64,128} and alpha ∈ {0.6,0.8,1.0}
├── evaluate.py              # Final evaluation → logs/final_evaluation.csv
├── cli/
│   └── process_video.py     # CLI tool: enhance a surveillance video
├── utils/
│   └── visualize.py         # Side-by-side comparison images
├── src/
│   ├── model.py             # SRCNN 9-5-5 architecture
│   ├── loss.py              # HybridLoss: alpha·MSE + (1-alpha)·(1-SSIM)
│   ├── degradation.py       # Bicubic downsample + Gaussian noise pipeline
│   └── train_utils.py       # Shared training loops, datasets, metrics
├── data/
│   └── pretrain/            # BSDS300 images (300 × natural images)
├── checkpoints/             # Saved model weights
└── logs/                    # CSV training logs and evaluation results
```

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
pip install opencv-python-headless   # for CLI video tool only
```

### 2 — Download training data

```bash
python - <<'EOF'
import urllib.request, tarfile, pathlib, shutil
url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
tgz = pathlib.Path("data/pretrain/BSDS300-images.tgz")
tgz.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, tgz)
with tarfile.open(tgz, "r:gz") as tf:
    tf.extractall(tgz.parent, filter="data")
for img in (tgz.parent/"BSDS300"/"images").rglob("*.jpg"):
    shutil.copy2(img, tgz.parent/img.name)
shutil.rmtree(tgz.parent/"BSDS300"); tgz.unlink()
EOF
```

### 3 — Pre-train on BSDS300

```bash
python pretrain.py               # uses data/pretrain/, saves checkpoints/pretrained.pth
python pretrain.py --smoke       # 1-epoch sanity check (no real images needed)
```

### 4 — Fine-tune on SCFace

```bash
python finetune.py               # requires data/scface/, saves checkpoints/finetuned_scface.pth
python finetune.py --smoke       # 1-epoch sanity check
```

### 5 — Hyperparameter search

```bash
python hparam_search.py          # full sweep on real images → logs/hparam_search.csv
python hparam_search.py --smoke  # 1-epoch smoke test
```

### 6 — Evaluate

```bash
python evaluate.py
# optional flags:
python evaluate.py --checkpoint checkpoints/pretrained.pth --image-dir data/pretrain
```

### 7 — Visual comparisons

```bash
python utils/visualize.py                  # saves 5 PNGs to results/comparisons/
python utils/visualize.py --n 10           # save 10 comparisons
```

---

## CLI video processing tool

Enhance a surveillance video frame-by-frame using the trained SRCNN.

```bash
python cli/process_video.py \
    --input  footage.mp4 \
    --output enhanced.mp4 \
    --scale  3

# With Gaussian pre-denoising (simulates DnCNN-style denoise step):
python cli/process_video.py \
    --input  footage.mp4 \
    --output enhanced.mp4 \
    --scale  3 \
    --denoise \
    --denoise-sigma 0.7
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input video |
| `--output` | *(required)* | Path to output video |
| `--scale` | `3` | Bicubic downscale factor used during training |
| `--denoise` | off | Enable Gaussian pre-denoising before SRCNN |
| `--denoise-sigma` | `0.5` | Gaussian σ for pre-denoising |
| `--checkpoint` | `checkpoints/finetuned_scface.pth` | Model weights to load |

---

## Results

### Hyperparameter search (`logs/hparam_search.csv`)

Locked config: **n1 = 64, alpha = 0.8**

| n1 | alpha | Val PSNR (dB) | Val SSIM |
|---:|------:|--------------:|---------:|
| 32 | 0.8 | 24.12 | 0.7235 |
| **64** | **0.8** | **24.65** | **0.7293** |
| 128 | 0.8 | 24.41 | 0.7149 |
| 64 | 0.6 | 24.41 | 0.7073 |
| 64 | 1.0 | 24.30 | 0.6856 |

### Final evaluation vs bicubic baseline (`logs/final_evaluation.csv`)

Evaluated on 30 BSDS300 validation images, 3× scale factor.

| Metric | Bicubic | SRCNN | Gain |
|--------|--------:|------:|-----:|
| PSNR (dB) | 25.240 | 25.302 | **+0.062** |
| SSIM | 0.6659 | 0.6928 | **+0.027** |

SRCNN exceeds the bicubic baseline on both PSNR and SSIM after 50 epochs of CPU training on BSDS300.  
Gains improve significantly with GPU training (200+ epochs) or a larger image corpus such as DIV2K.

### Sample comparison

`results/comparisons/187083_comparison.png`

![Sample comparison](results/comparisons/187083_comparison.png)

Four panels left to right: **LR Input** (bicubic-upsampled degraded image) · **Bicubic baseline** · **SRCNN output** · **HR ground truth**.

---

## Architecture

SRCNN 9-5-5 (Dong et al., 2014) — Y-channel only, no batch norm, no pooling:

```
Input (bicubic-upscaled LR)
  → Conv2d(1,  n1, 9×9, pad=4) → ReLU   # patch extraction
  → Conv2d(n1, 32,  5×5, pad=2) → ReLU   # non-linear mapping
  → Conv2d(32,  1,  5×5, pad=2)           # reconstruction
Output (HR estimate, same spatial size)
```

Weight init: Kaiming normal; biases: zero.

## Loss

```
L = alpha · MSE(pred, hr) + (1 − alpha) · (1 − SSIM(pred.clamp(0,1), hr))
```

Default: `alpha = 0.8` (hybrid MSE + SSIM outperforms pure MSE at `alpha = 1.0`).

## Training schedule

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Initial LR | 1 × 10⁻⁴ (pre-train), 1 × 10⁻⁵ (fine-tune) |
| LR schedule | StepLR — ×0.1 every 20 epochs |
| Patch size | 96 × 96 (pre-train), 33 × 33 (SCFace fine-tune) |
| Batch size | 16 |
| Y-channel only | yes |
