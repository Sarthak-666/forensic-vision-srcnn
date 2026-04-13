

# Forensic-Vision SRCNN

A Super-Resolution Convolutional Neural Network (SRCNN) optimized for 
forensic surveillance applications. Enhances low-resolution CCTV footage 
to recover facial detail, license plates, and scene context using deep learning.

## Results

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Bicubic Baseline | ~22.5 | ~0.62 |
| Forensic-Vision SRCNN | **24.65** | **0.7293** |

## How It Works

1. **Denoise** — DnCNN removes sensor noise and JPEG artefacts from the raw frame
2. **Enhance** — SRCNN upscales the clean image using a learned 3-layer CNN
3. **Output** — High-resolution frame ready for forensic review or face recognition

The model operates on the Y (luminance) channel of YCbCr color space and uses 
a hybrid MSE + SSIM loss function for perceptually accurate reconstruction.

## Architecture

Three convolutional layers, no pooling:
- **Layer 1** — 9×9 filters, 64 feature maps, ReLU (patch extraction)  
- **Layer 2** — 5×5 filters, 32 feature maps, ReLU (non-linear mapping)  
- **Layer 3** — 5×5 filters, 1 output channel, linear (reconstruction)

## Installation

```bash
git clone https://github.com/Sarthak-666/forensic-vision-srcnn.git
cd forensic-vision-srcnn
pip install -r requirements.txt
```

## Usage

### Enhance a single image
```bash
python cli/enhance_image.py --input image.jpg --output enhanced.jpg --scale 3
```

### With denoising (recommended for real CCTV footage)
```bash
python cli/enhance_image.py --input cctv_frame.jpg --output enhanced.jpg --scale 3 --denoise
```

### Process a video
```bash
python cli/process_video.py --input footage.mp4 --output enhanced.mp4 --scale 3 --denoise
```

## Training

Pre-trained on BSDS500 (~500 natural images), fine-tuned on SCFace 
(4,160 surveillance images across 5 camera types).

```bash
python pretrain.py      # Pre-train on BSDS500
python finetune.py      # Fine-tune on SCFace
python evaluate.py      # Evaluate on validation set
```

## Applications

- **Biometric Reconstruction** — Enhance faces for automated recognition systems
- **License Plate Recovery** — Recover characters from motion-blurred highway footage  
- **Crime Scene Context** — Clarify background detail in nighttime CCTV footage

## Tech Stack

Python · PyTorch · PIL · torchvision · tqdm · argparse

## Dataset Credits

- [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) — Berkeley Segmentation Dataset
- [SCFace](http://www.scface.org/) — Surveillance Cameras Face Database
