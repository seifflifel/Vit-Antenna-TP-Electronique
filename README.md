# Antenna Architecture Prediction with Vision Transformer

This project predicts circular antenna array architecture from radiation pattern images.
The model outputs 5 values, one per ring, where each value is the number of elements in that ring (0 to 10).

The workflow is fully script-driven:

1. Generate synthetic antenna data and polar images.
2. Split into training and test sets.
3. Train a multi-head Vision Transformer.
4. Run inference and evaluate electromagnetic metrics.

## What The Code Does

### Data generation

`dataset.py` generates 10,000 random antenna configurations (`nb_samples = 10000`) and computes:

- Target architecture matrix: `Moutput` (5 x N)
- Feature matrix: `Minput` (4 x N) containing main-lobe gain, side-lobe level, HPBW, and steering angle
- Polar image tensors used for ViT input

Saved outputs include `.npy` and `.mat` datasets plus generated sample images in `images/`.

### Train and test split

`repartition_training_test.py` shuffles the full dataset and creates:

- 85% training split
- 15% test split

Both numeric features/targets and image tensors are split and saved.

### Model training

`ViT_training.py` trains a multi-head classifier built on top of ViT:

- Backbone: `vit_b_16` from `torchvision` (fallback to `timm`)
- Heads: 5 independent classification heads
- Classes per head: 11 (`0..10` elements)
- Loss: average cross-entropy across the 5 heads
- Scheduler: warmup + cosine decay
- Optional checkpoint resume via `--resume`

The script saves:

- `best_vit_antenna_model.pth`
- `vit_last_checkpoint.pth`
- `vit_training_loss.png`

### Evaluation

`ViT_test.py` loads the trained model, predicts on the test set, and reports:

- Overall exact-match accuracy (all 5 rings correct)
- Per-ring accuracy
- Radiation metrics comparison (main-lobe gain, side-lobe level, HPBW)

It also creates `vit_test_comparison.png` with cartesian and polar plots.

## Project Files

- `dataset.py`
- `repartition_training_test.py`
- `ViT_training.py`
- `ViT_test.py`
- `requirements.txt`
- `.gitignore`

Generated artifacts are intentionally ignored by git (`.npy`, `.mat`, `.pth`, `.png`, `images/`, cache folders).

## Setup

### 1) Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Optional CUDA and GPU checks

Check NVIDIA driver / CUDA runtime:

```bash
nvidia-smi
```

Check PyTorch CUDA visibility:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If CUDA is not available, training still runs on CPU.

## Run Pipeline

### 1) Generate data

```bash
python dataset.py
```

### 2) Create train/test split

```bash
python repartition_training_test.py
```

### 3) Train model

```bash
python ViT_training.py
```

Resume from last checkpoint:

```bash
python ViT_training.py --resume
```

### 4) Test / inference

```bash
python ViT_test.py
```


## Notes

- `ViT_test.py` can run without a checkpoint, but predictions will be random if no trained model exists.
- This project was an introduction to machine learning and artificial intelligence in an electronics class at INSAT IIA3 course.

## Date & Context
Project Date: December 2025 Project Context: Function of electronics project for IIA3 at INSAT