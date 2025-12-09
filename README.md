# Antenna Architecture Prediction using Vision Transformer (ViT)

A deep learning project to predict antenna array architectures from radiation patterns using Vision Transformer models. This project migrates from a Feed-Forward Neural Network (FFNN) baseline to a state-of-the-art Vision Transformer for improved accuracy and feature learning from polar radiation patterns.

---

## 📋 Project Overview

**Objective:** Predict antenna element counts per ring (0-10 elements) from polar radiation pattern images.

**Key Features:**
- 🖼️ **Polar & Cartesian image generation** from antenna simulations
- 🤖 **Vision Transformer (ViT-B/16)** with multi-head classification
- 📊 **Multi-output classification** — 5 independent heads (one per ring)
- 🔄 **Data augmentation** — rotation, color jitter for robust learning
- 📈 **Comprehensive evaluation** — per-ring accuracy, radiation pattern metrics (HPBW, main-lobe gain, SSL)
- 💾 **Checkpointing & resumable training**

---

## 🗂️ Project Structure

```
VIT/
├── dataset.py                          # Generate synthetic antenna data & images
├── repartition_training_test.py        # Split data into 85% train / 15% test
├── ViT_training.py                     # Main ViT training script (100 epochs)
├── ViT_test.py                         # Evaluation & visualization on test set
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── Minput.npy                          # All features (4, 100): [MainLobe, SSL, HPBW, theta0]
├── Moutput.npy                         # All targets (5, 100): element counts per ring
├── Minput_images.npy                   # All cartesian images (100, 240, 240, 3)
├── Minput_images_polar.npy             # All polar images (100, 240, 240, 3)
│
├── Minput_training.npy                 # Training features (4, 85)
├── Moutput_training.npy                # Training targets (5, 85)
├── Minput_images_training.npy          # Training cartesian (85, 240, 240, 3)
├── Minput_images_polar_training.npy    # Training polar (85, 240, 240, 3) ← INPUT
│
├── Minput_test.npy                     # Test features (4, 15)
├── Moutput_test.npy                    # Test targets (5, 15)
├── Minput_images_test.npy              # Test cartesian (15, 240, 240, 3)
├── Minput_images_polar_test.npy        # Test polar (15, 240, 240, 3) ← INPUT
│
├── vit_best_model.pth                  # Best trained model (checkpoint)
├── vit_last_checkpoint.pth             # Last epoch checkpoint
├── vit_training_loss.png               # Loss curves
├── vit_test_comparison.png             # Radiation pattern comparison
│
└── images/                             # Generated antenna pattern images
    ├── sample_000.png, sample_000_polar.png
    ├── sample_001.png, sample_001_polar.png
    └── ...
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**
- **GPU:** NVIDIA RTX 3050+ with CUDA 12.6 (or adjust for your system)
- **Git** (for version control)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/antenna-vit.git
   cd antenna-vit
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU (optional but recommended):**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
   ```

---

## 📊 Workflow

### Step 1: Generate Synthetic Data
Generate 100 random antenna architectures with their radiation patterns and images:
```bash
python dataset.py
```
**Output:**
- `Minput.npy`, `Moutput.npy` — features & targets
- `Minput_images.npy`, `Minput_images_polar.npy` — cartesian & polar images
- `images/sample_*.png` — individual radiation pattern plots

### Step 2: Repartition into Train/Test
Split data into 85% training and 15% testing:
```bash
python repartition_training_test.py
```
**Output:**
- `Minput_training.npy`, `Moutput_training.npy`, etc.
- `Minput_images_polar_training.npy`, `Minput_images_polar_test.npy`

### Step 3: Train ViT Model
Train a Vision Transformer on polar images for 100 epochs:
```bash
python ViT_training.py
```
**Features:**
- Pretrained ViT-B/16 backbone (transfer learning)
- Multi-head classification: 5 heads × 11 classes
- Batch size: 16 (RTX 3050), 4 (CPU)
- Learning rate: 1e-4 with cosine annealing
- **Output:**
  - `vit_best_model.pth` — best model (lowest val loss)
  - `vit_last_checkpoint.pth` — last epoch (resumable)
  - `vit_training_loss.png` — loss curves

### Step 4: Evaluate on Test Set
Run inference and compare predictions vs ground truth:
```bash
python ViT_test.py
```
**Output:**
- Overall accuracy (all 5 rings correct)
- Per-ring accuracy for each ring
- Radiation pattern metrics (main-lobe gain, HPBW, SSL)
- Side-by-side cartesian + polar comparisons

---

## 📈 Expected Performance

- **Training time:** ~30 min (GPU RTX 3050), ~5-10 hours (CPU)
- **Per-ring accuracy:** 50-90% (depends on data and training)
- **Radiation metrics error:** typically 2-5% for well-trained models

---

## 🔧 Configuration

Edit parameters in each script:

**`dataset.py`:**
```python
nb_samples = 100         # Increase for more data
max_rings = 5            # Number of antenna rings
max_elements = 10        # Max elements per ring (0-10)
theta0_max_deg = 180     # Steering angle range
```

**`ViT_training.py`:**
```python
NUM_EPOCHS = 100         # Training epochs
BATCH_SIZE = 16          # Batch size (adjust for GPU memory)
LEARNING_RATE = 1e-4     # Learning rate
WEIGHT_DECAY = 1e-2      # L2 regularization
```

---

## 💾 Saving & Resuming Training

- **Automatic checkpointing:** every epoch saves `vit_last_checkpoint.pth`
- **Best model:** `vit_best_model.pth` (lowest val loss)
- **Resume training:** (requires custom code — currently starts fresh each run)

---

## 📝 Model Architecture

```
Input: Polar Image (240, 240, 3 RGB)
  ↓
ViT Backbone (pretrained, frozen initially)
  - Patch embedding (16×16 patches)
  - Transformer encoder (12 layers, 768 hidden dim)
  - Class token + learnable positional embeddings
  ↓
Output Feature Vector (768-dim)
  ↓
Multi-Head Classification (5 heads in parallel)
  ├─ Head 1 (Ring 1): 11-class softmax → [0..10 elements]
  ├─ Head 2 (Ring 2): 11-class softmax → [0..10 elements]
  ├─ Head 3 (Ring 3): 11-class softmax → [0..10 elements]
  ├─ Head 4 (Ring 4): 11-class softmax → [0..10 elements]
  └─ Head 5 (Ring 5): 11-class softmax → [0..10 elements]
  ↓
Output: Architecture prediction (5 values, each 0-10)
```

---

## 🤝 Contributing

1. **Clone & create a branch:**
   ```bash
   git clone https://github.com/your-username/antenna-vit.git
   cd antenna-vit
   git checkout -b feature/my-improvement
   ```

2. **Make changes & commit:**
   ```bash
   git add .
   git commit -m "Add feature X" -m "Details about the change"
   ```

3. **Push to GitHub:**
   ```bash
   git push origin feature/my-improvement
   ```

4. **Create a Pull Request** on GitHub for review & merge

---

## 📚 Reference Pipeline

The project includes a reference FFNN implementation for comparison:

```
Reference pipeline/
├── dataset.py              # Generate 100k samples (baseline)
├── repartition_training_test.py
├── FFNN_training.py        # FFNN with 3 hidden layers
└── test_model_FFNN.py      # FFNN evaluation
```

To train the FFNN baseline, run scripts in the `Reference pipeline/` folder in order.

---

## 🐛 Troubleshooting

**Q: CUDA not detected?**  
A: Ensure NVIDIA drivers are installed and PyTorch is configured for your CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Q: Out of memory?**  
A: Reduce `BATCH_SIZE` in `ViT_training.py` (e.g., 8 or 4).

**Q: Training too slow?**  
A: Use GPU (see CUDA setup above). On CPU, reduce `nb_samples` in `dataset.py` for quick tests.

**Q: Model not improving?**  
A: Check data quality, adjust learning rate, increase training epochs, or add more augmentation.

---

## 📄 License

This project is for educational purposes. Modify and share as needed within your team.

---

## 👥 Team Members

Add your teammates here:
- Your Name (@github-username) — role
- Colleague 1 (@username) — role
- Colleague 2 (@username) — role

---

## 📞 Contact

For questions or issues, create a GitHub Issue or contact the project maintainers.

---

**Last updated:** December 2025  
**Status:** ✅ ViT training functional, GPU-enabled
