# Reproducibility Guide & Verification Playbook
**Project:** Dual-Branch Spatio-Frequency Network with Wavelet-Packet Attention for Compression-Resilient Deepfake Image Forensics  
**Author / Consortium:** Team 14 Deepfake Forensics Research Group  
**Document Purpose:** Complete, step-by-step instructions for deterministic execution, environment setup, model training, evaluation, and artifact reproduction.

---

## 1. Environment & Hardware Specifications

The codebase has been verified on the following hardware and software configuration:

- **Operating System:** Windows 11 (64-bit) / Linux (Ubuntu 22.04 LTS compatible)
- **Host GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM)
- **CUDA Runtime:** CUDA 12.4
- **Python Version:** Python 3.12.4 (64-bit)
- **Deep Learning Framework:** PyTorch 2.6.0+cu124
- **Vision Backbones:** `timm` 1.0.30 (PyTorch Image Models)
- **Wavelet Library:** `PyWavelets` (pywt) 1.8.0
- **Scientific Stack:** NumPy 2.x / 1.26, SciPy 1.15.2, scikit-learn 1.6.1, OpenCV 4.11.0, Pillow 11.1.0

---

## 2. Deterministic Execution Setup

To ensure strict reproducibility across runs, the framework enforces deterministic random seeding across Python, NumPy, PyTorch CPU, and CUDA runtimes.

Include the following configuration at the entry point of every script:

```python
import os
import random
import numpy as np
import torch

def set_deterministic_seed(seed: int = 42) -> None:
    """Enforce complete deterministic execution across all subsystems."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN deterministic execution
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 2.x deterministic algorithms
    torch.use_deterministic_algorithms(False)  # True can error on certain non-deterministic CUDA ops
```

---

## 3. Step-by-Step Reproduction Workflow

### Step 1: Verify Environment & GPU Acceleration

Run the diagnostic command to verify that Python 3.12 and CUDA 12.4 are accessible:

```powershell
py -3.12 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"
```

*Expected Output:*
```text
PyTorch 2.6.0+cu124 | CUDA: True | GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

### Step 2: Execute Architecture Unit Tests

Verify the 3-level 2D WPD, HiFE block, Dual Cross-Attention, and the 6 ablation network configurations:

```powershell
# Run the complete Team 14 architectural test suite
py -3.12 tests/test_team14_architecture.py

# Run the 2D filter bank mathematical test suite
py -3.12 tests/test_differentiable_wpd.py
```

*Expected Output:*
```text
Ran 13 tests in ~3.3s. OK.
Ran 6 tests in ~0.7s. OK.
```

---

### Step 3: Dataset Ingestion & Partitioning

The repository supports both local static face datasets and external video datasets:

1. **Local Static Dataset (FFHQ vs. StyleGAN2):**
   ```text
   data/_raw/real_vs_fake/real-vs-fake/
   ├── train/  (100,000 images: 50k real, 50k fake)
   ├── valid/  (20,000 images: 10k real, 10k fake)
   └── test/   (20,000 images: 10k real, 10k fake)
   ```

2. **External FaceForensics++ Dataset Setup:**
   Download FF++ video sequences into `data/faceforensics/` and extract frames using:
   ```powershell
   py -3.12 scripts/extract_ffpp_frames.py --data_dir data/faceforensics --compression c23 --output_dir data/processed/ffpp_c23
   ```

---

### Step 4: Training Ablation Models

Train any of the 6 canonical ablation models using the unified training driver:

```powershell
# 1. Train Full Team 14 Network (Headline Model)
py -3.12 train.py --model_type full_team14 --epochs 20 --batch_size 32 --lr 1e-4 --seed 42

# 2. Train Spatial Branch Only (Baseline B1)
py -3.12 train.py --model_type spatial_only --epochs 20 --batch_size 32 --lr 1e-4 --seed 42

# 3. Train Wavelet Branch Only (Baseline B2)
py -3.12 train.py --model_type wpd_only --epochs 20 --batch_size 32 --lr 1e-4 --seed 42

# 4. Train Spatial + WPD Concat (Baseline B3)
py -3.12 train.py --model_type spatial_wpd_concat --epochs 20 --batch_size 32 --lr 1e-4 --seed 42

# 5. Train Spatial + WPD + HiFE (Baseline B4)
py -3.12 train.py --model_type spatial_wpd_hife --epochs 20 --batch_size 32 --lr 1e-4 --seed 42

# 6. Train Spatial + WPD + DCA (Baseline B5)
py -3.12 train.py --model_type spatial_wpd_dca --epochs 20 --batch_size 32 --lr 1e-4 --seed 42
```

---

### Step 5: Multi-Seed Statistical Evaluation

To generate statistically verified results across 5 seeds:

```powershell
py -3.12 scripts/run_multi_seed_eval.py --model_type full_team14 --seeds 0 1 2 3 4
```

---

### Step 6: JPEG Compression Sweep Benchmark

Evaluate trained checkpoints under deterministic JPEG quality factors:

```powershell
py -3.12 scripts/eval_compression_sweep.py `
    --checkpoint checkpoints/full_team14_best.pt `
    --quality_factors 100 90 75 50 30 20 10 `
    --output_dir eval_results/compression_sweep
```

---

### Step 7: Generate Publication Figures & Curves

Generate publication-ready figures (ROC curves, compression decay plots, sub-band heatmaps):

```powershell
py -3.12 scripts/generate_paper_plots.py --results_dir eval_results/
```

---

## 4. Checkpoint Directory & Artifact Integrity

All saved checkpoints are stored in `checkpoints/` with standardized naming:

| Checkpoint Name | Description | Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `full_team14_best.pt` | Best validation checkpoint of full Team 14 architecture | ~91.6 MB | Generated upon save |
| `spatial_only_best.pt` | Spatial branch baseline checkpoint | ~83.2 MB | Generated upon save |
| `wpd_only_best.pt` | Frequency branch baseline checkpoint | ~2.1 MB | Generated upon save |

---
*End of Reproducibility Guide.*
