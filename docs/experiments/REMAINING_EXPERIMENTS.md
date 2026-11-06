# Remaining Experiments & Execution Playbook
**Project:** Spatial–Frequency Deepfake Image Classification with Wavelet-Packet Features and Bidirectional Cross-Attention  
**Document Purpose:** Definitive catalog of pending experiments, compute requirements, execution scripts, and exact CLI commands needed to complete full empirical validation.

---

## 1. Summary of Experimental Status

| Experiment ID | Title | Target Hypotheses | Primary Dataset | Compute Budget | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **EXP-01** | Matched 6-Variant Ablation Suite ($N=5$ seeds) | $H_1, H_2, H_3, H_4$ | FFHQ vs. StyleGAN2 (140k) | ~12.5 GPU hours | **Ready to Execute** |
| **EXP-02** | Dense Compression Sweep ($Q = 100 \dots 10$) | $H_2, H_3$ | FFHQ vs. StyleGAN2 (20k test) | ~0.5 GPU hours | **Ready to Execute** |
| **EXP-03** | Factorial Architecture $\times$ Augmentation Study | $H_1$ | FFHQ vs. StyleGAN2 (140k) | ~8.0 GPU hours | **Ready to Execute** |
| **EXP-04** | Zero-Shot Cross-Generator Transfer (Diffusion) | $H_5$ | Stable Diffusion / Midjourney | ~0.5 GPU hours | **Pending Data Path** |
| **EXP-05** | Native FaceForensics++ Video Codec ($c0, c23, c40$) | $H_1, H_2$ | FF++ Video Frames | ~6.0 GPU hours | **Pending Video Download** |

---

## 2. Step-by-Step Execution Commands

### Experiment 1: Matched 6-Variant Multi-Seed Training (EXP-01)
To eliminate experimental confounds and establish whether $H_2, H_3,$ and $H_4$ are supported, train all 6 canonical variants under identical hyperparameters (AdamW, Cosine Annealing, 20 epochs, batch size 32) across 5 independent random seeds ($\text{seed} \in \{0, 1, 2, 3, 4\}$):

```powershell
# 1. Headline Model: Full Team 14 Architecture (Spatial + 3-Lvl WPD + HiFE + DCA)
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --lr 1e-4 --seed 0
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --lr 1e-4 --seed 1
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --lr 1e-4 --seed 2
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --lr 1e-4 --seed 3
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --lr 1e-4 --seed 4

# 2. Baseline B1: Spatial Xception Only
py -3.12 train.py --model-type spatial_only --epochs 20 --batch-size 32 --lr 1e-4 --seed 0

# 3. Baseline B2: Frequency 3-Level WPD + HiFE Only
py -3.12 train.py --model-type wpd_only --epochs 20 --batch-size 32 --lr 1e-4 --seed 0

# 4. Baseline B3: Spatial + WPD Naive Concatenation
py -3.12 train.py --model-type spatial_wpd_concat --epochs 20 --batch-size 32 --lr 1e-4 --seed 0

# 5. Ablation B4: Spatial + WPD + HiFE (without DCA)
py -3.12 train.py --model-type spatial_wpd_hife --epochs 20 --batch-size 32 --lr 1e-4 --seed 0

# 6. Ablation B5: Spatial + WPD + DCA (without HiFE)
py -3.12 train.py --model-type spatial_wpd_dca --epochs 20 --batch-size 32 --lr 1e-4 --seed 0
```

---

### Experiment 2: Dense JPEG Compression Sweep (EXP-02)
Evaluate the trained checkpoints under dense JPEG compression across 7 quality factors ($Q \in \{100, 90, 75, 50, 30, 20, 10\}$) to map the continuous degradation curve and quantify retention:

```powershell
py -3.12 scripts/eval_compression_sweep.py `
    --checkpoint checkpoints/full_team14_best.pt `
    --quality-factors 100 90 75 50 30 20 10 `
    --output-dir eval_results/compression_sweep
```

---

### Experiment 3: Factorial Architecture $\times$ Augmentation Grid (EXP-03)
To formally validate Hypothesis $H_1$ (whether WPD provides benefits beyond compression-aware data augmentation):

```powershell
# Proposed Model WITHOUT JPEG Augmentation
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --no-jpeg-aug --checkpoint-dir checkpoints/no_aug

# Proposed Model WITH JPEG Augmentation
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --checkpoint-dir checkpoints/with_aug

# Spatial Only WITHOUT JPEG Augmentation
py -3.12 train.py --model-type spatial_only --epochs 20 --batch-size 32 --no-jpeg-aug --checkpoint-dir checkpoints/no_aug

# Spatial Only WITH JPEG Augmentation
py -3.12 train.py --model-type spatial_only --epochs 20 --batch-size 32 --checkpoint-dir checkpoints/with_aug
```

---

### Experiment 4: Cross-Generator Transfer Evaluation (EXP-04)
Quantify the generalization drop ($\Delta \text{AUC}$) from StyleGAN2 to Diffusion synthetic faces:

```powershell
py -3.12 eval_generalization.py `
    --checkpoint checkpoints/full_team14_best.pt `
    --target-data data/diffusion_faces `
    --output-json eval_results/cross_generator_diffusion.json
```

---

### Experiment 5: FaceForensics++ Video Codec Benchmark (EXP-05)
Extract native video frames and evaluate across $c0$ (raw), $c23$ (light compression), and $c40$ (heavy compression):

```powershell
# 1. Extract frames from FF++ video sequences
py -3.12 scripts/extract_ffpp_frames.py --video-dir data/faceforensics/c23 --output-dir data/processed/ffpp_c23
py -3.12 scripts/extract_ffpp_frames.py --video-dir data/faceforensics/c40 --output-dir data/processed/ffpp_c40

# 2. Evaluate checkpoint
py -3.12 eval.py --checkpoint checkpoints/full_team14_best.pt --data-root data/processed/ffpp_c23 --output-json eval_results/ffpp_c23_eval.json
py -3.12 eval.py --checkpoint checkpoints/full_team14_best.pt --data-root data/processed/ffpp_c40 --output-json eval_results/ffpp_c40_eval.json
```

---
*End of Remaining Experiments & Execution Playbook.*
