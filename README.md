# Dual-Branch Spatio-Frequency Network with Wavelet-Packet Attention for Compression-Resilient Deepfake Image Forensics

**Research Project — Team 14 Deepfake Forensics Research Group**  
*Academic Journal Submission & Reproducibility Repository*

---

## 1. Architectural Overview

This repository implements the complete spatio-frequency deepfake detection architecture specified in the **Team 14 Interim Technical Report (ITR)**, rigorously revised to address all methodological, architectural, and experimental feedback from the faculty review.

```
                      Face Crop (224 × 224 RGB)
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        [Spatial Branch]                [Frequency Branch]
        Separable Xception              Luminance Y (0.299R+0.587G+0.114B)
        Middle Flow Stage 3                      │
                 │                      3-Level 2D db4 WPD (GPU)
                 │                      (64 sub-bands: 64 × 28 × 28)
                 │                               │
                 │                      [HiFE Module]
                 │                      - Dual-pool Channel Gating
                 │                      - Directional Depthwise Conv
                 │                      - Stride-2 Match Projection
                 │                               │
        F_spa (728 × 14 × 14)           F_freq (256 × 14 × 14)
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                   [Dual Cross-Attention (DCA)]
                   Bi-directional token attention (196 tokens)
                   A_{S->F} and A_{F->S} + Spatial Skip
                                 │
                     F_fused (984 × 14 × 14)
                                 │
                     Global Average Pooling (GAP)
                                 │
                         z ∈ ℝ^{984}
                                 │
                      MLP Head (984 → 512 → 1)
                                 │
                     Scalar Logit ŷ ∈ ℝ (BCE Loss)
```

---

## 2. Key Modules & Implementations

- **`models/team14_net.py`**: Full end-to-end detector (`Team14Net`) with native toggle support for all 6 ablation variants.
- **`models/wavelet_branch.py`**: Native PyTorch GPU/CPU 3-level 2D Daubechies-4 Wavelet Packet Decomposition (`DifferentiableWPD3Level`). Produces 64 sub-bands ($28 \times 28$) with exact dyadic downsampling and support for both log-magnitude $\log(|W| + \epsilon)$ and log-energy $\log(W^2 + \epsilon)$ transforms.
- **`models/hife.py`**: Learnable High-Frequency Enhancement (`HiFEBlock`) featuring dual-pooling channel gating, directional depthwise convolution, and stride-2 dimension matching ($28 \times 28 \to 14 \times 14$).
- **`models/dual_cross_attention.py`**: Bi-directional token-level cross-attention (`DualCrossAttention`) operating over $14 \times 14 = 196$ spatial tokens with spatial residual skip ($984$ channels).
- **`models/spatial_branch.py`**: Pretrained Separable Xception backbone tapping Middle Flow Stage 3 ($728 \times 14 \times 14$).

---

## 3. Standardized Ablation Matrix

All six canonical configurations can be instantiated directly via `--model-type`:

| Variant Flag | Architecture Components | Parameters | Description |
| :--- | :--- | :---: | :--- |
| `full_team14` | Spatial + 3-Lvl WPD + HiFE + DCA | ~22.9M | **Headline Model** (Complete ITR Architecture) |
| `spatial_only` | Spatial Xception alone | ~20.8M | Baseline B1: Spatial Context alone |
| `wpd_only` | 3-Lvl WPD + HiFE alone | ~0.5M | Baseline B2: Frequency Features alone |
| `spatial_wpd_concat` | Spatial + 3-Lvl WPD (Naive Concat) | ~21.3M | Baseline B3: Spatial + Frequency without Attention |
| `spatial_wpd_hife` | Spatial + 3-Lvl WPD + HiFE (Concat) | ~21.8M | Ablation: Measures impact of HiFE without DCA |
| `spatial_wpd_dca` | Spatial + 3-Lvl WPD + DCA (No HiFE) | ~22.4M | Ablation: Measures impact of DCA without HiFE |

---

## 4. Documentation & Research Artifacts

Detailed forensic reports and protocols are available in the repository:

1. **[`AUDIT_REPORT.md`](AUDIT_REPORT.md)**: Comprehensive audit addressing all 10 faculty review issues (model resolution, token alignment, lossy compression vs. video codecs, zero-shot generalization, and metric claims).
2. **[`LITERATURE_AUDIT.md`](LITERATURE_AUDIT.md)**: Scholarly review of 18 digital forensics papers, resolving theoretical questions regarding WPD vs. DWT/DCT and cross-attention vs. parallel gating.
3. **[`REVISED_RELATED_WORK.md`](REVISED_RELATED_WORK.md)**: Formal literature review structured around 5 testable, falsifiable scientific hypotheses (H1–H5).
4. **[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md)**: Standardized training budget, multi-seed statistical significance protocol, paired $t$-tests, and controlled JPEG degradation sweeps ($Q = 100 \dots 10$).
5. **[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**: System specifications, deterministic seeding instructions, execution commands, and checkpoint layout.

---

## 5. Quick Start & Execution

### 1. Environment Verification
```powershell
py -3.12 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Run Test Suite
```powershell
# Run the Team 14 Architecture Unit Test Suite (13 tests)
py -3.12 tests/test_team14_architecture.py

# Run the Wavelet Filter Bank Mathematical Suite (6 tests)
py -3.12 tests/test_differentiable_wpd.py
```

### 3. Model Training
```powershell
# Train the Full Team 14 Network
py -3.12 train.py --model-type full_team14 --epochs 20 --batch-size 32 --lr 1e-4

# Train Spatial Branch Baseline (B1)
py -3.12 train.py --model-type spatial_only --epochs 20 --batch-size 32 --lr 1e-4

# Train Wavelet Branch Baseline (B2)
py -3.12 train.py --model-type wpd_only --epochs 20 --batch-size 32 --lr 1e-4
```

---

## 6. Scientific Framing & Task Integrity

- **Binary Classification Task:** The network outputs scalar logit $\hat{y} \in \mathbb{R}$ mapped via sigmoid to $P(\text{Fake}) \in [0, 1]$.
- **Metric Reporting:** Evaluated via threshold-free ROC-AUC, Equal Error Rate (EER), and Accuracy @ 0.5.
- **Interpretability:** DCA cross-attention maps and Grad-CAM activations provide spatial qualitative attention heatmaps. The architecture does not contain a pixel-level segmentation mask decoder, and localization IoU is not claimed.
