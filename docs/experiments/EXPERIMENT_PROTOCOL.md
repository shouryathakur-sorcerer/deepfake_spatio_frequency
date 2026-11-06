# Standardized Experimental Protocol & Statistical Evaluation Standards
**Project:** Dual-Branch Spatio-Frequency Network with Wavelet-Packet Attention for Compression-Resilient Deepfake Image Forensics  
**Author / Consortium:** Team 14 Deepfake Forensics Research Group  
**Document Purpose:** Definitive, reproducible protocol governing training budgets, degradation benchmarks, statistical significance testing, and metric reporting.

---

## 1. Experimental Integrity & Non-Negotiable Research Standards

To eliminate experimental bias, data contamination, and unreplicable claims, all experiments conducted under the Team 14 research protocol must strictly adhere to the following principles:

1. **Zero Test-Set Peeking:** Hyperparameter tuning, early stopping, and checkpoint selection must be performed **exclusively** on the designated validation split (`val`). The test sets (`test`, compressed sweeps, cross-dataset sets) are evaluated only once per final model checkpoint.
2. **Matched Compute & Training Budget:** When comparing architectural ablations (B1–B6), all variants must share identical training hyperparameters: total sample exposures, optimizer (AdamW), learning rate schedule (Cosine Annealing with warm-up), batch size (32), and data augmentations. No ablation may receive extra epochs or unequal regularization.
3. **Multi-Seed Stability:** Point estimates from a single training run are considered scientifically inconclusive. All headline comparisons must be executed across $N = 5$ independent random seeds ($\text{seed} \in \{0, 1, 2, 3, 4\}$). Results must report the sample mean ($\mu$), sample standard deviation ($\sigma$), and 95% Student's $t$ confidence intervals.
4. **Distinction of Compression Modalities:**
   - **Static Image Compression:** Evaluated via libjpeg standard quantization tables across quality factors $Q \in \{100, 90, 75, 50, 30, 20, 10\}$.
   - **Video Codec Compression:** Evaluated on native H.264/AVC compressed streams at Constant Rate Factor (CRF) levels $c0$ (raw), $c23$ (light compression), and $c40$ (heavy compression).
   - Under no circumstances may JPEG results be conflated with or reported as H.264 video codec results.
5. **Honest Forensic Task Framing:** The Team 14 network is an image-level binary classifier producing a scalar logit $\hat{y} \in \mathbb{R}$ mapped via sigmoid $\sigma(\hat{y}) \in [0, 1]$ to class probabilities ($\text{Real} = 0, \text{Fake} = 1$). Grad-CAM and Dual Cross-Attention map activations provide qualitative interpretability. **Claims of manipulation bounding box or pixel mask localization IoU are invalid** unless a dedicated segmentation decoder is trained against ground-truth manipulation masks.

---

## 2. Dataset Partitioning & Leakage Prevention

### Primary Benchmark Datasets

| Dataset Identifier | Domain | Generation Mechanism / Source | Real Faces | Fake Faces | Total Images |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FFHQ vs. StyleGAN2 (140k)** | Static High-Res Faces | StyleGAN2 (NVidia) vs. Real FFHQ | 70,000 | 70,000 | 140,000 |
| **FaceForensics++ (FF++)** | Video Facial Sequences | Deepfakes, Face2Face, FaceSwap, NeuralTextures | 1,000 vids | 4,000 vids | ~1.8M frames |
| **Celeb-DF v2** | Challenging Face Swaps | Deepfake face-swapping with post-blend | 590 vids | 5,639 vids | ~2.2M frames |

### Leakage Controls
- **Subject / Identity Isolation:** For video datasets (FF++, Celeb-DF), frame sampling must never cross-contaminate video IDs between train, validation, and test splits. All splits are strictly partitioned by original video sequence index.
- **Hash De-duplication:** Verification of zero duplicate images across splits is enforced via SHA-256 hash checks and perceptual hashing (pHash distance $> 8$).

---

## 3. Matched Training Specification

All ablation models ($B_1 \dots B_6$) are trained using the standardized recipe below:

```yaml
Hardware Configuration:
  Device: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
  CUDA Version: 12.4
  PyTorch Version: 2.6.0+cu124
  Mixed Precision: Automatic Mixed Precision (torch.cuda.amp.autocast, FP16)

Optimization Parameters:
  Optimizer: AdamW (Loshchilov & Hutter, 2019)
  Base Learning Rate (Spatial Backbone): 5.0e-5
  Base Learning Rate (Wavelet/Fusion/Head): 1.0e-4
  Weight Decay: 1.0e-4
  Betas: (0.9, 0.999)
  Epsilon: 1.0e-8

Learning Rate Schedule:
  Type: CosineAnnealingLR with Linear Warmup
  Warmup Epochs: 2
  Total Epochs: 20
  Minimum Learning Rate: 1.0e-6

Batching & Preprocessing:
  Canonical Resolution: 224 x 224 pixels
  Batch Size: 32 (effective batch size 32 via gradient accumulation = 1)
  Normalization: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
  Luminance Formula: Y = 0.299 * R + 0.587 * G + 0.114 * B

Data Augmentations (Train Split Only):
  - Random Horizontal Flip (p = 0.5)
  - Color Jitter (Brightness = 0.1, Contrast = 0.1, Saturation = 0.1, Hue = 0.05) (p = 0.3)
  - Mild Gaussian Blur (kernel = 3x3, sigma in [0.1, 1.0]) (p = 0.2)
  - Random JPEG Compression (Q in [60, 100]) (p = 0.4)

Loss Function:
  Criterion: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
  Label Smoothing: epsilon = 0.05
  Output Convention: 0.0 = Real, 1.0 = Fake / Manipulated
```

---

## 4. Controlled Evaluation Protocols

### Protocol A: In-Distribution Performance
- **Input:** Canonical uncompressed test split ($224 \times 224$).
- **Metrics Reported:**
  1. **ROC-AUC (Area Under the Receiver Operating Characteristic):** Primary threshold-independent metric.
  2. **EER (Equal Error Rate):** Operating point where False Positive Rate equals False Negative Rate.
  3. **Accuracy (ACC @ threshold = 0.5):** Percentage of correctly classified instances.
  4. **F1-Score:** Harmonic mean of precision and recall.

### Protocol B: Compression Resilience Sweep (Image JPEG)
- **Input:** Test split subjected to deterministic JPEG recompression using Pillow/libjpeg:
  $$Q \in \{100, 90, 75, 50, 30, 20, 10\}$$
- **Procedure:** Original uncompressed test image $I \to \text{JPEG}(I, Q) \to \text{Evaluate}$.
- **Reporting:** Tabulate and plot AUC decay curve as a function of $Q$. Calculate Area Under Compression Curve (AUCC) and retention ratio:
  $$\text{Retention Ratio} = \frac{\text{AUC}_{Q=30}}{\text{AUC}_{Q=100}} \times 100\%$$

### Protocol C: Native Video Codec Evaluation (FF++ H.264)
- **Input:** Native video frames extracted from FaceForensics++ at compression factors:
  - Raw ($c0$)
  - High Quality ($c23$, default YouTube quality)
  - Low Quality ($c40$, severe mobile/social compression)
- **Reporting:** AUC across all 4 manipulation sub-types (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

### Protocol D: Zero-Shot Cross-Dataset / Cross-Generator Generalization
- **Training Source:** In-distribution dataset (e.g., FFHQ vs. StyleGAN2).
- **Target Evaluation:** Out-of-distribution datasets evaluated without fine-tuning:
  - Latent Diffusion (Stable Diffusion / Midjourney synthetic faces)
  - Celeb-DF v2
  - DeepFake Detection Challenge Preview (DFDC-P)
- **Metric:** Generalization Drop $\Delta \text{AUC} = \text{AUC}_{\text{in-domain}} - \text{AUC}_{\text{cross-domain}}$.

---

## 5. Statistical Significance Testing Framework

To rigorously validate hypotheses H1–H5:

1. **Paired Comparisons:** When comparing Model A (e.g., `full_team14`) against Model B (e.g., `spatial_wpd_concat` or `spatial_only`):
   - Evaluate both models on identical test sample instances ($x_i, y_i$).
   - For repeated seeds ($k = 1 \dots 5$), compute paired sample differences:
     $$d_k = \text{Metric}_A(k) - \text{Metric}_B(k)$$
2. **Hypothesis Testing:**
   - Check normality of differences via Shapiro-Wilk test ($\alpha = 0.05$).
   - If normally distributed: Conduct two-tailed **Paired Student’s t-test**.
   - If non-normal: Conduct non-parametric **Wilcoxon Signed-Rank Test**.
   - Report exact $p$-values. Reject null hypothesis if $p < 0.01$ (highly significant) or $p < 0.05$ (significant).
3. **Effect Size (Cohen’s $d$):**
   $$d = \frac{\bar{d}}{s_d}$$
   Where $\bar{d}$ is the mean paired difference and $s_d$ is the standard deviation of differences. Effect sizes are categorized as:
   - Small: $|d| \approx 0.2$
   - Medium: $|d| \approx 0.5$
   - Large: $|d| \ge 0.8$

---

## 6. Ablation Experiment Matrix

The 6 canonical ablation models must be evaluated under identical conditions:

| Variant ID | Name | Spatial Branch | Wavelet Branch | HiFE Module | Fusion Mechanism | Model Capacity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **B1** | `spatial_only` | Xception ($728 \times 14 \times 14$) | None | None | None (Head: 728 $\to$ 1) | ~20.8M |
| **B2** | `wpd_only` | None | 3-Lvl WPD (64 sub-bands) | HiFE ($256 \times 14 \times 14$) | None (Head: 256 $\to$ 1) | ~0.5M |
| **B3** | `spatial_wpd_concat`| Xception ($728 \times 14 \times 14$) | 3-Lvl WPD (64 sub-bands) | None (Conv $64 \to 256$) | Simple Concat (984 $\to$ 1) | ~21.3M |
| **B4** | `spatial_wpd_hife` | Xception ($728 \times 14 \times 14$) | 3-Lvl WPD (64 sub-bands) | HiFE ($256 \times 14 \times 14$) | Simple Concat (984 $\to$ 1) | ~21.8M |
| **B5** | `spatial_wpd_dca` | Xception ($728 \times 14 \times 14$) | 3-Lvl WPD (64 sub-bands) | None (Conv $64 \to 256$) | Dual Cross-Attention | ~22.4M |
| **B6** | `full_team14` | Xception ($728 \times 14 \times 14$) | 3-Lvl WPD (64 sub-bands) | HiFE ($256 \times 14 \times 14$) | Dual Cross-Attention | ~22.9M |

---
*End of Standardized Experimental Protocol.*
