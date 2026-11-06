# Master Research Dossier: Deepfake Image Detection Using Multi-Level Frequency Analysis

*This document is structured specifically for ingestion into Google NotebookLM, ChatGPT, or academic report synthesis tools.*

---

## 1. Executive Summary & Research Motivation

### Problem Statement
Standard deep learning facial forensics models operate predominantly in the spatial RGB domain (using convolutional neural networks or Vision Transformers). While these spatial models achieve high accuracy on pristine, uncompressed laboratory benchmarks, their detection efficacy drops catastrophically when subjected to lossy post-processing—most notably lossy JPEG compression automatically applied by social media channels (WhatsApp, X/Twitter, Instagram, Meta). Conversely, conventional global frequency analysis (such as Discrete Fourier Transforms or standard Discrete Cosine Transforms) lacks spatial localization and fails to distinguish between generative synthesis artifacts and benign high-frequency image compression noise.

### Proposed Solution
A Dual-Branch Spatio-Frequency Deepfake Detection Architecture that synergistically fuses:
1. Spatial Visual Context: Deep semantic facial features extracted via an ImageNet-pretrained convolutional backbone.
2. Multi-Scale Wavelet-Packet Frequency Representations: Directional spectral sub-band energies extracted via 2D Wavelet Packet Decomposition (WPD) with a log-energy transformation.
3. Cross-Domain Squeeze-and-Excitation (SE) Attention Fusion: An adaptive gating mechanism that dynamically recalibrates spatial versus frequency feature importance based on image compression severity.

---

## 2. End-to-End Architectural Specifications

### Module 1: Spatial Context Branch (models/spatial_branch.py)
* Backbone: legacy_xception (via PyTorch Image Models timm), initialized with ImageNet-pretrained weights.
* Architecture: Uses depthwise separable convolutional blocks with residual skip connections.
* Feature Extraction: Given an input image I in R^{B x 3 x 256 x 256}, the backbone extracts intermediate feature representations. The deepest feature map X_spatial in R^{B x 2048 x 8 x 8} is extracted.
* Global Average Pooling (GAP) collapses spatial dimensions across height and width:
  GAP(X_spatial)_c = (1 / (H * W)) * sum_{h=1}^H sum_{w=1}^W X_spatial[:, c, h, w]  in R^{B x 2048}
* Spatial Projection:
  F_spatial = ReLU( BatchNorm1D( W_proj_s * GAP(X_spatial) ) ) in R^{B x 512}

### Module 2: Multi-Scale Wavelet-Packet Branch (models/wavelet_branch.py)
* Wavelet Basis: Daubechies-4 (db4), orthogonal wavelet with compact support and 4 vanishing moments.
* Decomposition Depth: Level L = 2.
* Wavelet Packet vs. Standard Wavelet: Standard Discrete Wavelet Transform (DWT) only recursively splits the low-frequency approximation sub-band (LL). 2D WPD recursively decomposes both low-frequency (LL) and high-frequency (LH, HL, HH) detail sub-bands, producing 4^L = 4^2 = 16 directional sub-bands of dimension approx 64 x 64:
  W = { W_1, W_2, ..., W_16 }, where each sub-band W_i in R^{B x 64 x 64}
* Log-Energy Transformation: Extreme energy spikes in high-frequency detail bands are normalized using logarithmic absolute coefficient scaling:
  W_tilde_i = log( |W_i| + epsilon ), where epsilon = 1e-8
* Frequency CNN Stem: The 16-channel sub-band stack W_tilde in R^{B x 16 x 64 x 64} passes through a 3-stage convolutional hierarchy:
  * Stage 1: Conv2D(16 -> 64, k=3, s=2, p=1) -> BN -> ReLU  ==> (B, 64, 32, 32)
  * Stage 2: Conv2D(64 -> 128, k=3, s=2, p=1) -> BN -> ReLU ==> (B, 128, 16, 16)
  * Stage 3: Conv2D(128 -> 256, k=3, s=2, p=1) -> BN -> ReLU ==> (B, 256, 8, 8)
  * GAP(Stage 3) ==> (B, 256) -> Linear + BN + ReLU ==> F_frequency in R^{B x 512}

### Module 3: Cross-Domain Dual-Attention Fusion (models/fusion.py)
To prevent compression-induced noise from dominating under heavy quantization, Squeeze-and-Excitation (SE) channel attention computes dynamic importance vectors:
  alpha_spatial   = sigmoid( W_s2 * ReLU( W_s1 * F_spatial ) ) in R^{B x 512}
  alpha_frequency = sigmoid( W_f2 * ReLU( W_f1 * F_frequency ) ) in R^{B x 512}
where reduction ratio r = 8, W_1 in R^{64 x 512}, W_2 in R^{512 x 64}.
Features are recalibrated via element-wise Hadamard product, concatenated, and projected:
  F_hat_spatial = F_spatial * alpha_spatial
  F_hat_freq    = F_frequency * alpha_frequency
  F_fused = Dropout_0.3( ReLU( BatchNorm1D( W_fuse * [F_hat_spatial || F_hat_freq] ) ) ) in R^{B x 512}

### Module 4: Classification Head (models/dual_branch_net.py)
  y_hat = W_cls * Dropout_0.5( F_fused ) + b_cls in R^{B x 1}
  P(Fake | I) = sigmoid( y_hat ) = 1 / (1 + exp(-y_hat))

---

## 3. Loss Function & Optimization Details

### Binary Cross-Entropy with Logits
  L_BCE = - (1 / B) * sum_{i=1}^B [ y_i * log(sigmoid(y_hat_i)) + (1 - y_i) * log(1 - sigmoid(y_hat_i)) ]
Numerically stabilized computation:
  L_i = max(y_hat_i, 0) - y_hat_i * y_i + log( 1 + exp( -|y_hat_i| ) )

### Optimizer: AdamW
  LR = 1e-4, weight decay lambda = 1e-4, beta_1 = 0.9, beta_2 = 0.999.

### Learning Rate Schedule: Cosine Annealing with Warmup
  2-epoch linear warmup from 1e-5 to 1e-4, followed by Cosine Annealing decay down to eta_min = 1e-6:
  eta_t = eta_min + 0.5 * (eta_base - eta_min) * ( 1 + cos( pi * (t - 2) / 28 ) ), for epoch t > 2

---

## 4. Benchmark Dataset Profile & Augmentation

* Benchmark: 140k Real and Fake Faces.
* Real Class: 70,000 authentic portraits from Flickr-Faces-HQ (FFHQ).
* Fake Class: 70,000 synthetic photorealistic faces from StyleGAN / StyleGAN2.
* Partition: 100,000 train (50k/50k) | 20,000 val (10k/10k) | 20,000 test (10k/10k).
* Multi-Tier JPEG Augmentation: Simulates quality factors Q in [23, 100] on the fly during training.

---

## 5. Hardware Specifications & Convergence Profile

* Host Compute: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), CUDA 12.4.
* Execution Framework: PyTorch 2.6.0+cu124 on Python 3.12.4 (64-bit).
* Optimization Speed: 5.10 batches/sec (163.2 images/sec) with Automatic Mixed Precision.
* Convergence Progression:
  * Epoch 1: Train Loss = 0.5120 | Val Loss = 0.4124 | Val Acc = 81.45% | Val AUC = 0.8872
  * Epoch 2: Train Loss = 0.3621 | Val Loss = 0.2984 | Val Acc = 87.90% | Val AUC = 0.9340
  * Epoch 7: Train Loss = 0.0531 | Val Loss = 0.0211 | Val Acc = 99.29% | Val AUC = 0.9999

---

## 6. Empirical Test Results (20,000 Unseen Test Images)

* Raw (c0, Quality 100):
  * Accuracy: 99.52%
  * AUC-ROC: 1.0000
  * Precision: 0.9911
  * Recall: 0.9994
  * F1-Score: 0.9952

* Medium JPEG (c23, Quality 23):
  * Accuracy: 97.94%
  * AUC-ROC: 0.9990
  * Precision: 0.9645
  * Recall: 0.9953
  * F1-Score: 0.9797

* Heavy JPEG (c40, Quality 40):
  * Accuracy: 98.51%
  * AUC-ROC: 0.9996
  * Precision: 0.9730
  * Recall: 0.9979
  * F1-Score: 0.9853

---

## 7. Thesis Figures Summary (Generated in ./figures/)

* roc_curves.png / .pdf: Multi-tier ROC curves for c0, c23, c40 hugging the top-left boundary (AUC: 1.0000, 0.9990, 0.9996).
* confusion_matrices.png / .pdf: 1x3 normalized matrices showing raw counts and percentages. True Positive rates exceed 99.5%, False Negative rates below 0.5% under all tiers.
* score_distributions.png / .pdf: Bimodal probability histograms showing real images cluster at [0.00, 0.05], fake faces at [0.95, 1.00].
* metrics_comparison_bar.png / .pdf: Grouped bar chart showing horizontal stability of Accuracy, Precision, Recall, and F1 across c0, c23, and c40.

---

## 8. Two-Stage Inference Pipeline (predict.py)

1. Automatic Face Localization: Scans input images for face bounding boxes [x, y, w, h].
2. Contextual Expansion: Expands boxes by a 20% margin to incorporate hairline and jawline biomarkers.
3. Aspect-Ratio Preserving Fallback: If no face box is detected (e.g. tightly cropped portraits), it dynamically extracts the upper-center square to ensure facial features remain unstretched.
4. Visual Annotation: Overlays colored bounding boxes (Green = REAL, Red = FAKE) with confidence scores.

---

## 9. Operational Scope & Inherent Limitations

* Specialized Scope: Formulated strictly as a Human Facial Forensic Detector. Optimized to detect synthetic human facial artifacts and is intentionally not an open-domain AI illustration detector.
* GAN vs. Diffusion Artifacts: Excels on GAN upsampling artifacts (transposed convolutions). Stylized art or diffusion illustrations lack GAN grid lattice signatures and fall outside the human facial training manifold.
* Frame-Level Temporal Constraint: Evaluates single static images; does not model inter-frame optical flow.
