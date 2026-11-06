# Comprehensive Codebase, Methodology, and SOTA Gap Analysis

**Project Title:** Deepfake Image Detection Using Multi-Level Frequency Analysis  
**Repository Working Directory:** `c:\Users\Shourya Thakur\OneDrive\Documents\Watermark_Code\deepfake_spatio_frequency`  
**Target Venue:** Tier-1 / Q1 Academic Journal (e.g., *IEEE Transactions on Information Forensics and Security (T-IFS)*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, *IEEE Transactions on Circuits and Systems for Video Technology (T-CSVT)*, or *Pattern Recognition*)  
**Audit Date:** October 2026  
**Auditor:** Principal AI Research Engineer & PyTorch Specialist

---

## 1. Repository Inventory & Verification

The repository was systematically audited file by file. Below is the verified physical inventory and actual operational behavior of each component.

| Component / File Path | Actual Role in Codebase | Verification Status & Findings |
|---|---|---|
| [`config.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/config.py) | Central experiment configuration dataclass (`Config`). Defines paths, image dimensions ($256 \times 256$), batch size (32), optimizer hyperparameters (AdamW, $\text{LR}=10^{-4}$), spatial backbone (`legacy_xception`), wavelet parameters (`db4`, Level 2), and compression mappings. | **Verified.** Auto-detects Kaggle dataset path `./data/_raw/real_vs_fake/real-vs-fake`. Defines compression map `{"c0": (100, 100), "c23": (23, 23), "c40": (40, 40)}`. |
| [`dataset.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/dataset.py) | PyTorch `Dataset` (`DeepfakeDataset`) and DataLoader factory (`build_dataloaders`). Discovers image paths from subdirectories `train/`, `val/` (or `valid/`), and `test/`. Implements Albumentations augmentation pipeline with JPEG compression simulation. | **Verified.** Correctly loads image via OpenCV (`BGR \to RGB`), normalizes via ImageNet stats. Implements defensive 0-byte skipping and unreadable image fallback. |
| [`models/__init__.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/__init__.py) | Package initialization exposing public API: `SpatialBranch`, `WaveletBranch`, `CrossDomainFusion`, `DualBranchNet`. | **Verified.** Clean imports without circular dependencies. |
| [`models/spatial_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/spatial_branch.py) | Spatial context extractor wrapping `timm.create_model('legacy_xception', pretrained=True, features_only=True, out_indices=(-1,))`. Pools via `AdaptiveAvgPool2d(1)` and projects to 512-D via `Linear(2048, 512) -> BatchNorm1d -> ReLU`. | **Verified.** Input $(B, 3, 256, 256) \to$ Output $F_{\text{spatial}} \in \mathbb{R}^{B \times 512}$. Gradient flow is continuous throughout the backbone. |
| [`models/wavelet_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/wavelet_branch.py) | Frequency branch. De-normalizes RGB tensor, converts to grayscale via ITU-R BT.601, transfers batch to CPU NumPy array, computes 2D WPD Level 2 using PyWavelets (`pywt.WaveletPacket2D`, `wavelet='db4'`), applies $\log(|W| + 10^{-8})$, stacks sub-bands, converts back to GPU tensor, passes through a 3-stage Conv-BN-ReLU CNN stem and linear projection to 512-D. | **Verified with defect.** Input $(B, 3, 256, 256) \to$ Output $F_{\text{frequency}} \in \mathbb{R}^{B \times 512}$. **Critical finding:** CPU NumPy conversion breaks autograd graph (non-differentiable front-end). Actual sub-band spatial shape is $(69 \times 69)$, not $(64 \times 64)$ as documented. |
| [`models/fusion.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/fusion.py) | Feature fusion module. Implements two `_SEBlock(512, reduction=8)` modules to produce gating weights $\alpha_s \in \mathbb{R}^{B \times 512}$ and $\alpha_f \in \mathbb{R}^{B \times 512}$. Computes element-wise product, concatenates to 1024-D, and projects via `Linear(1024, 512) -> BatchNorm1d -> ReLU -> Dropout(0.3)`. | **Verified with critical conceptual defect.** The two SE blocks operate **strictly independently**. $\alpha_s = f(F_{\text{spatial}})$ and $\alpha_f = f(F_{\text{freq}})$. There is **zero cross-domain conditioning** in the attention computation. Claim of "cross-domain attention" is mathematically inaccurate. |
| [`models/dual_branch_net.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/dual_branch_net.py) | Unified network combining `SpatialBranch`, `WaveletBranch`, `CrossDomainFusion`, and classification head (`Dropout(0.5) -> Linear(512, 1)`). | **Verified.** Outputs raw scalar logit per sample. Forward pass verified on $(2, 3, 256, 256) \to (2, 1)$. |
| [`utils.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/utils.py) | Utilities for deterministic seeding (`seed_everything`), scikit-learn metric calculation (`compute_metrics`: accuracy, precision, recall, F1, ROC-AUC), checkpoint saving/loading, and parameter counting. | **Verified.** Total trainable parameters verified: 23,025,833. `compute_metrics` relies on hard threshold 0.5. |
| [`train.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/train.py) | Complete training loop with `BCEWithLogitsLoss`, `AdamW`, `LinearLR` warmup (2 epochs) + `CosineAnnealingLR` decay, PyTorch AMP (`torch.amp.autocast`), gradient scaling, gradient norm clipping (1.0), per-epoch validation, and best-AUC checkpoint persistence. | **Verified.** Trained on RTX 4060 GPU up to Epoch 7 (8th epoch), where checkpoint `best_model.pth` was recorded with validation AUC 0.999939. |
| [`eval.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/eval.py) | Robustness evaluation script. Iterates over test split under three JPEG quality factors: 100, 23, and 40 (labeled c0, c23, c40). Computes and prints tabular classification metrics. | **Verified.** Does not save serialized predictions (`y_true.npy`, `y_prob.npy`) to disk, limiting post-hoc statistical auditing without re-execution. |
| [`predict.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/predict.py) | Single-image / full-image inference pipeline. Detects face bounding boxes via OpenCV cascade (with defensive fallbacks for OpenCV 5.0 environments), expands margin by 20%, resizes to $256 \times 256$, runs inference, and writes visual bounding boxes to disk. | **Verified.** Successfully executed on test portrait `image.png` (1024x1536), outputting `REAL (100.00%)` and rendering `image_prediction.png`. |
| [`generate_thesis_plots.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/generate_thesis_plots.py) | Automated publication figure generation script. Renders multi-tier ROC curves, normalized confusion matrices, probability histograms, and grouped metric bar charts in 300 DPI PNG and vector PDF. | **Verified.** Successfully rendered all 4 figure pairs in `./figures/`. |
| [`generate_thesis_pdf.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/generate_thesis_pdf.py) | Standalone script using `matplotlib.backends.backend_pdf.PdfPages` to compile the 5-page research dossier PDF (`Thesis_Dossier_Deepfake_Spatio_Frequency.pdf`). | **Verified.** Successfully compiled 139 KB vector PDF. |
| [`prepare_data.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/prepare_data.py) | Script to download "140k Real and Fake Faces" dataset from Kaggle and split 70/15/15 into `data/faceforensics/`. | **Verified.** Downloaded Kaggle archive `data/_raw/140k-real-and-fake-faces.zip` (~4GB) and extracted to `data/_raw/real_vs_fake/real-vs-fake/`. |
| `checkpoints/best_model.pth` | Saved PyTorch model checkpoint. | **Verified.** Size: 276,816,348 bytes. Holds weights from **Epoch 7** with `best_auc = 0.999939`. |
| `data/_raw/` manifests (`train.csv`, `valid.csv`, `test.csv`) | CSV manifests provided with Kaggle 140k dataset. | **Verified.** Train: 100,000; Valid: 20,000; Test: 20,000. Verified zero overlap between splits. **Critical discrepancy:** Kaggle CSV uses `label=1` for real and `label=0` for fake; repository uses `label=0` for real and `label=1` for fake. |

---

## 2. Implementation vs. Thesis Audit

A point-by-point verification between the claimed thesis specifications and the actual code reveals several critical technical discrepancies that must be addressed prior to peer-reviewed submission.

### 2.1 Spatial Feature Extractor
* **Claim:** Pretrained Xception backbone extracts high-level spatial visual features, mapped to $F_{\text{spatial}} \in \mathbb{R}^{B \times D}$ ($D = 512$).
* **Actual Code:** [`models/spatial_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/spatial_branch.py) initializes `timm.create_model('legacy_xception', pretrained=True, features_only=True, out_indices=(-1,))`. Last stage output channels = 2048. Adaptive average pooling reduces to $(B, 2048)$, and linear projection reduces to $(B, 512)$.
* **Status:** **Satisfied.** Matches claimed design.

### 2.2 Two-Level 2D Daubechies-4 (db4) Wavelet Packet Decomposition
* **Claim:** Decomposes grayscale face crops into 16 uniform directional sub-bands of dimension $64 \times 64$ using `db4` wavelet basis.
* **Actual Code:** [`models/wavelet_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/wavelet_branch.py) lines 67–104 executes `pywt.WaveletPacket2D(data=gray_np, wavelet='db4', maxlevel=2)`. Extracts terminal leaf nodes via `wp.get_level(2, order="natural")`.
* **Discrepancy 1 (Tensor Spatial Dimensions):** Because `db4` has a filter tap length of 8, convolution with default symmetric padding yields Level 1 size $(256 + 8 - 1)//2 = 131$, and Level 2 size $(131 + 8 - 1)//2 = 69$. The actual sub-band shape is **$(69 \times 69)$**, not **$(64 \times 64)$** as stated in documentation. The CNN stem accommodates this because `AdaptiveAvgPool2d(1)` flattens the output regardless of input spatial dimensions, but writing $(64 \times 64)$ in a journal manuscript is factually incorrect.
* **Discrepancy 2 (Gradient Flow & Hardware Serialization):** In `batch_wpd`, line 150 executes `gray.detach().cpu().numpy()`, enters a Python `for` loop over batch items, calls PyWavelets on CPU, stacks NumPy arrays, and copies back to GPU tensor via `torch.from_numpy().to(device)`.
  * *Consequence A:* The decomposition is **strictly non-differentiable**. No gradients flow from the frequency branch back to the input image. (While wavelet filter coefficients are fixed, gradient isolation prevents end-to-end sensitivity analysis or input attribution methods like Integrated Gradients).
  * *Consequence B:* Massive host-device synchronization barrier. Every batch causes GPU execution to stall while CPU processes 32 WPD calculations sequentially in Python.

### 2.3 Log-Energy Sub-Band Transformation
* **Claim:** $\tilde{W} = \log(|W| + \epsilon)$ with $\epsilon = 10^{-8}$.
* **Actual Code:** [`models/wavelet_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/wavelet_branch.py) line 165: `torch.log(torch.abs(subbands_t) + 1e-8)`.
* **Status:** **Satisfied.**

### 2.4 Frequency CNN Stem
* **Claim:** Lightweight 3-stage Conv-BN-ReLU stem projecting 16 sub-bands to $F_{\text{frequency}} \in \mathbb{R}^{B \times 512}$.
* **Actual Code:** [`models/wavelet_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/wavelet_branch.py) lines 183–234: Channels progress $16 \to 64 \to 128 \to 256$, followed by `AdaptiveAvgPool2d(1)` and linear projection $256 \to 512$.
* **Status:** **Satisfied.**

### 2.5 Dual Squeeze-and-Excitation Gating vs. Claimed "Cross-Domain Attention"
* **Claim in Documentation:** *"Cross-Domain Dual-Attention Feature Fusion module that uses Channel Attention (SE-Block style) to dynamically balance $F_{\text{spatial}}$ and $F_{\text{frequency}}$ based on compression severity."*
* **Actual Code in [`models/fusion.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/fusion.py):**
  ```python
  alpha_s = self.attn_spatial(f_spatial)   # alpha_s = sigmoid(W2 * ReLU(W1 * f_spatial))
  alpha_f = self.attn_freq(f_freq)         # alpha_f = sigmoid(W2 * ReLU(W1 * f_freq))
  f_spatial_att = f_spatial * alpha_s
  f_freq_att = f_freq * alpha_f
  concat = torch.cat([f_spatial_att, f_freq_att], dim=1)
  fused = self.projection(concat)
  ```
* **Severe Methodological Discrepancy (Confirmed Defect):**
  $\alpha_s$ is strictly a function of $F_{\text{spatial}}$ alone. $\alpha_f$ is strictly a function of $F_{\text{frequency}}$ alone.
  Neither gate receives any input or signal from the opposing domain.
  Therefore, this module is **Independent Domain Self-Gating followed by Concatenation**, NOT Cross-Domain Attention.
  The central claim that *"the attention mechanism detects compression severity and automatically down-weights frequency features while shifting attention to spatial features"* is **unsupported by the computational graph**. For $\alpha_f$ to know that the frequency spectrum is corrupted by JPEG compression relative to the spatial content, $\alpha_f$ would need to be conditioned on spatial information (e.g., via multi-head cross-attention or a joint squeeze operation over $[F_{\text{spatial}} \,\|\, F_{\text{frequency}}]$). Claiming "cross-domain attention" in a Q1 submission with this implementation will result in immediate rejection by reviewers who inspect the equations or code.

### 2.6 Optimization, Loss, and Learning Rate Dynamics
* **Claim:** Trained for 30 epochs with AdamW, Cosine Annealing, and Linear Warmup.
* **Actual Code & Checkpoint:** Checkpoint `checkpoints/best_model.pth` contains state from `epoch = 7` (8th epoch). Training was halted at Epoch 8.
  * In Epoch 7, the learning rate was $9.24 \times 10^{-5}$ (the Cosine Annealing schedule had barely begun decaying towards its minimum of $10^{-6}$).
  * While the model converged to high accuracy on this specific dataset within 7 epochs, claiming a completed "30-epoch Cosine Annealing schedule" in a paper without disclosing early termination is an empirical inconsistency.

---

## 3. Dataset Quality, Leakage, and Compression Audit

### 3.1 Partition Leakage & Manifest Audit
We executed a complete hash and string intersection across all 140,000 samples listed in `data/_raw/train.csv` (100,000), `valid.csv` (20,000), and `test.csv` (20,000):
* Train $\cap$ Validation: **0 overlapping paths**
* Train $\cap$ Test: **0 overlapping paths**
* Validation $\cap$ Test: **0 overlapping paths**
* Total unique source paths: **140,000**
* Class distribution per split: Exactly 50.0% real, 50.0% fake across all three partitions.
* **Finding:** There is **zero partition overlap or identity leakage** between the training, validation, and test sets.

### 3.2 Label Inversion Discrepancy
* In Kaggle's original manifest CSVs: `label = 1` denotes **Real (FFHQ)**, and `label = 0` denotes **Fake (StyleGAN)**.
* In repository's `dataset.py`: `LABEL_MAP = {"real": 0, "fake": 1}`.
* *Audit Result:* Because `DeepfakeDataset._discover_samples()` parses directory names (`/real/` and `/fake/`) rather than reading the CSV `label` column directly, the internal training logic is consistent ($0 = \text{Real}, 1 = \text{Fake}$). However, any researcher attempting to validate predictions against `test.csv` directly will observe inverted confusion matrices unless this discrepancy is documented.

### 3.3 Generator-Specific Shortcut Learning (The "Wang / Frank" Effect)
* **Real Source:** 100% of real images are drawn from Flickr-Faces-HQ (FFHQ), which were captured by diverse consumer cameras and pre-processed with specific bilinear downsampling to $1024 \times 1024$ and JPEG/PNG compression profiles.
* **Fake Source:** 100% of fake images are drawn from a single Kaggle export ("1 Million Fake Faces"), generated by StyleGAN/StyleGAN2.
* **Critical Forensic Risk:** Pioneering studies by Wang et al. (*CVPR 2020*) and Frank et al. (*ICML 2020*) demonstrated that neural networks trained to classify FFHQ versus StyleGAN readily achieve $>99\%$ accuracy not because they learn generalized forgery clues, but because they memorize:
  1. The specific transposed-convolution filter grid periodicity of StyleGAN.
  2. Subtly distinct color gamut and clipping profiles between FFHQ and StyleGAN's synthesis pipeline.
  3. Preprocessing/resampling filter traces present in one collection but absent in the other.
* **Audit Finding:** The 99.52% test accuracy achieved on this dataset is an **in-distribution generator-specific metric**. It does not prove that the model can detect deepfakes from other generators. When the user tested a stylized image or an image from another generative distribution, the model failed. For a Q1 journal, evaluation on unseen generators (Diffusion models, GAN architectures other than StyleGAN, and identity-swap deepfakes) is mandatory.

### 3.4 Compression Protocol Audit: The H.264 CRF vs. JPEG Quality Inversion
The project documentation, scripts, and reports label evaluation tiers as **`c0`**, **`c23`**, and **`c40`**, asserting:
* `c0`: Raw / uncompressed (Quality = 100)
* `c23`: Medium compression (Quality = 23)
* `c40`: Heavy compression (Quality = 40)

#### The Conceptual and Mathematical Flaw:
1. **Origin of Terminology:** The labels `c0`, `c23`, `c40` were established by FaceForensics++ (Rossler et al., *ICCV 2019*) to denote **H.264 video compression** parameterized by Constant Rate Factor (CRF):
   * CRF 0: Raw lossless video.
   * CRF 23: Visually near-lossless video compression (H.264 default).
   * CRF 40: Severe, highly degraded video compression (higher CRF = higher quantization = worse quality).
2. **The Project's Implementation:** This repository operates on static images and applies **JPEG image compression** via Albumentations:
   * `c0` maps to JPEG Quality Factor $Q = 100$.
   * `c23` maps to JPEG Quality Factor $Q = 23$.
   * `c40` maps to JPEG Quality Factor $Q = 40$.
3. **The Inversion Error:** In standard JPEG quantization (ITU-T T.81 / IJG):
   * Quality Factor $Q \in [1, 100]$ represents image fidelity.
   * $Q = 100$ is maximum fidelity (minimal quantization).
   * $Q = 40$ represents moderate-to-heavy compression.
   * **$Q = 23$ represents substantially heavier compression than $Q = 40$!**
   * Quantization scale factor in IJG JPEG: For $Q < 50$, scale factor $S = 5000 / Q$. Thus:
     $$S_{Q=40} = \frac{5000}{40} = 125, \quad S_{Q=23} = \frac{5000}{23} \approx 217.4$$
     Quantization step sizes at $Q = 23$ are **$1.74 \times$ larger** than at $Q = 40$.
4. **Impact on Empirical Results:**
   Notice the reported test results:
   * $Q = 100$ ($c0$): Accuracy = **99.52%**
   * $Q = 40$ (labeled $c40$): Accuracy = **98.51%**
   * $Q = 23$ (labeled $c23$): Accuracy = **97.94%**
   The model's performance on $Q = 23$ is lower than on $Q = 40$ precisely because **$Q = 23$ is the more degraded condition**!
   Yet the documentation presents $c23$ as "medium" and $c40$ as "heavy".
5. **Reviewer Impact:** Presenting JPEG compression under FaceForensics++ video CRF terminology and claiming $Q=40$ is heavier compression than $Q=23$ will result in immediate negative reviewer feedback in a peer-reviewed journal.

---

## 4. Current State-of-the-Art (SOTA) Literature Comparison

To determine where this work sits in the literature, we audited eight leading peer-reviewed spatial-frequency deepfake detection papers published between 2020 and 2026.

| Method / Reference | Publication Venue & Year | Core Technical Contribution | Frequency Representation Used | Datasets & Generators Evaluated | Compression & Robustness Protocol | Overlap with Current Project | Gap / What This Project Must Prove |
|---|---|---|---|---|---|---|---|
| **F3-Net** (Qian et al.) | ECCV 2020 ([Paper](https://arxiv.org/abs/2007.09355)) | Two-stream collaborative frequency mining: Frequency-Aware Decomposition (FAD) + Local Frequency Statistics (LFS) with MixBlock interaction. | Discrete Cosine Transform (DCT) split into low-, mid-, high-frequency filters. | FaceForensics++, Celeb-DF, DeepfakeDetection. | Evaluated on real FF++ H.264 video compression ($c0, c23, c40$). | Dual-stream architecture; frequency decomposition combined with spatial features. | F3-Net uses genuine multi-level interactive MixBlocks across depths; this project uses late concatenation of independent SE embeddings. |
| **SPSL** (Liu et al.) | CVPR 2021 ([Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Spatial-Phase_Shallow_Learning_Rethinking_Face_Forgery_Detection_in_the_CVPR_2021_paper.html)) | Spatial-Phase Shallow Learning. Identifies that upsampling perturbations severely distort Fourier phase spectra; uses shallow CNNs to prevent high-level semantic overfitting. | Fast Fourier Transform (FFT) phase spectrum extraction. | FaceForensics++, Celeb-DF, WildDeepfake. | Cross-dataset evaluation, video compression ($c23, c40$). | Focuses on upsampling grid artifacts produced by generative models. | SPSL explicitly isolates phase information; this project isolates wavelet sub-band energy. SPSL evaluated cross-dataset transfer. |
| **SFDG** (Tan et al.) | IEEE T-IFS 2023 ([Paper](https://ieeexplore.ieee.org/document/10183827/)) | Spatial-Frequency Dynamic Graph Learning. Content-guided frequency extraction + dynamic graph reasoning for cross-domain spatial-frequency relation modeling. | Adaptive frequency mining + graph message passing. | FaceForensics++, Celeb-DF, DFDC. | Severe video compression, Gaussian blur, noise corruptions. | Aims at relation reasoning between spatial and frequency domains. | SFDG establishes true cross-domain relation modeling via graphs. This project claims cross-domain attention but computes independent channel gates. |
| **FreqNet** (Wang et al.) | AAAI / CVPR 2024 ([Paper](https://arxiv.org/abs/2401.07432)) | Universal frequency-aware detector learning source-agnostic high-frequency representations. Combines High-Frequency Representation of Images (HFRI), Features (HFRF), and Frequency Conv Layer (FCL) on amplitude and phase. | FFT amplitude + phase convolutional layer. | ProGAN, StyleGAN, StyleGAN2, BigGAN, Diffusion models. | Cross-generator transfer across 10+ distinct synthesis algorithms. | Frequency-domain representation designed to detect generative upsampling artifacts. | FreqNet evaluated cross-generator transfer across multiple GANs and diffusion models; this project evaluated only StyleGAN vs FFHQ. |
| **FSBI** (Sun et al.) | Image & Vision Computing (IVC) 2024 ([Paper](https://arxiv.org/abs/2405.07183)) | Frequency-Enhanced Self-Blended Images. Generates synthetic training pseudo-forgeries and uses DWT to expose boundary irregularities. | Discrete Wavelet Transform (DWT, `sym2`). | FaceForensics++, Celeb-DF-v2, CDF-v1. | Unseen manipulation generalization, cross-dataset benchmarks. | Uses wavelet transforms to expose subtle manipulation boundaries. | FSBI operates as a self-supervised data generation + DWT framework; this project relies on supervised binary classification of full synthetic faces. |
| **DF40** (Yan et al.) | NeurIPS 2024 ([Paper](https://arxiv.org/abs/2406.12685)) | Benchmark and taxonomy evaluating detectors across 40 contemporary manipulation methods (GANs, DiT, SiT, Midjourney, face swaps, reenactment). | Benchmark analyzing spatial, frequency, and artifact degradation. | 40 deepfake generation methods, >1M images/videos. | Standardized evaluation across multiple corruptions and unseen architectures. | Represents the modern benchmark against which any 2025–2026 detection paper is judged. | This project evaluates only 1 manipulation type (StyleGAN full-synthesis). Needs evaluation on a subset of DF40 or GenImage. |
| **WaveDIF** (Goel et al.) | CVPR Workshops 2022 ([Paper](https://openaccess.thecvf.com/content/CVPR2022W/WMF/html/Goel_WaveDIF_Wavelet_Sub-Band_Based_Deepfake_Identification_in_Frequency_Domain_CVPRW_2022_paper.html)) | Wavelet sub-band classification using Haar wavelet filters to isolate high-frequency noise in sub-bands (LL, LH, HL, HH). | 2D DWT (Haar wavelet sub-bands). | FaceForensics++, Celeb-DF. | Video compression ($c23, c40$). | Direct use of 2D wavelet sub-band energy for deepfake classification. | WaveDIF uses standard 4-subband DWT; this project uses 16-subband WPD. Project must prove WPD yields measurable gain over DWT. |
| **Wang et al.** (Wang et al.) | CVPR 2020 ([Paper](https://openaccess.thecvf.com/content/CVPR2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot_..._for_Now_CVPR_2020_paper.html)) | Foundational paper showing CNN-generated images share common systemic artifacts regardless of architecture. | Spatial ResNet-50 with heavy data augmentation (blur, JPEG). | ProGAN, StyleGAN, BigGAN, CycleGAN, DeepFakes. | Multi-level JPEG compression and Gaussian blurring. | Baseline study demonstrating that compression augmentation prevents frequency shortcut memorization. | Wang et al. already proved that training with JPEG augmentation achieves high cross-compression robustness. |

### Literature Synthesis & Positioning:
1. **The Task Taxonomy Must Be Kept Distinct:**
   * *Whole-Face Synthesis Detection:* Classifying real humans (FFHQ) vs fully synthesized faces (StyleGAN, Diffusion). This is the task this repository evaluates.
   * *Facial Manipulation / Identity-Swap Detection:* Classifying authentic video frames vs face-swapped/reenacted faces (FaceForensics++, Celeb-DF, DF40).
   * *Open-Domain AI Image Detection:* Classifying arbitrary scenes, illustrations, and objects (GenImage, DiffusionDB).
   * **Mandatory Action:** The thesis and manuscript must explicitly define the paper as addressing **whole-face synthesis forensic detection**. It must not cite FaceForensics++ numbers as direct apples-to-apples baseline comparisons without clarifying that FF++ evaluates face swapping, not full StyleGAN synthesis.

---

## 5. Research Novelty & Contribution Assessment

### 5.1 Component-Level Novelty Audit

| Architectural Component | Prior Literature Disclosure | Is It Novel in Isolation? | Verdict |
|---|---|---|---|
| **Xception Backbone** | Standard baseline in deepfake detection since Rossler et al. (*ICCV 2019*), F3-Net (*ECCV 2020*), Face X-ray (*CVPR 2020*). | **No.** | Standard off-the-shelf spatial extractor. |
| **2D Wavelet Decomposition** | Disclosed in WaveDIF (*CVPRW 2022*), FSBI (*IVC 2024*), WGN (*CVPRW 2023*), and dozens of forensic papers since 2020. | **No.** | Established signal processing tool. |
| **Wavelet Packet Decomposition (WPD)** | Explored in texture classification and steganalysis; sparsely applied to deepfakes compared to standard DWT. | **Moderate novelty.** | Using full 16-subband WPD rather than 4-subband DWT is technically defensible if an ablation proves WPD outperforms DWT. |
| **Log-Energy Transform $\log(|W| + \epsilon)$** | Standard in audio spectrogram processing, speech recognition, and seismic signal analysis. | **No.** | Standard dynamic range compression technique. |
| **CNN Stem for Sub-Bands** | Standard practice for consuming multi-channel frequency tensors (e.g. F3-Net FAD stream, WaveDIF). | **No.** | Standard engineering implementation. |
| **Independent SE Gating** | Hu et al. (*CVPR 2018*). Independent channel attention applied to two parallel streams. | **No.** | Off-the-shelf attention block. |
| **Late Concatenation & Linear Projection** | Standard multimodal and multi-stream fusion convention. | **No.** | Standard engineering fusion. |

### 5.2 Overall Novelty Classification
* **Novelty Type:** **Incremental Architectural Assembly with Empirical Focus.**
* **Honest Evaluation:** The combination of Xception + 2D WPD + independent SE gating + late concatenation does **not** represent a fundamental theoretical breakthrough. Each individual component is standard.
* **The Path to a Credible Q1 Contribution:**
  In top journals (IEEE T-IFS, T-CSVT), an incremental architecture can be accepted **only if** supported by:
  1. **Rigorous, clean ablation studies** proving that 2D WPD significantly outperforms standard DWT and FFT under equal parameter budgets.
  2. **A genuine cross-domain attention mechanism** where spatial and frequency branches truly interact and condition each other's gating weights.
  3. **Cross-generator and cross-dataset generalization experiments** demonstrating that the learned spatio-frequency representation transfers to unseen synthesis architectures (e.g., Diffusion models, ProGAN, StarGAN).
  4. **Precise, scientifically honest claims** regarding compression resilience without misleading nomenclature.

---

## 6. Required Experimental Baselines & Controlled Ablations

Currently, the repository contains only **one trained model** (`DualBranchNet` on FFHQ vs StyleGAN). To satisfy Q1 peer-review standards, the following controlled experimental suite is required.

### 6.1 Architectural Baselines (Mandatory for Comparative Claims)

| Baseline ID | Model Variant | Configuration / Implementation Details | Purpose in Paper |
|---|---|---|---|
| **B1** | **Spatial Branch Alone** | Pretrained Xception + GAP + Linear(2048, 512) + Classifier. Wavelet branch disabled. | Quantifies spatial baseline accuracy and demonstrates how much the spatial stream degrades under compression without frequency features. |
| **B2** | **Wavelet Branch Alone** | 2D WPD (`db4`, Level 2) + log-energy + CNN Stem + Classifier. Spatial branch disabled. | Demonstrates whether pure directional frequency energy is sufficient for detection. |
| **B3** | **Dual-Branch (Direct Concat)** | Spatial Branch + Wavelet Branch concatenated directly ($[F_{\text{spatial}} \,\|\, F_{\text{frequency}}]$) without SE gating. | Isolates the exact empirical contribution of channel attention gating versus plain concatenation. |
| **B4** | **Proposed Cross-Gated Model** | Spatial Branch + Wavelet Branch with true cross-domain attention/gating. | The proposed method. |
| **B5** | **FFT / DCT Baseline** | Replace WPD with 2D Fast Fourier Transform magnitude/phase or standard DCT (e.g., F3-Net style FAD). | Proves whether localized wavelet packets provide higher discriminability than global Fourier/DCT transforms. |

### 6.2 Controlled Ablation Studies

| Ablation ID | Hypothesis | Independent Variable | Controlled Variables | Metrics Reported |
|---|---|---|---|---|
| **A1: WPD vs. DWT** | Decomposing high-frequency sub-bands (WPD, 16 sub-bands) captures upsampling artifacts better than decomposing only low frequencies (DWT, 4 sub-bands). | Wavelet Transform Type: 2D WPD ($4^2 = 16$ sub-bands) vs. 2D DWT (4 sub-bands: LL, LH, HL, HH). | Same basis (`db4`), same stem parameter budget, same dataset, same training epochs. | Acc, AUC, F1 across $c0, c23, c40$. |
| **A2: Wavelet Basis Family** | Higher-order vanishing moments (`db4`, `sym4`) yield cleaner sub-band boundary separation than orthogonal Haar wavelets. | Wavelet Basis: `db4` vs. `sym4` vs. `haar` (`db1`) vs. `bior2.2`. | Level 2 WPD, same stem, same training seed. | Acc, AUC, F1 across $c0, c23, c40$. |
| **A3: Decomposition Level** | Level 2 ($16$ sub-bands) balances directional granularity and spatial resolution better than Level 1 ($4$ sub-bands) or Level 3 ($64$ sub-bands). | Decomposition Depth: $L = 1$ (4 sub-bands), $L = 2$ (16 sub-bands), $L = 3$ (64 sub-bands). | Same basis (`db4`), matched stem capacity. | Acc, AUC, Parameters, Inference Latency. |
| **A4: Log-Energy Transform** | $\log(|W| + \epsilon)$ prevents large gradient spikes from high-frequency sub-bands and improves convergence under compression. | Energy Transform: Log-Energy $\log(|W| + \epsilon)$ vs. Raw Absolute $|W|$ vs. Squared Energy $|W|^2$ vs. Standard Linear $W$. | Identical architecture and training hyperparameters. | Training loss convergence rate, Val Acc, Test AUC. |
| **A5: Compression Augmentation** | On-the-fly JPEG augmentation during training is the primary driver of compression robustness at test time. | Training Augmentation: With JPEG augmentation ($Q \in [23, 100]$) vs. Without JPEG augmentation (clean crops only). | Identical architecture, evaluated on identical test tiers (Raw, $Q=40$, $Q=23$). | Performance delta $\Delta(Q=100 \to Q=23)$. |

### 6.3 Generalization Benchmarks (Unseen Generators & Cross-Dataset)
To prevent rejection on grounds of "FFHQ vs StyleGAN shortcut memorization", the model must be tested (in zero-shot inference, without retraining) on:
1. **Unseen GANs:** ProGAN, StarGAN, BigGAN (available in Wang et al. benchmark).
2. **Diffusion Models:** Latent Diffusion / Stable Diffusion (SD v1.5, SDXL), Midjourney v5/v6 (available in GenImage or DiffusionDB face subsets).
3. **Face Swaps (Optional but valuable):** FaceForensics++ (Deepfakes, FaceSwap) or Celeb-DF-v2.

---

## 7. Results Verification & Empirical Audit

### 7.1 Existing Results vs. Verification Evidence
* **Reported Thesis Numbers:**
  * Raw ($c0$, Quality 100): Accuracy = **99.52%**, AUC-ROC = **1.0000**, Precision = **99.11%**, Recall = **99.94%**, F1 = **99.52%**
  * Medium ($c23$, Quality 23): Accuracy = **97.94%**, AUC-ROC = **0.9990**, Precision = **96.45%**, Recall = **99.53%**, F1 = **97.97%**
  * Heavy ($c40$, Quality 40): Accuracy = **98.51%**, AUC-ROC = **0.9996**, Precision = **97.30%**, Recall = **99.79%**, F1 = **98.53%**
* **Verification Audit:**
  * The terminal screenshot provided by the user confirms that these numbers were output directly by running `eval.py` with `checkpoints/best_model.pth`.
  * The checkpoint inspection confirms:
    `epoch: 7`, `best_auc: 0.9999390739073908`.
  * The figures in `./figures/` (`roc_curves.png`, `confusion_matrices.png`, `score_distributions.png`, `metrics_comparison_bar.png`) match these exact statistics.
* **Limitations of Evidence:**
  * Raw prediction vectors (`y_true.npy`, `y_prob.npy`) were not saved to disk.
  * `logs/` directory contains no text logs or TensorBoard/WandB event files.
  * While the figures and checkpoint confirm the metrics occurred, full auditability requires serialized prediction arrays and evaluation logs.

### 7.2 Metric Calculation Sanity Check
* Line 60 of [`utils.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/utils.py):
  `y_pred = (y_prob >= threshold).astype(np.int32)` with default `threshold = 0.5`.
* Accuracy, precision, recall, and F1 are calculated at threshold 0.5.
* ROC-AUC is calculated using `sklearn.metrics.roc_auc_score(y_true, y_prob)`.
* Because the dataset test split is balanced (10,000 real, 10,000 fake), accuracy and macro F1 are not artificially inflated by class imbalance.

---

## 8. Actionable Gap-Analysis Table

Below is the comprehensive audit matrix classifying every issue found across the repository, methodology, and experimental protocol.

| Area | Existing implementation or evidence | SOTA requirement | Gap found | Exact file or experiment affected | Recommended action | Priority |
|---|---|---|---|---|---|---|
| **Architecture / Fusion** | Independent SE gates: $\alpha_s = f(F_s)$, $\alpha_f = f(F_f)$ in [`models/fusion.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/fusion.py). | True cross-modal interaction where spatial cues inform frequency gating and vice versa. | **Confirmed defect.** Claim of "cross-domain attention" is mathematically false; gates are completely decoupled. | [`models/fusion.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/fusion.py), lines 143–154 | Implement true cross-attention (e.g. multi-head cross-attention or joint cross-gating $[F_s \,\|\, F_f] \to (\alpha_s, \alpha_f)$). | **P0** |
| **Methodology / Compression** | Labeled as $c0, c23, c40$, implemented as JPEG Quality Factors $100, 23, 40$. | Clear distinction between H.264 video CRF and JPEG Image Quality Factor; correct ordering of degradation severity. | **Confirmed defect.** Inverted compression severity (JPEG Q=23 is heavier compression than Q=40); video CRF notation conflated with static JPEG. | [`config.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/config.py), [`eval.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/eval.py), documentation | Rename conditions to explicit JPEG terms: `JPEG-Q100` (Uncompressed), `JPEG-Q50` (Moderate), `JPEG-Q23` (Severe). Re-order tables monotonically. | **P0** |
| **Implementation / Autograd** | PyWavelets CPU conversion in `batch_wpd` using `.detach().cpu().numpy()` and a Python `for` loop. | End-to-end GPU execution and tensor graph differentiability. | **Confirmed defect.** Breaks autograd gradient flow to input; creates severe host-device synchronization barrier. | [`models/wavelet_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/wavelet_branch.py), lines 150–163 | Implement 2D WPD using native PyTorch grouped convolutions with fixed db4 filter bank on GPU, or document CPU detachment as an intentional non-differentiable front-end. | **P1** |
| **Documentation / Geometry** | Documentation states 16 sub-bands of spatial size $64 \times 64$. | Mathematically exact reporting of tensor geometry. | **Documentation inconsistency.** Actual Level-2 `db4` WPD output on $256 \times 256$ input is $69 \times 69$ due to filter tap length 8 and symmetric padding. | [`models/wavelet_branch.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/wavelet_branch.py), `Thesis_Dossier_*.md`, `*.pdf` | Update all documentation and manuscript equations to state exact dimensions ($69 \times 69$) and explain why boundary filter taps produce this shape. | **P0** |
| **Dataset / Integrity** | In Kaggle manifest CSV: `label=1` is Real, `label=0` is Fake. In `dataset.py`: `label=0` is Real, `label=1` is Fake. | Consistency between data manifests and PyTorch DataLoader labels. | **Documentation inconsistency.** Internal training is consistent via directory names, but external CSV parsing will invert labels. | [`dataset.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/dataset.py), line 189, `data/_raw/*.csv` | Document label convention explicitly; provide a manifest generator that outputs repository-aligned CSVs. | **P3** |
| **Experimental Validity / Generalization** | Model trained and evaluated exclusively on FFHQ vs. StyleGAN/StyleGAN2. | Q1 journals require cross-generator generalization (testing on unseen GANs and Diffusion models). | **Missing experiment.** 99.5% accuracy reflects StyleGAN-specific fingerprint memorization. Zero empirical proof of general deepfake detection. | New evaluation script / experiment | Evaluate checkpoint on unseen generators: ProGAN, BigGAN, Latent Diffusion / Stable Diffusion (SD v1.5, SDXL), Midjourney. | **P1** |
| **Baseline Comparisons** | Only the proposed dual-branch model was evaluated. | Comparison against component baselines and established SOTA under identical conditions. | **Missing experiment.** No spatial-only baseline (Xception alone), no wavelet-only baseline, no simple concatenation baseline, no comparison to F3-Net/SPSL/FreqNet. | New training / evaluation runs | Train and evaluate Baselines B1 (Spatial-only), B2 (Wavelet-only), B3 (Direct Concat), and B5 (FFT/DCT). | **P1** |
| **Controlled Ablations** | No ablations executed in existing records. | Empirical validation of architectural design choices (WPD vs DWT, wavelet family, log-energy, decomposition depth). | **Missing experiment.** Claims that WPD is superior to DWT, or that db4 is superior to Haar/sym4, are currently theoretical conjectures without empirical proof. | New ablation suite | Run controlled ablations: A1 (WPD vs DWT), A2 (db4 vs sym4 vs Haar), A3 (Level 1 vs 2 vs 3), A4 (Log-energy vs raw). | **P1** |
| **Training Dynamics** | Checkpoint is from Epoch 7; Cosine Annealing schedule planned for 30 epochs was truncated. | Disclosed training completion or empirical justification for early stopping. | **Documentation inconsistency.** Documentation claims 30 epochs with Cosine Annealing; actual model was trained for only 7 epochs. | [`train.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/train.py), documentation | Explicitly report validation-loss-based early stopping at Epoch 7 with convergence curves in paper. | **P3** |
| **Reproducibility / Logging** | `eval.py` outputs only to console stdout; `logs/` directory is empty. | Persistent serialized evaluation logs and raw prediction arrays. | **Missing experiment / artifact.** Test predictions cannot be audited or re-analyzed without re-running 20,000 forward passes. | [`eval.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/eval.py) | Update `eval.py` to save `y_true.npy`, `y_prob.npy`, and `metrics.json` into `./results/`. | **P0** |
| **Statistical Reliability** | Metrics reported from a single seed (seed 42, 1 run). | Multi-run variance (mean $\pm$ standard deviation across $\ge 3$ seeds) or confidence intervals. | **Missing experiment.** Single-run results lack statistical error bounds required by top journals. | New experiment | Run 3 independent training runs with seeds 42, 43, 44 and report mean $\pm$ std for key metrics. | **P2** |
| **Task Scope Definition** | Project titled "Deepfake Detection" but exclusively detects whole-face GAN synthesis. | Precise delimitation of scope (Whole-Face Synthesis vs Face-Swap vs Inpainting). | **Documentation inconsistency.** Model fails on non-face art and face-swapping benchmarks without pre-processing. | All documentation, paper title/abstract | Explicitly define scope in abstract and title as "Facial Synthesis Detection" or "GAN-Generated Face Detection". | **P3** |
| **Dataset Leakage** | Audited 140,000 images across train, valid, test. | Zero identity or sample leakage between partitions. | **Already satisfied.** Confirmed zero path overlap between train, valid, and test sets. | `data/_raw/*.csv` | None required; keep splits verified. | **Already satisfied** |
| **Model Classification Head** | Uses `BCEWithLogitsLoss` with single logit output. | Numerically stable binary classification loss. | **Already satisfied.** Avoids sigmoid in loss; numerically stable log-sum-exp formulation. | [`models/dual_branch_net.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/models/dual_branch_net.py), [`train.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/train.py) | None required. | **Already satisfied** |
| **Inference Pipeline** | Two-stage pipeline in `predict.py` with bounding box annotations. | Full-frame inference with aspect-ratio preserving fallback. | **Already satisfied.** Successfully verified on 1024x1536 portrait `image.png`. | [`predict.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/predict.py) | None required. | **Already satisfied** |
