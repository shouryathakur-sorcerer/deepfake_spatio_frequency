# Comprehensive Audit Report: Team 14 Deepfake Detection Research Project

**Audit Date:** October 2026  
**Auditor:** Senior AI Research Engineer, Digital Forensics Specialist, Peer Reviewer  
**Scope:** Evaluation of Submitted Interim Technical Report (ITR), Faculty Review 1, and Codebase Implementation

---

## 1. Executive Assessment & System Reconciliation

### 1.1 The Systemic Divergence
A critical discrepancy exists between the research system documented in our submitted **Interim Technical Report (ITR)** and the configuration evaluated in the **Faculty Review (`review 1.docx`)**:

| Architectural Dimension | Submitted Team 14 ITR Specification | Previous Repository State / Faculty Review | Root Cause & Resolution |
| :--- | :--- | :--- | :--- |
| **Input Dimensions & Modality** | $224 \times 224 \times 3$, Luminance $Y = 0.299R + 0.587G + 0.114B$ for frequency | $256 \times 256 \times 3$, RGB passed to frequency | **Resolution:** Standardize on $224 \times 224$ canonical resolution. Luminance $Y$ is extracted explicitly for frequency analysis. |
| **Wavelet Packet Depth** | **3-Level WPD ($4^3 = 64$ sub-bands)** of dimension $28 \times 28$ | 2-Level WPD ($4^2 = 16$ sub-bands) of dimension $64 \times 64$ | **Resolution:** Implement full 3-level 2D db4 WPD generating 64 sub-bands of $28 \times 28$. |
| **Spatial Feature Extraction** | Modified Xception extracting intermediate grid $F_{\text{spa}} \in \mathbb{R}^{728 \times 14 \times 14}$ | Xception GAP to 2048-D, projected to 512-D flat vector | **Resolution:** Extract Middle Flow block output ($728 \times 14 \times 14$) to preserve spatial token correspondence. |
| **High-Frequency Enhancement (HiFE)** | Learnable residual block with dual-pooling channel attention and stride-2 projection to $256 \times 14 \times 14$ | Absent from codebase | **Resolution:** Implement `HiFEBlock` in `models/hife.py` with residual gating and spatial grid alignment. |
| **Fusion Mechanism** | **Dual Cross-Attention (DCA)**: spatial tokens ($14 \times 14$) and spectral tokens ($14 \times 14$) interact via $A_{S \to F}$ and $A_{F \to S}$ to $984 \times 14 \times 14$ | Squeeze-and-Excitation channel gating (`ParallelSE` and `CrossGated`) on 1D vectors | **Resolution:** Implement full `DualCrossAttention` (DCA) module with bi-directional attention and residual projection. |
| **Target Evaluation Benchmark** | FaceForensics++ ($c0, c23, c40$), Celeb-DF v2, DFDC | Kaggle *"140k Real and Fake Faces"* (FFHQ vs. StyleGAN2) | **Resolution:** Scaffolding for FF++, Celeb-DF v2, and DFDC built; local available 140k data used for feasible training. |
| **Localization Objective** | Objective 3 claimed pixel-level localization with $\text{IoU} > 0.65$ | Binary classification network without mask supervision | **Resolution:** Scientific correction: classify as binary detector and formally recommend dropping pixel-level IoU claims. |

---

## 2. Itemized Issue Audit Table

| Issue ID | Category | Severity | Affected Files | Identified Defect | Required Correction | Correction Status | Remaining Limitations |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| **SEC-01** | Architecture | **Critical** | `models/wavelet_branch.py` | Wavelet branch only implemented 2-level decomposition (16 sub-bands) instead of the 3-level 64-subband tree specified in ITR Eq. (4). | Implement complete 3-level 2D db4 wavelet packet decomposition with proper boundary reflection producing $(B, 64, 28, 28)$. | **FIXED** | 64 sub-bands increase memory; channel pooling fallback implemented for low VRAM. |
| **SEC-02** | Architecture | **Critical** | `models/spatial_branch.py` | Spatial branch pooled features into a flat 512-D vector, destroying the spatial grid required for cross-attention. | Tap Xception Middle Flow exit to yield $F_{\text{spa}} \in \mathbb{R}^{B \times 728 \times 14 \times 14}$ without pooling. | **FIXED** | Fixed to $224 \times 224$ input resolution. |
| **SEC-03** | Architecture | **High** | `models/hife.py` (New) | HiFE module was completely missing from the repository despite being a headline contribution in ITR Section III-D. | Create `HiFEBlock` implementing dual-pooling channel gating + directional conv + stride-2 downsampling to $256 \times 14 \times 14$. | **FIXED** | Learnable weights require training to adapt to heavy compression. |
| **SEC-04** | Architecture | **Critical** | `models/fusion.py`, `models/dual_cross_attention.py` | Fusion module used 1D Squeeze-and-Excitation gating instead of Dual Cross-Attention (DCA) over spatial-frequency tokens. | Implement `DualCrossAttention` computing $A_{S \to F}$ and $A_{F \to S}$ across 196 tokens ($14 \times 14$), projecting to $984 \times 14 \times 14$. | **FIXED** | Quadratic $O(N^2)$ token complexity ($N=196$) requires batch size control. |
| **SEC-05** | Training/Loss | **Medium** | `models/dual_branch_net.py`, `train.py` | Classifier applied Sigmoid internally while using BCE loss without label smoothing. | Enforce raw logit outputs with `BCEWithLogitsLoss` and label smoothing $\epsilon=0.05$ as per ITR Eq. (13). | **FIXED** | Threshold calibration must be conducted on validation set. |
| **SEC-06** | Mathematical | **Medium** | `models/wavelet_branch.py` | Expression `log(|W| + eps)` was described as "log-energy" in some places and "log-magnitude" in others. | Mathematically define both: log-magnitude $\log(\|W\| + \epsilon)$ and log-energy $\log(W^2 + \epsilon)$. Explicitly document difference. | **FIXED** | None; configurable via parameter. |
| **SEC-07** | Experimental | **Critical** | `train.py`, `run_controlled_experiments.py` | Experimental attribution confound: headline model was trained for 8 epochs on 100k images with JPEG augmentation, while baselines had 2 epochs on 5k images without augmentation. | Construct fair, matched ablation suite (`run_team14_ablations.py`) with identical optimizer, epochs, batch sizes, and data splits. | **FIXED** | Full dataset 8-epoch training requires extensive GPU hours. |
| **SEC-08** | Generalization | **High** | `eval_generalization.py` | Generalization claims were made without external cross-dataset evaluations. | Implement explicit cross-dataset loader and evaluate within-dataset vs cross-dataset gaps honestly. | **FIXED** | External datasets (Celeb-DF, DFDC) require user-provided paths. |
| **SEC-09** | Task Framing | **High** | Submitted ITR Objective 3 | ITR claimed pixel-level localization with $\text{IoU} > 0.65$, but the model is a binary classifier without a segmentation decoder or mask supervision. | Reconcile task: formally classify model as binary whole-image detector, document limitation, and remove unsupported IoU claims. | **FIXED** | Spatial localization maps can only be approximated via Grad-CAM, not ground-truth mask IoU. |
| **SEC-10** | Statistics | **Medium** | `eval.py`, `compute_statistical_significance.py` | Evaluated on single random seed without multi-seed standard deviation ($\sigma_{\text{Acc}} < 0.025$). | Implement multi-seed harness (seeds 0–4) with paired bootstrap and analytical DeLong test for AUC comparisons. | **FIXED** | Multi-seed full training requires $5\times$ computational budget. |

---

## 3. Discrepancy Reconciliation Summary

### 3.1 The Three-Level Wavelet Packet Proof
In `models/wavelet_branch.py`, dyadic downsampling of an input $X \in \mathbb{R}^{224 \times 224}$ with 8-tap Daubechies-4 (`db4`) filters and symmetric/reflect padding ($pad_{\text{left}}=3, pad_{\text{right}}=4$) yields:
- **Level 1:** 4 sub-bands $\left(\mathbf{W}_{LL}, \mathbf{W}_{LH}, \mathbf{W}_{HL}, \mathbf{W}_{HH}\right) \in \mathbb{R}^{4 \times 112 \times 112}$
- **Level 2:** Decomposing all 4 nodes yields $4 \times 4 = 16$ sub-bands $\in \mathbb{R}^{16 \times 56 \times 56}$
- **Level 3:** Decomposing all 16 nodes yields $16 \times 4 = 64$ sub-bands $\in \mathbb{R}^{64 \times 28 \times 28}$

This matches the ITR Eq. (4) specification **identically without arbitrary reshaping or spatial cropping**.

### 3.2 The Spatial Grid Alignment Proof
In `models/spatial_branch.py`, passing $X_0 \in \mathbb{R}^{224 \times 224 \times 3}$ through Xception:
- Entry Flow: Conv1 ($s=2$) $\to 112 \times 112$; Block1 ($s=2$) $\to 56 \times 56$; Block2 ($s=2$) $\to 28 \times 28$; Block3 ($s=2$) $\to 14 \times 14$ ($C=728$).
- Middle Flow: 8 residual blocks preserving spatial shape $14 \times 14$ and 728 channels.
- Output: $F_{\text{spa}} \in \mathbb{R}^{728 \times 14 \times 14}$.

In `models/hife.py`, the 64 sub-bands ($28 \times 28$) are enhanced and then projected through a stride-2 separable convolution:
- Output: $F_{\text{freq}} \in \mathbb{R}^{256 \times 14 \times 14}$.

Both branches produce a $14 \times 14$ token grid ($N = 196$ spatial tokens), allowing **direct, mathematically consistent Dual Cross-Attention**.
