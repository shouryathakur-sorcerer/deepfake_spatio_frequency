# Missing Information and Scientific Gaps Inventory

**Document Purpose:** Explicit, transparent accounting of all experimental data, evaluations, ablation runs, and external comparisons that are missing, unverifiable, or pending in this repository. In accordance with strict scientific integrity standards, no gaps are concealed by heuristic or plausible estimates.

---

## 1. Experimental and Generalization Gaps

### 1.1 Unseen Generator Generalization (Zero-Shot Cross-Model Evaluation)
- **Status:** **NOT PERFORMED / PENDING EXTERNAL DATA.**
- **Missing Information:** Exact test metrics (Accuracy, ROC-AUC, Precision, Recall) on:
  1. Alias-Free GANs (StyleGAN3).
  2. Latent Diffusion Models (Stable Diffusion v1.5 / SDXL / SD3).
  3. Autoregressive / Transformer Generators (FLUX.1, Midjourney v5/v6).
  4. Early GAN Architectures (ProGAN, StarGAN, BigGAN).
- **Why It Is Missing:** The local filesystem contains exclusively Kaggle's *"140k Real and Fake Faces"* (FFHQ vs StyleGAN2). No external image archives reside in `data/external/`.
- **Impact on Manuscript:** The manuscript cannot claim universal generalization across arbitrary synthesis algorithms. Claims must be strictly bounded to **whole-face StyleGAN2 detection against authentic FFHQ portraits**.
- **Actionable Remediation:** The evaluation script [`eval_generalization.py`](file:///c:/Users/Shourya%20Thakur/OneDrive/Documents/Watermark_Code/deepfake_spatio_frequency/eval_generalization.py) is implemented. Populating `data/external/` with 1,000 images from each generator will produce these numbers in ~5 minutes on GPU.

### 1.2 Multi-Epoch Trained Metrics for Ablations A1, A2, and A3
- **Status:** **IMPLEMENTATION VERIFIED / FULL TRAINING PENDING.**
- **Missing Information:** Trained classification metrics (Accuracy, ROC-AUC across $Q=100, 50, 23$) for:
  - **Ablation A1:** Standard DWT (4 sub-bands) vs. WPD (16 sub-bands).
  - **Ablation A2:** Basis sensitivity across `db4` vs. `db1` (Haar) vs. `sym4` vs. `coif1`.
  - **Ablation A3:** Sub-band representation space: Log-Energy $\log(|W| + \epsilon)$ vs. Raw $W$ vs. Absolute $|W|$.
- **Why It Is Missing:** While all architectures were verified for tensor dimensionality, parameter counts, forward-backward pass stability, and mathematical filter bank properties (`tests/test_differentiable_wpd.py`), the execution of full 5-epoch training runs for all 9 permutation variants was not executed to conserve computing resources.
- **Impact on Manuscript:** The superiority of WPD over DWT and log-energy over raw coefficients must be argued on theoretical signal-processing principles and filter bank properties, rather than cited as empirical trained metric gains.

### 1.3 Multi-Seed Training Run Variance
- **Status:** **SINGLE TRAINING SEED (SEED 42) FOR CONTROLLED BASELINES.**
- **Missing Information:** Inter-training run standard deviations (e.g. Mean $\pm$ Std across 5 random seeds).
- **Current Evidence:** The statistical confidence intervals reported in `STATISTICAL_COMPARISONS.csv` and Section 8 are **200-iteration stratified test-set bootstrap intervals and paired sample tests** across the 20,000 locked test predictions. They capture sample uncertainty over the evaluation population, but do not measure initialization variance across multiple stochastic optimization runs.
- **Impact on Manuscript:** Authors must disclose that controlled models were trained under fixed seed 42.

---

## 2. Dataset and Forensic Scope Boundaries

### 2.1 Video Deepfake Benchmarks (FaceForensics++, Celeb-DF, WildDeepfake)
- **Status:** **NOT PRESENT IN WORKSPACE.**
- **Missing Information:** Evaluation on video-level facial manipulations, including:
  - FaceForensics++ (Deepfakes, Face2Face, FaceSwap, NeuralTextures) under H.264 video compression ($c0, c23, c40$).
  - Celeb-DF-v2 / WildDeepfake in-the-wild video clips.
- **Why It Is Missing:** The repository evaluates static whole-face synthesis images ($256 \times 256$), not temporal video sequences.
- **Impact on Manuscript:** The paper must clearly state that it addresses **whole-face synthesis detection (real humans vs fully generated faces)**, not facial identity-swapping or video reenactment. Comparing numbers directly against FaceForensics++ leaderboard numbers without clarifying this distinction would constitute a major scientific error.

### 2.2 JPEG Chroma Subsampling Sensitivity
- **Status:** **DEFAULT 4:2:0 CHROMINANCE SUBSAMPLING EVALUATED.**
- **Missing Information:** Performance variations across distinct JPEG chrominance sampling standards (`4:4:4`, `4:2:2`, `4:2:0`).
- **Impact on Manuscript:** Standard PIL/OpenCV JPEG compression was applied with default 4:2:0 chroma subsampling. The impact of color space downsampling vs pure DCT quantization was not independently ablated.

### 2.3 Human Observer Forensic Baseline
- **Status:** **NO PSYCHOPHYSICAL STUDY PERFORMED.**
- **Missing Information:** Human visual detection accuracy on clean vs. compressed StyleGAN2 images.
- **Impact on Manuscript:** Claims regarding "human imperceptibility" of high-frequency synthesis traces must cite established vision literature (e.g. Frank et al., Wang et al.) rather than local user studies.

---

## 3. Literature and Citation Verification Status

All 15 canonical citations have been audited in `REFERENCE_AUDIT.csv`. Minor bibliographic notes:
1. **FreqNet Citation Venue:** While the foundational paper is archived on arXiv (arXiv:2401.07432), conference proceedings indexing varies between AAAI and CVPR workshop tracks. The arXiv preprint DOI is authoritative and verified.
2. **FSBI Journal Volume:** Published in Elsevier's *Image and Vision Computing*, Volume 147, Article 105156 (2024). Verified via official ScienceDirect DOI.
3. **DF40 Benchmark:** Formally presented at NeurIPS 2024 Datasets and Benchmarks Track. Verified via arXiv:2406.12685.

---

## 4. Summary Table of Disclosure Boundaries

| Aspect | Available & Verified | Missing / Unverified | Mandatory Manuscript Phrasing |
| :--- | :--- | :--- | :--- |
| **Generators** | StyleGAN2 vs FFHQ ($N=140,000$) | StyleGAN3, Diffusion, FLUX | "Evaluated on large-scale StyleGAN2 whole-face synthesis; zero-shot transfer across diffusion families remains future work." |
| **Ablations** | A4 (Fusion Mechanisms) fully trained | A1, A2, A3 trained metrics | "Ablations A1–A3 are validated via architectural and filter bank verification; full multi-epoch ablation across bases is ongoing." |
| **Variance** | Paired McNemar, bootstrap 95% CIs | Multi-seed training variance | "Statistical tests reflect paired bootstrap confidence intervals over the 20,000 test images under seed 42." |
| **Compression** | Standard JPEG ($Q=100, 50, 23$) | H.264 video CRF ($c0, c23, c40$) | "Evaluated under standard image JPEG compression quality factors $Q \in \{100, 50, 23\}$." |
| **Model Scope** | Whole-face static images | Identity swapping, video clips | "Focused on high-resolution static synthetic face forensics." |
