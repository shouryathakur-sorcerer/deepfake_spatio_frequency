# Revised Related Work and Theoretical Hypotheses
**Manuscript Title:** Dual-Branch Spatio-Frequency Network with Wavelet-Packet Attention for Compression-Resilient Deepfake Image Forensics  
**Authors:** Team 14 Research Consortium  
**Section Purpose:** Publication-grade Related Work and Formal Scientific Hypotheses (H1–H5).

---

## 1. Spatial Domain Face Forgery Forensics

Initial deepfake detection research focused almost exclusively on spatial domain artifacts. Following the introduction of deep convolutional networks for image classification, forensic methods adopted spatial backbones such as MesoNet [Afchar et al., 2018], ResNet [He et al., 2016], and Xception [Chollet, 2017; Rössler et al., 2019]. These architectures excel at detecting low-level blending inconsistencies, color discrepancies, specular reflection mismatches, and boundary artifacts at the facial perimeter. In the comprehensive FaceForensics++ benchmark [Rössler et al., 2019], Xception with depthwise separable convolutions emerged as the leading standard baseline across Deepfakes, Face2Face, FaceSwap, and NeuralTextures manipulations.

However, spatial-domain models exhibit severe degradation under real-world transmission channels. When facial images undergo lossy compression (such as JPEG block quantization or H.264 video compression), high-frequency spatial gradients, boundary transitions, and pixel-level interpolation cues are smoothed or destroyed. Furthermore, modern generative architectures—such as StyleGAN2 [Karras et al., 2020], StyleGAN3, and latent diffusion models [Rombach et al., 2022]—generate seamless facial boundaries and photorealistic textures that leave minimal spatial traces, precipitating a steep decline in cross-generator generalization [Li et al., 2020].

---

## 2. Frequency Domain Forensics: From FFT to Wavelet Packets

To overcome spatial vulnerability, digital forensics turned to spectral representations. Seminal investigations by Frank et al. [2020] and Durall et al. [2020] demonstrated that convolutional upsampling operations—specifically transposed convolutions and nearest-neighbor interpolations—induce severe structural violations of the Nyquist-Shannon sampling theorem. These structural violations manifest as periodic grid patterns and high-frequency spectral replications visible in the 2D Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT) spectra.

Despite their diagnostic utility, standard global transforms possess critical theoretical drawbacks:
1. **Loss of Spatial Localization:** The Fourier transform integrates globally over the entire spatial canvas ($e^{-j(u x + v y)}$), discarding all spatial position information. Consequently, a small localized manipulation (such as swapped eyes or a modified mouth) cannot be spatially isolated from global facial frequency distributions.
2. **Dyadic DWT Limitations:** While the Discrete Wavelet Transform (DWT) provides joint time-frequency localization [Mallat, 1989], standard dyadic multiresolution analysis decomposes only the low-frequency approximation sub-band ($LL$). The high-frequency detail octaves ($LH, HL, HH$) remain unpartitioned. As established by Wolter et al. [2022], this coarse high-frequency resolution cannot isolate narrow-band spectral discrepancies.
3. **Wavelet Packet Decomposition (WPD):** Wavelet Packet Decomposition [Coifman & Wickerhauser, 1992] addresses this limitation by recursively decomposing **both** low-frequency and high-frequency nodes across all decomposition levels. For a 3-level 2D decomposition using an orthogonal basis such as Daubechies-4 (db4), WPD yields a complete quadtree of $4^3 = 64$ sub-bands. This creates a uniform tiling of the 2D spatial-frequency plane, enabling simultaneous narrow-band frequency selectivity and localized spatial support.

---

## 3. High-Frequency Degradation and Compression Resilience

When media traverses social platforms and messaging networks, lossy compression is applied aggressively. As shown by Wang et al. [2023] and Gao et al. [2024], standard JPEG compression utilizes discrete $8 \times 8$ block DCT quantization matrices that divide high-frequency coefficients by large divisors, truncating subtle generative artifacts to zero. Video compression standards (H.264/AVC) apply intra-frame DCT quantization coupled with inter-frame motion estimation, exacerbating forensic signal loss.

Consequently, naive frequency detectors that depend strictly on unweighted high-frequency energy suffer catastrophic performance drops under high compression ($Q \le 50$ or CRF $\ge 30$). To maintain compression resilience, a detector must:
1. Retain the luminance channel ($Y_0$), which preserves fine structural information that chroma channels ($Cr, Cb$) discard during 4:2:0 subsampling.
2. Adaptively re-weight spectral sub-bands. As demonstrated by Gao et al. [2024], mid-frequency packets ($0.25\pi$ to $0.75\pi$) frequently retain generative footprints even when extreme high frequencies are obliterated.

---

## 4. Cross-Modal Fusion: From Squeeze-and-Excitation to Cross-Attention

A central challenge in dual-branch spatio-frequency architectures is the fusion mechanism. Earlier architectures applied simple vector concatenation or branch-independent gating, such as Parallel Squeeze-and-Excitation (Parallel SE) [Hu et al., 2018]. However, as proved by Kashiani et al. [2025], dual-branch networks trained with uncoupled fusion frequently suffer from **spatial dominance**: because spatial backbones (such as ImageNet-pretrained Xception) converge faster than frequency backbones, the spatial branch dominates the loss gradient, reducing the frequency branch to an inactive residual.

True cross-modal interaction requires bidirectional query-key mapping. In Dual Cross-Attention (DCA) [Vaswani et al., 2017; Wang et al., 2024; Ghosh et al., 2026], spatial tokens query the frequency packet representations ($A_{S \to F} = \text{Softmax}(Q_S K_F^T / \sqrt{d}) V_F$), while frequency tokens query spatial context ($A_{F \to S} = \text{Softmax}(Q_F K_S^T / \sqrt{d}) V_S$). This bidirectional exchange guarantees gradient flow to both representations and enforces contextual alignment between spatial features and frequency anomalies.

---

## 5. Formal Scientific Hypotheses (H1–H5)

To ensure scientific rigor and empirical falsifiability, Team 14 structures this research around five formal hypotheses:

### Hypothesis 1 (Frequency Complementarity)
> **$H_1$:** *Spatial and frequency representations encode complementary forensic signatures: the spatial branch captures macro-level semantic and boundary inconsistencies, while the wavelet-packet branch captures micro-level upsampling and compression-resilient spectral fingerprints. Consequently, a dual-branch network will achieve strictly superior classification performance (measured by ROC-AUC and EER) compared to either branch operating in isolation ($B_{\text{spatial}} < B_{\text{dual}}$ and $B_{\text{freq}} < B_{\text{dual}}$).*

### Hypothesis 2 (Wavelet Packet Granularity)
> **$H_2$:** *A 3-level 2D Daubechies-4 Wavelet Packet Decomposition producing 64 uniform sub-bands preserves forensic signatures under aggressive compression ($Q \le 50$) more effectively than a standard 2-level DWT (16 sub-bands) or a 1-level DWT (4 sub-bands), because the uniform frequency tiling isolates mid-frequency octaves that survive JPEG quantization tables.*

### Hypothesis 3 (Learnable High-Frequency Enhancement)
> **$H_3$:** *A learnable High-Frequency Enhancement (HiFE) module with channel-attention gating dynamically re-weights attenuated spectral sub-bands, mitigating the degradation caused by lossy compression and achieving significantly higher AUC ($p < 0.01$) under low-quality compression ($Q \in \{10, 20, 30\}$) than unweighted, fixed-filter wavelet representations.*

### Hypothesis 4 (Bidirectional Attentive Fusion vs. Gated Concatenation)
> **$H_4$:** *Dual Cross-Attention (DCA) with spatial residual bypass enforces bidirectional gradient propagation and spatial-frequency feature alignment, outperforming both naive feature concatenation and parallel channel gating (Parallel SE) across compressed test sets.*

### Hypothesis 5 (Generalization and Domain Shift)
> **$H_5$:** *Because generative upsampling artifacts are universal across convolutional synthesis architectures, training with multi-band wavelet packet representations reduces overfitting to generator-specific spatial artifacts, providing superior cross-dataset and cross-generator generalization compared to spatial-only networks.*

---
*End of Revised Related Work and Formal Hypotheses.*
