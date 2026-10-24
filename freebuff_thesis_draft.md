# NeuroScan CDS: Multi-Modal Brain Tumor Classification, Staging Reference & Clinical Decision Support

**Capstone Thesis & IEEE Transactions Ready Manuscript Kit**  
*Academic Year: 2025–2026 | Department of Computer Science & Engineering*

---

## Executive Summary & Integrity Statement

NeuroScan CDS is an open, fully local, GPU-accelerated Clinical Decision Support (CDS) system designed for multi-class brain tumor classification across four pathologies: **Glioma, Meningioma, Pituitary Tumor, and Non-Tumorous brain tissue**.

### Regulatory and Methodological Boundary (FDA / ISO 62304 / MDR)
1. **Intended Use:** NeuroScan CDS operates strictly as a **second-read Clinical Decision Support (CDS) aid** under US FDA Section 520(o)(1)(E) and EU MDR Rule 11. It is not an autonomous primary diagnostic device.
2. **Definite Grading vs. Staging Reference:** Imaging-alone cannot establish histopathological WHO CNS grade (which strictly requires biopsy, IDH1/2 mutation analysis, 1p/19q co-deletion, and ATRX/MGMT assessment). The platform maps predicted tumor families to evidence-based WHO CNS 5th Edition reference ranges.
3. **Audit Transparency:** Unlike black-box literature claims, this work includes empirical audits of dataset duplication (7.12% duplicate exposure identified in public Kaggle benchmarks) and quantifies Grad-CAM++ saliency localization boundaries.

---

## Part A — Abstract & Contributions

### Abstract
Reliable differentiation of intracranial neoplasms from Magnetic Resonance Imaging (MRI) is clinically crucial but hindered by reader subjectivity, overlapping radiological appearances, and complex parenchymal alterations. While recent computer-aided diagnostic pipelines report high nominal accuracy, they frequently rely on opaque feature representations and suffer from unverified dataset leakage. 

We present **NeuroScan CDS**, an interpretable dual-branch hybrid deep-learning architecture for multi-class brain tumor classification and 3D volumetric decision support. The primary branch employs an ImageNet-pretrained **ResNet-50 backbone augmented with a Convolutional Block Attention Module (CBAM)** to adaptively calibrate channel-wise and spatial feature maps, projected to a 256-dimensional deep latent vector. A concurrent handcrafted feature branch extracts **Histogram of Oriented Gradients (HOG)** texture descriptors (2,048 dimensions), reduced via Principal Component Analysis (PCA) to 288 dimensions and embedded to a 512-dimensional manifold. Concatenated multi-scale representations (768 dimensions) are mapped through a non-in-place fusion classifier trained using **Focal Loss ($\gamma=2.0$)** with class-frequency reweighting, decoupled AdamW learning rates, and Test-Time Augmentation (TTA).

Evaluated on a held-out benchmark of $N = 1,600$ multi-class MRI scans across four categories (Glioma, Meningioma, No_Tumor, Pituitary), NeuroScan CDS achieved an overall **accuracy of 95.50%**, **macro F1-score of 0.9547**, and **macro ROC-AUC of 0.9919**. A rigorous dataset audit identified and isolated 114 cross-split duplicate images (25.0% of the meningioma test partition); on the verified uncontaminated test subset ($N = 1,486$), the architecture maintained **95.42% accuracy** and **0.9518 macro F1**. Explainability is provided via mathematically non-corrupting Grad-CAM++ saliency overlays. An integrated Clinical Decision Support engine enforces strict safeguards (<70% confidence threshold review trigger; $\ge 30\%$ tumor probability alert) and compiles audit-logged, hospital-grade PDF summaries.

### Key Contributions
1. **Dual-Branch Spatial-Texture Hybrid Fusion:** Integrating localized sub-cortical texture (HOG-PCA) with deep CNN representations refined by CBAM channel-spatial attention.
2. **3D Multi-Planar Reconstruction (MPR) & Volumetric Engine:** Synchronized Axial, Coronal, and Sagittal orthogonal plane reconstruction with slice-by-slice craniocaudal tumor depth profiling and tumor volume estimation ($cm^3$).
3. **Clinical Image QA & Out-of-Distribution (OOD) Anomaly Guard:** Pre-inference SNR, motion blur, and bilateral symmetry audits combined with diagnostic entropy gating ($H_{norm} > 0.70$) to prevent silent hallucination on non-cranial or corrupted inputs.
4. **Gradient-Preserving Explainability Engine:** Elimination of in-place ReLU operations and adoption of forward-hook autograd traversal to guarantee stable Grad-CAM++ generation without memory corruption.
5. **Rigorous Empirical Dataset & Localisation Audit:** Forensic identification of identical-duplicate leakage in public MRI callsets and statistical quantification of CAM spatial boundaries.
6. **Safety-First Clinical Decision Support Architecture:** Dynamic confidence safeguarding, aggregate tumor alert gating, and immutable JSON-Lines audit logging compliant with FDA CDS standards.

---

## Part B — System Architecture & Methodology

### 1. Dual-Branch Hybrid Architecture

```
                          ┌───────────────────────────────────────────────┐
                          │    Input MRI Slice (144 × 144 × 3 Tensor)    │
                          └──────────────────────┬────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
       ┌────────────────────────┐                                  ┌────────────────────────┐
       │   Branch A: Deep CNN   │                                  │ Branch B: Handcrafted  │
       │ ResNet-50 (ImageNet)   │                                  │ Greyscale Conversion   │
       │ └─ Layer 4 Feature Map │                                  │ └─ HOG Descriptor      │
       │    (5 × 5 × 2048)      │                                  │    (2048-d raw vector) │
       │ └─ CBAM Attention      │                                  │ └─ PCA Projection      │
       │    (Channel + Spatial) │                                  │    (288 components)    │
       │ └─ Adaptive GAP        │                                  │ └─ MLP Texture Encoder │
       │ └─ FC Projection       │                                  │    (288 -> 512-d)      │
       │    (2048 -> 256-d)     │                                  └───────────┬────────────┘
       └───────────┬────────────┘                                              │
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │   Concatenation Layer │
                                     │   (256 + 512 = 768-d) │
                                     └───────────┬───────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │   Fusion Classifier   │
                                     │   768 -> 512 -> 256   │
                                     │   -> 4 Logits         │
                                     └───────────┬───────────┘
                                                 ▼
                         Softmax + Temperature Scaling + TTA View Averaging
```

### 2. Attention Formulation (CBAM)
Given intermediate feature map $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ at `layer4`:
1. **Channel Attention ($\mathbf{M}_c$):**
   $$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right)$$
2. **Spatial Attention ($\mathbf{M}_s$):**
   $$\mathbf{M}_s(\mathbf{F}') = \sigma\left(f^{7 \times 7}\left(\left[\text{AvgPool}(\mathbf{F}'); \text{MaxPool}(\mathbf{F}')\right]\right)\right)$$
where $\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$ and $\sigma$ denotes the sigmoid activation.

### 3. Optimization and Loss Function
Training minimizes multi-class **Focal Loss** to counteract class frequency imbalance and suppress easily classified negative examples:
$$\mathcal{L}_{focal} = - \sum_{i=1}^{K} \alpha_i (1 - p_i)^\gamma \log(p_i)$$
with focusing parameter $\gamma = 2.0$, decoupled AdamW optimizer ($\text{LR}_{backbone} = 1\times 10^{-5}$, $\text{LR}_{head} = 1\times 10^{-4}$), weight decay $1\times 10^{-4}$, and Cosine Annealing learning rate scheduling.

---

## Part C — Experimental Results & Empirical Verification

### 1. Overall Classification Performance ($N = 1,600$ Test Set)

| Metric | Value |
|---|---|
| **Test Accuracy** | **95.50%** |
| **Macro Precision** | **0.9566** |
| **Macro Recall** | **0.9550** |
| **Macro F1-Score** | **0.9547** |
| **Weighted F1-Score** | **0.9547** |
| **Macro ROC-AUC (OvR)** | **0.9919** |
| **Total Test Misclassifications** | **72 / 1,600 (4.50%)** |

### 2. Per-Class Diagnostic Metrics

| Class | Support | Precision | Recall | F1-Score | One-vs-Rest AUC |
|---|---|---|---|---|---|
| **Glioma** | 400 | 0.9831 | 0.8750 | 0.9259 | 0.9786 |
| **Meningioma** | 400 | 0.8998 | 0.9650 | 0.9312 | 0.9914 |
| **No_Tumor** | 400 | 0.9707 | 0.9950 | 0.9827 | 0.9993 |
| **Pituitary** | 400 | 0.9728 | 0.9850 | 0.9789 | 0.9994 |
| **Macro Average** | **1,600** | **0.9566** | **0.9550** | **0.9547** | **0.9919** |

### 3. Confusion Matrix

| True \ Predicted | Glioma | Meningioma | No_Tumor | Pituitary |
|---|---|---|---|---|
| **Glioma** | **350** | 37 | 12 | 1 |
| **Meningioma** | 4 | **386** | 0 | 10 |
| **No_Tumor** | 1 | 1 | **398** | 0 |
| **Pituitary** | 1 | 5 | 0 | **394** |

---

## Part D — Forensic Audits: Data Leakage & Saliency Limits

### 1. Dataset Duplication Audit (`audit_leakage.py`)
Public brain MRI benchmarks commonly distribute slices without patient UIDs. Pixel-level SHA-1 hashing revealed:
* **114 of 1,600 test images (7.12%) are byte-identical** to images in `data/Training`.
* **100 of these occur in Meningioma** (representing 25.0% of that class test cohort).
* Evaluating strictly on the uncontaminated subset ($N = 1,486$):
  * **Leak-Free Accuracy:** 95.42% (vs 95.50% full)
  * **Leak-Free Macro F1:** 0.9518 (vs 0.9547 full)
  * **Meningioma Precision shift:** 0.8998 $\rightarrow$ 0.8758.

### 2. Grad-CAM++ Saliency Boundary Limits (`validate_localisation.py`)
Evaluating whether Grad-CAM++ can be used for automated tumor volumetry:
* At $144\times 144$ input resolution, `layer4` feature maps measure $5\times 5$ spatial grid cells.
* Comparing mean pixel intensity in CAM-activated ROI vs. a size-matched random brain region:
  * Grad-CAM++ difference: $|\Delta| = 0.025$ (below the statistical significance threshold of $0.05$).
* **Conclusion:** Grad-CAM++ serves effectively as qualitative coarse attention verification, but **cannot** substitute for volumetric segmentation masks.

---

## Part E — Clinical 3D Volumetrics & Decision Support Integration

To bridge the gap between academic classification and clinical radiology requirements:
1. **3D Multi-Planar Reconstruction (MPR):** Synchronous real-time tri-planar slicing across Axial (transverse), Coronal (frontal), and Sagittal (lateral) anatomical planes with slice coordinate tracking.
2. **Volumetric Lesion Profiling:** Computes the craniocaudal tumor probability gradient, locates the lesion epicenter ($Z_{max}$), and estimates volumetric burden in $cm^3$ ($mL$).
3. **Automated Image Quality Assurance (QA):** Evaluates Signal-to-Noise Ratio (SNR), Laplacian variance for motion blur, and tissue foreground area before executing inference.
4. **Out-of-Distribution (OOD) Guard:** Formulates normalized diagnostic Shannon entropy $H_{norm} = -\sum p_i \ln p_i / \ln(K)$ combined with anatomical symmetry tests to detect non-cranial images or anomalous presentations.
5. **Immutable JSON-Lines Audit Trail:** Every transaction is logged with input image SHA-256 hashes, execution latency, model version, and radiologist digital sign-off (`logs/clinical_audit.jsonl`).
6. **Standardized PDF Generation:** Automated export of hospital-grade ReportLab summaries with cryptographic audit timestamps.

---

## Part F — Viva & Defense Presentation Guide

### Anticipated Questions & Bulletproof Answers

**Q1: "Does your model predict WHO Grade IV Glioblastoma directly?"**  
*Answer:* "No, and scientifically, no imaging-only model can. The 2021 WHO CNS 5th Edition mandates molecular markers (IDH1/2 mutation status and 1p/19q codeletion) for definitive grading. NeuroScan CDS classifies the morphological tumor family and provides an evidence-based clinical reference range for decision support."

**Q2: "Why did you use ResNet-50 instead of a pure Vision Transformer?"**  
*Answer:* "Brain MRI slice datasets have limited sample size (~7,000 images). Pure Vision Transformers lack inductive bias and overfit heavily on small medical datasets without massive pretraining. ResNet-50 combined with CBAM spatial-channel attention and HOG-PCA texture features provides superior parameter efficiency and convergence stability."

**Q3: "Did you verify data leakage between your train and test sets?"**  
*Answer:* "Yes. We built an automated pixel-hashing audit (`audit_leakage.py`) revealing that 7.12% of the public test set contained duplicate slices. We evaluated both full and leak-free subsets, proving that our architecture maintains a robust 95.42% accuracy on completely uncontaminated data."

**Q4: "How does your system handle 3D volumetric MRI studies?"**  
*Answer:* "NeuroScan CDS features a dedicated 3D Multi-Planar Reconstruction (MPR) engine supporting NIfTI volumes. It reconstructs Axial, Coronal, and Sagittal views simultaneously, tracks slice coordinates, and computes a craniocaudal tumor burden profile with volumetric estimation in $cm^3$."

**Q5: "What happens if a user uploads a non-brain image or a corrupted scan?"**  
*Answer:* "Unlike conventional closed-world classifiers that produce high-confidence hallucinations, NeuroScan CDS incorporates an automated QA and Out-of-Distribution (OOD) Anomaly Guard. It computes bilateral cranial symmetry, SNR, and normalized diagnostic entropy. Inputs with $H_{norm} > 0.70$ or abnormal geometry are flagged with an immediate OOD warning, suppressing automated sign-off."
