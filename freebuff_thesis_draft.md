# FreeBuff Capstone Thesis + IEEE Paper Writing Kit

This document gives you a ready-to-fill structure for both a thesis report and an IEEE-style paper. Treat every `[...]` as a slot to populate from your actual project artifacts.

---

## Part A — Abstract and Introduction

### Abstract skeleton (≈150–250 words)

**Fill-in version**

Brain tumor assessment from multi-sequence MRI is clinically important but
difficult because no single contrast fully captures lesion morphology,
enhancement pattern, and peritumoral signal. Many existing deep learning
systems rely on a single modality or simple fusion, which limits robustness
and interpretability. FreeBuff is a multi-modal brain tumor classification
system that jointly analyzes synchronized T1c, T2, and FLAIR MRI scans
using a hybrid architecture that combines EfficientNet-based CNN feature
extraction, Vision Transformer global encoding, and CBAM attention
refinement. The model produces a fused deep embedding that is combined with
PCA-compressed HOG features into a compact hybrid representation for
classification. The system is packaged as a reusable inference engine with
explainability support. On a held-out test set of `[N]` cases spanning
`[K]` classes, FreeBuff achieved `[accuracy]` overall accuracy and
`[macro F1]` macro F1, with the strongest performance on `[class]`.
These results support the premise that structured multi-modal fusion can
improve discriminative consistency for research-grade tumor classification.

**What to collect before finalizing**

- `N`: total test cases
- `K`: number of classes
- overall accuracy
- macro F1
- strongest class and its F1 if notable
- one sentence on clinical framing (decision-support, not diagnosis)

---

### Introduction skeleton

**Paragraph 1 — Clinical context**

Brain tumors require careful imaging interpretation because their appearance
varies across sequences and pathologies. MRI protocols commonly use multiple
contrasts, including T1-weighted post-contrast, T2-weighted, and FLAIR, each
emphasizing different tissue properties. In practice, interpretation is
labor-intensive, reader-dependent, and constrained by workload.

**Paragraph 2 — Diagnostic bottleneck**

Even with multi-sequence studies, reliable discrimination among tumor types
such as glioma, meningioma, pituitary lesions, and non-tumor findings remains
challenging. Subtle enhancement patterns, edema, and overlapping signal
characteristics can produce ambiguous cases. Automated decision-support tools
can help by surfacing consistent pattern-based predictions, especially when
they can combine complementary contrasts rather than relying on a single
image type.

**Paragraph 3 — Research gap**

A common limitation in medical imaging ML is the reliance on single-modality
inputs or shallow fusion. Single-sequence models discard clinically useful
context, while naive fusion may not fully exploit the complementary nature of
local texture, enhancement, and fluid-sensitive signals. There is also value
in architectures that combine strong localized feature extraction with global
context and explicit attention mechanisms.

**Paragraph 4 — What FreeBuff does**

FreeBuff addresses this gap with a multi-modal classification pipeline that
ingests synchronized T1c, T2, and FLAIR scans and fuses them through a hybrid
model combining EfficientNet-style CNN features, Vision Transformer encoding,
and CBAM attention modules. The system uses a hybrid representation that joins
deep embeddings with PCA-compressed handcrafted features, then classifies
through a mature inference engine that also supports explainability outputs.

**Paragraph 5 — Contributions**

Concrete contributions you can list:

- A multi-modal MRI classification architecture combining EfficientNet, ViT,
  and CBAM across T1c, T2, and FLAIR.
- A hybrid deep + handcrafted feature representation under a fixed
  dimensionality contract.
- A serialized, reusable inference engine with accelerated execution and
  saliency-based explainability.
- A structured evaluation on a multi-class brain tumor dataset with
  per-class and overall metrics.

**Paragraph 6 — Report roadmap**

Briefly state how the report is organized: background, methodology,
experiments, discussion, conclusion.

---

## Part B — Literature Review Summary

### Themes to cover

1. **Brain tumor imaging and tumor types**
   - Why MRI is central, what T1c, T2, and FLAIR emphasize.
   - The clinical relevance of glioma, meningioma, pituitary, and no-tumor
     discrimination.

2. **Traditional and CNN-based approaches**
   - Earlier pipelines often used handcrafted features or single-modality CNNs.
   - CNNs improved representation learning but can under-use complementary
     contrasts when trained on one sequence alone.

3. **Hybrid and transformer-based models**
   - Vision Transformers and hybrid CNN–Transformer designs can capture both
     local detail and broader context.
   - Multi-branch designs are increasingly used when inputs are complementary.

4. **Attention mechanisms**
   - Attention modules such as CBAM refine feature maps along channel and
     spatial dimensions.
   - In medical imaging, attention can help focus on salient regions and
     improve interpretability.

5. **Multi-modal fusion**
   - Fusion can be early, late, or intermediate.
   - The strongest systems often fuse at the feature level after modality-specific
     encoding, especially when sequences provide non-redundant information.

6. **Handcrafted + deep feature combinations**
   - Some systems augment deep features with structured descriptors.
   - Dimensionality control (e.g., PCA) is important when combining large
     handcrafted features with deep embeddings.

### Positioning paragraph for FreeBuff

Many existing systems classify brain tumors from one sequence or use
fairly simple fusion. FreeBuff instead combines modality-specific
EfficientNet-based CNN features with Vision Transformer encoding and CBAM
attention, then fuses the modalities through a cross-modality transformer.
It also augments the deep representation with PCA-compressed HOG features to
form a compact hybrid vector, and exposes explainability outputs to support
interpretation. This places FreeBuff between single-modality CNNs and more
general multi-modal fusion systems, with emphasis on structured hybrid
representation and interpretability.

---

## Part C — System Design and Methodology

### 1. System overview

FreeBuff is organized as three cooperating stages:

1. **Preprocessing and data loading**
   - Standardize every MRI slice to the project resolution and format.
   - Handle both common images and DICOM where supported.
   - Load synchronized T1c, T2, and FLAIR sequences by class folder structure,
     with ground truth taken from the folder label.

2. **Feature extraction and fusion**
   - Each modality passes through a dual-branch encoder:
     - CNN branch with EfficientNet-style backbone and CBAM attention.
     - Transformer branch with patch embedding and encoder blocks.
   - Per-modality vectors are fused by a cross-modality transformer into a
     compact deep embedding.
   - A handcrafted HOG branch is computed per modality, concatenated, and
     PCA-compressed.
   - The deep and handcrafted branches are joined into a fixed hybrid feature
     vector.

3. **Classification and inference**
   - The hybrid vector is classified through an SVM-based decision pipeline
     with optional stacked super-learner augmentation.
   - At inference time, the engine loads serialized artifacts, supports
     accelerated execution paths where available, and can produce probability
     outputs and saliency/explainability maps.

---

### 2. Data and preprocessing

**Input specification**

- Three synchronized MRI sequences: T1c, T2, FLAIR.
- Standardization to 144×144 spatial resolution.
- Each modality is represented as a 144×144×3 input for its encoder.

**Preprocessing points to mention**

- Uniform spatial standardization to keep feature contracts stable.
- Modality-parallel loading so the three sequences stay aligned by case.
- Folder-based ground truth: each case inherits the class label from its
  containing folder.
- Inference-time handling of missing modalities by replicating a present
  sequence when necessary, with logging for transparency.

**Dataset structure**

Use folder layout such as:

- `glioma/`
- `meningioma/`
- `notumor/`
- `pituitary/`

with each folder containing images for that class.

---

### 3. Model architecture

**Per-modality dual-branch encoder**

For each modality:

- A CNN branch extracts localized features from an ImageNet-derived backbone
  and applies CBAM attention to refine channel and spatial responses.
- A transformer branch embeds the image into patches and processes tokens
  through encoder blocks, providing global context.
- The two branches are combined into a per-modality vector.

**Fusion**

- The three modality vectors are stacked and combined through a cross-modality
  transformer.
- The fused representation is projected into a compact deep embedding.

**Hybrid feature construction**

- Deep embedding: `[deep dim]` dimensions.
- Handcrafted branch: per-modality HOG with `[orientations]` orientations,
  `[pixels_per_cell]`-pixel cells, and `[cells_per_block]`-cell blocks.
- HOG raw dimensionality per modality: `[HOG dim]`.
- After concatenation and PCA compression: `[PCA HOG dim]` dimensions.
- Hybrid vector: deep + PCA-HOG = `[hybrid dim]` dimensions.

**Classifier**

- The hybrid vector is classified using a support vector machine classifier
  with probability estimation.
- If present, a super-learner ensemble can be used to combine multiple base
  learners for the final decision.

**Dimension contract**

State the contract explicitly because it is the backbone of reproducibility:

- deep embedding + PCA-HOG = hybrid vector dimension.
- This fixed dimensionality is the input contract for the classifier.

---

### 4. Training and reproducibility

**Training design**

- Use a train/validation split and, if applicable, cross-validation.
- Describe the optimization objective and any class-imbalance handling.
- Mention whether augmentation was used and why it is clinically reasonable.
- Note regularization choices such as dropout or early stopping where relevant.

**Reproducibility controls**

- Fixed random seed.
- Documented preprocessing standard.
- Saved model artifacts with a manifest.
- Versioned feature contract.

---

### 5. Inference pipeline

**Engine contract**

- The inference engine loads the trained artifacts from a model directory.
- It accepts modality-specific inputs and returns predictions and
  probabilities.
- Missing modalities are handled explicitly rather than silently.

**Execution and explainability**

- The engine is designed for best-effort accelerated execution, such as XLA
  compilation or ONNX Runtime when available.
- Explainability backends can generate saliency-style maps for inspection,
  supporting qualitative interpretation of predictions.

---

## Part D — Experimental Results and Discussion

### 1. Evaluation setup

State:

- Dataset name/layout and number of classes.
- Number of test cases per class.
- Metrics used: accuracy, precision, recall, F1, macro and weighted variants,
  confusion matrix, and ROC-AUC where applicable.
- Whether the test set was held out from training.

---

### 2. Headline results table

**Table 1 — Overall performance**

| Metric | Value |
|---|---|
| Accuracy | `[value]` |
| Macro Precision | `[value]` |
| Macro Recall | `[value]` |
| Macro F1 | `[value]` |
| Weighted F1 | `[value]` |
| Macro ROC-AUC | `[value or n/a]` |

---

### 3. Per-class results table

**Table 2 — Per-class performance**

| Class | Support | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| glioma | `[n]` | `[v]` | `[v]` | `[v]` | `[v]` |
| meningioma | `[n]` | `[v]` | `[v]` | `[v]` | `[v]` |
| notumor | `[n]` | `[v]` | `[v]` | `[v]` | `[v]` |
| pituitary | `[n]` | `[v]` | `[v]` | `[v]` | `[v]` |

If AUC is not available for all classes, say so explicitly instead of
omitting the column silently.

---

### 4. Confusion matrix

Include a confusion matrix figure and describe the main patterns in words:

- Which classes are most often confused.
- Whether errors are asymmetric.
- Whether any class is systematically harder.

A strong discussion does not just say “the model is good”; it explains
*where* and *why* the model succeeds or fails.

---

### 5. Qualitative and explainability discussion

If you include saliency maps or overlays:

- Show a few representative cases.
- Describe whether highlighted regions appear consistent with lesion location
  and modality characteristics.
- Be cautious: explainability supports interpretation, it does not prove
  clinical correctness.

---

### 6. Discussion points to cover

- **Strengths**: multi-modal fusion, hybrid representation, interpretability.
- **Limitations**: dataset size/source, single-institution or single-source
  nature if applicable, possible class imbalance, dependency on preprocessing
  quality, and the gap between research evaluation and clinical deployment.
- **Error analysis**: most common confusions and likely causes.
- **Generalization**: what would be needed to trust the model on new data
  sources.

---

## Part E — Conclusion and Future Scope

### Conclusion paragraph

FreeBuff demonstrates a structured multi-modal approach to brain tumor
classification from synchronized T1c, T2, and FLAIR MRI. By combining
EfficientNet-based CNN features, Vision Transformer encoding, and CBAM
attention within a hybrid deep + handcrafted feature pipeline, the system
produces compact, reproducible representations for classification. The
evaluation suggests that fusing complementary MRI contrasts can improve
discriminative consistency across tumor classes, while attached explainability
mechanisms support interpretability. Importantly, FreeBuff is designed as a
research and decision-support tool, not a standalone clinical diagnostic
device.

### Future work

1. **Automated tumor segmentation masks**
   - Add a segmentation stage to localize suspicious regions and support
     visual correlation.
   - Explore segmentation-guided feature extraction or attention.

2. **Real-time or cloud API deployment**
   - Package the inference engine behind a REST or gRPC service.
   - Consider containerization and batching for throughput.
   - Evaluate latency, GPU sharing, and on-premise vs. cloud trade-offs.

3. **Broader validation**
   - Test on external datasets or institutional data.
   - Expand tumor types or MRI sequences if data allows.
   - Study calibration and uncertainty more rigorously.

4. **Clinical workflow integration**
   - Design for assistive use with clear disclaimers and human oversight.
   - Include reporting hooks and structured outputs for research pipelines.

---

## Part F — IEEE Paper Mapping

If you need an IEEE-format draft later, map the thesis chapters to:

1. **Introduction**
2. **Related Work**
3. **Methodology**
4. **Experiments and Results**
5. **Discussion**
6. **Conclusion**

Suggested tables/figures:

- Table: dataset composition
- Table: model configuration summary
- Table: overall metrics
- Table: per-class metrics
- Figure: architecture overview
- Figure: confusion matrix
- Figure: per-class F1
- Figure: explainability example

---

## Placeholder manifest

Before submission, fill these in:

- `[N]`: test set size
- `[K]`: number of classes
- `[accuracy]`, `[macro F1]`, `[class]` for abstract
- `[deep dim]`, `[HOG dim]`, `[PCA HOG dim]`, `[hybrid dim]`
- `[orientations]`, `[pixels_per_cell]`, `[cells_per_block]`
- per-class numbers and overall metrics
- figure files and captions

If you want, I can turn this outline into a fully populated draft once you
provide the exact evaluation numbers and the figures you plan to include.
