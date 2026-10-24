# FreeBuff v4 — Technical Specification

**Version 4.0.0 · Python 3.11 · PyTorch 2.5.1+cu121 · Target hardware: NVIDIA RTX 3050 (laptop GPU, 4–6 GB VRAM), Windows 10/11**

> Research prototype for an academic engineering project — not a medical device.
> Every clinical claim in this document is qualified by `REGULATORY_DISCLAIMER`
> in `config.py`, which is embedded in every generated PDF report.

---

## 1. System Overview

FreeBuff v4 converts multi-modal brain MRI studies into a clinically
actionable decision-support package: a four-way tumour-family classification,
a per-class confidence spectrum, visual explanations of each decision, a WHO
grading reference, safety flags, and an auditable PDF report. It is a fully
local system — no cloud dependency, no external API keys.

**Processing chain**

```
Upload (T1c/T2/FLAIR, any subset)
   → replicate missing modalities (logged)
   → per-modality preprocessing: brain-field crop, [0,1] intensity
     normalization, resize 144×144, grayscale→RGB, ImageNet normalization
   → composite = pixel-wise mean of the (up to 3) sequences
   → hybrid model (Branch A ⊕ Branch B)
   → TTA probability averaging (2 views)
   → sensitivity-first decision rule + confidence band
   → Grad-CAM++ per modality
   → WHO staging lookup + narrative generation
   → dashboard rendering + PDF export
```

---

## 2. Data Layer

### 2.1 Dataset (`dataset.py`)

| Item | Value |
|---|---|
| Directory contract | `data/<class_name>/<image files>` |
| Classes (fixed order) | `Glioma`, `Meningioma`, `No_Tumor`, `Pituitary` |
| Supported inputs | PNG/JPG/BMP/TIFF, DICOM slices (`.dcm`), NIfTI volumes (`.nii`) — middle slice taken |
| Training resolution | 144 × 144, 3 channels |
| HOG geometry | cell 16×16, blocks 2×2, block stride 1, 8 orientation bins → **2048-d** raw descriptor |
| HOG-PCA | fitted on **training split only** (no leakage), target 288 components, capped at `min(n−1, 2048)` |

### 2.2 Splits

Stratified 60/20/20 train/val/test via scikit-learn `train_test_split`
(two-stage: holdout then val/test partition), `random_state = 42` for full
reproducibility. Class balance is preserved through both stages.

### 2.3 Augmentation (training only)

`preprocessing.augment_mri_image` applies geometric and photometric transforms
on the PIL image before tensorization. Validation and test transforms are
deterministic — only resize + normalize. All ranges are config-driven
(`AUGMENT_*` in `config.py`) so the policy is tunable without editing code.

| Axis | Transform | Range |
|---|---|---|
| Geometric | random resized crop (scale × aspect) | 0.75–1.00 × 0.90–1.10 |
| | in-plane rotation | ± 12° |
| | translation | ± 6 % of size |
| | horizontal flip | p = 0.5 |
| Photometric | brightness | 0.85–1.15 |
| | contrast | 0.85–1.15 |
| | gamma | 0.80–1.25 |
| | Gaussian noise | σ = 0.015 |

**Only the horizontal flip is used among flips, and rotations stay small.**
An axial brain MRI is roughly bilaterally symmetric, so a mirror is a
plausible scan; a vertical flip or a quarter-turn is not, and training on
inputs no scanner produces wastes capacity. The TTA policy in §6.1 is kept
inside these same ranges for the same reason.

The photometric axis matters more than the geometric one on this dataset:
scanner, sequence and centre differences appear mainly as intensity and
bias-field variation, which is the dominant train/test domain gap.

---

## 3. Model Architecture (`model.py`)

### 3.1 Branch A — ResNet50 + CBAM

- Backbone: torchvision ResNet50, ImageNet1K_V1 weights (download once at
  training time; inference checkpoints load without any download).
- The torchvision FC head is removed; `layer4` output is **2048-channel**.
- **CBAM** is applied after `layer4`:
  - *Channel attention*: shared 2-layer MLP (2048 → 128 → 2048, LayerNorm +
    ReLU) over GAP- and GMP-pooled channel descriptors; sigmoid gate added
    element-wise.
  - *Spatial attention*: 7×7 conv over the concatenated channel-wise
    mean/max maps → single (B, 1, h, w) sigmoid map.
- Global average pooling → MLP 2048 → 1024 → 256 (**spatial embedding**).

**Shape trace (144×144 input):**

```
input          (B, 3, 144, 144)
conv1/maxpool  (B, 64, 36, 36)
layer1         (B, 256, 36, 36)
layer2         (B, 512, 18, 18)
layer3         (B, 1024, 9, 9)
layer4         (B, 2048, 5, 5)
CBAM           (B, 2048, 5, 5)
GAP + MLP      (B, 256)
```

### 3.2 Branch B — HOG-PCA texture descriptor

Raw 2048-d HOG → PCA → 288-d → MLP 288 → 512 (**HOG embedding**). The PCA
transformer is fitted **only** on the training split and persisted
(`saved_models/freebuff_pca.pkl`).

**Degradation rule (critical):** if no PCA artifact is available, or the
artifact's `n_features_in_` does not match the live extractor's output, the
branch is *disabled* rather than fed mismatched dimensions — the model then
runs on the deep branch alone with a zero HOG embedding (implemented in
`FreeBuffInference._hog_for` and `HOGBranch.forward`). A legacy artifact
fitted on 6144-d (v1 three-modality concatenation) is detected and rejected
with an explicit log line.

### 3.3 Fusion head

Concatenate [256-d spatial ⊕ 512-d HOG] = 768-d → Linear 512 → ReLU →
Dropout 0.3 → Linear 256 → ReLU → Dropout 0.3 → Linear 4.

**Parameter budget:** ≈ 27.07 M total (≈ 23.5 M backbone, ≈ 3.5 M
attention + HOG + fusion), all trainable unless `freeze_backbone`.

### 3.4 Non-inplace activation policy

Every `ReLU` and `Dropout` in attention, embeddings, and the fusion head is
constructed with `inplace=False`. In-place operators overwrite tensor
storage that the autograd engine may still need; with Grad-CAM++ we compute
`torch.autograd.grad(score, features)` after the forward pass, and an
in-place op anywhere on the score→features path raises
*"one of the variables needed for gradient computation has been modified by
an inplace operation"*. The non-inplace policy makes the entire network
Grad-CAM-safe by construction.

---

## 4. Training Pipeline (`train.py`)

### 4.1 Loss — Focal Loss (γ = 2.0)

Implemented via the cross-entropy expansion:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t),   p_t = exp(-CE)
```

Class weighting: α = inverse frequency, normalized to mean 1:

```
α_c = (1/count_c) / mean(1/count)
```

γ = 2.0 down-weights examples with p_t > ≈ 0.7, focusing gradient signal on
the hard boundary cases that dominate clinical error. `alpha` is registered
as a buffer so `.to(device)` relocates it automatically.

### 4.2 Optimizer & schedule

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8) | decoupled weight decay ≠ L2-in-gradient |
| Backbone LR | 1e-4 | pretrained features need gentle adaptation |
| Head LR | 1e-3 | fresh layers converge faster at higher LR |
| Weight decay | 1e-4 | mild regularization |
| Gradient clipping | max-norm 1.0 | stability with fp16 |
| Schedule | CosineAnnealingLR, T_max = 60, η_min = 1e-6 | smooth decay without stage tuning |
| Early stopping | patience 12 epochs on val weighted F1 | 4 GB card budget |

### 4.3 Automatic Mixed Precision (RTX 3050)

Modern namespace (`torch.amp.autocast(device_type="cuda")` +
`torch.amp.GradScaler`) with a legacy fallback for older torch:

```python
with autocast(device_type="cuda", dtype=torch.float16):
    logits = model(images, hog_pca)["logits"]
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                       # clip in fp32 space
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Benefits on the 4–6 GB card: tensor-core fp16 GEMMs roughly halve activation
memory (batch 32 fits comfortably), ~1.6–2× throughput. GradScaler prevents
fp16 underflow: it multiplies the loss, checks for inf/NaN gradients, skips
poisoned steps, and rescales dynamically. BatchNorm running statistics stay
fp32 under autocast by framework design.

### 4.4 Validation & persistence

Per epoch: mean loss, top-1 accuracy, **weighted F1** (primary model
selection metric — robust to the class imbalance typical of tumour sets).
Best-F1 checkpoint is written to `checkpoints/freebuff_v4.pt` (weights +
optimizer + scheduler + epoch + metrics + `model_config` + `format_version`)
and copied to `saved_models/freebuff_v4.pt`. History is exported to
`evaluation_results/training_history.csv`.

### 4.5 Reproducibility

`config.set_all_seeds(42)` seeds Python `random`, NumPy, torch (CPU and all
CUDA devices) and sets `cudnn.deterministic = True`,
`cudnn.benchmark = False`.

---

## 5. Error-Proof Checkpoint Loading

`model.load_checkpoint` is engineered so a checkpoint can never crash the
app, in seven ordered stages:

1. **Container tolerance** — `.pt`/`.pth` payloads and TorchScript archives
   (`torch.jit.load` fallback on `RuntimeError`).
2. **Envelope unwrapping** — accepts `model_state_dict`, `state_dict`,
   `model`, `generator`, `module`, `weights` keys, or a raw tensor dict.
3. **Legacy key stripping** — fragment rules rename v1-era conventions
   (`module.backbone.` → `backbone.`, `resnet.` → `backbone.`, old CBAM
   names → current ones); exact aliases map `fc./classifier.` heads onto
   `head.net.6.*`; stale leftovers are dropped.
4. **Bookkeeping removal** — `num_batches_tracked` keys removed (they are
   BN buffers, rematerialised automatically, and a classic strict-load
   breaker).
5. **Shape matching** — every remaining tensor is compared against the live
   model; mismatches are skipped with a logged report instead of raising
   `RuntimeError: size mismatch`.
6. **strict→partial fallback** — strict load attempted first; on failure it
   retries with `strict=False` and reports the count of randomly-initialised
   keys.
7. **Optimizer/scheduler restore** — wrapped in try/except so a corrupt
   optimizer block cannot poison the restored weights.

The deployment loader (`FreeBuffInference.from_dir`) always calls it with
`strict=False` and `pretrained=False` (weights come from the file; no
ImageNet download at the point of care).

---

## 6. Inference & Explainability (`inference.py`)

### 6.1 Test-Time Augmentation

Shipped policy: 2 flips (identity, horizontal mirror) × 1 rotation (0°) ×
1 scale (1.00) = **2 deterministic views**. Each view's logits are
temperature-scaled (`τ = 1.0` default), softmaxed, and **averaged in
probability space** — probability averaging is less sensitive to a single
pathological view than logit averaging or max selection. The deterministic
even-subsample in `build_tta_transforms` (`max_views`) can reduce the set
further for tight VRAM budgets.

**Why the policy is this narrow.** An earlier version used 4 flips ×
4 rotations (0/90/180/270°) × 3 scales = 48 views. Vertical flips and
quarter-turns are anatomically invalid for an axial brain MRI — the model
never saw such inputs in training, so its predictions on them are noise that
gets averaged into the result. Measured on the full 1600-image held-out set,
with the policy selected on a held-out split of `data/Training` and confirmed
once on `data/Testing`:

| policy | views | val acc | test acc | test macro F1 |
|---|---|---|---|---|
| no TTA | 1 | 0.9925 | 0.9513 | 0.9508 |
| **hflip only (shipped)** | **2** | **0.9950** | **0.9556** | **0.9552** |
| hflip × rot ±8° | 6 | 0.9875 | 0.9456 | 0.9450 |
| hflip × scale | 6 | 0.9867 | 0.9394 | 0.9386 |
| hflip × rot × scale | 18 | 0.9367 | 0.8906 | 0.8884 |
| previous 8-view default | 8 | — | 0.8894 | 0.8888 |
| previous 48-view grid | 48 | — | 0.8869 | 0.8859 |

The 48-view grid scored below no-TTA. The TTA policy is therefore kept
consistent with the training augmentation ranges in `config.py`: a view that
falls outside the training distribution costs accuracy rather than reducing
variance.

### 6.2 Sensitivity-first decision rule

```
tumour_prob = Σ p_c for c ∈ {Glioma, Meningioma, Pituitary}
if tumour_prob ≥ 0.30 and argmax == No_Tumor:
        surface the highest-probability tumour class + review alert
else:   keep the argmax
```

Raw probabilities are always reported unchanged — the rule only re-orders
what is *surfaced*, never what is *hidden*.

### 6.3 Grad-CAM++ — implementation notes

- **Forward-only hook** on `layer4` captures the exact output tensor (no
  clone, no detach): it must remain the same graph node the class score
  flows through.
- Gradients via `torch.autograd.grad(score, features, retain_graph=True)`.
  `register_full_backward_hook` was deliberately abandoned: when the hooked
  module's output is reshaped downstream (view/clone/cat), the hook fires
  with stale gradient shapes and has produced CUDA illegal-memory-access
  crashes on this stack.
- **Grad-CAM++ weights** (second-order):

```
α_k^c = Σ_{i,j} g²_{ijk} / (Σ g² + 2 Σ g²)     (per-channel spatial weights)
L_c    = ReLU( Σ_k α_k^c · ReLU(Σ_{i,j} g_{ijk} · f_{ijk}) )
```

  implemented with 1e-8 guards against zero denominators; heatmap
  bilinearly upsampled to input resolution and min-max normalized.
- **Guards**: `class_idx` clamped into range (CUDA OOB protection);
  explicit RuntimeError if features were captured under
  `inference_mode`/`no_grad` (gradients impossible); hooks removed in
  `finally`; a grey fallback overlay is returned on any internal failure so
  the clinical UI cannot crash.
- Overlay blending uses `matplotlib.colormaps[...]` with a
  `plt.get_cmap` fallback (matplotlib ≥ 3.9 removed `cm.get_cmap`), and
  normalizes/contiguous-ifies tensors before CUDA element-wise ops.

### 6.4 Explainability suite

| Method | Mechanism | Use |
|---|---|---|
| Grad-CAM++ | gradient-based, layer4 | primary explanation |
| Score-CAM | channel-mask score deltas (softmax-weighted, ≤ 64 channels, even subsample) | gradient-free baseline |
| Attention rollout | CBAM spatial-attention map (hooks `backbone.cbam.spatial_att.conv`, whose output **is** the (B,1,h,w) map) | attention visualisation |
| `explain_maps(mode=...)` | `auto` → Grad-CAM++ with Score-CAM fallback | dispatch used by `explain.py` |

### 6.5 WHO staging & narrative

`lookup_who_stage` resolves a class (+ optional grade 1–4) against the
`WHO_STAGING` table in `config.py` (grade label, typical imaging features,
malignancy, growth pattern, recommended intervention protocol). `staging.py`
adds the legally defensible framing: the engine reports the *possible grade
range* for the family — imaging alone cannot certify a WHO grade.
`generate_clinical_reasoning` assembles a deterministic narrative (findings,
probability bars, staging block, disclaimer); `narrative.py` optionally
upgrades it with a local transformer backend and always falls back to the
template.

### 6.6 Confidence safeguarding

| Band | Range | Dashboard behaviour |
|---|---|---|
| High | ≥ 90 % | green confirmation, verification reminder |
| Moderate | 75–90 % | amber caution |
| Low | 60–75 % | caution + review recommended |
| Very Low | < 60 % | red |

`is_low_confidence` (< **0.70**) triggers the mandatory red LOW-CONFIDENCE
SAFEGUARD banner, provisional-result labelling, hardened report language,
and the secondary-review flag in the PDF.

---

## 7. Clinical Dashboard (`app.py`) & API

- Standalone Streamlit app (loads the model directly via
  `st.cache_resource`) plus a FastAPI backend (`app.py` legacy API module /
  `api/main.py`) consumed by `frontend.py`.
- Uploads are staged to temp files, preprocessed through the engine's own
  contract (`FreeBuffInference.preprocess_path`), and deleted in a `finally`
  block.
- Results render in numbered clinical sections (finding → safeguards →
  spectrum → overlays → staging → reasoning → report). PDF generation
  reuses `utils.generate_pdf_report` (ReportLab: cover, prediction summary,
  confidence chart, Grad-CAM overlay, WHO staging, reasoning, disclaimer,
  checksum audit line).
- The API validates uploads (decode, size, non-black content), rejects
  corrupt payloads with 422, and returns the full schema the tests assert.

---

## 8. Evaluation Methodology (`evaluate.py`)

Artifacts produced per run: confusion matrix (counts + normalized), per-class
F1, one-vs-rest ROC-AUC curves with micro average, confidence histogram,
per-class accuracy, misclassification ledger, and Grad-CAM++ exports for up
to N misclassified cases (computed **outside** `torch.no_grad()`). Model
selection uses validation weighted F1; the test set is touched exactly once
per run.

---

## 9. Design Decisions & Limitations

**Decisions.** (1) Composite-mean multi-modal input lets one 2D classifier
serve the 3-sequence protocol on a 4 GB card; (2) Focal Loss over plain CE
for hard-example emphasis under imbalance; (3) probability-space TTA
averaging for decision stability; (4) strict partial-loading policy to
prioritise availability at the point of care; (5) temperature-scaling hook
(`fit_temperature`) exposed but off by default for reproducibility.

**Limitations (acknowledged, not hidden).** Single 2D slice per sequence —
no volumetric context (3D CNNs are VRAM-prohibitive on this GPU); dataset
bias (public-cohort demographics and scanners); composite averaging may
dilute sequence-specific cues; softmax confidence ≠ calibrated probability
absent temperature scaling; WHO grades are *reference ranges*, never
certified grades (histopathology + molecular markers required); image-level
outputs are not patient-level diagnoses.

---

## 10. Reproducibility Checklist

- Seeds: `set_all_seeds(42)` — random / NumPy / torch / CUDA / cuDNN.
- Splits: stratified, fixed `random_state`.
- PCA: training-split-only fit, artifact persisted.
- Class weights: derived deterministically from the training label counts.
- History: `evaluation_results/training_history.csv` (epoch-level metrics).
- Environment: `requirements.txt` pins torch 2.5.1 + torchvision 0.20.1
  (cu121 index); Python 3.11.
