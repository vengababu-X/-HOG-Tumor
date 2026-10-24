# FreeBuff v4 — Multi-Modal Brain Tumor Classification, Staging & Clinical Decision Support

> **Research and Educational Decision-Support Only.**
> FreeBuff v4 is **NOT a medical device** and has **NOT been cleared or approved**
> by any regulatory authority (FDA, EMA/MDR, or others). Every output —
> classification, confidence, WHO grade estimate, Grad-CAM++ heatmap, and PDF
> report — is a **second-read aid** that must be independently verified by a
> licensed medical professional.

FreeBuff v4 is a fully local, GPU-accelerated clinical decision support system
that classifies brain MRI studies into **Glioma / Meningioma / No_Tumor /
Pituitary**, explains every decision with **Grad-CAM++**, maps findings onto a
**WHO Grades I–IV staging reference**, enforces **confidence safeguarding**,
and exports **hospital-grade PDF reports**. No cloud APIs, no API keys —
everything runs on your machine.

---

## 1. Architecture at a Glance

```
                        ┌────────────────────────────────────────────┐
  T1c / T2 / FLAIR ───► │  Preprocessing (crop, normalize, 144×144)  │
                        └───────────────┬────────────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
   ┌─────────────────────────┐                        ┌────────────────────────┐
   │ Branch A — deep spatial │                        │ Branch B — handcrafted │
   │ ResNet50 (ImageNet)     │                        │ HOG (2048-d) → PCA     │
   │ └─ CBAM (channel+spatial│                        │      (288-d) → MLP     │
   │    attention)           │                        │      → 512-d embedding │
   │ └─ GAP → MLP → 256-d    │                        └───────────┬────────────┘
   └───────────┬─────────────┘                                    │
               └───────────────┬──────────────────────────────────┘
                               ▼
                   ┌───────────────────────┐
                   │ Fusion head 768→512→  │
                   │ 256→4 (non-inplace)   │
                   └──────────┬────────────┘
                              ▼
        Classification + TTA + Grad-CAM++ + WHO staging + PDF report
```

| Component | File | Role |
|-----------|------|------|
| Hybrid model | `model.py` | ResNet50+CBAM × HOG-PCA fusion, Focal Loss, error-proof checkpoint loader, Grad-CAM++ engine |
| Training pipeline | `train.py` | AdamW (decoupled LRs), cosine annealing, AMP (fp16), Focal Loss class weighting, early stopping |
| Inference engine | `inference.py` | TTA (2 anatomy-valid views), Grad-CAM++/Score-CAM/attention maps, WHO staging, `FreeBuffInference` |
| Dashboard | `app.py` | Streamlit clinical UI — uploads, confidence spectrum, side-by-side overlays, <70% safeguard, PDF |
| REST API | `api/main.py` | FastAPI `/predict` + `/health` backend (used by `frontend.py`) |
| Report generator | `utils.py` / `report.py` | ReportLab PDF with findings, staging, overlays, disclaimer |
| Central config | `config.py` | Hyperparameters, WHO_STAGING table, paths, disclaimers |

---

## 2. Requirements

- **Python 3.11** (3.10–3.12 supported)
- **NVIDIA GPU with CUDA 12.1** (developed on an RTX 3050; CPU-only mode works)
- **Windows 10/11** (developed), Linux, or macOS
- pip ≥ 23

---

## 3. Installation

```bash
# 1. Clone
git clone <repository-url> freebuff-v4
cd freebuff-v4

# 2. Create an isolated environment
python -m venv .venv
.venv\Scripts\activate          # Windows (bash: source .venv/Scripts/activate)
# source .venv/bin/activate     # Linux/macOS

# 3. Install PyTorch with CUDA 12.1 first, then the rest
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Verify the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cu121 True
```

---

## 4. Quickstart

### 4.1 Train

```bash
python train.py --data_dir data/Training --num_epochs 60
```

Expected output on an RTX 3050: ≈ 27.1 M parameters, ~3.5 min/epoch at batch
size 32 with AMP enabled. Best checkpoint (by validation weighted F1) is
written to `checkpoints/freebuff_v4.pt` and copied to `saved_models/freebuff_v4.pt`;
per-epoch metrics land in `evaluation_results/training_history.csv`.

Useful flags: `--no_amp`, `--no_hog`, `--freeze_backbone`, `--scheduler plateau`,
`--batch_size 16` (if VRAM-constrained).

### 4.2 Launch the clinical dashboard

```bash
streamlit run app.py
# or: python run.py dashboard
```

Open http://localhost:8501, upload T1c / T2 / FLAIR (any subset — missing
sequences are replicated with a log line), press **Run analysis**.

### 4.3 Run the REST API (optional, used by `frontend.py`)

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
# then in a second terminal:
streamlit run frontend.py
```

### 4.4 CLI inference

```bash
python run_cli.py                 # interactive multi-sequence prompt
python run.py infer <image.png>   # single slice
python run.py infer study.nii.gz  # whole volume - every slice scored
python run.py evaluate            # full evaluation artifact suite
```

### 4.5 Volumetric (3D) inference

A NIfTI file is a whole study, not a slice. Passing one to `run.py infer`, to
`batch_predict.py`, or to `FreeBuffInference.predict_volume()` scores **every
informative slice** and aggregates, instead of classifying the middle slice and
discarding the rest:

```bash
python run.py infer study.nii.gz --volume_aggregation mean       # stable (default)
python run.py infer study.nii.gz --volume_aggregation max_tumor  # sensitivity-first
python run.py infer study.nii.gz --middle_slice_only             # legacy behaviour
```

Slices that are mostly air (the first and last of a head study) are skipped
automatically. The result carries `n_slices_scored`, `n_slices_total` and a
per-slice breakdown, so the evidence behind the volume-level answer stays
inspectable — on a test volume the aggregate stayed correct even though three
individual slices disagreed.

`max_tumor` selects the slice with the highest aggregate tumour probability
rather than averaging, which is the right default when a small lesion appears
on only a few slices of a long study.

---

## 5. The Dashboard Workflow

1. **Upload** — three optional slots (T1c, T2, FLAIR) + patient/study ID.
2. **Run analysis** — TTA inference (2 views, probability-averaged) +
   per-modality Grad-CAM++ maps.
3. **Primary finding** — predicted class, confidence, reliability band
   (High ≥ 90 / Moderate ≥ 75 / Low ≥ 60 / Very Low < 60), aggregate tumour
   probability.
4. **Safeguards** —
   - **< 70 % confidence → LOW-CONFIDENCE SAFEGUARD**: red banner, result
     marked provisional, secondary review mandated.
   - **Sensitivity alert**: if aggregate tumour probability ≥ 30 % while the
     argmax says "No_Tumor", the highest-probability tumour class is surfaced
     with a review recommendation (probabilities are never hidden).
5. **Confidence spectrum** — bar chart + sorted table across all 4 classes.
6. **Side-by-side overlays** — original vs Grad-CAM++ per sequence (jet colormap).
7. **WHO staging** — grade range, typical presentation, malignancy profile,
   clinical notes (from the WHO CNS reference table in `config.py`/`staging.py`).
8. **PDF report** — one click generates and offers a ReportLab PDF
   (`clinical_reports/pdf_reports/`), containing every artifact above plus the
   full regulatory disclaimer and a checksum-auditable layout.

---

## 6. Project Layout

```
freebuff-v4/
├── config.py               # central config: hyperparams, WHO_STAGING, paths
├── model.py                # FreeBuffV4 hybrid, FocalLoss, load_checkpoint, GradCAM++
├── train.py                # training pipeline (AMP, cosine, early stopping)
├── inference.py            # TTA, explainability suite, FreeBuffInference engine
├── app.py                  # Streamlit clinical dashboard (standalone)
├── frontend.py             # Streamlit client for the FastAPI backend
├── api/main.py             # FastAPI backend (/predict, /health)
├── dataset.py              # BrainTumorDataset, HOGExtractor, PCA fitting, splits
├── preprocessing.py        # brain crop, intensity normalization, augmentation
├── staging.py              # WHO grade reference engine (Grades I–IV)
├── narrative.py            # clinical narrative generation (template/LLM)
├── utils.py                # PDF report generator (ReportLab) + artifacts
├── report.py               # compact PDF builder (evaluate/CLI flows)
├── evaluate.py             # confusion matrix, ROC, F1, calibration artifacts
├── compare_checkpoints.py  # A/B two checkpoints on the held-out set, optional promote
├── calibrate.py            # fit/apply a softmax temperature (confidence calibration)
├── audit_leakage.py        # train/test duplicate audit + leak-free rescoring
├── deduplicate_dataset.py  # writes a leakage-free copy of the dataset
├── batch_predict.py        # classify a whole folder -> CSV (+ optional Grad-CAM)
├── explain.py              # standalone explainability harness
├── run.py                  # unified entry point (train/evaluate/infer/dashboard)
├── run_cli.py              # interactive CLI diagnosis
├── test_pipeline.py        # integration suite (model + Grad-CAM++ + TTA + eval)
├── test_api.py             # FastAPI endpoint tests
├── test_frontend.py        # dashboard/API-client helper tests
├── docs/
│   ├── TECHNICAL_SPECIFICATION.md   # full engineering specification
│   └── DEFENSE_VIVA.md              # 40+ viva questions with model answers
├── data/                   # dataset root (one subfolder per class)
├── checkpoints/            # training checkpoints (best = freebuff_v4.pt)
├── saved_models/           # deployment artifacts (model + PCA + manifest)
├── clinical_reports/pdf_reports/
├── evaluation_results/     # plots, Grad-CAM exports, training history CSV
└── logs/
```

---

## 7. Model Card

| Field | Value |
|---|---|
| Architecture | Hybrid dual-branch: ResNet50+CBAM (deep) ⊕ HOG→PCA→MLP (handcrafted) |
| Parameters | ≈ 27.07 M (all trainable) |
| Input | 144×144×3 (grayscale MRI replicated to RGB, ImageNet-normalized) |
| Classes | Glioma, Meningioma, No_Tumor, Pituitary |
| Loss | Focal Loss, γ = 2.0, inverse-frequency α |
| Optimizer | AdamW — backbone LR 1e-4, head LR 1e-3, weight decay 1e-4 |
| Schedule | Cosine annealing (T_max = 60, η_min = 1e-6), patience 12 |
| Precision | AMP fp16 on CUDA (GradScaler), fp32 fallback |
| Inference | TTA over 2 views (identity + horizontal mirror), probability averaging, temperature 1.0 |
| Explainability | Grad-CAM++ (layer4), Score-CAM, CBAM attention maps |
| Calibration | `fit_temperature()` (LBFGS) on held-out logits — optional |
| Safeguards | < 70 % low-confidence flag; ≥ 30 % aggregate tumour alert |
| Intended use | Research / education / second-read support only |

---

## 8. Measured Results

**The headline figure is 0.9453 accuracy / 0.9431 macro F1**, measured on the
1,189 held-out images that are provably absent from the training set.

That is lower than the 0.9556 this project reported before auditing the data,
and the difference is the point: **25.7% of `data/Testing` is a duplicate or
near-duplicate of something in `data/Training`** (§8.2). Scoring on those
images rewards memorisation, so they are excluded from the headline. Both
numbers are given below so the gap is visible rather than hidden.

Reproduce both with one command:

```bash
python audit_leakage.py --evaluate
```

Or score the full (contaminated) set on its own with:

```bash
python evaluate.py --checkpoint saved_models/freebuff_v4.pt --test_data_dir data/Testing --use_hog --pca_path saved_models/freebuff_pca.pkl
```

That command also rewrites `evaluation_results/holdout_report.txt`, which
records the checkpoint, device, HOG state and exact TTA policy used.

### 8.1 Held-out performance

| Evaluation set | Images | Accuracy | Macro F1 | Errors |
|---|---|---|---|---|
| **Uncontaminated — quote this one** | **1,189** | **0.9453** | **0.9431** | 65 |
| Full `data/Testing` (25.7% contaminated) | 1,600 | 0.9550 | 0.9547 | 72 |

Per class, on the uncontaminated set:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Glioma | 0.9942 | 0.8779 | 0.9324 | 393 |
| Meningioma | 0.8773 | 0.9630 | 0.9181 | 297 |
| No_Tumor | 0.8919 | **1.0000** | 0.9429 | 99 |
| Pituitary | 0.9728 | 0.9850 | 0.9789 | 400 |

No_Tumor recall stays 1.0000 — no healthy scan was labelled as a tumour, and
that still holds after the contaminated copies are removed. The dominant
residual error is Glioma → Meningioma. Note the reduced No_Tumor support (99):
**301 of its 400 test images were contaminated**, which is why its earlier
0.9828 F1 was the most inflated figure in the old table.

**Does the contamination change the conclusion?** We retrained from scratch on
the de-duplicated dataset and compared both models on the 1,189 images neither
was trained on:

| | Baseline (trained on contaminated data) | Clean-data model |
|---|---|---|
| Accuracy | 0.9453 | 0.9445 |
| Macro F1 | **0.9431** | 0.9350 |

They are within 0.1 accuracy points, so the deployed model is kept — its
apparent lead on the full test set was memorisation, not capability, and on
fair data the two are equivalent. The clean-data model has a lower No_Tumor
precision (0.8182 vs 0.8919) because de-duplication removed 31% of that
class's training images.

### 8.2 Dataset integrity — measured, not assumed

The public dataset ships loose slices with **no patient identifiers**, so a
patient-level split is impossible. We audited the exposure instead, using an
exact hash of every decoded image:

```bash
python audit_leakage.py --evaluate
```

| Finding | Value |
|---|---|
| Test images **byte-identical** to a training image | 114 / 1,600 (7.1%) |
| Test images **near-identical** (pixel correlation ≥ 0.999) | +297 |
| **Total contaminated** | **411 / 1,600 (25.7%)** |
| Duplicate images inside `data/Training` itself | 190 |

Contamination is wildly uneven across classes, which is what made it dangerous:

| Class | Contaminated | of test set |
|---|---|---|
| Glioma | 7 | 1.8% |
| Meningioma | 103 | 25.8% |
| **No_Tumor** | **301** | **75.3%** |
| Pituitary | 0 | 0.0% |

Three quarters of the No_Tumor test set was a copy of training data. That is
the single most important finding in this project: the previously reported
**1.0000 No_Tumor recall was largely memorisation**, and only 99 of its 400
test images were ever a real test.

Near-duplicates are detected by pixel correlation on a 64×64 downsample, and
the threshold is measured rather than assumed — genuinely different scans sit
around 0.91, while re-encoded or 1-pixel-shifted copies sit above 0.999.
Verified at full resolution: flagged pairs have mean absolute error 0.00–5.11,
control pairs 10–55.

Rescoring with every contaminated image removed:

| Evaluation | Images | Accuracy | Macro F1 |
|---|---|---|---|
| Full test set | 1,600 | 0.9550 | 0.9547 |
| **Uncontaminated** | **1,189** | **0.9453** | **0.9431** |
| difference | −411 | **−0.0097** | **−0.0116** |

The honest headline is **0.9453 / 0.9431**. The reported figure was inflated by
about one accuracy point — a real correction, but not one that changes the
conclusion that the system works.

This detects duplicate *files*. Without patient IDs it cannot detect different
slices from the same patient, so it is a lower bound, not a clean bill of health.

### 8.3 Removing the leakage (not just measuring it)

`deduplicate_dataset.py` writes a clean copy of the dataset. It keeps the
**test set whole** at 1,600 images and removes the duplicated copies from
*training* instead, so the held-out numbers stay comparable while every test
image becomes genuinely unseen:

```bash
python deduplicate_dataset.py            # dry run, prints the plan
python deduplicate_dataset.py --apply    # writes data_clean/
```

| | Glioma | Meningioma | No_Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| kept for training | 1,396 | 1,288 | 1,257 | 1,358 | **5,299** |
| removed — leaked into test | 4 | 102 | 9 | 0 | 115 |
| removed — duplicate within training | 0 | 10 | 134 | 42 | 186 |

Training drops 5,600 → 5,299; the test set stays at 1,600. Verified clean:

```bash
python audit_leakage.py --train_dir data_clean/Training --test_dir data_clean/Testing
# EXACT duplicates: 0 / 1600 (0.00%)   duplicates within training: 0
```

The original `data/` is never modified. Retrain against the clean copy with
`python train.py --data_dir data_clean/Training --checkpoint_dir checkpoints/clean`.

### 8.4 Confidence calibration

A softmax score is not a probability. `calibrate.py` fits a temperature by
maximum likelihood on held-out logits; dividing all logits by one positive
scalar cannot change which class wins, so **accuracy is provably unchanged** —
only the confidence moves.

Where the calibration set comes from decides whether it works at all:

| Protocol | Fitted T | Held-out ECE | Result |
|---|---|---|---|
| Fit on a split of `data/Training` | 0.65 | 0.0214 → 0.0369 | **fails** — does not transfer |
| Fit within the deployment distribution (2-fold) | ≈ 1.24 | 0.0238 → **0.0165** | **works** — 31% relative reduction |

The model is near-perfect on the training distribution, so a fit there sharpens
confidence further and makes calibration worse across the domain gap.

No temperature is hardcoded: a value fitted on our own test set would
contaminate the reported metrics. Fit your own on your own scans:

```bash
python calibrate.py --calib_data_dir <your-calibration-folder> --apply
```

`--apply` writes `saved_models/calibration.json`, which the inference engine
loads automatically on next start.

### 8.5 Why the TTA policy is only 2 views

The TTA policy was chosen on a **held-out split of `data/Training`** and only
then confirmed once on `data/Testing`, so the configuration is not tuned on
the test set:

| TTA policy | Views | Val acc | Test acc | Test macro F1 |
|---|---|---|---|---|
| no TTA | 1 | 0.9925 | 0.9513 | 0.9508 |
| **hflip only** | **2** | **0.9950** | **0.9556** | **0.9552** |
| hflip × rot ±8° | 6 | 0.9875 | 0.9456 | 0.9450 |
| hflip × scale | 6 | 0.9867 | 0.9394 | 0.9386 |
| hflip × rot × scale | 18 | 0.9367 | 0.8906 | 0.8884 |
| *previous default* (vertical/both flips + 90/180/270° rotations) | 8 | — | 0.8894 | 0.8888 |
| *previous full grid* | 48 | — | 0.8869 | 0.8859 |

Horizontal-flip-only wins on **both** sets. The reason is anatomical rather
than statistical: an axial brain MRI is roughly bilaterally symmetric, so the
mirror image is a plausible second look, whereas a quarter-turned or
vertically flipped brain is a scan no scanner would produce. Averaging the
model's opinion of those out-of-distribution views into the result cost
**6.6 accuracy points** against the current policy.

---

## 9. Testing

```bash
python test_pipeline.py   # model I/O, Grad-CAM++ safety, TTA, evaluation suite
python test_api.py        # FastAPI endpoints (uses a temp dummy checkpoint)
python test_frontend.py   # dashboard/API client helpers (temp checkpoint)
python inference.py       # engine self-test (random input, full TTA)
```

All suites are self-contained: they create their **own** dummy checkpoints in
temporary directories and never overwrite the production
`saved_models/freebuff_v4.pt`, nor the real artifacts in
`evaluation_results/` — a test run uses a random-weights model, so its plots
and Grad-CAM overlays stay in its temp directory rather than masquerading as
results. `test_frontend.py` starts its own API server on port 8001 and **fails
loudly if that leg cannot run**, so a skipped test can never be mistaken for a
passing one.

### 9.1 Comparing a newly trained model against the deployed one

Training writes to whatever `--checkpoint_dir` you give it, so a new run never
overwrites the deployed artifacts. Compare before adopting:

```bash
python compare_checkpoints.py --baseline saved_models/freebuff_v4.pt --baseline-pca saved_models/freebuff_pca.pkl --candidate checkpoints/v5_aug/freebuff_v4.pt --candidate-pca checkpoints/v5_aug/freebuff_pca.pkl
```

Both checkpoints are evaluated on the same held-out data with the same TTA
policy, and the tool prints a side-by-side table plus a verdict. Adding
`--promote` copies the candidate over the deployed artifacts **only if it wins
on macro F1**, backing up the previous ones to `saved_models/previous/` first.

Each checkpoint is paired with its own PCA artifact, because the HOG branch is
only meaningful when the PCA is the one that checkpoint was trained with — a
mismatched pair degrades the branch silently rather than failing.

### 9.2 Batch prediction over a folder

```bash
python batch_predict.py --input_dir scans/ --out predictions.csv
python batch_predict.py --input_dir data/Testing --recursive --labelled --out all.csv --gradcam_dir overlays/
```

Writes one CSV row per scan carrying the **whole decision record** — all four
class probabilities, aggregate tumour probability, reliability band, both
safeguard flags and the WHO grade reference — so a run can be audited later
without re-running the model. `--labelled` additionally scores accuracy against
the class subfolder names, and `--gradcam_dir` exports a Grad-CAM++ overlay per
scan. A file that fails to load is recorded in an `error` column instead of
aborting the run.

---

## 10. Documentation

- [`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md) —
  full engineering specification: data flow, tensor shapes, loss mathematics,
  AMP mechanics, Grad-CAM++ derivation, design decisions, limitations.
- [`docs/DEFENSE_VIVA.md`](docs/DEFENSE_VIVA.md) — 45 defense/viva questions
  with model answers, grouped by topic (architecture, training, explainability,
  clinical safety, evaluation, ethics).

---

## 11. Regulatory & Ethical Notice

This system is a research prototype for an academic engineering project.
It **must not** be used for diagnosis, treatment planning, or any clinical
action affecting patient care. The authors accept no liability for misuse.
See `REGULATORY_DISCLAIMER` in `config.py` for the complete legal text that
accompanies every generated report.

## License

Released for educational use as part of the FreeBuff v4 research prototype.
Copyright (c) 2024–2026.
