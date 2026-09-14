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
| Inference engine | `inference.py` | TTA (48 views), Grad-CAM++/Score-CAM/attention maps, WHO staging, `FreeBuffInference` |
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
python train.py --data_dir data --num_epochs 60
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
python run.py infer <image.png>   # single image
python run.py evaluate            # full evaluation artifact suite
```

---

## 5. The Dashboard Workflow

1. **Upload** — three optional slots (T1c, T2, FLAIR) + patient/study ID.
2. **Run analysis** — TTA inference (48 views, probability-averaged) +
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
| Inference | TTA over 48 views, probability averaging, temperature 1.0 |
| Explainability | Grad-CAM++ (layer4), Score-CAM, CBAM attention maps |
| Calibration | `fit_temperature()` (LBFGS) on held-out logits — optional |
| Safeguards | < 70 % low-confidence flag; ≥ 30 % aggregate tumour alert |
| Intended use | Research / education / second-read support only |

---

## 8. Testing

```bash
python test_pipeline.py   # model I/O, Grad-CAM++ safety, TTA, evaluation suite
python test_api.py        # FastAPI endpoints (uses a temp dummy checkpoint)
python test_frontend.py   # dashboard/API client helpers (temp checkpoint)
python inference.py       # engine self-test (random input, full TTA)
```

All suites are self-contained: they create their **own** dummy checkpoints in
temporary directories and never overwrite the production
`saved_models/freebuff_v4.pt`.

---

## 9. Documentation

- [`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md) —
  full engineering specification: data flow, tensor shapes, loss mathematics,
  AMP mechanics, Grad-CAM++ derivation, design decisions, limitations.
- [`docs/DEFENSE_VIVA.md`](docs/DEFENSE_VIVA.md) — 45 defense/viva questions
  with model answers, grouped by topic (architecture, training, explainability,
  clinical safety, evaluation, ethics).

---

## 10. Regulatory & Ethical Notice

This system is a research prototype for an academic engineering project.
It **must not** be used for diagnosis, treatment planning, or any clinical
action affecting patient care. The authors accept no liability for misuse.
See `REGULATORY_DISCLAIMER` in `config.py` for the complete legal text that
accompanies every generated report.

## License

Released for educational use as part of the FreeBuff v4 research prototype.
Copyright (c) 2024–2026.
