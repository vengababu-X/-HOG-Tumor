#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4: Multi-Modal Brain Tumor Classification & Clinical Decision Support
# ==============================================================================
# Central configuration module for hyperparameters, paths, WHO staging, and
# global constants used across training, inference, evaluation, and reporting.
# ==============================================================================
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import torch


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("freebuff_v4")


# ==============================================================================
# RANDOM SEEDING FOR REPRODUCIBILITY
# ==============================================================================
def set_all_seeds(seed: int = 42) -> None:
    """
    Enforce deterministic behaviour across PyTorch, NumPy, and stdlib random.

    Args:
        seed: Integer seed to use for all RNG subsystems.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# DATASET & LABEL DEFINITIONS
# ==============================================================================
# Ordered class list used consistently across datasets, models, and reports.
CLASSES: List[str] = ["Glioma", "Meningioma", "No_Tumor", "Pituitary"]

# Integer label -> human-readable class name mapping (0-indexed).
LABEL_TO_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(CLASSES)}

# Human-readable class name -> integer label mapping.
NAME_TO_LABEL: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}

# Number of target classification categories.
NUM_CLASSES: int = len(CLASSES)

# Background label used for segmentation mask semantics (not used by classifier).
BACKGROUND_LABEL: int = 0

# ==============================================================================
# IMAGE & VOLUME DIMENSIONS
# ==============================================================================
# Target spatial resolution for all pre-processed 2D slices (square).
IMG_SIZE: int = 144

# Tuple form used by torchvision transforms and model input contracts.
INPUT_SIZE: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)

# Number of input channels for the raw MRI slice (grayscale converted to RGB
# for ImageNet compatibility when loading pretrained backbones).
NUM_CHANNELS: int = 3

# Original DICOM/magnetic resonance spatial resolution hints (for metadata).
DEFAULT_SLICE_THICKNESS_MM: float = 5.0
DEFAULT_PIXEL_SPACING: Tuple[float, float] = (0.9375, 0.9375)

# ==============================================================================
# PRE-PROCESSING & HOG PARAMETERS
# ==============================================================================
# HOG cell geometry aligned to the 144x144 input grid.
HOG_CELL_SIZE: int = 16
HOG_BLOCK_SIZE: int = 2          # cells per block (2x2)
HOG_BLOCK_STEP: int = 1          # block stride in cells
HOG_ORIENTATIONS: int = 8        # number of orientation bins

# Total HOG feature dimension before PCA reduction (computed internally).
HOG_DIMS_RAW: int = 2048

# PCA projection target dimension for the handcrafted texture branch.
HOG_PCA_COMPONENTS: int = 288

# Binarization threshold for converting probability to hard class decision.
DEFAULT_NDCG_THRESHOLD: float = 0.5

# Intensity normalization ranges (min, max) applied after clipping.
INTENSITY_CLIP_MIN: float = 0.0
INTENSITY_CLIP_MAX: float = 255.0
INTENSITY_MEAN: float = 0.485   # ImageNet mean for 3-channel RGB-like tensors
INTENSITY_STD: float = 0.229

# Intensity normalization constants used by torchvision Normalize (RGB order).
NORMALIZE_MEAN: Tuple[float, float, float] = (INTENSITY_MEAN, INTENSITY_MEAN, INTENSITY_MEAN)
NORMALIZE_STD: Tuple[float, float, float] = (INTENSITY_STD, INTENSITY_STD, INTENSITY_STD)

# Intensity windowing for MRI brain window defaults (legacy clinical reference,
# applied only if explicitly requested by a preprocessing flag).
MRI_WINDOW_LEVEL: float = 40.0
MRI_WINDOW_WIDTH: float = 80.0

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
# Default batch size for DataLoaders. Tuned for ASUS RTX 3050 VRAM (4GB).
BATCH_SIZE: int = 32

# Number of worker processes for parallel data loading (0 on Windows for
# compatibility with fork-safety concerns).
NUM_WORKERS: int = 0

# Maximum number of training epochs.
NUM_EPOCHS: int = 60

# Early stopping patience measured in validation epochs without improvement.
EARLY_STOPPING_PATIENCE: int = 12

# Learning rate for the classifier head / fusion layers.
HEAD_LR: float = 1e-3

# Learning rate for the ResNet50 backbone (decoupled from head).
BACKBONE_LR: float = 1e-4

# Combined default learning rate when decoupled scheduling is not active.
DEFAULT_LR: float = 1e-4

# AdamW epsilon term for numerical stability.
ADAMW_EPS: float = 1e-8

# AdamW weight decay (L2 regularization) applied to all trainable params.
WEIGHT_DECAY: float = 1e-4

# Global gradient clipping maximum norm to prevent exploding gradients.
GRADIENT_CLIP_NORM: float = 1.0

# Beta parameters for AdamW first/second moment estimates.
ADAMW_BETA1: float = 0.9
ADAMW_BETA2: float = 0.999

# ==============================================================================
# LOSS FUNCTION PARAMETERS
# ==============================================================================
# Focal Loss modulating factor (gamma) for hard-example emphasis.
FOCAL_GAMMA: float = 2.0

# Focal Loss balancing factor (alpha). None = no class rebalancing.
FOCAL_ALPHA: float | None = None

# Label smoothing epsilon applied inside the cross-entropy term (disabled when
# using pure Focal Loss without CE smoothing overlap).
LABEL_SMOOTHING: float = 0.0

# Entropy penalty weight on the classifier output (used optionally for
# confidence calibration, disabled by default).
ENTROPY_REG_WEIGHT: float = 0.0

# ==============================================================================
# LEARNING RATE SCHEDULER
# ==============================================================================
# Cosine annealing scheduler parameters.
COSINE_T_MAX: int = 60          # total number of epochs for one cosine cycle
COSINE_ETA_MIN: float = 1e-6    # minimum learning rate at the cosine nadir

# Reduce-on-plateau scheduler parameters (fallback path).
PLATEAU_FACTOR: float = 0.5
PLATEAU_PATIENCE: int = 6
PLATEAU_COOLDOWN: int = 2

# Milestones for multi-step scheduler variants (unused by default).
MILESTONE_EPOCHS: List[int] = []

# ==============================================================================
# CLINICAL CONFIDENCE SAFEGUARDS
# ==============================================================================
# Predictions below this softmax confidence trigger the low-confidence
# safeguard in the Streamlit dashboard / API (mandatory second-read banner,
# hardened report language, and secondary-review flag).
LOW_CONFIDENCE_THRESHOLD: float = 0.70

# Aggregate tumour probability at/above which the sensitivity-first safety
# alert fires (sum of non-'notumor' class probabilities).
TUMOR_ALERT_THRESHOLD: float = 0.30

# ==============================================================================
# NARRATIVE GENERATION (narrative.py)
# ==============================================================================
# Master switch for transformer-backed narrative generation. When False (or
# when the optional backend is unavailable) narrative.py falls back to the
# deterministic clinical template so the pipeline never hard-fails.
NARRATIVE_ENABLED: bool = False

# Attempt the local transformer backend before the deterministic template.
NARRATIVE_MODEL_BACKEND: str = "template"     # "template" | "transformers"

# Optional second-pass rephrasing model (kept off by default for speed and
# reproducibility on the 4 GB RTX 3050).
NARRATIVE_REPHRASER_BACKEND: str = "template"  # "template" | "transformers"
NARRATIVE_REPHRASER_MODEL_NAME: str = "microsoft/phi-3-mini-4k-instruct"

# If the transformer backend errors out, degrade to the template instead of
# raising (clinical-facing surfaces must always produce a narrative).
NARRATIVE_FALLBACK_TO_TEMPLATE: bool = True

# Phrase injected when the model output is judged inconclusive.
NARRATIVE_INCONCLUSIVE_PHRASE: str = (
    "The automated narrative could not be reliably generated for this case; "
    "the templated clinical summary below applies."
)

# Extra cautionary sentence appended when confidence < LOW_CONFIDENCE_THRESHOLD.
NARRATIVE_LOW_CONFIDENCE_WARN: str = (
    "Model confidence for this case is below the 70% reliability threshold; "
    "the finding should be treated as provisional and requires full "
    "radiologist re-review before any clinical weight is placed on it."
)

# Decoding / token-budget limits for the optional local LLM backend.
NARRATIVE_MAX_INPUT_TOKEN_BUDGET: int = 2048
NARRATIVE_MAX_NEW_TOKENS: int = 512
NARRATIVE_TEMPERATURE: float = 0.2
NARRATIVE_TOP_P: float = 0.9

# ==============================================================================
# TEST-TIME AUGMENTATION (TTA) CONFIGURATION
# ==============================================================================
# TTA policy set applied during evaluation and inference to reduce variance.
TTA_FLIPS: List[str] = ["none", "horizontal", "vertical", "both"]
TTA_ROTATIONS_DEG: List[int] = [0, 90, 180, 270]
TTA_SCALE_FACTORS: List[float] = [1.0, 1.05, 0.95]

# Whether to enable full TTA (flips + rotations + scale) or a reduced set.
TTA_FULL: bool = True

# Number of TTA samples aggregated per input image (deterministic subset).
TTA_NUM_SAMPLES: int = 8

# Smoothing factor applied to TTA-averaged logits (post-softmax temperature).
TTA_SOFTMAX_TEMP: float = 1.0

# ==============================================================================
# GRAD-CAM++ EXPLAINABILITY PARAMETERS
# ==============================================================================
# Target convolutional layer for Grad-CAM++ gradient computation.
GRADCAM_TARGET_LAYER_NAME: str = "layer4"

# Interpolation mode used when resizing heatmaps back to original image size.
GRADCAM_INTERPOLATION_MODE: str = "bilinear"

# Alpha blending weight between the original image and the heatmap overlay.
GRADCAM_ALPHA: float = 0.5

# Colormap name used to render the heatmap (matplotlib colour map identifier).
GRADCAM_COLORMAP: str = "jet"

# Scaling range for the normalized heatmap intensity (min, max).
GRADCAM_INTENSITY_CLIP: Tuple[float, float] = (0.0, 1.0)

# ==============================================================================
# STAGING & CLINICAL DECISION SUPPORT
# ==============================================================================
# WHO grade definitions for each tumour class (imaging-based ranges).
# Each grade entry stores: grade_label, typical_imaging_features, malignancy,
# growth_pattern, and recommended_intervention_protocol.
WHO_STAGING: Dict[str, Dict[int, Dict[str, str]]] = {
    "Glioma": {
        1: {
            "grade_label": "WHO Grade I (Pilocytic Astrocytoma)",
            "typical_imaging_features": (
                "Well-circumscribed, cystic with mural nodule, "
                "homogeneous post-contrast enhancement, minimal peritumoral "
                "oedema, no necrosis, atypical only in paediatric posterior fossa."
            ),
            "malignancy": "Low-grade, infrequently malignant; potential for "
                           "complete surgical cure with gross-total resection.",
            "growth_pattern": "Slow-growing, expansile, usually non-infiltrative.",
            "recommended_intervention_protocol": (
                "Surgical resection with intent for gross-total removal; "
                "observation with serial MRI if subtotal resection in asymptomatic "
                "patient; adjuvant therapy generally not required for Grade I."
            ),
        },
        2: {
            "grade_label": "WHO Grade II (Diffuse Astrocytoma / Oligodendroglioma)",
            "typical_imaging_features": (
                "Ill-defined margins, infiltrative, mild-to-moderate oedema, "
                "heterogeneous or minimal enhancement, possible calcification "
                "(oligodendroglioma), no significant necrosis."
            ),
            "malignancy": "Low-grade but diffusely infiltrative; high rate of "
                           "progression to higher-grade glioma over time.",
            "growth_pattern": "Infiltrative along white matter tracts; slow but "
                              "persistent growth with tendency to recur.",
            "recommended_intervention_protocol": (
                "Maximal safe surgical resection followed by observation or "
                "early adjuvant radiotherapy/chemotherapy depending on molecular "
                "profile (IDH mutation, 1p/19q co-deletion); close MRI surveillance."
            ),
        },
        3: {
            "grade_label": "WHO Grade III (Anaplastic Glioma)",
            "typical_imaging_features": (
                "Irregular enhancement, increased cellularity, moderate-to-severe "
                "peritumoral oedema, possible central necrosis, mass effect."
            ),
            "malignancy": "Malignant, rapidly progressive, higher recurrence risk.",
            "growth_pattern": "Aggressive, infiltrative, anaplastic cellular "
                              "features on histology correlating with imaging.",
            "recommended_intervention_protocol": (
                "Aggressive surgical debulking followed by radiotherapy with "
                "concurrent and adjuvant temozolomide (Stupp protocol) or "
                "PCV chemotherapy based on molecular markers; multidisciplinary "
                "neuro-oncology review."
            ),
        },
        4: {
            "grade_label": "WHO Grade IV (Glioblastoma / Glioblastoma Multiforme)",
            "typical_imaging_features": (
                "Ring-enhancing lesion with thick irregular wall, central "
                "necrosis, extensive peritumoral oedema, significant mass "
                "effect, crossing corpus callosum (butterfly appearance)."
            ),
            "malignancy": "Most malignant primary brain tumour; highly "
                           "aggressive, rapid progression, poor prognosis "
                           "despite maximal therapy.",
            "growth_pattern": "Rapid, neovascularized, necrotic, highly "
                              "infiltrative with microscopic extension beyond "
                              "the radiographic margin.",
            "recommended_intervention_protocol": (
                "Maximal safe surgical resection followed by radiotherapy with "
                "concurrent and adjuvant temozolomide (Stupp protocol); consider "
                "tumour treating fields (TTFields), clinical trials, and "
                "palliative care integration; multidisciplinary neuro-oncology "
                "and neurosurgery decision-making essential."
            ),
        },
    },
    "Meningioma": {
        1: {
            "grade_label": "WHO Grade I (Benign Meningioma)",
            "typical_imaging_features": (
                "Extra-axial, dural-based, homogeneous enhancement, "
                "well-circumscribed, dural tail sign, calcifications common, "
                "no or minimal oedema."
            ),
            "malignancy": "Benign, slow-growing, favourable prognosis after "
                           "complete resection.",
            "growth_pattern": "Slow, expansile, compressive rather than "
                              "infiltrative; remodelling of adjacent bone.",
            "recommended_intervention_protocol": (
                "Observation with serial MRI for small asymptomatic tumours; "
                "surgical resection for symptomatic or growing lesions; "
                "radiosurgery (Gamma Knife / SBRT) for residual or recurrent "
                "small tumours; adjuvant therapy generally not required."
            ),
        },
        2: {
            "grade_label": "WHO Grade II (Atypical Meningioma)",
            "typical_imaging_features": (
                "Higher enhancement heterogeneity, possible bony invasion, "
                "more extensive oedema, indistinct borders, higher mitotic "
                "activity correlates with imaging aggressiveness."
            ),
            "malignancy": "Atypical with higher recurrence risk after subtotal "
                           "resection; occasional malignant transformation.",
            "growth_pattern": "More infiltrative than Grade I, higher propensity "
                              "for local recurrence.",
            "recommended_intervention_protocol": (
                "Aggressive surgical resection with wide dural margin; adjuvant "
                "radiotherapy (external beam or stereotactic) for subtotal "
                "resection or high-risk features; closer MRI follow-up interval."
            ),
        },
        3: {
            "grade_label": "WHO Grade III (Anaplastic / Malignant Meningioma)",
            "typical_imaging_features": (
                "Rapid growth, necrosis, heterogeneous enhancement, "
                "significant peritumoral oedema, local invasion, distant "
                "metastasis rare but possible."
            ),
            "malignancy": "Rare, aggressively malignant, high recurrence rate, "
                           "poor prognosis.",
            "growth_pattern": "Rapidly progressive, invasive, destructive to "
                              "adjacent structures.",
            "recommended_intervention_protocol": (
                "Aggressive surgical resection with goal of complete removal; "
                "post-operative radiotherapy (often with higher dose); consider "
                "chemotherapy and clinical trials; palliative care discussion "
                "early in management."
            ),
        },
        4: {
            "grade_label": "WHO Grade IV (Not routinely assigned for Meningioma)",
            "typical_imaging_features": (
                "Reserved for rare sarcomatous or pro-dromal transformation "
                "patterns; extremely aggressive imaging features analogous to "
                "high-grade glial lesions."
            ),
            "malignancy": "Extremely rare, highly aggressive phenotype; "
                           "management extrapolated from high-grade glioma and "
                           "other aggressive sarcoma protocols.",
            "growth_pattern": "Extremely rapid, destructive, with potential for "
                              "CSF spread.",
            "recommended_intervention_protocol": (
                "Highly individualized, multidisciplinary decision-making; "
                "aggressive resection if feasible; consider high-dose "
                "radiotherapy, systemic therapy trials, and early palliative "
                "care integration. Explicit neurosurgical and neuro-oncology "
                "review required."
            ),
        },
    },
    "Pituitary": {
        1: {
            "grade_label": "WHO Grade I (Benign Non-Functioning / Small Adenoma)",
            "typical_imaging_features": (
                "Well-circumscribed sellar/suprasellar lesion, homogeneous or "
                "mildly heterogeneous enhancement, possible microadenoma "
                "<10 mm, minimal or no cavernous sinus invasion."
            ),
            "malignancy": "Benign pituitary adenoma; rarely malignant. Hormone "
                           "secretion status drives management more than grade.",
            "growth_pattern": "Slow, expansile, may compress optic chiasm.",
            "recommended_intervention_protocol": (
                "Endocrine evaluation and hormone panel; observation for "
                "incidental asymptomatic microadenomas; surgical resection "
                "(transsphenoidal) for visual compromise or hormone-secreting "
                "tumours; medical therapy (dopamine agonists for prolactinomas); "
                "radiation reserved for refractory cases."
            ),
        },
        2: {
            "grade_label": "WHO Grade II (Atypical / Functioning Adenoma with Invasion)",
            "typical_imaging_features": (
                "Larger macroadenoma with cavernous sinus invasion (Knosp "
                "grade 3-4), heterogeneous enhancement, possible cystic "
                "components, compression of adjacent structures."
            ),
            "malignancy": "Atypical behaviour, higher recurrence, potential for "
                           "hormonal hypersecretion syndromes.",
            "growth_pattern": "Invasive toward cavernous sinus and skull base; "
                              "slower than malignancy but locally aggressive.",
            "recommended_intervention_protocol": (
                "Multidisciplinary endocrinology and neurosurgery planning; "
                "surgical debulking for decompression; medical therapy for "
                "hormone control; stereotactic radiosurgery for residual "
                "invasive component; long-term endocrine surveillance."
            ),
        },
        3: {
            "grade_label": "WHO Grade III (Aggressive / Potentially Malignant Adenoma)",
            "typical_imaging_features": (
                "Rapidly growing pituitary lesion, extensive local invasion, "
                "necrosis, haemorrhage (pituitary apoplexy), marked "
                "perilesional oedema."
            ),
            "malignancy": "Rare pituitary carcinoma or aggressive atypical adenoma "
                           "with high recurrence; metastatic disease rare.",
            "growth_pattern": "Rapid, invasive, potential for CSF or systemic "
                              "spread in carcinoma subtypes.",
            "recommended_intervention_protocol": (
                "Urgent neurosurgical evaluation; emergency decompression if "
                "apoplexy with visual/neurological compromise; aggressive "
                "surgical resection; adjuvant radiotherapy; systemic therapy "
                "consideration; endocrine emergency management as needed."
            ),
        },
        4: {
            "grade_label": "WHO Grade IV (Pituitary Carcinoma / Metastatic)",
            "typical_imaging_features": (
                "Extremely rare malignant pituitary carcinoma; evidence of "
                "craniospinal or systemic metastases, aggressive local "
                "destruction, necrosis, oedema."
            ),
            "malignancy": "Rare malignant pituitary carcinoma with metastatic "
                           "potential; extremely poor prognosis.",
            "growth_pattern": "Highly aggressive, metastatic, rapid progression.",
            "recommended_intervention_protocol": (
                "Multidisciplinary neuro-oncology, neurosurgery, radiation "
                "oncology, and medical oncology coordination; surgical "
                "decompression; radiotherapy; systemic chemotherapy trials; "
                "palliative and supportive care integration."
            ),
        },
    },
    "No_Tumor": {
        1: {
            "grade_label": "No Evidence of Tumour (Normal / Benign Non-Neoplastic)",
            "typical_imaging_features": (
                "Normal brain parenchyma, no enhancing mass, no midline shift, "
                "no mass effect, normal ventricles, no abnormal signal "
                "intensities consistent with neoplasm."
            ),
            "malignancy": "No malignancy; incidental findings (e.g., small "
                           "vascular variants, benign cysts) handled accordingly.",
            "growth_pattern": "N/A (no neoplastic growth).",
            "recommended_intervention_protocol": (
                "Routine follow-up if incidental findings present; no oncologic "
                "intervention required. Clinical correlation recommended for any "
                "non-neoplastic incidental abnormality."
            ),
        },
        2: {
            "grade_label": "N/A (No tumour detected by imaging classification)",
            "typical_imaging_features": (
                "No additional characteristics; classification confidence and "
                "clinical context determine further workup."
            ),
            "malignancy": "No malignancy attributable to classifier output.",
            "growth_pattern": "N/A.",
            "recommended_intervention_protocol": (
                "No tumour-directed protocol; standard clinical care and "
                "surveillance as indicated by symptoms and non-neoplastic "
                "findings."
            ),
        },
        3: {
            "grade_label": "N/A (No tumour detected by imaging classification)",
            "typical_imaging_features": (
                "No tumour-related imaging features present in the studied "
                "slice/volume."
            ),
            "malignancy": "None identified.",
            "growth_pattern": "N/A.",
            "recommended_intervention_protocol": (
                "No tumour-directed protocol; clinician interpretation "
                "essential for incidental or symptomatic non-neoplastic findings."
            ),
        },
        4: {
            "grade_label": "N/A (No tumour detected by imaging classification)",
            "typical_imaging_features": (
                "No high-grade or aggressive features attributable to any "
                "neoplastic process."
            ),
            "malignancy": "None identified.",
            "growth_pattern": "N/A.",
            "recommended_intervention_protocol": (
                "No tumour-directed protocol; clinical correlation and "
                "non-oncologic workup as appropriate."
            ),
        },
    },
}

# ==============================================================================
# REGULATORY & MEDICAL DISCLAIMER CONSTANTS
# ==============================================================================
REGULATORY_DISCLAIMER: str = (
    "IMPORTANT REGULATORY AND MEDICAL DISCLAIMER\n\n"
    "FreeBuff v4 is a COMPUTER-AIDED DETECTION AND CLASSIFICATION research "
    "system intended for RESEARCH, EDUCATIONAL, AND INFORMATIONAL PURPOSES ONLY.\n\n"
    "This software is NOT a medical device, has NOT been cleared or approved by "
    "any regulatory authority (including but not limited to the U.S. FDA, "
    "European EMA/MDR notified bodies, or other national health regulatory "
    "agencies), and MUST NOT be used as the sole basis for clinical diagnosis, "
    "treatment decisions, surgical planning, medication prescribing, or any "
    "other clinical action affecting patient care.\n\n"
    "All outputs, classifications, confidence scores, WHO grade estimations, "
    "staging text, Grad-CAM++ heatmaps, and PDF reports generated by FreeBuff v4 "
    "ARE PROPOSED AS SECONDARY READING AIDS ONLY and must be independently "
    "verified, interpreted, and confirmed by a licensed, qualified medical "
    "professional using the complete clinical context, full imaging series, "
    "histopathology where available, and all standard-of-care diagnostic "
    "workflows.\n\n"
    "The system may produce FALSE POSITIVES, FALSE NEGATIVES, MISCLASSIFICATIONS, "
    "or UNRELIABLE CONFIDENCE ESTIMATES, particularly for out-of-distribution "
    "scans, rare pathologies, atypical presentations, or image quality degraded "
    "by motion, artefact, low signal-to-noise ratio, or protocol variation.\n\n"
    "No warranty of accuracy, completeness, fitness for a particular clinical "
    "purpose, or regulatory compliance is provided. Use entirely at the user's "
    "own risk and under the supervision of appropriate qualified professionals.\n\n"
    "Copyright (c) 2024-2026 FreeBuff v4 research prototype. All rights reserved."
)

REGULATORY_SHORTHAND: str = (
    "FOR RESEARCH USE ONLY. NOT FOR CLINICAL DIAGNOSTIC USE. "
    "NOT FDA-CLEARED OR APPROVED. VERIFY WITH A QUALIFIED MEDICAL PROFESSIONAL."
)

# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent

CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
DATA_DIR: Path = PROJECT_ROOT / "data"
EVALUATION_DIR: Path = PROJECT_ROOT / "evaluation_results"
REPORTS_DIR: Path = PROJECT_ROOT / "clinical_reports"
SAVED_MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Default paths for the trained model weights and scaler artifacts.
DEFAULT_CHECKPOINT_PATH: Path = CHECKPOINTS_DIR / "freebuff_v4.pt"
DEFAULT_MODEL_SAVE_PATH: Path = SAVED_MODELS_DIR / "freebuff_v4.pt"
DEFAULT_SCALER_PATH: Path = SAVED_MODELS_DIR / "freebuff_scaler.pt"
DEFAULT_PCA_PATH: Path = SAVED_MODELS_DIR / "freebuff_pca.pkl"

# Evaluation visualization output directory.
EVAL_VIS_DIR: Path = EVALUATION_DIR / "visualizations"

# Clinical PDF report output directory.
PDF_REPORT_DIR: Path = REPORTS_DIR / "pdf_reports"

# Grad-CAM++ heatmap export directory.
GRADCAM_EXPORT_DIR: Path = EVALUATION_DIR / "gradcam_overlays"


# ==============================================================================
# DEVICE & CUDA CONFIGURATION
# ==============================================================================
CUDA_DEVICE_ID: int = 0
CUDA_DEVICE_STRING: str = f"cuda:{CUDA_DEVICE_ID}"

# ==============================================================================
# DIRECTORY INITIALIZATION
# ==============================================================================
def initialize_directories() -> None:
    """
    Create all required project directories if they do not already exist,
    and log the resulting directory tree for transparency.

    This includes checkpoint, data, evaluation, report, saved-model, log,
    visualization, PDF, and Grad-CAM export directories.
    """
    dirs: Tuple[Path, ...] = (
        CHECKPOINTS_DIR,
        DATA_DIR,
        EVALUATION_DIR,
        REPORTS_DIR,
        SAVED_MODELS_DIR,
        LOG_DIR,
        EVAL_VIS_DIR,
        PDF_REPORT_DIR,
        GRADCAM_EXPORT_DIR,
    )
    created: List[str] = []
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory.resolve()))
    logger.info("Initialized FreeBuff v4 directory structure:\n%s", "\n".join(created))


# ==============================================================================
# VERSION INFO
# ==============================================================================
__version__: str = "4.0.0"
__author__: str = "FreeBuff v4 Research Prototype"
__description__: str = (
    "Multi-modal brain tumor classification, segmentation, staging & "
    "clinical decision support system with ResNet50+CBAM + HOG-PCA fusion "
    "and Graduate-CAM++ explainability."
)
