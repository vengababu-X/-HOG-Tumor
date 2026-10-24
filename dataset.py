#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Dataset & Preprocessing Pipeline
# ==============================================================================
# Robust data loading for MRI brain scans (DICOM, NIfTI, PNG, JPEG) with
# resizing to the canonical 144x144 grid, aligned HOG extraction, PCA fitting
# on the training split, and deterministic train / validation / test splitting.
# ==============================================================================
from __future__ import annotations

import logging
import math
import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Resize, ToTensor

from config import (
    BATCH_SIZE,
    HOG_BLOCK_SIZE,
    HOG_CELL_SIZE,
    HOG_DIMS_RAW,
    HOG_ORIENTATIONS,
    HOG_PCA_COMPONENTS,
    IMG_SIZE,
    INPUT_SIZE,
    INTENSITY_CLIP_MAX,
    INTENSITY_CLIP_MIN,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CHANNELS,
    NUM_CLASSES,
    NAME_TO_LABEL,
    CLASSES,
)
from preprocessing import augment_mri_image, image_to_tensor

logger = logging.getLogger("freebuff_v4.dataset")


# ==============================================================================
# PIL / NUMPY IMAGE LOADING HELPER
# ==============================================================================
def _load_pil_image(path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk and return a PIL Image in 'L' or 'RGB' mode.

    Supports common medical image formats saved as PNG / JPEG / TIFF as well as
    any format the PIL Imaging library can decode. DICOM/NIfTI are handled by
    separate loaders in the BrainTumorDataset class.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        img = Image.open(path)
    except Exception as e:
        raise IOError(f"Failed to open {path} as an image: {e}") from e
    # Convert to RGB if the file is palette/gray/LA/CMYK mode so subsequent
    # transforms are safe; preserve L for single-channel when later code needs
    # it by adding explicit handling at the transform stage.
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return img


# ==============================================================================
# DICOM LOADER (NUMPY BACKING) USING PYDICOM OR FALLBACK
# ==============================================================================
def load_dicom_slice(path: Union[str, Path]) -> np.ndarray:
    """
    Load a single DICOM slice and return a 2D numpy array (float32,
    H x W) with pixel values preserved as stored.

    Uses pydicom when available and falls back gracefully if the module is
    absent. No rescaling/resampling is applied here; intensity normalization
    occurs in the transform pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    try:
        import pydicom  # type: ignore[import-untyped]

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if "RescaleSlope" in ds and "RescaleIntercept" in ds:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            arr = arr * slope + intercept
        return arr
    except ImportError:
        raise ImportError(
            "pydicom is required to load DICOM files. Install with: pip install pydicom"
        ) from None


def load_nifti_volume(path: Union[str, Path]) -> np.ndarray:
    """
    Load a NIfTI volume (single file) and return a 3D numpy array (float32,
    D x H x W). Uses nibabel if available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    try:
        import nibabel as nib  # type: ignore[import-untyped]

        img = nib.load(str(path))
        # Reorient to canonical RAS+ so the third axis is the axial direction
        # regardless of how the file was stored.
        try:
            img = nib.as_closest_canonical(img)
        except Exception as exc:  # non-standard affine: use the file order
            logger.debug("NIfTI canonical reorientation skipped: %s", exc)
        arr = img.get_fdata().astype(np.float32)
        # nibabel yields (H, W, D) — spatial dims first, slices last — but this
        # function's contract (and every caller, which takes volume[D // 2]) is
        # D-first. Without this transpose the "middle axial slice" was actually
        # a row taken across the volume, i.e. not an anatomical slice at all.
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return arr
    except ImportError:
        raise ImportError(
            "nibabel is required to load NIfTI files. Install with: pip install nibabel"
        ) from None


# ==============================================================================
# DICOM METADATA -> REPORT LINE FORMATTING
# ==============================================================================
def dicom_meta_to_report_lines(dicom_meta: Optional[Dict[str, Any]]) -> List[str]:
    """
    Convert a DICOM metadata dictionary into human-readable report lines.

    Used by narrative.py and the PDF report generator to embed acquisition
    context (modality, TR/TE, slice thickness, etc.) into clinical outputs.
    Missing/None fields are silently skipped; an empty/None input yields an
    empty list so callers can render nothing gracefully.

    Args:
        dicom_meta: Optional mapping of DICOM tag names to values. Recognised
            keys (case-insensitive) include: patient_id, study_date, modality,
            series_description, manufacturer, magnetic_field_strength,
            slice_thickness, pixel_spacing, repetition_time, echo_time,
            institution_name.

    Returns:
        List of formatted "Key: value" strings.
    """
    if not dicom_meta:
        return []

    # Canonical display order + friendly labels for known tags.
    labelled_keys: Tuple[Tuple[str, str], ...] = (
        ("patient_id", "Patient ID"),
        ("patient_name", "Patient Name"),
        ("study_date", "Study Date"),
        ("modality", "Modality"),
        ("series_description", "Series Description"),
        ("manufacturer", "Manufacturer"),
        ("magnetic_field_strength", "Field Strength (T)"),
        ("slice_thickness", "Slice Thickness (mm)"),
        ("pixel_spacing", "Pixel Spacing (mm)"),
        ("repetition_time", "TR (ms)"),
        ("echo_time", "TE (ms)"),
        ("institution_name", "Institution"),
    )

    # Case-insensitive lookup over the provided dictionary.
    lowered: Dict[str, Any] = {
        str(k).strip().lower(): v for k, v in dict(dicom_meta).items()
    }

    lines: List[str] = []
    seen: set = set()
    for key, label in labelled_keys:
        if key in lowered and lowered[key] not in (None, ""):
            lines.append(f"{label}: {lowered[key]}")
            seen.add(key)

    # Pass through any additional non-empty tags not in the canonical set.
    for key, value in lowered.items():
        if key not in seen and value not in (None, ""):
            lines.append(f"{str(key).replace('_', ' ').title()}: {value}")

    return lines


# ==============================================================================
# INTENSITY NORMALIZATION UTILITIES
# ==============================================================================
def normalize_intensity(arr: np.ndarray, low: float = INTENSITY_CLIP_MIN,
                        high: float = INTENSITY_CLIP_MAX) -> np.ndarray:
    """
    Clip and scale a 2D/3D numpy array into the [0, 1] range after min-max
    normalization within the supplied intensity window.

    Args:
        arr: Input numpy array (any shape).
        low: Minimum intensity value for clipping (before min-max scaling).
        high: Maximum intensity value for clipping.

    Returns:
        Normalized array of same shape with values in [0, 1].
    """
    arr = arr.astype(np.float32, copy=True)
    arr = np.clip(arr, low, high)
    denom = (high - low)
    if denom <= 0.0:
        warnings.warn(f"Intensity range [{low}, {high}] has zero width; returning zeros.")
        return np.zeros_like(arr)
    arr = (arr - low) / denom
    return arr


def normalize_three_channel(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet-style per-channel normalization to a 3-channel float tensor
    of shape (3, H, W) in [0, 1] range.

    Args:
        tensor: Float tensor (3, H, W), values in [0, 1].

    Returns:
        Normalized tensor with same shape.
    """
    for c in range(tensor.shape[0]):
        tensor[c] = (tensor[c] - NORMALIZE_MEAN[c]) / NORMALIZE_STD[c]
    return tensor


# ==============================================================================
# HOG FEATURE EXTRACTION (NUMPY IMPLEMENTATION)
# ==============================================================================
class HOGExtractor:
    """
    Histogram of Oriented Gradients extractor operating on a 2D grayscale
    numpy array and enforcing alignment with the 144x144 input resolution.

    Cell layout: HOG_CELL_SIZE x HOG_CELL_SIZE (16x16 cells), blocks are
    HOG_BLOCK_SIZE x HOG_BLOCK_SIZE cells (2x2), block stride of 1 cell.
    8 orientation bins are used per cell.

    The resulting raw feature vector is 2048-dimensional, which is subsequently
    reduced via PCA (fitted on the training set) to HOG_PCA_COMPONENTS (288).
    """

    def __init__(
        self,
        cell_size: int = HOG_CELL_SIZE,
        block_size: int = HOG_BLOCK_SIZE,
        block_step: int = 1,
        orientations: int = HOG_ORIENTATIONS,
        pixels_per_cell: Optional[Tuple[int, int]] = None,
        cells_per_block: Optional[Tuple[int, int]] = None,
        feature_vector_size: int = HOG_DIMS_RAW,
    ) -> None:
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_step = block_step
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell or (cell_size, cell_size)
        self.cells_per_block = cells_per_block or (block_size, block_size)
        self.feature_vector_size = feature_vector_size

    # ------------------------------------------------------------------
    def _gradient_mag_and_angle(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient magnitude and angle via central differences.

        Args:
            image: 2D numpy array (H, W), float32.

        Returns:
            Tuple of (magnitude, angle in radians) arrays of same shape.
        """
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)
        gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) * 0.5
        gy[1:-1, :] = (image[2:, :] - image[:-2, :]) * 0.5
        magnitude = np.hypot(gx, gy)
        angle = np.arctan2(gy, gx)  # radians in [-pi, pi]
        return magnitude, angle

    # ------------------------------------------------------------------
    def _cell_histogram(self, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
        """
        Build the orientation histogram for one cell.

        Args:
            mag: Magnitude patch for the cell (cell_h, cell_w).
            ang: Angle patch for the cell (cell_h, cell_w).

        Returns:
            1D array of length `orientations`.
        """
        hist = np.zeros(self.orientations, dtype=np.float32)
        bin_size = (2.0 * np.pi) / self.orientations
        # Map angle from [-pi, pi] to [0, 2*pi]
        ang = ang % (2.0 * np.pi)
        bins = (ang / bin_size).astype(np.int32) % self.orientations
        flat_mag = mag.ravel()
        flat_bins = bins.ravel()
        for m, b in zip(flat_mag, flat_bins):
            hist[b] += float(m)
        return hist

    # ------------------------------------------------------------------
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the HOG descriptor for a 2D grayscale image.

        Args:
            image: 2D numpy array (H, W), float32, normalized to [0, 1].

        Returns:
            1D numpy array of length self.feature_vector_size (2048).
        """
        if image.ndim != 2:
            raise ValueError(f"HOG expects a 2D grayscale image; got shape {image.shape}")
        h, w = image.shape
        mag, ang = self._gradient_mag_and_angle(image)

        cell_h = self.pixels_per_cell[0]
        cell_w = self.pixels_per_cell[1]

        n_cells_y = h // cell_h
        n_cells_x = w // cell_w

        # Build per-cell histograms grid: (n_cells_y, n_cells_x, orientations)
        cell_hist = np.zeros((n_cells_y, n_cells_x, self.orientations), dtype=np.float32)
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0 = cy * cell_h
                y1 = y0 + cell_h
                x0 = cx * cell_w
                x1 = x0 + cell_w
                cell_hist[cy, cx, :] = self._cell_histogram(mag[y0:y1, x0:x1], ang[y0:y1, x0:x1])

        # Block normalization (L2-Hys) over cells_per_block x cells_per_block windows
        block_h = self.cells_per_block[0]
        block_w = self.cells_per_block[1]
        block_stride = self.block_step

        n_blocks_y = (n_cells_y - block_h) // block_stride + 1
        n_blocks_x = (n_cells_x - block_w) // block_stride + 1

        features: List[np.ndarray] = []
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y0 = by * block_stride
                y1 = y0 + block_h
                x0 = bx * block_stride
                x1 = x0 + block_w
                block = cell_hist[y0:y1, x0:x1, :].ravel()
                # L2-Hys normalization
                norm = np.sqrt(np.sum(block ** 2) + 1e-6)
                block = block / norm
                block = np.clip(block, 0.0, 0.2)
                norm2 = np.sqrt(np.sum(block ** 2) + 1e-6)
                if norm2 > 0.0:
                    block = block / norm2
                features.append(block)

        feat = np.concatenate(features, dtype=np.float32)
        expected_len = self.feature_vector_size
        if feat.shape[0] < expected_len:
            feat = np.pad(feat, (0, expected_len - feat.shape[0]), mode="constant", constant_values=0.0)
        elif feat.shape[0] > expected_len:
            feat = feat[:expected_len]
        return feat.astype(np.float32)


# ==============================================================================
# BRAIN TUMOUR DATASET
# ==============================================================================
class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumour classification.

    Supports:
      - Single-modality folder-of-images layout (CSV mapping file -> folder).
      - CSV with one path column and one label column.
      - Pandas DataFrame with 'image_path' and 'label' columns.

    Each sample yields:
      - image: torch.Tensor of shape (3, IMG_SIZE, IMG_SIZE) normalized.
      - label: int64 scalar (0..NUM_CLASSES-1).
      - hog_pca: torch.Tensor of shape (HOG_PCA_COMPONENTS,) or None if PCA
        not fitted / disabled.
      - path: the original file path (str) for traceability.
      - metadata: dict with imaging metadata (slice index, modality hint, etc.).
    """

    def __init__(
        self,
        samples: Sequence[Union[str, Path, Tuple[Union[str, Path], int], Dict[str, Any]]],
        transform: Optional[Callable] = None,
        hog_extractor: Optional[HOGExtractor] = None,
        pca: Optional[SklearnPCA] = None,
        use_hog: bool = True,
        augment: bool = False,
        augment_fn: Optional[Callable] = None,
        label_column: str = "label",
        image_column: str = "image_path",
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            samples: Iterable of file paths OR (path, label) tuples OR dicts with
                'image_path' and 'label' keys. Used when df is None.
            transform: Callable to apply to the PIL image before tensor conversion.
            hog_extractor: Fitted or unfitted HOGExtractor; must be provided if
                use_hog is True.
            pca: Fitted sklearn PCA object for projecting HOG features. Pass None
                to disable HOG projection (raw HOG returned as tensor instead).
            use_hog: Whether to compute and attach HOG features.
            augment: If True, apply random augmentations on the fly (mainly for
                validation consistency this is False). Controlled by caller.
            augment_fn: Optional callable applied in addition to transform.
            label_column: Column name for labels in df.
            image_column: Column name for image paths in df.
            df: Optional pandas DataFrame with at least image_column and
                label_column.
        """
        self.transform = transform
        self.hog_extractor = hog_extractor
        self.pca = pca
        self.use_hog = use_hog
        self.augment = augment
        self.augment_fn = augment_fn
        self.label_column = label_column
        self.image_column = image_column
        self.df = df

        self.samples_list: List[Dict[str, Any]] = []
        self._load_samples(samples)

        if self.use_hog and self.hog_extractor is None:
            self.hog_extractor = HOGExtractor()
        logger.info("BrainTumorDataset ready with %d samples.", len(self.samples_list))

    # ------------------------------------------------------------------
    def _load_samples(self, samples: Sequence[Any]) -> None:
        """Populate self.samples_list from mixed input types."""
        if self.df is not None:
            self.df = self.df.reset_index(drop=True)
            for _, row in self.df.iterrows():
                img_path = row[self.image_column]
                label = int(row[self.label_column])
                self.samples_list.append({
                    "image_path": str(img_path),
                    "label": label,
                })
            return

        for sample in samples:
            if isinstance(sample, dict):
                path = sample.get("image_path") or sample.get("path")
                label = int(sample.get("label", sample.get("target", -1)))
                if path is None or label < 0:
                    raise ValueError(f"Invalid sample dict: {sample}")
                self.samples_list.append({"image_path": str(path), "label": label})
            elif isinstance(sample, (str, Path)):
                self.samples_list.append({"image_path": str(sample), "label": 0})
            elif isinstance(sample, tuple) and len(sample) == 2:
                self.samples_list.append({"image_path": str(sample[0]), "label": int(sample[1])})
            else:
                raise ValueError(f"Unsupported sample type: {type(sample)}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples_list)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single data sample as a dictionary.

        Keys:
            "image": torch.Tensor (3, IMG_SIZE, IMG_SIZE)
            "label": int64
            "hog_pca": torch.Tensor (HOG_PCA_COMPONENTS,) or None
            "path": str
            "metadata": dict
        """
        entry = self.samples_list[idx]
        path = Path(entry["image_path"])
        label = int(entry["label"])

        # Load image (supports DICOM, NIfTI, PNG, JPEG, generic PIL)
        img = self._load_image(path)
        # Augment before deterministic resize if augmentation is on
        if self.augment and self.augment_fn is not None:
            img = self.augment_fn(img)
        # Apply standard transforms (resize, to tensor, normalize)
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = self._default_transform(img)

        metadata: Dict[str, Any] = {
            "path": str(path),
            "label": label,
            "slice_idx": idx,
        }

        hog_pca: Optional[torch.Tensor] = None
        if self.use_hog and self.hog_extractor is not None:
            hog_raw = self.hog_extractor.extract(img_tensor_to_numpy_gray(img_tensor))
            if self.pca is not None:
                hog_proj = self.pca.transform(hog_raw.reshape(1, -1)).ravel()
                hog_pca = torch.from_numpy(hog_proj).float()
            else:
                hog_pca = torch.from_numpy(hog_raw).float()

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "hog_pca": hog_pca,
            "path": str(path),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> Image.Image:
        """Load an image from disk, handling DICOM/NIfTI/PIL formats."""
        ext = path.suffix.lower()
        if ext in (".dcm", ".dicom"):
            arr = load_dicom_slice(path)
            # Normalize to [0,1] and convert to PIL grayscale
            arr = normalize_intensity(arr)
            arr = (arr * 255.0).astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        # Path.suffix of "scan.nii.gz" is ".gz", not ".nii.gz", so the compound
        # extension has to be matched on the full name — otherwise the most
        # common NIfTI form fell through to the generic PIL loader.
        if ext in (".nii", ".nifti") or path.name.lower().endswith(".nii.gz"):
            vol = load_nifti_volume(path)
            slice_2d = _select_middle_slice(vol)
            slice_2d = normalize_intensity(slice_2d)
            slice_2d = (slice_2d * 255.0).astype(np.uint8)
            return Image.fromarray(slice_2d, mode="L")
        # Generic PIL load (PNG, JPEG, TIFF, etc.)
        img = _load_pil_image(path)
        if img.mode == "L":
            return img
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ------------------------------------------------------------------
    def _default_transform(self, img: Image.Image) -> torch.Tensor:
        """Default foreground-crop, robust-normalization pipeline."""
        return image_to_tensor(img, size=INPUT_SIZE)

    # ------------------------------------------------------------------
    def get_sample_by_path(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Locate a sample dict by path (useful for targeted inspection)."""
        target = str(Path(path).resolve())
        for entry in self.samples_list:
            if str(Path(entry["image_path"]).resolve()) == target:
                return self.__getitem__(self.samples_list.index(entry))
        return None


# ==============================================================================
# HELPERS
# ==============================================================================
def _select_middle_slice(volume: np.ndarray) -> np.ndarray:
    """
    Select the middle axial slice from a 3D volume.

    If volume is 3D (D, H, W) -> return volume[D//2].
    If already 2D, return as-is.
    """
    volume = np.asarray(volume)
    if volume.ndim == 3:
        mid = volume.shape[0] // 2
        return volume[mid]
    if volume.ndim == 2:
        return volume
    raise ValueError(f"Expected 2D or 3D array, got shape {volume.shape}")


def img_tensor_to_numpy_gray(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (3, H, W) RGB tensor in [0,1] to a 2D grayscale numpy array
    normalized to [0, 1].
    """
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        gray = 0.2989 * tensor[0] + 0.5870 * tensor[1] + 0.1140 * tensor[2]
    elif tensor.ndim == 2:
        gray = tensor
    else:
        gray = tensor.squeeze()
    return gray.detach().cpu().numpy().astype(np.float32)


# ==============================================================================
# DATA SPLITTING
# ==============================================================================
def split_dataset_into_train_val_test(
    dataset: BrainTumorDataset,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split a BrainTumorDataset into train / validation / test subsets.

    Args:
        dataset: Source dataset.
        val_frac: Fraction of total samples allocated to validation.
        test_frac: Fraction allocated to test.
        seed: Random seed for deterministic split.
        stratify: Whether to preserve class balance in splits.

    Returns:
        Tuple of (train_subset, val_subset, test_subset) as torch Subsets.
    """
    n = len(dataset)
    labels = np.array([dataset.samples_list[i]["label"] for i in range(n)])

    if stratify:
        try:
            train_idx, temp_idx, train_labels, temp_labels = train_test_split(
                np.arange(n), labels,
                test_size=(val_frac + test_frac),
                random_state=seed,
                stratify=labels,
            )
            # Further split temp into val and test preserving stratification
            val_ratio_in_temp = val_frac / (val_frac + test_frac)
            val_idx, test_idx = train_test_split(
                temp_idx, labels[temp_idx] if stratify else None,
                test_size=(1.0 - val_ratio_in_temp),
                random_state=seed,
                stratify=temp_labels if stratify else None,
            )
        except ValueError:
            # Fallback to non-stratified split if stratification fails
            train_idx, temp_idx = train_test_split(
                np.arange(n), test_size=(val_frac + test_frac),
                random_state=seed,
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=(test_frac / (val_frac + test_frac)),
                random_state=seed,
            )
    else:
        train_idx, temp_idx = train_test_split(
            np.arange(n), test_size=(val_frac + test_frac),
            random_state=seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(test_frac / (val_frac + test_frac)),
            random_state=seed,
        )

    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())

    logger.info(
        "Split dataset: train=%d, val=%d, test=%d (stratified=%s).",
        len(train_idx), len(val_idx), len(test_idx), stratify,
    )
    return train_subset, val_subset, test_subset


# ==============================================================================
# DATA LOADER FACTORY
# ==============================================================================
def make_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders from Datasets.

    Args:
        train_dataset: Training Dataset.
        val_dataset: Validation Dataset.
        test_dataset: Test Dataset.
        batch_size: Batch size for all loaders.
        num_workers: Worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        shuffle_train: Shuffle training loader each epoch.
        drop_last_train: Drop last incomplete batch in training.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader, test_loader


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of sample dicts into batched tensors.

    Handles variable-length metadata lists and optional HOG features.
    """
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    paths = [b["path"] for b in batch]
    metadata = [b["metadata"] for b in batch]

    hog_feats: Optional[torch.Tensor] = None
    has_hog = all(b.get("hog_pca") is not None for b in batch)
    if has_hog:
        hog_feats = torch.stack([b["hog_pca"] for b in batch])

    return {
        "image": images,
        "label": labels,
        "hog_pca": hog_feats,
        "path": paths,
        "metadata": metadata,
    }


# ==============================================================================
# PCA FITTING ON TRAINING HOG FEATURES
# ==============================================================================
def fit_pca_on_training_hog(
    train_dataset: BrainTumorDataset,
    pca_components: int = HOG_PCA_COMPONENTS,
    hog_extractor: Optional[HOGExtractor] = None,
    random_state: int = 42,
) -> SklearnPCA:
    """Fit a scikit-learn PCA on the HOG feature vectors extracted from the
    training split only.

    Args:
        train_dataset: Training BrainTumorDataset, or a torch Subset wrapping one.
        pca_components: Number of PCA components to retain.
            Capped at min(n_samples - 1, n_features) for solver compatibility.
        hog_extractor: HOGExtractor instance; if None, a default HOGExtractor
            is created.
        random_state: Random seed for PCA SVD solver.

    Returns:
        Fitted SklearnPCA object.
    """
    if hog_extractor is None:
        hog_extractor = HOGExtractor()

    logger.info("Extracting HOG features from training set for PCA fitting...")
    hog_matrix: List[np.ndarray] = []
    n = len(train_dataset)
    base = train_dataset
    while hasattr(base, "dataset") and hasattr(base, "indices"):
        base = base.dataset
    for i in range(n):
        idx = getattr(train_dataset, "indices", None)
        if idx is not None:
            sample = base.samples_list[idx[i]]
        else:
            sample = base.samples_list[i]
        path = Path(sample["image_path"])
        img = _load_pil_image(path)
        # Resize to expected dims for HOG alignment
        img = Resize(INPUT_SIZE, interpolation=Image.BILINEAR)(img)
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        feat = hog_extractor.extract(arr)
        hog_matrix.append(feat)
        if (i + 1) % 200 == 0:
            logger.info("Processed %d/%d training samples for HOG.", i + 1, n)

    hog_matrix = np.stack(hog_matrix, dtype=np.float32)
    logger.info("Collected HOG matrix of shape %s.", hog_matrix.shape)

    # Cap the number of components so SVD solvers do not reject the request.
    n_samples, n_features = hog_matrix.shape
    max_components = min(n_samples - 1, n_features)
    actual_components = int(min(pca_components, max_components))
    if actual_components < pca_components:
        logger.warning(
            "Requested %d PCA components but only %d are feasible with %d samples; "
            "using %d components instead.",
            pca_components, max_components, n_samples, actual_components,
        )

    pca = SklearnPCA(
        n_components=actual_components,
        random_state=random_state,
        svd_solver="full",
    )
    pca.fit(hog_matrix)
    explained = sum(pca.explained_variance_ratio_)
    logger.info(
        "PCA fitted: %d components (requested %d), cumulative explained variance = %.4f.",
        actual_components, pca_components, explained,
    )
    return pca


# ==============================================================================
# DATASET STATISTICS UTILITY
# ==============================================================================
def print_dataset_statistics(datasets: Dict[str, Dataset]) -> None:
    """
    Log class distribution and total sample count for each split.

    Args:
        datasets: Dict with keys 'train', 'val', 'test' mapping to Datasets.
    """
    for split_name, ds in datasets.items():
        if isinstance(ds, Subset):
            base_dataset = ds.dataset
            labels = [
                base_dataset.samples_list[index]["label"]
                for index in ds.indices
            ]
        else:
            labels = [ds.samples_list[i]["label"] for i in range(len(ds))]
        counts = Counter(labels)
        logger.info("Dataset statistics for %s split:", split_name)
        for cls_idx in sorted(counts.keys()):
            cls_name = CLASSES[cls_idx]
            logger.info("  %s (label %d): %d samples", cls_name, cls_idx, counts[cls_idx])
        logger.info("  Total: %d samples", len(labels))


# ==============================================================================
# BULK DATASET CONSTRUCTION FROM FOLDER TREE
# ==============================================================================
def _matches_extension(path: Union[str, Path], extensions: Sequence[str]) -> bool:
    """True when `path` ends with any of `extensions`, compound ones included.

    `Path.suffix` returns only the LAST component, so "scan.nii.gz" yields
    ".gz" and never matches a ".nii.gz" entry. Comparing against the full
    lowercase filename handles both simple and compound extensions.
    """
    name = Path(path).name.lower()
    return any(name.endswith(ext.lower()) for ext in extensions)


def build_dataset_from_folder(
    data_dir: Union[str, Path],
    label_map: Optional[Dict[str, int]] = None,
    extension_whitelist: Optional[Sequence[str]] = None,
) -> BrainTumorDataset:
    """
    Construct a BrainTumorDataset from a directory tree where each subdirectory
    name maps to a class label.

    Typical structure:
        data_dir/
            Glioma/
                img001.png
                img002.dcm
            Meningioma/
                ...
            No_Tumor/
                ...
            Pituitary/
                ...

    Args:
        data_dir: Root directory containing class subfolders.
        label_map: Optional explicit mapping of folder name -> integer label.
            If None, uses NAME_TO_LABEL (expects folder names matching CLASSES).
        extension_whitelist: Optional sequence of file extensions to include
            (e.g. ['.png', '.dcm']). If None, includes all files.

    Returns:
        BrainTumorDataset with all samples.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")
    if label_map is None:
        label_map = NAME_TO_LABEL
    normalized_label_map = {
        str(name).casefold().replace("_", ""): int(label)
        for name, label in label_map.items()
    }
    normalized_label_map.update(
        {
            "glioma": NAME_TO_LABEL["Glioma"],
            "meningioma": NAME_TO_LABEL["Meningioma"],
            "notumor": NAME_TO_LABEL["No_Tumor"],
            "pituitary": NAME_TO_LABEL["Pituitary"],
        }
    )
    if extension_whitelist is None:
        extension_whitelist = (".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".nii", ".nii.gz")

    samples: List[Dict[str, Any]] = []
    skipped: int = 0
    for class_folder in data_dir.iterdir():
        if not class_folder.is_dir():
            continue
        folder_name = class_folder.name
        normalized_folder_name = folder_name.casefold().replace("_", "")
        label = normalized_label_map.get(normalized_folder_name)
        if label is None:
            logger.warning("Skipping unrecognized class folder: %s", folder_name)
            skipped += 1
            continue
        for file_path in class_folder.iterdir():
            if not file_path.is_file():
                continue
            # Path.suffix of "case.nii.gz" is ".gz", so a plain suffix test
            # silently dropped every gzipped NIfTI even though ".nii.gz" is in
            # the whitelist. Match the full name for compound extensions.
            if extension_whitelist and not _matches_extension(file_path, extension_whitelist):
                continue
            samples.append({"image_path": str(file_path), "label": label})

    logger.info(
        "Built dataset from %s: %d samples collected, %d folders skipped.",
        data_dir, len(samples), skipped,
    )
    if not samples:
        raise ValueError(
            f"No supported labeled images found in {data_dir}. "
            f"Expected class folders: {', '.join(CLASSES)}."
        )
    return BrainTumorDataset(samples=samples)
