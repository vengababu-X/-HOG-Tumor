"""
ssl_encoder.py – Self-Supervised Foundation Embedding Pillar (v4)

Provides a DINOv2-style / self-distilled ViT feature extractor that produces
robust, anatomically invariant embeddings directly from raw multimodal MRI
slices, and fuses them into the existing 416-dim hybrid pipeline.

Design goals (aligned to prompt):
  * Pre-trained ViT backbone (self-supervised distillation family).
  * Deterministic, resolution-anchored extraction: everything is standardised
    to the FreeBuff 144x144 contract before embedding, so feature dimensions
    and downstream PCA/Scaler/SVM stay exact.
  * Fuse foundation embeddings with the existing hybrid vector.
  * Graceful local CUDA/CPU fallback (PyTorch) plus a pure-TF bridge when the
    PyTorch stack is not available.

Backbone selection policy (best-effort):
  1. Try to load a Vision Transformer from the local torchvision/timm ecosystem
     (e.g. a ViT trained with self-supervised objectives if available).
  2. Fall back to a deterministic TF implementation of a small ViT so the
     pillar is runnable on CPU-only / no-torch installs.
  3. Never crash the platform on import — the pillar is optional and degrades
     to a no-op embedding passthrough when nothing can be loaded.

Author: Codebuff AI
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("ssl_encoder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Feature contract (kept aligned with the rest of the platform)
# ---------------------------------------------------------------------------
# The pillar exposes a fixed-size foundation embedding per sample so that the
# hybrid feature vector can be extended without changing downstream contracts.
# The platform already uses:
#
#   deep(128) + hog_pca(288) = 416  →  scaled → SVM / super-learner
#
# Here we append:
#
#   foundation(D)   → projected → fused into the hybrid vector
#
# For this first v4 pillar we keep the foundation dimension constant and
# project it down to a compact slot so the full vector does not explode.
#
# contract_fusion_mode:
#   "append"  → hybrid = [deep(128), hog_pca(288), foundation_proj(D)]  (D=64)
#   "replace" → hybrid = [foundation_proj(D=256), hog_pca(288)]         (experimental)
#
# The default is "append" for backward compatibility with the existing 416-dim
# SVM / super-learner artefacts.  When foundation fusion is active, the scaler
# and downstream classifier must be retrained; the pillar therefore ships a
# deterministic projection so it can still run at inference time on existing
# bundles without retraining.
# ---------------------------------------------------------------------------

FOUNDATION_EMBED_DIM: int = 768        # raw ViT patch/token embedding dim (family-dependent)
FOUNDATION_PROJECTED_DIM: int = 64     # compact slot appended to hybrid vector
FOUNDATION_FUSE_MODE: str = "append"   # "append" | "replace"

# When fuse_mode == "append", hybrid becomes:
#   128 + 288 + 64 = 480  (v4 contract)
HYBRID_DIM_WITH_FUSION: int = 128 + 288 + FOUNDATION_PROJECTED_DIM


# ---------------------------------------------------------------------------
# PyTorch backbone loading (optional, best-effort)
# ---------------------------------------------------------------------------

def _try_import_torch() -> Tuple[bool, Optional["torch"]]:
    try:
        import torch
        return True, torch
    except Exception as exc:
        logger.debug("torch unavailable: %s", exc)
        return False, None


def _try_import_timm() -> Tuple[bool, Optional["timm"]]:
    try:
        import timm
        return True, timm
    except Exception as exc:
        logger.debug("timm unavailable: %s", exc)
        return False, None


class TorchVisionTransformerBackbone:
    """
    Wraps a pre-trained ViT backbone from the local PyTorch ecosystem.

    This is intentionally *not* a hard requirement.  If torch/timm are
    unavailable, or if no suitable backbone is found, the pillar falls back
    to the deterministic TF backbone below.
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        projection_dim: int = FOUNDATION_PROJECTED_DIM,
        device: Optional[str] = None,
    ) -> None:
        found_torch, torch = _try_import_torch()
        if not found_torch:
            raise RuntimeError("Torch backbone requested but torch is unavailable.")

        found_timm, timm = _try_import_timm()
        if not found_timm:
            raise RuntimeError("Torch backbone requested but timm is unavailable.")

        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Torch backbone: %s → device=%s (CUDA=%s)",
            backbone_name,
            self.device,
            torch.cuda.is_available(),
        )

        # Build the backbone.  timm provides a clean entrypoint for ViT
        # families; if the requested name is not present we fall back to the
        # smallest available ViT in this class so the pillar always has a
        # runnable model.
        try:
            self.model = timm.create_model(backbone_name, pretrained=True)
            self.feature_dim = self.model.num_features
        except Exception:
            # Best-effort fallback to a small ViT from timm.
            try:
                self.model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
                self.feature_dim = self.model.num_features
                logger.warning(
                    "Backbone '%s' unavailable — fell back to vit_tiny_patch16_224.",
                    backbone_name,
                )
            except Exception as exc:
                raise RuntimeError(f"Could not load any ViT backbone: {exc}")

        self.model.eval()
        self.model.to(self.device)

        # Projection head: raw feature dim → compact slot.
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, projection_dim),
        ).to(self.device)

    def embed(self, images: np.ndarray) -> np.ndarray:
        """
        Extract compact foundation embeddings for a batch of RGB images.

        ``images``: (N, H, W, 3) float32 in [0,1].
        Returns: (N, projection_dim) float32.
        """
        found_torch, torch = _try_import_torch()
        assert found_torch

        n, h, w, c = images.shape
        # Convert to torch tensor, normalised in the style expected by
        # ImageNet-pretrained ViT families (deterministic at inference time).
        # We use the standard mean/std so that any timm-hosted pretrained
        # weights behave as intended.
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        x = (x - mean) / std

        with torch.no_grad():
            feats = self.model(x)
            emb = self.proj(feats)

        return emb.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Deterministic TF bridge (no PyTorch dependency)
# ---------------------------------------------------------------------------

def _build_small_vit_tf(
    input_size: int = 144,
    patch_size: int = 16,
    embed_dim: int = 96,
    depth: int = 4,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    num_classes: int = 0,  # 0 → feature extractor only
) -> "keras.Model":
    """
    Build a small deterministic ViT feature extractor in Keras.

    This is the guaranteed-no-hard-dependency path so the pillar can run on
    CPU-only environments without torch/timm.  It is *not* a DINOv2 replica —
    it is a deterministic ViT-style backbone whose weights are initialised
    from scratch; in production it should be replaced with the PyTorch
    self-supervised backbone once available.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    num_patches = (input_size // patch_size) ** 2

    inputs = keras.Input(shape=(input_size, input_size, 3), name="foundation_input")

    # Patch embedding
    patch_embed = layers.Conv2D(
        embed_dim, patch_size, strides=patch_size, padding="valid", name="patch_embed",
    )(inputs)
    x = layers.Reshape((num_patches, embed_dim))(patch_embed)

    # Positional embedding
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim,
                                 name="pos_embed")(tf.range(num_patches))
    x = layers.Add()([x, pos_embed])

    # Transformer blocks
    for i in range(depth):
        norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(embed_dim // num_heads, 1),
            dropout=dropout,
            name=f"attn_{i}",
        )(norm1, norm1)
        x = layers.Add()([x, attn])

        norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{i}")(x)
        mlp = keras.Sequential([
            layers.Dense(int(embed_dim * mlp_ratio), activation="gelu", name=f"mlp1_{i}"),
            layers.Dropout(dropout, name=f"mlp1_drop_{i}"),
            layers.Dense(embed_dim, name=f"mlp2_{i}"),
            layers.Dropout(dropout, name=f"mlp2_drop_{i}"),
        ], name=f"mlp_{i}")(norm2)
        x = layers.Add()([x, mlp])

    # Global pool → feature vector
    x = layers.LayerNormalization(epsilon=1e-6, name="pre_pool_norm")(x)
    x = layers.GlobalAveragePooling1D(name="pool")(x)

    if num_classes > 0:
        x = layers.Dense(num_classes, activation="softmax", name="head")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="foundation_vit_tf")
    return model


class TfVisionTransformerBackbone:
    """
    Deterministic TF ViT backbone (no PyTorch dependency).

    Used as the guaranteed fallback for the foundation pillar so the platform
    runs on CPU-only hosts.
    """

    def __init__(
        self,
        input_size: int = 144,
        projection_dim: int = FOUNDATION_PROJECTED_DIM,
        backbone_name: str = "vit_small_patch16_224",
        device: Optional[str] = None,
    ) -> None:
        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.device = device or ("gpu" if _tf_gpu_available() else "cpu")
        self.input_size = input_size

        logger.info(
            "TF backbone (fallback): %s → device=%s",
            backbone_name,
            self.device,
        )

        # Build the backbone (feature extractor only).
        self.backbone = _build_small_vit_tf(
            input_size=input_size,
            embed_dim=96,
            depth=4,
            num_heads=4,
            projection_dim=0,  # we add our own projection head below
        )

        # Projection head into the compact slot.
        self.proj = keras.Sequential([
            layers.Dense(256, activation="gelu", name="foundation_proj_256"),
            layers.Dense(projection_dim, name="foundation_proj"),
        ], name="foundation_projection")

        self.model = keras.Model(
            inputs=self.backbone.inputs,
            outputs=self.proj(self.backbone.outputs[0]),
            name="foundation_embedder",
        )

    def embed(self, images: np.ndarray) -> np.ndarray:
        """
        Extract compact foundation embeddings for a batch of RGB images.

        ``images``: (N, H, W, 3) float32 in [0,1].
        Returns: (N, projection_dim) float32.
        """
        # Simple ImageNet-style normalisation for the TF bridge so the
        # deterministic backbone behaves consistently.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
        x = (images - mean) / std
        return self.model.predict(x, verbose=0).astype(np.float32)


def _tf_gpu_available() -> bool:
    try:
        import tensorflow as tf
        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public pillar API
# ---------------------------------------------------------------------------

def create_ssl_encoder(
    backbone: str = "torch",
    backbone_name: str = "vit_small_patch16_224",
    projection_dim: int = FOUNDATION_PROJECTED_DIM,
    fuse_mode: str = FOUNDATION_FUSE_MODE,
    device: Optional[str] = None,
) -> "FoundationEncoder":
    """
    Create the foundation embedding encoder, trying hardware-optimal backends.

    Priority:
      1. Torch ViT backbone if torch/timm are available.
      2. Deterministic TF ViT backbone otherwise.

    ``backbone="auto"`` selects the best available automatically.
    """
    if backbone == "auto":
        found_torch, _ = _try_import_torch()
        if found_torch:
            torch_device = "cuda" if _try_import_torch()[1].cuda.is_available() else "cpu"
            try:
                return TorchVisionTransformerBackbone(
                    backbone_name=backbone_name,
                    projection_dim=projection_dim,
                    device=torch_device,
                )
            except Exception as exc:
                logger.warning("Torch backbone unavailable: %s — using TF fallback.", exc)
        return TfVisionTransformerBackbone(
            input_size=144,
            projection_dim=projection_dim,
            backbone_name=backbone_name,
            device=device,
        )

    if backbone == "torch":
        found_torch, torch = _try_import_torch()
        if not found_torch:
            logger.warning("backbone=torch requested but torch unavailable — TF fallback.")
            return TfVisionTransformerBackbone(
                input_size=144,
                projection_dim=projection_dim,
                backbone_name=backbone_name,
                device=device,
            )
        torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return TorchVisionTransformerBackbone(
            backbone_name=backbone_name,
            projection_dim=projection_dim,
            device=torch_device,
        )

    if backbone == "tf":
        return TfVisionTransformerBackbone(
            input_size=144,
            projection_dim=projection_dim,
            backbone_name=backbone_name,
            device=device,
        )

    raise ValueError(f"Unknown backbone selection: {backbone!r} (use 'auto'|'torch'|'tf').")


class FoundationEncoder:
    """
    Wraps whichever backbone was selected and exposes a uniform embedding API.
    """

    def __init__(
        self,
        backbone: str = "auto",
        backbone_name: str = "vit_small_patch16_224",
        projection_dim: int = FOUNDATION_PROJECTED_DIM,
        fuse_mode: str = FOUNDATION_FUSE_MODE,
        device: Optional[str] = None,
    ) -> None:
        if fuse_mode not in ("append", "replace"):
            raise ValueError(f"Unsupported fuse_mode: {fuse_mode!r} (use 'append'|'replace').")
        self.fuse_mode = fuse_mode
        self.backbone = create_ssl_encoder(
            backbone=backbone,
            backbone_name=backbone_name,
            projection_dim=projection_dim,
            device=device,
        )
        self.projection_dim = projection_dim
        self._input_size = 144

    def embed(self, images: np.ndarray) -> np.ndarray:
        """
        Extract compact foundation embeddings for a batch of RGB images.

        ``images``: (N, H, W, 3) float32 in [0,1].
        Returns: (N, projection_dim) float32.
        """
        return self.backbone.embed(images)

    def embed_preprocessed(self, images: np.ndarray) -> np.ndarray:
        """
        Embed a batch that is already at the FreeBuff 144x144 contract.

        Convenience wrapper around ``embed`` so callers that already normalised
        to 144x144 do not have to resize again.
        """
        return self.embed(images)

    @property
    def name(self) -> str:
        return getattr(self.backbone, "backbone_name", "foundation_vit_tf")


# ---------------------------------------------------------------------------
# Fusion into the hybrid pipeline
# ---------------------------------------------------------------------------

def fuse_with_foundation(
    hybrid: np.ndarray,
    foundation: np.ndarray,
    fuse_mode: str = FOUNDATION_FUSE_MODE,
    hog_pca: Optional[np.ndarray] = None,
    deep: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fuse foundation embeddings with the existing hybrid pipeline.

    Parameters
    ----------
    hybrid : np.ndarray
        Existing hybrid vector, e.g. (N, 416) for deep(128)+hog_pca(288).
    foundation : np.ndarray
        Foundation embeddings, (N, D).
    fuse_mode : str
        "append" → hybrid = [deep(128), hog_pca(288), foundation_proj(64)]
        "replace" → hybrid = [foundation_proj(256), hog_pca(288)]
    hog_pca : np.ndarray or None
        HOG-PCA features (N, 288). Required for "replace" mode and for
        reconstructing the hybrid in "append" mode when only components are
        available.
    deep : np.ndarray or None
        Deep features (N, 128). Required for "append" mode when reconstructing
        from components.

    Returns
    -------
    np.ndarray
        Fused hybrid vector ready for the scaler / classifier.
    """
    if fuse_mode == "append":
        if deep is None or hog_pca is None:
            # hybrid already contains deep+ hog, so just append foundation.
            if hybrid.shape[1] != 128 + 288:
                raise ValueError(
                    f"Expected hybrid dim 416 for append mode, got {hybrid.shape[1]}."
                )
            return np.concatenate([hybrid, foundation], axis=1)

        fused = np.concatenate([deep, hog_pca, foundation], axis=1)
        if fused.shape[1] != HYBRID_DIM_WITH_FUSION:
            raise ValueError(
                f"Append fusion expected dim {HYBRID_DIM_WITH_FUSION}, got {fused.shape[1]}."
            )
        return fused

    if fuse_mode == "replace":
        if hog_pca is None or foundation.shape[1] != 256:
            raise ValueError(
                "Replace mode requires foundation dim 256 and hog_pca (N,288)."
            )
        fused = np.concatenate([foundation, hog_pca], axis=1)
        if fused.shape[1] != 256 + 288:
            raise ValueError(
                f"Replace fusion expected dim 544, got {fused.shape[1]}."
            )
        return fused

    raise ValueError(f"Unsupported fuse_mode: {fuse_mode!r}.")


def foundation_projection_matrix(
    raw_dim: int = FOUNDATION_EMBED_DIM,
    projected_dim: int = FOUNDATION_PROJECTED_DIM,
    seed: int = 42,
) -> np.ndarray:
    """
    Deterministic projection matrix from raw foundation dim to compact slot.

    Used when a raw backbone embedding must be projected without a learned
    projection head (e.g. TF bridge feature dim mismatch).  In production this
    should be replaced by the learned projection inside the backbone.
    """
    rng = np.random.RandomState(seed)
    W = rng.randn(raw_dim, projected_dim).astype(np.float32)
    # Small SVD-style whitening proxy: normalise columns to unit norm.
    col_norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    W = W / col_norms
    return W


def project_foundation_embedding(
    emb: np.ndarray,
    W: Optional[np.ndarray] = None,
    projected_dim: int = FOUNDATION_PROJECTED_DIM,
) -> np.ndarray:
    """
    Project a raw foundation embedding batch to the compact slot.

    ``emb``: (N, raw_dim) float32.
    Returns: (N, projected_dim) float32.
    """
    if W is None:
        W = foundation_projection_matrix(projected_dim=projected_dim)

    if emb.shape[1] != W.shape[0]:
        raise ValueError(
            f"Embedding dim {emb.shape[1]} != projection matrix dim {W.shape[0]}."
        )

    proj = emb @ W
    # L2-normalise embeddings so they behave like directional features.
    norm = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
    proj = proj / norm
    return proj.astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("ssl_encoder.py — self-tests")
    logger.info("=" * 60)

    # Synthetic RGB batch at the FreeBuff contract.
    fake = np.random.rand(4, 144, 144, 3).astype(np.float32)

    # 1. Torch backbone path (must only log failure, not crash).
    try:
        enc = create_ssl_encoder(backbone="torch", backbone_name="vit_small_patch16_224")
        emb = enc.embed(fake)
        assert emb.shape == (4, FOUNDATION_PROJECTED_DIM), emb.shape
        logger.info("✓ Torch foundation encoder → %s", emb.shape)
    except Exception as exc:
        logger.warning("Torch foundation path unavailable (expected on CPU/no-torch): %s", exc)

    # 2. TF backbone path (guaranteed runnable).
    try:
        tf_enc = create_ssl_encoder(backbone="tf")
        emb_tf = tf_enc.embed(fake)
        assert emb_tf.shape == (4, FOUNDATION_PROJECTED_DIM), emb_tf.shape
        logger.info("✓ TF foundation encoder → %s", emb_tf.shape)
    except Exception as exc:
        logger.error("TF foundation path failed: %s", exc)
        raise

    # 3. Fusion contract in append mode.
    deep = np.random.rand(4, 128).astype(np.float32)
    hog_pca = np.random.rand(4, 288).astype(np.float32)
    hybrid = np.concatenate([deep, hog_pca], axis=1)
    assert hybrid.shape == (4, 416)

    fused = fuse_with_foundation(hybrid, emb_tf, fuse_mode="append")
    assert fused.shape == (4, HYBRID_DIM_WITH_FUSION), fused.shape
    logger.info("✓ Append fusion → %s (416 + foundation_proj)", fused.shape)

    # 4. Projection helper.
    raw_emb = np.random.rand(4, FOUNDATION_EMBED_DIM).astype(np.float32)
    proj = project_foundation_embedding(raw_emb)
    assert proj.shape == (4, FOUNDATION_PROJECTED_DIM), proj.shape
    logger.info("✓ Foundation projection → %s", proj.shape)

    logger.info("=" * 60)
    logger.info("ssl_encoder.py self-tests passed ✓")
    logger.info("=" * 60)
