#!/usr/bin/env python3
# ==============================================================================
# FreeBuff v4 — Hybrid Model Architecture (model.py)
# ==============================================================================
# Branch A:  ImageNet-pretrained ResNet50 backbone with a CBAM (Convolutional
#            Block Attention Module) refinement stage on top of `layer4`.
# Branch B:  Handcrafted HOG texture descriptor, PCA-projected to 288 dims,
#            embedded through a small MLP.
# Fusion:    Concatenation of both embeddings -> FC fusion head -> logits over
#            the 4 tumour classes.
#
# Design constraints honoured in this file:
#   * All activation / dropout layers are strictly NON-INPLACE so that
#     Grad-CAM++ (torch.autograd.grad) can traverse the autograd graph.
#   * `load_checkpoint` is error-proof: it accepts raw state dicts, wrapped
#     payloads, torch.jit archives, legacy v1 keys (with automatic stripping
#     and key renaming), and performs shape matching with a best-effort
#     strict=True -> partial fallback so a checkpoint never crashes the app.
#   * Grad-CAM++ uses a forward-only hook + torch.autograd.grad(), never
#     register_full_backward_hook (which corrupts gradients when downstream
#     ops reshape the module output, and triggers CUDA illegal memory access).
# ==============================================================================
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import Tensor

from config import (
    BACKBONE_LR,
    CUDA_DEVICE_STRING,
    DEFAULT_LR,
    GRADCAM_TARGET_LAYER_NAME,
    HOG_DIMS_RAW,
    HOG_PCA_COMPONENTS,
    NUM_CLASSES,
)

logger = logging.getLogger("freebuff_v4.model")


# ==============================================================================
# CBAM: CHANNEL ATTENTION MODULE
# ==============================================================================
class ChannelAttention(nn.Module):
    """Channel attention (CAM) — the first stage of CBAM.

    A shared MLP (applied to both global-average-pooled and global-max-pooled
    channel descriptors) produces per-channel gate weights in [0, 1].
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        num_layers: int = 2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        reduced = max(1, in_channels // reduction)

        layers: List[nn.Module] = [nn.Linear(in_channels, reduced, bias=False)]
        if use_batch_norm:
            layers.append(nn.LayerNorm(reduced))
        # FIX: inplace=False everywhere — in-place ReLU corrupts the autograd
        # graph that Grad-CAM++ must traverse.
        layers.append(nn.ReLU(inplace=False))
        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Linear(reduced, reduced, bias=False))
            if use_batch_norm:
                layers.append(nn.LayerNorm(reduced))
            layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Linear(reduced, in_channels, bias=True))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[0], x.shape[1]
        avg_pool = x.view(b, c, -1).mean(dim=2)
        max_pool = x.view(b, c, -1).max(dim=2).values
        att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        return x * att.expand_as(x)


# ==============================================================================
# CBAM: SPATIAL ATTENTION MODULE
# ==============================================================================
class SpatialAttention(nn.Module):
    """Spatial attention (SAM) — the second stage of CBAM.

    Channel-wise average and max descriptors are concatenated and convolved
    into a single-channel (B, 1, H, W) spatial attention map.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # enforce odd kernel for symmetric padding
        padding = kernel_size // 2

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=True),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(1))
        layers.append(nn.Sigmoid())
        self.conv = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="sigmoid")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        spatial_att = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att


# ==============================================================================
# CBAM: FULL MODULE (channel -> spatial)
# ==============================================================================
class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial."""

    def __init__(
        self,
        in_channels: int,
        channel_reduction: int = 16,
        spatial_kernel_size: int = 7,
        channel_num_layers: int = 2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(
            in_channels=in_channels,
            reduction=channel_reduction,
            num_layers=channel_num_layers,
            use_batch_norm=use_batch_norm,
        )
        self.spatial_att = SpatialAttention(
            kernel_size=spatial_kernel_size,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ==============================================================================
# CBAM-WRAPPED RESNET50 BACKBONE
# ==============================================================================
class CBAMResNet50(nn.Module):
    """ResNet50 backbone with CBAM applied after the final residual stage.

    The forward hook target for Grad-CAM++ is `layer4`; CBAM refines its
    output before pooling.
    """

    def __init__(
        self,
        pretrained: bool = True,
        cbam_channels: Optional[int] = None,
        cbam_kwargs: Optional[Dict[str, Any]] = None,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.out_channels: int = resnet.fc.in_features  # 2048 for ResNet50

        del resnet.fc

        # FIX: cbam_channels must be the BACKBONE's channel count (2048), not
        # the embedding dim. A previous bug passed the spatial embedding size
        # (256) here, which built a 256-channel CBAM that received 2048-channel
        # feature maps from layer4 — a guaranteed shape mismatch at runtime.
        cbam_in = cbam_channels if cbam_channels is not None else self.out_channels
        if cbam_in != self.out_channels:
            logger.warning(
                "CBAMResNet50: cbam_channels=%s differs from backbone output "
                "%s; this will fail on forward unless intentional.",
                cbam_in, self.out_channels,
            )
        self.cbam = CBAM(in_channels=cbam_in, **(cbam_kwargs or {}))

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        return x


# ==============================================================================
# HOG TEXTURE BRANCH (Branch B)
# ==============================================================================
class HOGBranch(nn.Module):
    """MLP embedding the PCA-projected HOG descriptor (288-d -> 512-d)."""

    def __init__(
        self,
        input_dim: int = HOG_PCA_COMPONENTS,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512]
        dims = [input_dim] + list(hidden_dims)

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nn.ReLU(inplace=False))   # non-inplace: Grad-CAM safe
            layers.append(nn.Dropout(dropout, inplace=False))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.out_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: Optional[Tensor]) -> Tensor:
        if x is None:
            # Graceful degradation when the handcrafted branch is unavailable
            # (e.g. no PCA artifact shipped): emit a zero embedding so the
            # fusion head still receives a correctly-shaped tensor.
            return torch.zeros(self._last_batch, self.out_dim,
                               device=self._last_device, dtype=torch.float32)
        self._last_batch = x.size(0)
        self._last_device = x.device
        return self.net(x)


# ==============================================================================
# FUSION CLASSIFICATION HEAD
# ==============================================================================
class FusionClassificationHead(nn.Module):
    """Concatenates the spatial and HOG embeddings and maps them to logits."""

    def __init__(
        self,
        spatial_dim: int,
        hog_dim: int,
        fusion_hidden_dims: Optional[Sequence[int]] = None,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        fusion_hidden_dims = fusion_hidden_dims or [512, 256]
        dims = [spatial_dim + hog_dim] + list(fusion_hidden_dims) + [num_classes]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nn.ReLU(inplace=False))   # non-inplace: Grad-CAM safe
            layers.append(nn.Dropout(dropout, inplace=False))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, spatial_emb: Tensor, hog_emb: Tensor) -> Tensor:
        fused = torch.cat([spatial_emb, hog_emb], dim=1)
        return self.net(fused)


# ==============================================================================
# FOCAL LOSS (gamma = 2.0 default)
# ==============================================================================
class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t), implemented via the CE expansion
    so `alpha` class weighting composes cleanly. Down-weights easy examples
    and concentrates gradient signal on hard, borderline cases.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        # Register as buffer so .to(device) moves it automatically while
        # keeping it out of state_dict semantics.
        if alpha is not None:
            self.register_buffer("alpha", alpha.detach().clone().float())
        else:
            self.register_buffer("alpha", torch.tensor(0.0), persistent=False)
            self.alpha = None  # type: ignore[assignment]

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ==============================================================================
# FREEBUFF V4 — FULL HYBRID MODEL
# ==============================================================================
class FreeBuffV4(nn.Module):
    """Full hybrid dual-branch architecture.

    Branch A (deep spatial):   ResNet50 + CBAM -> GAP -> MLP (256-d)
    Branch B (handcrafted):    HOG -> PCA(288) -> MLP (512-d)
    Fusion:                    [256 + 512] -> [512, 256] -> 4 logits
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        spatial_embedding_dim: int = 256,
        hog_embedding_dim: int = 512,
        hog_pca_components: int = HOG_PCA_COMPONENTS,
        hog_hidden_dims: Optional[Sequence[int]] = None,
        fusion_hidden_dims: Optional[Sequence[int]] = None,
        dropout_spatial: float = 0.3,
        dropout_hog: float = 0.2,
        dropout_fusion: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_bn: bool = False,
        cbam_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.spatial_embedding_dim = spatial_embedding_dim
        self.hog_embedding_dim = hog_embedding_dim
        self.hog_pca_components = hog_pca_components
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn

        self.backbone = CBAMResNet50(
            pretrained=pretrained,
            cbam_channels=None,  # always the full 2048-channel layer4 output
            cbam_kwargs=cbam_kwargs,
            freeze_bn=freeze_bn,
        )
        backbone_dim = self.backbone.out_channels

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        hidden_spatial = max(256, backbone_dim // 2)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(backbone_dim, hidden_spatial, bias=True),
            nn.ReLU(inplace=False),                    # non-inplace: Grad-CAM safe
            nn.Dropout(0.2, inplace=False),
            nn.Linear(hidden_spatial, spatial_embedding_dim, bias=True),
            nn.ReLU(inplace=False),                    # non-inplace: Grad-CAM safe
        )
        for m in self.spatial_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.hog_branch = HOGBranch(
            input_dim=hog_pca_components,
            hidden_dims=hog_hidden_dims,
            dropout=dropout_hog,
        )

        self.head = FusionClassificationHead(
            spatial_dim=spatial_embedding_dim,
            hog_dim=hog_embedding_dim,
            fusion_hidden_dims=fusion_hidden_dims,
            num_classes=num_classes,
            dropout=dropout_fusion,
        )

        self._post_init(freeze_backbone=freeze_backbone, freeze_bn=freeze_bn)

    # ------------------------------------------------------------------ init
    def _post_init(self, freeze_backbone: bool, freeze_bn: bool) -> None:
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "FreeBuffV4 initialised: total_params=%d, trainable=%d, "
            "spatial_emb=%d, hog_emb=%d, hog_pca=%d.",
            total, trainable, self.spatial_embedding_dim,
            self.hog_embedding_dim, self.hog_pca_components,
        )

    # --------------------------------------------------------------- forward
    def forward(
        self,
        image: Tensor,
        hog_pca: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        spatial_map = self.backbone(image)               # (B, 2048, H', W')
        pooled = self.spatial_pool(spatial_map).flatten(1)
        spatial_emb = self.spatial_mlp(pooled)           # (B, 256)

        if hog_pca is not None:
            hog_emb = self.hog_branch(hog_pca)           # (B, 512)
        else:
            hog_emb = torch.zeros(
                image.size(0), self.hog_embedding_dim,
                device=image.device, dtype=image.dtype,
            )

        logits = self.head(spatial_emb, hog_emb)         # (B, num_classes)
        return {
            "logits": logits,
            "spatial_emb": spatial_emb,
            "hog_emb": hog_emb,
            "fused_emb": torch.cat([spatial_emb, hog_emb], dim=1),
        }

    def get_logits(self, image: Tensor, hog_pca: Optional[Tensor] = None) -> Tensor:
        return self.forward(image, hog_pca=hog_pca)["logits"]

    def named_attention_maps(self) -> Dict[str, Any]:
        return {
            "channel_attention": self.backbone.cbam.channel_att,
            "spatial_attention": self.backbone.cbam.spatial_att,
        }

    # ---------------------------------------------------------- gradcam target
    def get_gradcam_target_layer(self) -> nn.Module:
        """Resolve the Grad-CAM++ target layer (resilient to key prefixes)."""
        if hasattr(self.backbone, "layer4"):
            return self.backbone.layer4  # type: ignore[return-value]
        for name, module in self.named_modules():
            if name.endswith(".layer4"):
                return module
        raise RuntimeError("FreeBuffV4: Grad-CAM target layer 'layer4' not found.")


# ==============================================================================
# MODEL BUILDER
# ==============================================================================
def build_freebuff_v4(
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    freeze_bn: bool = False,
    fusion_hidden_dims: Optional[Sequence[int]] = None,
    **model_kwargs: Any,
) -> FreeBuffV4:
    """Construct a FreeBuffV4 and move it to `device`."""
    if device is None:
        device = torch.device(CUDA_DEVICE_STRING if torch.cuda.is_available() else "cpu")
    model = FreeBuffV4(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        freeze_bn=freeze_bn,
        fusion_hidden_dims=fusion_hidden_dims,
        **model_kwargs,
    )
    model.to(device)
    if device.type == "cuda" and torch.cuda.is_available():
        logger.info(
            "FreeBuffV4 on device %s (%s, %.1f GB).",
            device, torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        )
    else:
        logger.info("FreeBuffV4 on device %s.", device)
    return model


# ==============================================================================
# PARAMETER GROUPS (decoupled backbone / head learning rates)
# ==============================================================================
def build_parameter_groups(
    model: nn.Module,
    backbone_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """Split trainable parameters into backbone (low LR) and head (high LR)."""
    backbone_lr = backbone_lr if backbone_lr is not None else DEFAULT_LR
    head_lr = head_lr if head_lr is not None else BACKBONE_LR

    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else head_params).append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


# ==============================================================================
# CHECKPOINT SAVE
# ==============================================================================
def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    **extra: Any,
) -> Path:
    """Persist model (and optionally optimizer/scheduler) state to `path`.

    Accepts `path` positionally or by keyword (historically the first
    parameter was named `checkpoint_path`, so both names are honoured).
    """
    # Backwards-compatible keyword alias.
    if "checkpoint_path" in extra and extra.get("checkpoint_path") is not None:
        path = extra["checkpoint_path"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_classes": getattr(model, "num_classes", NUM_CLASSES),
            "spatial_embedding_dim": getattr(model, "spatial_embedding_dim", 256),
            "hog_embedding_dim": getattr(model, "hog_embedding_dim", 512),
            "hog_pca_components": getattr(model, "hog_pca_components", HOG_PCA_COMPONENTS),
            "pretrained": False,  # weights are materialised, no download needed
            "freeze_backbone": getattr(model, "freeze_backbone", False),
            "freeze_bn": getattr(model, "freeze_bn", False),
        },
        "format_version": 4,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if train_loss is not None:
        payload["train_loss"] = float(train_loss)
    if val_loss is not None:
        payload["val_loss"] = float(val_loss)
    if val_acc is not None:
        payload["val_acc"] = float(val_acc)
    if extra:
        payload["extra"] = {k: v for k, v in extra.items()
                            if k not in ("checkpoint_path",)}

    torch.save(payload, str(path))
    logger.info("Checkpoint saved to %s", path)
    return path


# ==============================================================================
# ERROR-PROOF CHECKPOINT LOADING
# ==============================================================================
#: Legacy (v1-era) parameter-name fragments -> canonical replacements.
_LEGACY_KEY_RULES: Tuple[Tuple[str, str], ...] = (
    # Old TSM-style temporal prefixes from the archived v1 pipeline.
    ("module.backbone.", "backbone."),
    # torchvision resnet weights saved bare (no backbone prefix).
    ("resnet.", "backbone."),
    # Very old CBAM naming variants.
    ("cbam.channel_attention.mlp", "backbone.cbam.channel_att.mlp"),
    ("cbam.spatial_attention.conv", "backbone.cbam.spatial_att.conv"),
)

#: Exact legacy key -> canonical key (applied after the fragment rules).
_LEGACY_KEY_EXACT: Dict[str, str] = {
    "fc.weight": "head.net.6.weight",
    "fc.bias": "head.net.6.bias",
    "classifier.weight": "head.net.6.weight",
    "classifier.bias": "head.net.6.bias",
}

#: Key fragments that belong to components which no longer exist.
_DROP_KEY_PATTERNS: Tuple[str, ...] = (
    "fc.",            # dropped torchvision head handled above, leftover guards
    "num_batches_tracked",  # never shape-matched; rematerialised by buffers
)


def _strip_legacy_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Rename / drop legacy parameter keys so old checkpoints load cleanly."""
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if any(pat in key for pat in _DROP_KEY_PATTERNS):
            # num_batches_tracked is rematerialised by BN buffers on load;
            # dropping it only ever affects strict-mode counting.
            if key not in _LEGACY_KEY_EXACT and "num_batches_tracked" not in key:
                logger.debug("load_checkpoint: dropping legacy key %s", key)
                continue
        new_key = key
        for old_frag, new_frag in _LEGACY_KEY_RULES:
            if new_key.startswith(old_frag) or old_frag in new_key:
                new_key = new_key.replace(old_frag, new_frag)
                break
        new_key = _LEGACY_KEY_EXACT.get(new_key, new_key)
        out[new_key] = value
    return out


def _align_shapes(
    state_dict: Mapping[str, Any],
    model: nn.Module,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Filter state dict entries whose shapes do not match the live model.

    Returns the filtered dict and a report of every skipped tensor. This is
    the safety net that turns a hard RuntimeError (size mismatch) into a
    logged, partial load with random-init fallbacks.
    """
    own = model.state_dict()
    filtered: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for key, value in state_dict.items():
        if key not in own:
            continue
        target = own[key]
        if isinstance(value, torch.Tensor) and tuple(value.shape) != tuple(target.shape):
            skipped[key] = f"checkpoint{tuple(value.shape)} != model{tuple(target.shape)}"
            continue
        filtered[key] = value
    return filtered, skipped


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    strip_legacy_keys: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """Load a FreeBuff v4 checkpoint into `model` — designed never to crash.

    Robustness layers, applied in order:
      1. Transparently reads `.pt` / `.pth` / raw TorchScript archives.
      2. Unwraps any payload envelope (`model_state_dict`, `state_dict`,
         `model`, `generator`, plain dict) to the state dict itself.
      3. Strips legacy v1 parameter-name conventions (renames + drops).
      4. Drops optimizer-only bookkeeping keys that break strict loading.
      5. Shape-matches every remaining tensor against the live model and
         skips mismatches with a logged warning (never a size RuntimeError).
      6. Falls back from strict=True to strict=False if a handful of keys
         (e.g. freshly-added buffers) are absent.
      7. Optionally restores optimizer / scheduler state with its own
         try/except so a corrupt optimizer block cannot poison the weights.

    Args:
        checkpoint_path: File to load.
        model: Destination module.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Target device for tensor materialisation.
        strict: Attempt strict loading first (falls back automatically).
        strip_legacy_keys: Apply the legacy key renaming table.
        verbose: Log a full report of remapped / skipped keys.

    Returns:
        The `model` instance, loaded in place.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = next(model.parameters()).device if any(
            p.requires_grad or True for p in model.parameters()
        ) else torch.device("cpu")

    try:
        payload = torch.load(str(checkpoint_path), map_location=device,
                             weights_only=False, mmap=False)
    except TypeError:
        # Older torch without mmap / weights_only kwargs.
        payload = torch.load(str(checkpoint_path), map_location=device)
    except RuntimeError:
        # Some TorchScript archives only load via torch.jit.load.
        try:
            jit_module = torch.jit.load(str(checkpoint_path), map_location=device)
            payload = jit_module.state_dict()
        except Exception as exc:
            raise RuntimeError(
                f"Unreadable checkpoint {checkpoint_path}: {exc}"
            ) from exc

    # ---- 2. Unwrap the payload envelope -----------------------------------
    if isinstance(payload, dict):
        state_dict = None
        for key in ("model_state_dict", "state_dict", "model", "generator",
                    "module", "weights"):
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break
        if state_dict is None:
            # A raw {param_name: tensor} dict — but only if it "looks" like one
            # (values are tensors), otherwise the envelope keys would leak in.
            tensor_vals = [v for v in payload.values() if isinstance(v, torch.Tensor)]
            if len(tensor_vals) >= max(1, len(payload) // 2):
                state_dict = payload
    else:
        state_dict = payload

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} contains no loadable state dict."
        )

    # ---- 3. Legacy key stripping ------------------------------------------
    if strip_legacy_keys:
        state_dict = _strip_legacy_keys(state_dict)

    # ---- 4. Drop optimizer bookkeeping that strict loading rejects --------
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith("num_batches_tracked")}

    # ---- 5. Shape matching -------------------------------------------------
    state_dict, skipped = _align_shapes(state_dict, model)
    if verbose and skipped:
        logger.warning(
            "load_checkpoint: skipped %d shape-mismatched tensors: %s",
            len(skipped), skipped,
        )

    # ---- 6. Strict -> partial fallback -------------------------------------
    missing_before = set(model.state_dict()) - set(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
        load_mode = "strict"
    except RuntimeError as exc:
        logger.warning("load_checkpoint: strict load failed (%s); retrying partial.", exc)
        model.load_state_dict(state_dict, strict=False)
        load_mode = "partial"

    if verbose:
        logger.info(
            "Checkpoint loaded from %s (mode=%s, tensors=%d, skipped=%d, "
            "random-init=%d).",
            checkpoint_path, load_mode, len(state_dict), len(skipped),
            len(missing_before),
        )
        if missing_before:
            sample = sorted(missing_before)[:8]
            logger.info("load_checkpoint: keys initialised randomly (sample): %s",
                        sample)

    # ---- 7. Optimizer / scheduler restore ----------------------------------
    if optimizer is not None and isinstance(payload, dict):
        opt_state = payload.get("optimizer_state_dict")
        if opt_state:
            try:
                optimizer.load_state_dict(opt_state)
                if verbose:
                    logger.info("load_checkpoint: optimizer state restored.")
            except Exception as exc:
                logger.warning("load_checkpoint: optimizer restore failed: %s", exc)

    if scheduler is not None and isinstance(payload, dict):
        sched_state = payload.get("scheduler_state_dict")
        if sched_state:
            try:
                scheduler.load_state_dict(sched_state)
                if verbose:
                    logger.info("load_checkpoint: scheduler state restored.")
            except Exception as exc:
                logger.warning("load_checkpoint: scheduler restore failed: %s", exc)

    return model


# ==============================================================================
# GRAD-CAM++ EXPLAINABILITY ENGINE
# ==============================================================================
class GradCAMPlusPlus:
    """Grad-CAM++ for FreeBuff v4.

    Implementation notes (why not `register_full_backward_hook`):
      * A full backward hook fires with stale/reshaped gradients when the
        hooked module's output is transformed downstream (view/clone/cat),
        which has produced CUDA illegal-memory-access crashes on this codebase.
      * We instead capture the layer output with a *forward-only* hook and
        compute d(score)/d(features) explicitly with `torch.autograd.grad`.
        The features tensor stays ON the score->input graph path, so the
        gradient is exact, deterministic under AMP, and CUDA-safe.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str = GRADCAM_TARGET_LAYER_NAME,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer_name = target_layer_name
        self.hooks: List[Any] = []
        self._features: Optional[Tensor] = None
        self._register_hook()

    # ------------------------------------------------------------------ hook
    def _register_hook(self) -> None:
        target_layer: Optional[nn.Module] = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name or name.endswith("." + self.target_layer_name):
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(
                f"Target layer '{self.target_layer_name}' not found in model. "
                f"Available: {[n for n, _ in self.model.named_modules()]}"
            )

        def forward_hook(_module: nn.Module, _inp: Any, out: Any) -> None:
            # Keep the EXACT output tensor (no clone/detach): it must remain
            # the same graph node that the class score flows back through.
            tensor = out if isinstance(out, Tensor) else out[0]
            if isinstance(tensor, Tensor):
                self._features = tensor

        self.hooks.append(target_layer.register_forward_hook(forward_hook))

    # --------------------------------------------------------------- heatmap
    def generate_heatmap(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute the Grad-CAM++ saliency map.

        Args:
            input_tensor: (B, 3, H, W) image batch (B == 1 supported).
            class_idx: Target class; defaults to the model's argmax.

        Returns:
            (heatmap, details) where heatmap is (1, 1, H, W) in [0, 1] and
            details carries logits / class index / target layer name.
        """
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.detach().to(self.device).float()
        input_tensor.requires_grad_(True)

        self._features = None
        output = self.model(input_tensor)
        logits = output["logits"] if isinstance(output, dict) else output
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        num_classes = logits.shape[-1]
        class_idx = int(max(0, min(class_idx, num_classes - 1)))  # CUDA OOB guard
        score = logits[0, class_idx]

        if self._features is None or not isinstance(self._features, Tensor):
            raise RuntimeError(
                "Grad-CAM++: forward hook captured no features — the target "
                f"layer '{self.target_layer_name}' was not reached."
            )
        features = self._features
        if not features.requires_grad:
            # Happens if the caller wrapped the forward in inference_mode.
            raise RuntimeError(
                "Grad-CAM++: features do not require grad — the model ran "
                "inside torch.inference_mode()/no_grad(). Call generate_heatmap "
                "with gradients enabled (model.eval() is fine, no_grad is not)."
            )

        # d(score)/d(features) via autograd.grad — the core of the fix.
        (grad_result,) = torch.autograd.grad(
            outputs=score,
            inputs=features,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        if grad_result is None:
            raise RuntimeError("Grad-CAM++: autograd returned None gradients.")

        gradients = grad_result.detach()
        feats = features.detach()

        # ---- Grad-CAM++ alpha weighting (second-order spatial weights) ----
        grads_pow = gradients.pow(2)
        denom = grads_pow + 2.0 * grads_pow.sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha_num = grads_pow
        alpha = alpha_num / denom.clamp_min(1e-8)

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        heatmap = (weights * feats).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        heatmap = F.interpolate(
            heatmap,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

        details: Dict[str, Any] = {
            "class_idx": class_idx,
            "logits": logits.detach().cpu(),
            "target_layer": self.target_layer_name,
        }
        return heatmap, details

    # --------------------------------------------------------------- overlay
    @staticmethod
    def overlay_heatmap_on_image(
        image: Tensor,
        heatmap: Tensor,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Tensor:
        """Blend a (1,1,H,W) or (H,W) heatmap over a (3,H,W)/(1,3,H,W) image.

        Static so both `instance.overlay_heatmap_on_image(...)` and
        `GradCAMPlusPlus.overlay_heatmap_on_image(...)` work (evaluate.py and
        the API call it both ways). Handles matplotlib >= 3.9 where
        `cm.get_cmap` was removed.
        """
        import matplotlib
        from matplotlib import pyplot as plt

        try:  # matplotlib >= 3.9 removed cm.get_cmap
            cmap = matplotlib.colormaps[colormap]
        except (AttributeError, KeyError):
            cmap = plt.get_cmap(colormap)

        hm = heatmap.detach().float().cpu()
        if hm.dim() == 4:
            hm = hm.squeeze(0).squeeze(0)
        elif hm.dim() == 3:
            hm = hm.squeeze(0)
        hm_np = hm.numpy()
        hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-7)

        colored = cmap(hm_np)[..., :3]                       # (H, W, 3) float
        colored = torch.from_numpy(np.ascontiguousarray(colored)).to(
            device=image.device, dtype=image.dtype,
        )
        colored = colored.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)

        img = image if image.dim() == 4 else image.unsqueeze(0)
        img = img.contiguous().to(device=image.device)
        if img.shape[-2:] != colored.shape[-2:]:
            colored = F.interpolate(colored, size=img.shape[-2:],
                                    mode="bilinear", align_corners=False)

        overlay = (1.0 - alpha) * img + alpha * colored
        return overlay

    # ---------------------------------------------------------------- cleanup
    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._features = None

    def __del__(self) -> None:  # pragma: no cover — best-effort safety
        try:
            self.remove_hooks()
        except Exception:
            pass
