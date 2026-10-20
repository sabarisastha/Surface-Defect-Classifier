"""
model.py — Model builder for the defect classifier.

Supported architectures (configured via config.ARCHITECTURE):
  • resnet18      — lightweight, fast
  • resnet50      — deeper, more capacity
  • mobilenet_v2  — efficient, suitable for deployment

All backbones are loaded with ImageNet pretrained weights.
The classification head is replaced with a custom Dropout + Linear layer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────
# 1. Build model
# ─────────────────────────────────────────────────────────────

def build_model(
    architecture: str = config.ARCHITECTURE,
    num_classes: int  = len(config.NEU_CLASSES),
    dropout: float    = config.DROPOUT_RATE,
    pretrained: bool  = True,
) -> nn.Module:
    """
    Return a pretrained backbone with a custom classification head.

    The final Linear layer is replaced by:
        Dropout(p) → Linear(in_features, num_classes)

    Args:
        architecture: One of 'resnet18', 'resnet50', 'mobilenet_v2'.
        num_classes:  Number of output classes.
        dropout:      Dropout probability in the head.
        pretrained:   Whether to load ImageNet weights.

    Returns:
        An nn.Module ready for training.

    Raises:
        ValueError: If an unsupported architecture is specified.
    """
    weights_arg = "DEFAULT" if pretrained else None

    if architecture == "resnet18":
        model = models.resnet18(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif architecture == "resnet50":
        model = models.resnet50(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Choose from: resnet18, resnet50, mobilenet_v2."
        )

    return model


# ─────────────────────────────────────────────────────────────
# 2. Parameter freezing helpers
# ─────────────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module, architecture: str = config.ARCHITECTURE) -> None:
    """
    Freeze all backbone parameters; leave only the head trainable.

    Used in Phase 1 of training to warm-up the new classification head.

    Args:
        model:        The full model.
        architecture: Architecture name (needed to identify the head module).
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the new head
    head = _get_head(model, architecture)
    for param in head.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze every parameter in the model for full fine-tuning (Phase 2).

    Args:
        model: The model whose parameters should all be set trainable.
    """
    for param in model.parameters():
        param.requires_grad = True


def _get_head(model: nn.Module, architecture: str) -> nn.Module:
    """Return the classification head sub-module."""
    if architecture in ("resnet18", "resnet50"):
        return model.fc
    elif architecture == "mobilenet_v2":
        return model.classifier
    raise ValueError(f"Unknown architecture: {architecture}")


def get_head_params(model: nn.Module, architecture: str = config.ARCHITECTURE) -> list:
    """Return parameter list for the classification head only."""
    return list(_get_head(model, architecture).parameters())


def get_backbone_params(model: nn.Module, architecture: str = config.ARCHITECTURE) -> list:
    """Return parameter list for backbone only (excludes head)."""
    head_ids = {id(p) for p in _get_head(model, architecture).parameters()}
    return [p for p in model.parameters() if id(p) not in head_ids]


# ─────────────────────────────────────────────────────────────
# 3. Checkpoint I/O
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_loss: float,
    val_acc: float,
    class_names: List[str],
    path: Path,
    architecture: str = config.ARCHITECTURE,
) -> None:
    """
    Save model weights and metadata to a .pth file.

    Args:
        model:        Trained model.
        epoch:        Current epoch number (0-indexed).
        val_loss:     Validation loss at this checkpoint.
        val_acc:      Validation accuracy at this checkpoint.
        class_names:  List of class names (for loading later).
        path:         File path to write the checkpoint.
        architecture: Architecture name (stored in metadata).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "architecture": architecture,
            "num_classes": len(class_names),
            "class_names": class_names,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[nn.Module, List[str], Dict]:
    """
    Load a saved checkpoint and return the model, class names, and metadata.

    Args:
        path:   Path to the .pth checkpoint.
        device: Target device.

    Returns:
        (model, class_names, metadata_dict)
    """
    ckpt = torch.load(path, map_location=device)
    architecture = ckpt["architecture"]
    num_classes  = ckpt["num_classes"]
    class_names  = ckpt["class_names"]

    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,   # weights come from checkpoint
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, class_names, metadata


# ─────────────────────────────────────────────────────────────
# 4. Model info
# ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters.

    Args:
        model: Any nn.Module.

    Returns:
        Dict with keys 'trainable' and 'total'.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total}
