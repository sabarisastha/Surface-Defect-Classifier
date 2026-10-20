"""
train.py — Two-phase training loop with early stopping and checkpointing.

Phase 1 (head warm-up): backbone frozen, only the new head trains.
Phase 2 (fine-tuning):  entire network trains at a lower learning rate.

Usage:
    python src/train.py [--arch resnet18] [--epochs-head 5] [--epochs-ft 25]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Resolve src/ on path before local imports ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from dataset import get_dataloaders, save_sample_grid, save_class_distribution, NEUDefectDataset
from model import (
    build_model,
    freeze_backbone,
    unfreeze_all,
    get_head_params,
    get_backbone_params,
    save_checkpoint,
    count_parameters,
)


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# One epoch helpers
# ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
    desc: str = "",
) -> Tuple[float, float]:
    """
    Run one training or evaluation epoch.

    Args:
        model:     The neural network.
        loader:    DataLoader for this split.
        criterion: Loss function.
        optimizer: Optimizer (None during evaluation).
        device:    Target device.
        training:  If True, performs gradient updates.
        desc:      Description string for the progress bar.

    Returns:
        (mean_loss, accuracy) for this epoch.
    """
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# Training phase
# ─────────────────────────────────────────────────────────────

def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    patience: int,
    class_names: List[str],
    checkpoint_path: Path,
    phase_name: str,
    history: Dict,
) -> Tuple[float, float]:
    """
    Train for up to num_epochs, with early stopping on val_loss.

    Args:
        model:            The model to train.
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        optimizer:        Configured optimizer.
        scheduler:        LR scheduler.
        criterion:        Loss function.
        device:           Target device.
        num_epochs:       Maximum epochs for this phase.
        patience:         Early-stopping patience.
        class_names:      For checkpoint metadata.
        checkpoint_path:  Where to save the best model.
        phase_name:       Label for logging (e.g. 'head' or 'finetune').
        history:          Mutable dict that accumulates loss/acc lists.

    Returns:
        (best_val_loss, best_val_acc)
    """
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    no_improve    = 0

    print(f"\n{'='*60}")
    print(f" Phase: {phase_name}  |  max_epochs={num_epochs}  |  patience={patience}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            training=True, desc=f"[{phase_name}] E{epoch+1:02d} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            training=False, desc=f"[{phase_name}] E{epoch+1:02d} val  ",
        )

        elapsed = time.time() - t0

        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:3d}/{num_epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"acc {train_acc:.4f}/{val_acc:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Append to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Checkpoint on improvement
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            no_improve    = 0
            save_checkpoint(
                model=model,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                class_names=class_names,
                path=checkpoint_path,
            )
            print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered after {epoch+1} epochs.")
                break

    return best_val_loss, best_val_acc


# ─────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict, out_path: Path) -> None:
    """
    Save loss and accuracy curves to a PNG file.

    Args:
        history:  Dict with keys train_loss, val_loss, train_acc, val_acc.
        out_path: File path for the saved figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curves")
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy Curves")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[train] Curves saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(config.RANDOM_SEED)

    # ── GPU / CUDA Check ──────────────────────────────────────
    print("\n" + "="*60)
    print(" GPU Diagnostic Check")
    print("="*60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name:       {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:   {torch.version.cuda}")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Save class names for Flask
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_names_path = config.MODELS_DIR / config.CLASS_NAMES_FILE
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[train] Class names saved → {class_names_path}")

    # ── EDA plots ─────────────────────────────────────────────
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.USE_MNIST:
        from dataset import NEUDefectDataset
        full_ds_for_eda = NEUDefectDataset(
            root=config.RAW_DIR,
            class_names=class_names,
            transform=__import__("dataset", fromlist=["get_transforms"]).get_transforms("val"),
        )
        if len(full_ds_for_eda) > 0:
            save_sample_grid(
                full_ds_for_eda, class_names,
                config.OUTPUTS_DIR / config.SAMPLE_GRID,
            )
            save_class_distribution(
                full_ds_for_eda.targets, class_names,
                config.OUTPUTS_DIR / config.CLASS_DIST_PLOT,
            )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(
        architecture=args.arch,
        num_classes=len(class_names),
    )
    model.to(device)

    info = count_parameters(model)
    print(f"[train] {args.arch}: {info['total']:,} total params, "
          f"{info['trainable']:,} trainable at init")

    criterion = nn.CrossEntropyLoss()
    checkpoint_path = config.MODELS_DIR / config.BEST_CHECKPOINT

    history: Dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Phase 1: head warm-up ─────────────────────────────────
    freeze_backbone(model, args.arch)
    head_opt = torch.optim.AdamW(
        get_head_params(model, args.arch),
        lr=args.lr_head,
        weight_decay=config.WEIGHT_DECAY,
    )
    head_sched = torch.optim.lr_scheduler.CosineAnnealingLR(head_opt, T_max=args.epochs_head)

    train_phase(
        model, train_loader, val_loader,
        head_opt, head_sched, criterion, device,
        num_epochs=args.epochs_head,
        patience=config.PATIENCE,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        phase_name="head-warmup",
        history=history,
    )

    # ── Phase 2: full fine-tune ────────────────────────────────
    unfreeze_all(model)
    ft_opt = torch.optim.AdamW(
        [
            {"params": get_backbone_params(model, args.arch), "lr": args.lr_ft * 0.1},
            {"params": get_head_params(model, args.arch),     "lr": args.lr_ft},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    if config.SCHEDULER == "cosine":
        ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=args.epochs_ft)
    else:
        ft_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            ft_opt, mode="min", patience=3, factor=0.5
        )

    train_phase(
        model, train_loader, val_loader,
        ft_opt, ft_sched, criterion, device,
        num_epochs=args.epochs_ft,
        patience=config.PATIENCE,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        phase_name="fine-tune",
        history=history,
    )

    # ── Save curves ────────────────────────────────────────────
    plot_training_curves(history, config.OUTPUTS_DIR / config.TRAINING_CURVES)
    print("\n[train] Training complete.")
    print(f"        Best checkpoint → {checkpoint_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the surface defect classifier.")
    parser.add_argument("--arch",         type=str, default=config.ARCHITECTURE,
                        choices=["resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--epochs-head",  type=int,   default=config.HEAD_EPOCHS)
    parser.add_argument("--epochs-ft",    type=int,   default=config.FINETUNE_EPOCHS)
    parser.add_argument("--lr-head",      type=float, default=config.HEAD_LR)
    parser.add_argument("--lr-ft",        type=float, default=config.FINETUNE_LR)
    parser.add_argument("--batch-size",   type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--num-workers",  type=int,   default=config.NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
