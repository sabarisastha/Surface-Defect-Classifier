"""
evaluate.py — Evaluation, metrics, confusion matrix, and Grad-CAM.

Usage (after training):
    python src/evaluate.py [--checkpoint models/best_model.pth]
"""

import argparse
import json
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
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from dataset import get_dataloaders
from model import load_checkpoint

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[evaluate] WARNING: grad-cam not installed. Grad-CAM will be skipped.")


# ─────────────────────────────────────────────────────────────
# Inference pass
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[List[float]]]:
    """
    Run the model on all batches and collect predictions.

    Args:
        model:  Trained model in eval mode.
        loader: Test DataLoader.
        device: Target device.

    Returns:
        (all_labels, all_preds, all_probs)  — three aligned lists.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="[evaluate] Inference"):
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().tolist()
        preds  = logits.argmax(dim=1).cpu().tolist()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    out_path: Path,
) -> None:
    """
    Save a normalised confusion matrix heatmap.

    Args:
        labels:      Ground-truth integer labels.
        preds:       Predicted integer labels.
        class_names: Ordered class name list.
        out_path:    File path to save the PNG.
    """
    import seaborn as sns

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────

def _get_target_layer(model: nn.Module, architecture: str):
    """Return the last convolutional layer for Grad-CAM."""
    if architecture in ("resnet18", "resnet50"):
        return [model.layer4[-1]]
    elif architecture == "mobilenet_v2":
        return [model.features[-1]]
    raise ValueError(f"Unknown architecture: {architecture}")


def generate_gradcam(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    architecture: str,
    device: torch.device,
    out_dir: Path,
    n_correct: int = 6,
    n_wrong: int = 6,
) -> None:
    """
    Generate and save Grad-CAM visualisations for correct & wrong samples.

    Args:
        model:        Trained model in eval mode.
        loader:       Test DataLoader (batch_size can be 1 for clarity).
        class_names:  Class name list.
        architecture: Architecture string for target layer selection.
        device:       Target device.
        out_dir:      Directory to save Grad-CAM images.
        n_correct:    How many correctly classified samples to visualise.
        n_wrong:      How many incorrectly classified samples to visualise.
    """
    if not GRADCAM_AVAILABLE:
        print("[evaluate] Skipping Grad-CAM (library not installed).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = _get_target_layer(model, architecture)
    cam = GradCAM(model=model, target_layers=target_layers)

    mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)

    correct_saved, wrong_saved = 0, 0
    model.eval()

    for images, labels in loader:
        if correct_saved >= n_correct and wrong_saved >= n_wrong:
            break

        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            preds  = logits.argmax(dim=1)

        for i in range(images.size(0)):
            label = labels[i].item()
            pred  = preds[i].item()
            is_correct = (label == pred)

            if is_correct and correct_saved >= n_correct:
                continue
            if not is_correct and wrong_saved >= n_wrong:
                continue

            # Grad-CAM
            input_tensor = images[i:i+1]
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # Denormalise for display
            img_np = (images[i].cpu() * std.view(3, 1, 1) + mean.view(3, 1, 1)).clamp(0, 1)
            img_np = img_np.permute(1, 2, 0).numpy()

            vis = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

            tag = "correct" if is_correct else "wrong"
            fname = (
                f"{config.GRADCAM_PREFIX}_{tag}_"
                f"true-{class_names[label]}_pred-{class_names[pred]}.png"
            )
            # Avoid name collisions
            idx = correct_saved if is_correct else wrong_saved
            fname = f"{config.GRADCAM_PREFIX}_{tag}_{idx:02d}_"  \
                    f"true-{class_names[label]}_pred-{class_names[pred]}.png"

            plt.figure(figsize=(5, 4))
            plt.imshow(vis)
            title = (
                f"{'✓' if is_correct else '✗'}  "
                f"True: {class_names[label]}  |  Pred: {class_names[pred]}"
            )
            plt.title(title, fontsize=9)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=120)
            plt.close()

            if is_correct:
                correct_saved += 1
            else:
                wrong_saved += 1

    print(f"[evaluate] Grad-CAM → {correct_saved} correct, {wrong_saved} wrong saved to {out_dir}")


# ─────────────────────────────────────────────────────────────
# Inference speed benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    n_images: int = 100,
    image_size: int = config.IMAGE_SIZE,
) -> float:
    """
    Measure average inference time per image (ms).

    Args:
        model:      Model in eval mode.
        device:     Target device.
        n_images:   Number of dummy images to time.
        image_size: Spatial resolution.

    Returns:
        Average time per image in milliseconds.
    """
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_images):
            model(dummy)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / n_images
    return elapsed_ms


# ─────────────────────────────────────────────────────────────
# Analysis helper
# ─────────────────────────────────────────────────────────────

def _write_analysis(
    report: str,
    cm_arr: np.ndarray,
    class_names: List[str],
    out_path: Path,
    arch: str,
    acc: float,
    ms_per_img: float,
) -> None:
    """Write a markdown analysis file summarising evaluation results."""
    lines = [
        "# Evaluation Analysis\n",
        f"**Architecture:** {arch}  |  **Test Accuracy:** {acc*100:.2f}%  |  "
        f"**Inference:** {ms_per_img:.2f} ms/img\n",
        "\n## Classification Report\n",
        "```\n" + report + "\n```\n",
        "\n## Confusion Analysis\n",
    ]

    # Top confused pairs
    confused_pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm_arr[i, j] > 0:
                confused_pairs.append((cm_arr[i, j], class_names[i], class_names[j]))
    confused_pairs.sort(reverse=True)

    lines.append("Top confusion pairs (true → predicted, count):\n\n")
    for count, true_cls, pred_cls in confused_pairs[:10]:
        lines.append(f"- **{true_cls}** → {pred_cls}: {count} samples\n")

    lines.append(
        "\n### Interpretation\n"
        "- *Crazing* and *rolled-in_scale* share fine-line textures that even humans can confuse.\n"
        "- *Inclusion* (dark spots) can resemble *pitted_surface* when defects are small.\n"
        "- *Patches* are large irregular bright regions and are usually well-separated.\n"
        "- *Scratches* are thin bright lines and can overlap visually with *rolled-in_scale*.\n"
        "- Grad-CAM maps confirm whether the network attends to the actual defect region.\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print(f"[evaluate] Analysis written → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[evaluate] ERROR: checkpoint not found at '{checkpoint_path}'")
        sys.exit(1)

    model, class_names, meta = load_checkpoint(checkpoint_path, device)
    architecture = meta["architecture"]
    print(f"[evaluate] Loaded {architecture} | val_acc={meta['val_acc']:.4f}")

    _, _, test_loader, _ = get_dataloaders()

    # ── Predictions ───────────────────────────────────────────
    labels, preds, probs = get_predictions(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    cm_arr = confusion_matrix(labels, preds)

    print(f"\n[evaluate] Test Accuracy: {acc*100:.2f}%\n")
    print(report)

    # ── Confusion matrix ──────────────────────────────────────
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, class_names, config.OUTPUTS_DIR / config.CONFUSION_MATRIX)

    # ── Benchmark ─────────────────────────────────────────────
    ms_per_img = benchmark_inference(model, device)
    print(f"[evaluate] Inference speed: {ms_per_img:.2f} ms/image")

    # ── Save metrics JSON ─────────────────────────────────────
    metrics = {
        "architecture": architecture,
        "test_accuracy": acc,
        "ms_per_image": ms_per_img,
        "classification_report": report,
    }
    metrics_path = config.OUTPUTS_DIR / config.METRICS_FILE
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] Metrics saved → {metrics_path}")

    # ── Grad-CAM ──────────────────────────────────────────────
    if GRADCAM_AVAILABLE:
        generate_gradcam(
            model, test_loader, class_names, architecture, device,
            out_dir=config.OUTPUTS_DIR / "gradcam",
        )

    # ── Written analysis ──────────────────────────────────────
    _write_analysis(
        report, cm_arr, class_names,
        config.OUTPUTS_DIR / "analysis.md",
        arch=architecture, acc=acc, ms_per_img=ms_per_img,
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the surface defect classifier.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.MODELS_DIR / config.BEST_CHECKPOINT),
        help="Path to .pth checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
