"""
dataset.py — Data loading, augmentation, and stratified splitting.

Supports two NEU-DET layouts:
  1. Class-per-folder  →  data/raw/<class>/<image>.jpg
  2. Filename-prefix   →  data/raw/<class>_<idx>.jpg  (flat directory)

Set USE_MNIST=True in config.py to use MNIST as a fallback.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────
# 1. Transforms
# ─────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return torchvision transforms for the given split.

    Training augmentation: random flip, rotation, colour jitter.
    Val/Test: deterministic resize + normalize only.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        A Compose transform pipeline.
    """
    resize_norm = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(config.AUG_ROTATION_DEG),
            transforms.ColorJitter(
                brightness=config.AUG_BRIGHTNESS,
                contrast=config.AUG_CONTRAST,
            ),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    else:
        return transforms.Compose(resize_norm)


# ─────────────────────────────────────────────────────────────
# 2. NEU-DET custom Dataset
# ─────────────────────────────────────────────────────────────

class NEUDefectDataset(Dataset):
    """
    Dataset for the NEU Surface Defect Database.

    Supports two disk layouts automatically:
      • Folder layout: data/raw/<class_name>/<img>.jpg
      • Flat prefix  : data/raw/<class_name>_<idx>.jpg

    Args:
        root:       Path to the raw data directory.
        class_names: Ordered list of class labels.
        indices:    Optional subset of global indices (used by the splitter).
        transform:  torchvision transform to apply.
    """

    def __init__(
        self,
        root: Path,
        class_names: List[str],
        indices: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.root        = Path(root)
        self.class_names = class_names
        self.transform   = transform
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(class_names)}

        self._samples: List[Tuple[Path, int]] = []   # (path, label)
        self._discover_samples()

        if indices is not None:
            self._samples = [self._samples[i] for i in indices]

    # ------------------------------------------------------------------
    def _discover_samples(self) -> None:
        """Auto-detect layout and populate self._samples."""
        # ── Layout 1: class sub-directories ───────────────────────────
        subdir_hits = sum(
            1 for c in self.class_names if (self.root / c).is_dir()
        )
        if subdir_hits > 0:
            for cls in self.class_names:
                cls_dir = self.root / cls
                if not cls_dir.is_dir():
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                    for img_path in sorted(cls_dir.glob(ext)):
                        self._samples.append((img_path, self.class_to_idx[cls]))
            return

        # ── Layout 2: flat directory with class-prefix filenames ───────
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for img_path in sorted(self.root.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            stem = img_path.stem.lower()
            matched = False
            for cls in sorted(self.class_names, key=len, reverse=True):
                if stem.startswith(cls.lower()):
                    self._samples.append((img_path, self.class_to_idx[cls]))
                    matched = True
                    break
            if not matched:
                # Try matching with space/underscore variants
                for cls in sorted(self.class_names, key=len, reverse=True):
                    normalised = cls.replace("-", "_").replace(" ", "_").lower()
                    if stem.startswith(normalised):
                        self._samples.append((img_path, self.class_to_idx[cls]))
                        matched = True
                        break

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self._samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def targets(self) -> List[int]:
        """Return all labels (needed by the stratified splitter)."""
        return [s[1] for s in self._samples]


# ─────────────────────────────────────────────────────────────
# 3. Splitter
# ─────────────────────────────────────────────────────────────

def stratified_split(
    dataset: NEUDefectDataset,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    seed:        int   = config.RANDOM_SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (train_indices, val_indices, test_indices) via stratified split.

    The test set is separated first so it is never touched during training.

    Args:
        dataset:     The full NEUDefectDataset (no transform applied yet).
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation.
        seed:        Random seed for reproducibility.

    Returns:
        Three lists of integer indices.
    """
    labels = dataset.targets
    all_idx = list(range(len(labels)))

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: separate test
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_ratio,
        stratify=[labels[i] for i in all_idx],
        random_state=seed,
    )

    # Second split: separate val from train+val
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=[labels[i] for i in train_val_idx],
        random_state=seed,
    )

    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────
# 4. Public factory
# ─────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build and return (train_loader, val_loader, test_loader, class_names).

    Respects config.USE_MNIST for quick testing without the real dataset.

    Args:
        data_root:   Override for the raw data directory.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        seed:        Random seed.

    Returns:
        Tuple of three DataLoaders and the list of class names.
    """
    if config.USE_MNIST:
        return _get_mnist_loaders(batch_size, num_workers)

    root = data_root or config.RAW_DIR

    # ── Discover all samples without any transform ─────────────────────
    full_ds = NEUDefectDataset(
        root=root,
        class_names=config.NEU_CLASSES,
        transform=None,
    )

    if len(full_ds) == 0:
        raise RuntimeError(
            f"No images found under '{root}'.\n"
            "Please see README.md → 'Dataset Setup' for download instructions."
        )

    print(f"[dataset] Found {len(full_ds)} images across {len(config.NEU_CLASSES)} classes.")

    train_idx, val_idx, test_idx = stratified_split(full_ds, seed=seed)
    print(f"[dataset] Split → train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Wrap each split with its own transform ─────────────────────────
    train_ds = NEUDefectDataset(root, config.NEU_CLASSES, train_idx, get_transforms("train"))
    val_ds   = NEUDefectDataset(root, config.NEU_CLASSES, val_idx,   get_transforms("val"))
    test_ds  = NEUDefectDataset(root, config.NEU_CLASSES, test_idx,  get_transforms("test"))

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader, config.NEU_CLASSES


# ─────────────────────────────────────────────────────────────
# 5. MNIST fallback
# ─────────────────────────────────────────────────────────────

def _get_mnist_loaders(
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Return MNIST loaders that mimic the NEU pipeline (same shapes)."""
    mnist_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    mnist_dir = config.DATA_DIR / "mnist"
    train_ds = datasets.MNIST(mnist_dir, train=True,  download=True, transform=mnist_transform)
    test_ds  = datasets.MNIST(mnist_dir, train=False, download=True, transform=mnist_transform)

    # carve out val from train
    n_val = int(0.15 * len(train_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    class_names = [str(i) for i in range(10)]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────────────────────
# 6. EDA utilities  (save plots to outputs/)
# ─────────────────────────────────────────────────────────────

def save_sample_grid(
    dataset: NEUDefectDataset,
    class_names: List[str],
    out_path: Path,
    n_per_class: int = 5,
) -> None:
    """
    Save a grid of n_per_class images per class to out_path.

    Args:
        dataset:     A NEUDefectDataset (transforms already applied).
        class_names: Ordered list of class labels.
        out_path:    File path for the saved PNG.
        n_per_class: Number of samples to show per class.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2, n_classes * 2))

    # Bucket indices by class
    buckets: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(dataset)):
        _, lbl = dataset._samples[idx]
        buckets[lbl].append(idx)

    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

    for row, cls_idx in enumerate(range(n_classes)):
        samples = random.sample(buckets[cls_idx], min(n_per_class, len(buckets[cls_idx])))
        for col, sample_idx in enumerate(samples):
            img_tensor, _ = dataset[sample_idx]
            img_np = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[row][col] if n_classes > 1 else axes[col]
            ax.imshow(img_np)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_names[cls_idx], fontsize=8, rotation=45, ha="right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[dataset] Sample grid saved → {out_path}")


def save_class_distribution(
    targets: List[int],
    class_names: List[str],
    out_path: Path,
    title: str = "Class Distribution",
) -> None:
    """
    Bar chart of per-class sample counts.

    Args:
        targets:     Integer labels for the full dataset.
        class_names: Ordered list of class labels.
        out_path:    File path for the saved PNG.
        title:       Chart title.
    """
    counts = [targets.count(i) for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, counts, color="steelblue", edgecolor="white")
    ax.bar_label(bars)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[dataset] Class distribution saved → {out_path}")
