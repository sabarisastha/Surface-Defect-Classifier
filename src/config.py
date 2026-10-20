"""
config.py — Central configuration for the defect classifier project.

All hyperparameters, paths, and flags live here so every other module
imports from a single source of truth.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent   # project root
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"                         # place NEU-DET here
MODELS_DIR  = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ─────────────────────────────────────────────────────────────
# Dataset settings
# ─────────────────────────────────────────────────────────────
# Set USE_MNIST=True to use MNIST as a quick sanity-check fallback.
# When False the loader targets NEU Surface Defect data under data/raw/.
USE_MNIST: bool = False
FORCE_GRAYSCALE_INFERENCE: bool = True  # Convert all inference images to grayscale (L -> RGB) before classifying

NEU_CLASSES: list[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Stratified split ratios (must sum to 1.0)
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

RANDOM_SEED: int = 42

# ─────────────────────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE: int = 224          # final H×W fed to the network
NUM_CHANNELS: int = 3          # grayscale → RGB so pretrained weights work

# ImageNet statistics (used even for grayscale-converted images)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────
ARCHITECTURE: str = "resnet18"   # choices: resnet18 | resnet50 | mobilenet_v2

BATCH_SIZE: int  = 32
NUM_WORKERS: int = 0             # set > 0 if you have enough RAM / not on Windows with issues

# Phase-1: train only the new head
HEAD_EPOCHS: int   = 5
HEAD_LR:     float = 1e-3

# Phase-2: fine-tune entire network at a lower LR
FINETUNE_EPOCHS: int   = 25
FINETUNE_LR:     float = 1e-4

WEIGHT_DECAY: float  = 1e-4
DROPOUT_RATE: float  = 0.3

# Scheduler: "cosine" | "plateau"
SCHEDULER: str = "plateau"

# Early stopping
PATIENCE: int = 7           # epochs without improvement before stopping

# ─────────────────────────────────────────────────────────────
# Augmentation knobs (training only)
# ─────────────────────────────────────────────────────────────
AUG_ROTATION_DEG: int        = 10
AUG_BRIGHTNESS:   float      = 0.2
AUG_CONTRAST:     float      = 0.2

# ─────────────────────────────────────────────────────────────
# Flask / deployment
# ─────────────────────────────────────────────────────────────
FLASK_HOST:       str  = "0.0.0.0"
FLASK_PORT:       int  = 5000
FLASK_DEBUG:      bool = False

ALLOWED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_CONTENT_LENGTH: int       = 10 * 1024 * 1024   # 10 MB

# ─────────────────────────────────────────────────────────────
# Output artefact names
# ─────────────────────────────────────────────────────────────
BEST_CHECKPOINT:    str = "best_model.pth"
CLASS_NAMES_FILE:   str = "class_names.json"
METRICS_FILE:       str = "test_metrics.json"
CONFUSION_MATRIX:   str = "confusion_matrix.png"
TRAINING_CURVES:    str = "training_curves.png"
SAMPLE_GRID:        str = "sample_grid.png"
CLASS_DIST_PLOT:    str = "class_distribution.png"
GRADCAM_PREFIX:     str = "gradcam"
