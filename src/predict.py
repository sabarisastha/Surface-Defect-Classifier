"""
predict.py — Single-image inference used by both Flask and CLI.

Loads the saved checkpoint once (call load_model()) and provides
a predict() function that accepts a PIL Image or file path.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import base64
import io
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from model import load_checkpoint


# ─────────────────────────────────────────────────────────────
# Inference transform (identical to val/test)
# ─────────────────────────────────────────────────────────────

_inference_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────
# Image preprocessing helpers
# ─────────────────────────────────────────────────────────────

def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """
    Converts an image to grayscale and back to 3-channel RGB.
    Handles EXIF orientation, 16-bit/TIFF modes, and transparency correctly.
    """
    # Fix EXIF orientation
    img = ImageOps.exif_transpose(img)
    
    # Handle transparency by compositing onto white
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=alpha)
        img = bg.convert('RGB')
    else:
        # Avoid crashing on 16-bit or TIFF by forcing to standard RGB
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
            
    # Convert to grayscale (L) then back to 3-channel RGB
    return img.convert('L').convert('RGB')

# ─────────────────────────────────────────────────────────────
# Loader (call once at startup)
# ─────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, List[str], torch.device]:
    """
    Load the best checkpoint and return (model, class_names, device).

    Intended to be called once at application startup.

    Args:
        checkpoint_path: Path to the .pth file. Defaults to config default.
        device:          Override device. Defaults to CUDA if available.

    Returns:
        (model, class_names, device)

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = config.MODELS_DIR / config.BEST_CHECKPOINT

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'.\n"
            "Run 'python src/train.py' first to train and save the model."
        )

    model, class_names, _ = load_checkpoint(checkpoint_path, device)
    model.eval()
    return model, class_names, device


# ─────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────

def predict(
    image_input: Union[str, Path, Image.Image],
    model: nn.Module,
    class_names: List[str],
    device: torch.device,
    top_k: int = 3,
) -> Dict:
    """
    Run inference on a single image.

    Args:
        image_input:  A file path (str/Path) or an already-opened PIL Image.
        model:        The loaded model (must be in eval mode).
        class_names:  Ordered list of class names matching the model output.
        device:       Device to run inference on.
        top_k:        Number of top predictions to return.

    Returns:
        A dict with keys:
          - 'predicted_class': str
          - 'confidence': float  (0‒1)
          - 'top_k': list of {class, probability} dicts, sorted descending

    Raises:
        ValueError: If the image cannot be opened or is corrupt.
    """
    # ── Open image ────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        try:
            img = Image.open(image_input)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Cannot open image '{image_input}': {exc}") from exc
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # ── Preprocess ────────────────────────────────────────────
    if getattr(config, "FORCE_GRAYSCALE_INFERENCE", False):
        img = to_grayscale_rgb(img)
    else:
        img = ImageOps.exif_transpose(img).convert("RGB")
        
    # Generate base64 preview of exactly what was fed into the transform
    buffered = io.BytesIO()
    img.copy().thumbnail((224, 224)) # resize for preview efficiency
    img.save(buffered, format="JPEG")
    preview_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # ── Transform & forward ───────────────────────────────────
    tensor = _inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

    # ── Build result ──────────────────────────────────────────
    ranked = sorted(
        zip(class_names, probs), key=lambda x: x[1], reverse=True
    )

    top_k_results = [
        {"class": cls, "probability": round(prob, 5)}
        for cls, prob in ranked[:top_k]
    ]

    return {
        "predicted_class": ranked[0][0],
        "confidence":      round(ranked[0][1], 5),
        "top_k":           top_k_results,
        "processed_preview": f"data:image/jpeg;base64,{preview_b64}"
    }


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict defect class for a single image.")
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.MODELS_DIR / config.BEST_CHECKPOINT))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model, class_names, device = load_model(Path(args.checkpoint))
    result = predict(args.image, model, class_names, device, top_k=args.top_k)

    print(json.dumps(result, indent=2))
