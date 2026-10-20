"""
app.py — Flask web application for the surface defect classifier.

Endpoints:
  GET  /          → Upload page (HTML)
  POST /predict   → Returns JSON with predicted class and top-3 probabilities
  GET  /health    → Health check

The model is loaded once at startup.

Run:
    python app/app.py
"""

import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

# ── Add src/ to path so we can import from it ────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import config
from predict import load_model, predict as run_predict


# ─────────────────────────────────────────────────────────────
# Flask setup
# ─────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

UPLOAD_FOLDER = APP_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────

try:
    MODEL, CLASS_NAMES, DEVICE = load_model()
    print(f"[app] Model loaded — {len(CLASS_NAMES)} classes on {DEVICE}")
except FileNotFoundError as e:
    MODEL      = None
    CLASS_NAMES = config.NEU_CLASSES
    DEVICE     = None
    print(f"[app] WARNING: {e}")
    print("[app] /predict will return an error until the model is trained.")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    """Return True if the file extension is in ALLOWED_EXTENSIONS."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _save_upload(file_obj) -> Tuple[Path, str]:
    """
    Safely save an uploaded file and return (saved_path, safe_filename).

    - Uses secure_filename to strip any path traversal.
    - Prepends a UUID to avoid collisions and never expose the original name.
    """
    original_name = secure_filename(file_obj.filename or "upload.jpg")
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "jpg"
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_FOLDER / safe_name
    file_obj.save(save_path)
    return save_path, safe_name


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
    })


@app.get("/")
def index():
    """Render the upload page."""
    return render_template("index.html", class_names=CLASS_NAMES)


@app.post("/predict")
def predict_endpoint():
    """
    Accept an image upload and return the prediction as JSON.

    Expected request: multipart/form-data with a 'file' field.

    Returns JSON:
      {
        "predicted_class": "scratches",
        "confidence": 0.92,
        "top_k": [
          {"class": "scratches",        "probability": 0.92},
          {"class": "rolled-in_scale",  "probability": 0.05},
          {"class": "crazing",          "probability": 0.02}
        ],
        "image_url": "/static/uploads/<uuid>.jpg"
      }
    """
    # ── Validate model availability ────────────────────────────
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded. Please train the model first: python src/train.py"
        }), 503

    # ── Validate file presence ─────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "No file selected."}), 400

    # ── Validate extension ─────────────────────────────────────
    if not _allowed_file(file.filename):
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return jsonify({
            "error": f"File type not allowed. Accepted: {allowed}."
        }), 400

    # ── Save file ──────────────────────────────────────────────
    save_path, safe_name = _save_upload(file)

    # ── Validate image integrity ───────────────────────────────
    try:
        img = Image.open(save_path)
        img.verify()                # check for corruption
        img = Image.open(save_path) # reopen after verify
    except (UnidentifiedImageError, OSError) as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": f"Corrupt or unreadable image: {exc}"}), 400

    # ── Run inference ──────────────────────────────────────────
    try:
        result = run_predict(img, MODEL, CLASS_NAMES, DEVICE, top_k=3)
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    result["image_url"] = f"/static/uploads/{safe_name}"
    return jsonify(result)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
