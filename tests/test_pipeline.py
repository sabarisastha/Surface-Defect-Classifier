"""
tests/test_pipeline.py — Pytest suite covering preprocessing, model, and Flask endpoint.

Run with:
    pytest tests/ -v
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ── Resolve imports ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
APP  = ROOT / "app"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(APP))

import config
from dataset import get_transforms, NEUDefectDataset
from model import build_model, count_parameters, freeze_backbone, unfreeze_all


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_pil_image() -> Image.Image:
    """Return a 200×200 grayscale PIL image (mimics NEU-DET)."""
    arr = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture(scope="module")
def dummy_rgb_pil_image() -> Image.Image:
    """Return a 224×224 RGB PIL image."""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="module")
def resnet18_model() -> torch.nn.Module:
    """Return an untrained ResNet18 with the correct head."""
    return build_model(
        architecture="resnet18",
        num_classes=len(config.NEU_CLASSES),
        pretrained=False,   # avoid downloading weights in CI
    )


# ─────────────────────────────────────────────────────────────
# 1. Preprocessing / transforms
# ─────────────────────────────────────────────────────────────

class TestTransforms:
    def test_train_output_shape(self, dummy_pil_image: Image.Image) -> None:
        """Training transform should output [3, 224, 224] tensor."""
        transform = get_transforms("train")
        tensor = transform(dummy_pil_image)
        assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE), (
            f"Expected (3,{config.IMAGE_SIZE},{config.IMAGE_SIZE}), got {tensor.shape}"
        )

    def test_val_output_shape(self, dummy_pil_image: Image.Image) -> None:
        """Val/test transform should output [3, 224, 224] tensor."""
        transform = get_transforms("val")
        tensor = transform(dummy_pil_image)
        assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def test_test_output_shape(self, dummy_pil_image: Image.Image) -> None:
        """Test transform should output [3, 224, 224] tensor."""
        transform = get_transforms("test")
        tensor = transform(dummy_pil_image)
        assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def test_normalisation_range(self, dummy_pil_image: Image.Image) -> None:
        """
        After ImageNet normalisation the tensor values can exceed [-1, 1]
        slightly, but must be finite.
        """
        transform = get_transforms("val")
        tensor = transform(dummy_pil_image)
        assert torch.isfinite(tensor).all(), "Tensor contains NaN or Inf after normalisation."

    def test_grayscale_to_3channel(self, dummy_pil_image: Image.Image) -> None:
        """Grayscale input must be converted to 3 channels."""
        transform = get_transforms("val")
        tensor = transform(dummy_pil_image)
        assert tensor.shape[0] == 3, "Channel count must be 3 after Grayscale(num_output_channels=3)."

    def test_rgb_input_unchanged_channels(self, dummy_rgb_pil_image: Image.Image) -> None:
        """RGB input must still produce a 3-channel output."""
        transform = get_transforms("val")
        tensor = transform(dummy_rgb_pil_image)
        assert tensor.shape[0] == 3

    def test_grayscale_conversion(self) -> None:
        """Test the to_grayscale_rgb helper in predict.py."""
        from predict import to_grayscale_rgb, _inference_transform
        
        # 1. Synthetic color image
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        color_img = Image.fromarray(arr, mode="RGB")
        
        out_img = to_grayscale_rgb(color_img)
        assert out_img.mode == "RGB"
        
        # All three channels should be identical
        r, g, b = out_img.split()
        assert list(r.getdata()) == list(g.getdata()) == list(b.getdata())
        
        # Final tensor shape
        tensor = _inference_transform(out_img).unsqueeze(0)
        assert tensor.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
        # 2. RGBA image with transparency
        rgba_arr = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
        out_rgba = to_grayscale_rgb(rgba_img)
        assert out_rgba.mode == "RGB"
        
        # 3. Palette PNG with transparency
        p_img = Image.new("P", (50, 50))
        p_img.info["transparency"] = 0
        out_p = to_grayscale_rgb(p_img)
        assert out_p.mode == "RGB"


# ─────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────

class TestModel:
    def test_forward_pass_shape(self, resnet18_model: torch.nn.Module) -> None:
        """Model forward pass must output [batch, num_classes] logits."""
        resnet18_model.eval()
        dummy_batch = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            logits = resnet18_model(dummy_batch)
        assert logits.shape == (2, len(config.NEU_CLASSES)), (
            f"Expected (2,{len(config.NEU_CLASSES)}), got {logits.shape}"
        )

    def test_freeze_backbone(self, resnet18_model: torch.nn.Module) -> None:
        """After freeze_backbone, only head params should require grad."""
        freeze_backbone(resnet18_model, "resnet18")
        head_params = set(id(p) for p in resnet18_model.fc.parameters())
        for name, param in resnet18_model.named_parameters():
            if id(param) in head_params:
                assert param.requires_grad, f"Head param '{name}' should be trainable."
            else:
                assert not param.requires_grad, f"Backbone param '{name}' should be frozen."

    def test_unfreeze_all(self, resnet18_model: torch.nn.Module) -> None:
        """After unfreeze_all, every parameter should require grad."""
        freeze_backbone(resnet18_model, "resnet18")
        unfreeze_all(resnet18_model)
        for name, param in resnet18_model.named_parameters():
            assert param.requires_grad, f"Param '{name}' should be unfrozen."

    def test_parameter_counts(self, resnet18_model: torch.nn.Module) -> None:
        """Parameter count utility must return positive integers."""
        unfreeze_all(resnet18_model)
        info = count_parameters(resnet18_model)
        assert info["total"] > 0
        assert info["trainable"] > 0
        assert info["trainable"] <= info["total"]

    def test_mobilenet_forward(self) -> None:
        """MobileNetV2 forward pass must also produce correct output shape."""
        model = build_model("mobilenet_v2", num_classes=6, pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 6)

    def test_resnet50_forward(self) -> None:
        """ResNet50 forward pass test."""
        model = build_model("resnet50", num_classes=6, pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 6)

    def test_invalid_architecture(self) -> None:
        """Building with an unknown architecture must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("vgg99", num_classes=6, pretrained=False)


# ─────────────────────────────────────────────────────────────
# 3. Flask /predict endpoint
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flask_client():
    """Return a Flask test client."""
    from app import app  # noqa: F401 — imports the Flask app
    app.config["TESTING"] = True
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    with app.test_client() as client:
        yield client


class TestFlaskEndpoints:
    def test_health_endpoint(self, flask_client) -> None:
        """GET /health should return 200 with status 'ok'."""
        resp = flask_client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_index_page(self, flask_client) -> None:
        """GET / should return 200 with HTML content."""
        resp = flask_client.get("/")
        assert resp.status_code == 200
        assert b"Defect" in resp.data or b"defect" in resp.data.lower()

    def test_predict_no_file(self, flask_client) -> None:
        """POST /predict with no file should return 400."""
        resp = flask_client.post("/predict")
        assert resp.status_code in (400, 503)

    def test_predict_bad_extension(self, flask_client) -> None:
        """POST /predict with a .exe file should be rejected."""
        data = {"file": (io.BytesIO(b"not an image"), "malware.exe")}
        resp = flask_client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code in (400, 503)

    def test_predict_corrupt_image(self, flask_client) -> None:
        """POST /predict with corrupt bytes labelled as .jpg should return 400 or 503."""
        data = {"file": (io.BytesIO(b"\xff\xd8\xff corrupt data"), "bad.jpg")}
        resp = flask_client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        # 503 if model not loaded, 400 if image is corrupt and model is loaded
        assert resp.status_code in (400, 503)

    def test_predict_valid_image(self, flask_client, dummy_rgb_pil_image: Image.Image) -> None:
        """
        POST /predict with a valid PNG should return either:
          - 200 with predicted_class + top_k  (model loaded)
          - 503 with an error message          (model not yet trained)
        """
        buf = io.BytesIO()
        dummy_rgb_pil_image.save(buf, format="PNG")
        buf.seek(0)

        data = {"file": (buf, "test_image.png")}
        resp = flask_client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code in (200, 503)

        if resp.status_code == 200:
            body = resp.get_json()
            assert "predicted_class" in body
            assert "confidence" in body
            assert "top_k" in body
            assert isinstance(body["top_k"], list)
            assert len(body["top_k"]) == 3
            # Probabilities must sum to ≈ 1
            total = sum(item["probability"] for item in body["top_k"])
            assert total <= 1.0 + 1e-3  # top-3 can't exceed total
