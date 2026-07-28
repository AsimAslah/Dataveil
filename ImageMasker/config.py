"""Configuration for the in-memory ImageMasker service."""

from __future__ import annotations

from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = MODULE_DIR / "yolov8n-seg.pt"

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_RECOVERY_BYTES = 48 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
DEFAULT_CONFIDENCE = 0.35
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.9

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}
ALLOWED_EFFECTS = {
    "blur",
    "pixelate",
    "mosaic",
    "black",
    "solid",
    "noise",
    "sticker",
    "silhouette",
}
