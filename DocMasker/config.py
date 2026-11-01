"""Central configuration for the local-only DocMasker application."""

from __future__ import annotations

APP_NAME = "DataVeil DocMasker"
APP_VERSION = "0.1.0"
LANGUAGE = "en"
SPACY_MODEL = "en_core_web_sm"

# Privacy-sensitive requests are deliberately bounded. The API also rejects a
# clearly oversized Content-Length before parsing JSON.
MAX_TEXT_LENGTH = 100_000
MAX_REQUEST_BYTES = 400_000
MAX_RESTORATION_ENTRIES = 250
MAX_OCCURRENCES_PER_PLACEHOLDER = 250
MAX_PLACEHOLDER_LENGTH = 128
MAX_ENTITY_TYPE_LENGTH = 64

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DETECTION_SCORE_THRESHOLD = 0.35
