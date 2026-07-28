"""Application service coordinating detection, masking, and restoration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .detector import EntityDetector, PresidioDetector
from .masking import (
    MaskingResult,
    mask_detected_text,
    resolve_overlaps,
)
from .restoration import restore_text


class InvalidEntitySelection(ValueError):
    """Raised when a client submits stale or unknown detection identifiers."""


class MaskingService:
    def __init__(self, detector: EntityDetector | None = None) -> None:
        self._detector = detector or PresidioDetector()

    def detect(self, text: str):
        return resolve_overlaps(text, self._detector.detect(text))

    def mask(
        self,
        text: str,
        *,
        selected_entity_ids: list[str] | None = None,
        masking_type: str = "placeholder",
    ) -> MaskingResult:
        candidates = self.detect(text)
        if selected_entity_ids is not None:
            selected = set(selected_entity_ids)
            available = {
                f"entity-{index}" for index in range(len(candidates))
            }
            if not selected <= available:
                raise InvalidEntitySelection(
                    "A selected entity is no longer available."
                )
            candidates = [
                candidate
                for index, candidate in enumerate(candidates)
                if f"entity-{index}" in selected
            ]
        return mask_detected_text(
            text,
            candidates,
            replacement_style=masking_type,
        )

    def restore(
        self, masked_text: str, placeholder_mapping: Mapping[str, Any]
    ) -> str:
        return restore_text(masked_text, placeholder_mapping)
