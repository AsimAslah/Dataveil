"""Validated API request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import MAX_TEXT_LENGTH


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MaskRequest(StrictModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    selected_entity_ids: list[str] | None = None
    masking_type: Literal[
        "placeholder",
        "redact",
        "pseudonym",
        "generalize",
        "remove",
    ] = "placeholder"


class DetectRequest(StrictModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)


class DetectionPreview(StrictModel):
    id: str
    entity_type: str
    original_value: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class DetectResponse(StrictModel):
    text: str
    detected_entities: list[DetectionPreview]
    detection_count: int = Field(ge=0)


class DetectedEntity(StrictModel):
    entity_type: str
    original_value: str
    placeholder: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class PlaceholderMappingEntry(StrictModel):
    entity_type: str
    # Values are stored per occurrence so restoration can preserve differences
    # such as "Alice" versus "ALICE" while using one consistent placeholder.
    original_values: list[str]


class MaskResponse(StrictModel):
    original_text: str
    masked_text: str
    detected_entities: list[DetectedEntity]
    placeholder_mapping: dict[str, PlaceholderMappingEntry]
    detection_count: int = Field(ge=0)


class RestoreRequest(StrictModel):
    masked_text: str = Field(max_length=MAX_TEXT_LENGTH)
    placeholder_mapping: dict[str, PlaceholderMappingEntry]


class RestoreResponse(StrictModel):
    restored_text: str
