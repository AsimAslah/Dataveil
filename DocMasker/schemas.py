"""Validated API request and response models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .config import (
    MAX_ENTITY_TYPE_LENGTH,
    MAX_OCCURRENCES_PER_PLACEHOLDER,
    MAX_PLACEHOLDER_LENGTH,
    MAX_RESTORATION_ENTRIES,
    MAX_TEXT_LENGTH,
)


Placeholder = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_PLACEHOLDER_LENGTH)
]
EntityType = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_ENTITY_TYPE_LENGTH)
]
OriginalValue = Annotated[str, StringConstraints(max_length=MAX_TEXT_LENGTH)]


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
    entity_type: EntityType
    # Values are stored per occurrence so restoration can preserve differences
    # such as "Alice" versus "ALICE" while using one consistent placeholder.
    original_values: list[OriginalValue] = Field(
        max_length=MAX_OCCURRENCES_PER_PLACEHOLDER
    )


class MaskResponse(StrictModel):
    original_text: str
    masked_text: str
    detected_entities: list[DetectedEntity]
    placeholder_mapping: dict[str, PlaceholderMappingEntry]
    detection_count: int = Field(ge=0)


class RestoreRequest(StrictModel):
    masked_text: str = Field(max_length=MAX_TEXT_LENGTH)
    placeholder_mapping: dict[Placeholder, PlaceholderMappingEntry] = Field(
        max_length=MAX_RESTORATION_ENTRIES
    )

    @model_validator(mode="after")
    def bound_original_values(self):
        total = sum(
            len(value)
            for entry in self.placeholder_mapping.values()
            for value in entry.original_values
        )
        if total > MAX_TEXT_LENGTH:
            raise ValueError("The recovery mapping exceeds the allowed size.")
        return self


class RestoreResponse(StrictModel):
    restored_text: str
