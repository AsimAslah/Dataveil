"""Validated response and recovery models for ImageMasker."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ImageRegion(StrictModel):
    id: str
    label: str
    person_label: str | None
    kind: Literal["face", "body"]
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    mask_type: Literal["instance", "semantic", "fallback"] | None = None
    mask_preview_png_base64: str | None = None


class ImageDetectResponse(StrictModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    person_count: int = Field(ge=0)
    regions: list[ImageRegion]


class ManualRegion(StrictModel):
    id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ImageRecoveryPackage(StrictModel):
    format: Literal["dataveil-image-recovery"]
    version: Literal[1]
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    mask_png_base64: str
    reference_png_base64: str
    protected_pixels_png_base64: str
    selected_region_ids: list[str]


class ImageMaskResponse(StrictModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    media_type: Literal["image/png"]
    image_base64: str
    effect: str
    strength: int = Field(ge=1, le=100)
    selected_region_ids: list[str]
    recovery: ImageRecoveryPackage
