"""APIRouter for DataVeil image detection, masking, and restoration."""

from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from .config import (
    DEFAULT_CONFIDENCE,
    MAX_IMAGE_BYTES,
    MAX_RECOVERY_BYTES,
)
from .schemas import (
    ImageDetectResponse,
    ImageMaskResponse,
    ManualRegion,
)
from .service import ImageMaskingService, ImageProcessingError


MODULE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(MODULE_DIR / "templates"))
router = APIRouter(tags=["ImageMasker"])


@lru_cache(maxsize=1)
def get_image_masking_service() -> ImageMaskingService:
    return ImageMaskingService()


async def _read_upload(
    upload: UploadFile,
    limit: int,
    *,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"The uploaded {label} is too large.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_string_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError
    return parsed


def _parse_manual_regions(value: str | None) -> list[ManualRegion]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError
    return [ManualRegion.model_validate(item) for item in parsed]


@router.get(
    "/imagemasker", response_class=HTMLResponse, name="imagemasker"
)
async def imagemasker_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@router.post("/api/image/detect", response_model=ImageDetectResponse)
async def detect_image(
    image: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    service: ImageMaskingService = Depends(get_image_masking_service),
) -> ImageDetectResponse:
    try:
        data = await _read_upload(image, MAX_IMAGE_BYTES, label="image")
        return await run_in_threadpool(
            service.detect_upload,
            data,
            filename=image.filename,
            content_type=image.content_type,
            confidence=confidence,
        )
    except ImageProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await image.close()


@router.post("/api/image/mask", response_model=ImageMaskResponse)
async def mask_image(
    image: UploadFile = File(...),
    selected_region_ids: str | None = Form(None),
    manual_regions: str | None = Form(None),
    effect: str = Form("blur"),
    strength: int = Form(65),
    color: str = Form("#101815"),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    service: ImageMaskingService = Depends(get_image_masking_service),
) -> ImageMaskResponse:
    try:
        data = await _read_upload(image, MAX_IMAGE_BYTES, label="image")
        selected = _parse_string_list(selected_region_ids)
        manual = _parse_manual_regions(manual_regions)
        return await run_in_threadpool(
            service.mask_upload,
            data,
            filename=image.filename,
            content_type=image.content_type,
            selected_region_ids=selected,
            manual_regions=manual,
            effect=effect,
            strength=strength,
            color=color,
            confidence=confidence,
        )
    except (ImageProcessingError, ValueError, json.JSONDecodeError) as exc:
        detail = (
            str(exc)
            if isinstance(exc, ImageProcessingError)
            else "The selected image regions are invalid."
        )
        raise HTTPException(status_code=400, detail=detail) from exc
    finally:
        await image.close()


@router.post("/api/image/restore")
async def restore_image(
    edited_image: UploadFile = File(...),
    recovery_file: UploadFile = File(...),
    service: ImageMaskingService = Depends(get_image_masking_service),
):
    try:
        edited_data = await _read_upload(
            edited_image, MAX_IMAGE_BYTES, label="edited image"
        )
        recovery_data = await _read_upload(
            recovery_file, MAX_RECOVERY_BYTES, label="recovery file"
        )
        restored, alignment = await run_in_threadpool(
            service.restore_upload,
            edited_data,
            edited_filename=edited_image.filename,
            edited_content_type=edited_image.content_type,
            recovery_data=recovery_data,
        )
        response = StreamingResponse(
            BytesIO(restored),
            media_type="image/png",
            headers={
                "Content-Disposition": (
                    'attachment; filename="dataveil-restored.png"'
                ),
                "X-DataVeil-Alignment": (
                    "applied" if alignment.get("applied") else "resize-only"
                ),
            },
        )
        return response
    except ImageProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await edited_image.close()
        await recovery_file.close()
