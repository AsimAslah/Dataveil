"""APIRouter for DataVeil text and document masking."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from .extraction import (
    MAX_DOCUMENT_BYTES,
    DocumentExtractionError,
    extract_document,
)
from .restoration import InvalidRestorationMapping
from .schemas import (
    DetectRequest,
    DetectResponse,
    DetectedEntity,
    DetectionPreview,
    MaskRequest,
    MaskResponse,
    PlaceholderMappingEntry,
    RestoreRequest,
    RestoreResponse,
)
from .service import InvalidEntitySelection, MaskingService


MODULE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(MODULE_DIR / "templates"))
router = APIRouter(tags=["DocMasker"])


@lru_cache(maxsize=1)
def get_masking_service() -> MaskingService:
    return MaskingService()


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail="The uploaded document is too large.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _detect_response(text: str, service: MaskingService) -> DetectResponse:
    candidates = service.detect(text)
    entities = [
        DetectionPreview(
            id=f"entity-{index}",
            entity_type=candidate.entity_type,
            original_value=text[candidate.start : candidate.end],
            start=candidate.start,
            end=candidate.end,
            confidence_score=round(candidate.score, 4),
        )
        for index, candidate in enumerate(candidates)
    ]
    return DetectResponse(
        text=text,
        detected_entities=entities,
        detection_count=len(entities),
    )


def _mask_response(payload: MaskRequest, service: MaskingService) -> MaskResponse:
    result = service.mask(
        payload.text,
        selected_entity_ids=payload.selected_entity_ids,
        masking_type=payload.masking_type,
    )
    return MaskResponse(
        original_text=result.original_text,
        masked_text=result.masked_text,
        detected_entities=[
            DetectedEntity(
                entity_type=entity.entity_type,
                original_value=entity.original_value,
                placeholder=entity.placeholder,
                start=entity.start,
                end=entity.end,
                confidence_score=entity.confidence_score,
            )
            for entity in result.detected_entities
        ],
        placeholder_mapping={
            replacement: PlaceholderMappingEntry.model_validate(entry)
            for replacement, entry in result.placeholder_mapping.items()
        },
        detection_count=result.detection_count,
    )


@router.get("/docmasker", response_class=HTMLResponse, name="docmasker")
async def docmasker_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"max_text_length": 100_000},
    )


@router.post("/api/doc/detect", response_model=DetectResponse)
def detect_text(
    payload: DetectRequest,
    service: MaskingService = Depends(get_masking_service),
) -> DetectResponse:
    return _detect_response(payload.text, service)


@router.post("/api/doc/detect-file", response_model=DetectResponse)
async def detect_document(
    document: UploadFile = File(...),
    service: MaskingService = Depends(get_masking_service),
) -> DetectResponse:
    try:
        data = await _read_upload(document, MAX_DOCUMENT_BYTES)
        text = await run_in_threadpool(
            extract_document, document.filename, data
        )
        return await run_in_threadpool(_detect_response, text, service)
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await document.close()


@router.post("/api/doc/mask", response_model=MaskResponse)
def mask_text(
    payload: MaskRequest,
    service: MaskingService = Depends(get_masking_service),
) -> MaskResponse:
    try:
        return _mask_response(payload, service)
    except InvalidEntitySelection as exc:
        raise HTTPException(
            status_code=400,
            detail="A selected text entity is no longer available.",
        ) from exc


@router.post("/api/doc/mask-file", response_model=MaskResponse)
async def mask_document(
    document: UploadFile = File(...),
    masking_type: str = Form("placeholder"),
    selected_entity_ids: str | None = Form(None),
    service: MaskingService = Depends(get_masking_service),
) -> MaskResponse:
    try:
        data = await _read_upload(document, MAX_DOCUMENT_BYTES)
        text = await run_in_threadpool(
            extract_document, document.filename, data
        )
        selected: list[str] | None = None
        if selected_entity_ids:
            parsed = json.loads(selected_entity_ids)
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) for item in parsed
            ):
                raise ValueError
            selected = parsed
        payload = MaskRequest(
            text=text,
            masking_type=masking_type,
            selected_entity_ids=selected,
        )
        return await run_in_threadpool(_mask_response, payload, service)
    except (DocumentExtractionError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The document or masking selections are invalid.",
        ) from exc
    finally:
        await document.close()


@router.post("/api/doc/restore", response_model=RestoreResponse)
def restore_masked_text(
    payload: RestoreRequest,
    service: MaskingService = Depends(get_masking_service),
) -> RestoreResponse:
    try:
        restored = service.restore(
            payload.masked_text, payload.placeholder_mapping
        )
    except InvalidRestorationMapping as exc:
        raise HTTPException(
            status_code=400,
            detail="The recovery mapping cannot restore this text.",
        ) from exc
    return RestoreResponse(restored_text=restored)
