"""Bounded, in-memory extraction for supported document uploads."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .config import MAX_TEXT_LENGTH


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
MAX_DOCUMENT_BYTES = 15 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 5_000
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 100 * 1024 * 1024


class DocumentExtractionError(ValueError):
    """Raised when a document cannot be safely extracted."""


def _validate_office_archive(data: bytes) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise DocumentExtractionError(
                    "The document archive contains too many files."
                )
            expanded_size = sum(entry.file_size for entry in entries)
            if expanded_size > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
                raise DocumentExtractionError(
                    "The expanded document is too large."
                )
    except BadZipFile as exc:
        raise DocumentExtractionError(
            "The uploaded Office document is invalid."
        ) from exc


def _extract_pdf(data: bytes) -> str:
    if not data.startswith(b"%PDF-"):
        raise DocumentExtractionError("The uploaded PDF is invalid.")

    import fitz

    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
    except Exception as exc:
        raise DocumentExtractionError(
            "The uploaded PDF could not be read."
        ) from exc
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _extract_docx(data: bytes) -> str:
    _validate_office_archive(data)
    from docx import Document

    try:
        document = Document(BytesIO(data))
        blocks = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        for table in document.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                if any(values):
                    blocks.append(" | ".join(values))
    except Exception as exc:
        raise DocumentExtractionError(
            "The uploaded DOCX document could not be read."
        ) from exc
    return "\n".join(blocks)


def _extract_pptx(data: bytes) -> str:
    _validate_office_archive(data)
    from pptx import Presentation

    try:
        presentation = Presentation(BytesIO(data))
        blocks: list[str] = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            slide_blocks = [
                shape.text.strip()
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text.strip()
            ]
            if slide_blocks:
                blocks.append(
                    f"Slide {slide_index}\n" + "\n".join(slide_blocks)
                )
    except Exception as exc:
        raise DocumentExtractionError(
            "The uploaded PPTX presentation could not be read."
        ) from exc
    return "\n\n".join(blocks)


def extract_document(filename: str | None, data: bytes) -> str:
    if not filename:
        raise DocumentExtractionError("A document filename is required.")
    if not data:
        raise DocumentExtractionError("The uploaded document is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentExtractionError("The uploaded document is too large.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentExtractionError(
            "Supported document types are PDF, DOCX, and PPTX."
        )

    if extension == ".pdf":
        text = _extract_pdf(data)
    elif extension == ".docx":
        text = _extract_docx(data)
    else:
        text = _extract_pptx(data)

    text = text.strip()
    if not text:
        raise DocumentExtractionError(
            "No extractable text was found in this document."
        )
    if len(text) > MAX_TEXT_LENGTH:
        raise DocumentExtractionError(
            "The extracted text exceeds the allowed size."
        )
    return text
