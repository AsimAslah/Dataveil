from __future__ import annotations

from io import BytesIO

import fitz
from docx import Document
from fastapi.testclient import TestClient
from pptx import Presentation

from DocMasker.detector import DetectionCandidate
from DocMasker.extraction import extract_document
from DocMasker.router import get_masking_service
from DocMasker.service import MaskingService
from main import app


class ApiStubDetector:
    def detect(self, text: str) -> list[DetectionCandidate]:
        start = text.find("Alice")
        if start < 0:
            return []
        return [
            DetectionCandidate(
                entity_type="PERSON",
                start=start,
                end=start + len("Alice"),
                score=0.91,
            )
        ]


def pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def pptx_bytes(text: str) -> bytes:
    presentation = Presentation()
    slide = presentation.slides.add_slide(
        presentation.slide_layouts[5]
    )
    box = slide.shapes.add_textbox(100, 100, 400, 100)
    box.text = text
    output = BytesIO()
    presentation.save(output)
    return output.getvalue()


def test_supported_documents_extract_in_memory() -> None:
    assert "Alice" in extract_document(
        "private.pdf", pdf_bytes("Alice")
    )
    assert extract_document("private.docx", docx_bytes("Alice")) == "Alice"
    assert "Alice" in extract_document(
        "private.pptx", pptx_bytes("Alice")
    )


def test_document_api_accepts_valid_pdf() -> None:
    service = MaskingService(detector=ApiStubDetector())
    app.dependency_overrides[get_masking_service] = lambda: service
    with TestClient(app) as client:
        response = client.post(
            "/api/doc/detect-file",
            files={
                "document": (
                    "private.pdf",
                    pdf_bytes("Alice"),
                    "application/pdf",
                )
            },
        )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["detection_count"] == 1
