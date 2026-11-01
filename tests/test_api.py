from __future__ import annotations

from fastapi.testclient import TestClient

from DocMasker.config import (
    MAX_OCCURRENCES_PER_PLACEHOLDER,
    MAX_TEXT_LENGTH,
)
from DocMasker.detector import DetectionCandidate
from DocMasker.router import get_masking_service
from DocMasker.service import MaskingService
from main import app


class ApiStubDetector:
    def detect(self, text: str) -> list[DetectionCandidate]:
        output: list[DetectionCandidate] = []
        for marker, entity_type in (
            ("Alice", "PERSON"),
            ("alice@example.com", "EMAIL"),
        ):
            start = text.find(marker)
            if start >= 0:
                output.append(
                    DetectionCandidate(
                        entity_type=entity_type,
                        start=start,
                        end=start + len(marker),
                        score=0.91,
                    )
                )
        return output


def make_client() -> TestClient:
    service = MaskingService(detector=ApiStubDetector())
    app.dependency_overrides[get_masking_service] = lambda: service
    return TestClient(app)


def clear_overrides() -> None:
    app.dependency_overrides.clear()


def test_landing_page_links_both_workspaces() -> None:
    with make_client() as client:
        response = client.get("/")
    clear_overrides()

    assert response.status_code == 200
    assert 'href="/docmasker"' in response.text
    assert 'href="/imagemasker"' in response.text
    assert response.headers["cache-control"] == "no-store"


def test_docmasker_page_contains_selective_interface() -> None:
    with make_client() as client:
        response = client.get("/docmasker")
    clear_overrides()

    assert response.status_code == 200
    assert "Scan private data" in response.text
    assert "Restore after AI" in response.text


def test_detect_endpoint_returns_stable_entity_ids() -> None:
    with make_client() as client:
        response = client.post(
            "/api/doc/detect",
            json={"text": "Alice emailed alice@example.com."},
        )
    clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["detection_count"] == 2
    assert [item["id"] for item in payload["detected_entities"]] == [
        "entity-0",
        "entity-1",
    ]


def test_mask_endpoint_returns_required_response_fields() -> None:
    text = "Alice is here."
    with make_client() as client:
        response = client.post("/api/doc/mask", json={"text": text})
    clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["original_text"] == text
    assert payload["masked_text"] == "<PERSON_1> is here."
    assert payload["detection_count"] == 1
    assert set(payload) == {
        "original_text",
        "masked_text",
        "detected_entities",
        "placeholder_mapping",
        "detection_count",
    }


def test_user_can_exclude_an_individual_detection() -> None:
    text = "Alice emailed alice@example.com."
    with make_client() as client:
        response = client.post(
            "/api/doc/mask",
            json={
                "text": text,
                "selected_entity_ids": ["entity-1"],
                "masking_type": "redact",
            },
        )
    clear_overrides()

    assert response.status_code == 200
    assert response.json()["masked_text"] == (
        "Alice emailed <REDACTED_EMAIL_1>."
    )


def test_empty_input_is_valid() -> None:
    with make_client() as client:
        response = client.post("/api/doc/mask", json={"text": ""})
    clear_overrides()

    assert response.status_code == 200
    assert response.json()["detection_count"] == 0


def test_restore_endpoint_uses_returned_mapping() -> None:
    with make_client() as client:
        masked_response = client.post(
            "/api/doc/mask", json={"text": "Alice is here."}
        ).json()
        restore_response = client.post(
            "/api/doc/restore",
            json={
                "masked_text": masked_response["masked_text"],
                "placeholder_mapping": masked_response[
                    "placeholder_mapping"
                ],
            },
        )
    clear_overrides()

    assert restore_response.status_code == 200
    assert restore_response.json()["restored_text"] == "Alice is here."


def test_oversized_input_returns_safe_error_without_text() -> None:
    private_value = "SENSITIVE_VALUE"
    oversized = private_value + ("x" * MAX_TEXT_LENGTH)
    with make_client() as client:
        response = client.post(
            "/api/doc/mask", json={"text": oversized}
        )
    clear_overrides()

    assert response.status_code == 422
    assert private_value not in response.text
    assert "invalid or exceeds" in response.json()["detail"]


def test_restore_rejects_unbounded_mapping_without_reflecting_values() -> None:
    private_value = "SENSITIVE_VALUE"
    with make_client() as client:
        response = client.post(
            "/api/doc/restore",
            json={
                "masked_text": "<PERSON_1>",
                "placeholder_mapping": {
                    "<PERSON_1>": {
                        "entity_type": "PERSON",
                        "original_values": [private_value]
                        * (MAX_OCCURRENCES_PER_PLACEHOLDER + 1),
                    }
                },
            },
        )
    clear_overrides()

    assert response.status_code == 422
    assert private_value not in response.text


def test_restore_rejects_output_amplification() -> None:
    with make_client() as client:
        response = client.post(
            "/api/doc/restore",
            json={
                "masked_text": "<PERSON_1>x",
                "placeholder_mapping": {
                    "<PERSON_1>": {
                        "entity_type": "PERSON",
                        "original_values": ["a" * MAX_TEXT_LENGTH],
                    }
                },
            },
        )
    clear_overrides()

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The recovery mapping cannot restore this text."
    )
