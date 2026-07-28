from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from ImageMasker.router import get_image_masking_service
from ImageMasker.schemas import (
    BoundingBox,
    ImageDetectResponse,
    ImageMaskResponse,
    ImageRecoveryPackage,
    ImageRegion,
)
from main import app


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class StubImageService:
    def detect_upload(self, *_args, **_kwargs) -> ImageDetectResponse:
        return ImageDetectResponse(
            width=1,
            height=1,
            person_count=1,
            regions=[
                ImageRegion(
                    id="person:p0:body",
                    label="P0 person — body",
                    person_label="p0",
                    kind="body",
                    confidence=0.9,
                    bbox=BoundingBox(x=0, y=0, width=1, height=1),
                )
            ],
        )

    def mask_upload(self, *_args, **_kwargs) -> ImageMaskResponse:
        encoded = base64.b64encode(VALID_PNG).decode("ascii")
        recovery = ImageRecoveryPackage(
            format="dataveil-image-recovery",
            version=1,
            width=1,
            height=1,
            mask_png_base64=encoded,
            reference_png_base64=encoded,
            protected_pixels_png_base64=encoded,
            selected_region_ids=["person:p0:body"],
        )
        return ImageMaskResponse(
            width=1,
            height=1,
            media_type="image/png",
            image_base64=encoded,
            effect="blur",
            strength=65,
            selected_region_ids=["person:p0:body"],
            recovery=recovery,
        )

    def restore_upload(self, *_args, **_kwargs):
        return VALID_PNG, {"applied": True}


def make_client() -> TestClient:
    app.dependency_overrides[get_image_masking_service] = (
        lambda: StubImageService()
    )
    return TestClient(app)


def test_imagemasker_page_loads() -> None:
    with make_client() as client:
        response = client.get("/imagemasker")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Detect private regions" in response.text
    assert "Restore after AI" in response.text


def test_image_detect_api_accepts_a_valid_png() -> None:
    with make_client() as client:
        response = client.post(
            "/api/image/detect",
            files={"image": ("person.png", VALID_PNG, "image/png")},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["regions"][0]["id"] == "person:p0:body"


def test_image_mask_api_exposes_effect_and_recovery_data() -> None:
    with make_client() as client:
        response = client.post(
            "/api/image/mask",
            files={"image": ("person.png", VALID_PNG, "image/png")},
            data={
                "selected_region_ids": '["person:p0:body"]',
                "effect": "blur",
                "strength": "65",
            },
        )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["media_type"] == "image/png"
    assert payload["recovery"]["format"] == "dataveil-image-recovery"


def test_image_restore_api_returns_a_downloadable_png() -> None:
    with make_client() as client:
        response = client.post(
            "/api/image/restore",
            files={
                "edited_image": ("edited.jpg", b"\xff\xd8stub", "image/jpeg"),
                "recovery_file": (
                    "recovery.json",
                    b"{}",
                    "application/json",
                ),
            },
        )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-dataveil-alignment"] == "applied"
