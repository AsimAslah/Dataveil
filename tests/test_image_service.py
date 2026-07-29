from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from ImageMasker.schemas import ManualRegion
from ImageMasker.service import ImageMaskingService, decode_uploaded_image
from ImageMasker.unified_masker import apply_effect


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_decoder_accepts_png_jpg_and_jpeg_extensions() -> None:
    image = np.full((24, 32, 3), (20, 80, 140), dtype=np.uint8)
    png = png_bytes(image)
    ok, jpeg_encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = jpeg_encoded.tobytes()

    assert decode_uploaded_image(
        png, filename="photo.png", content_type="image/png"
    ).shape == image.shape
    assert decode_uploaded_image(
        jpeg, filename="photo.jpg", content_type="image/jpeg"
    ).shape == image.shape
    assert decode_uploaded_image(
        jpeg, filename="photo.jpeg", content_type="image/jpeg"
    ).shape == image.shape
    # The repository's preserved TestImage.png is JPEG-encoded despite its
    # extension. Both codecs are allowed, so safe decoding follows the content.
    assert decode_uploaded_image(
        jpeg, filename="legacy.png", content_type="image/png"
    ).shape == image.shape


@pytest.mark.parametrize(
    "effect",
    [
        "blur",
        "pixelate",
        "mosaic",
        "black",
        "solid",
        "noise",
        "sticker",
        "silhouette",
    ],
)
def test_all_masking_effects_change_selected_pixels(effect: str) -> None:
    image = np.full((80, 80, 3), 180, dtype=np.uint8)
    image[20:60, 20:60] = np.random.default_rng(7).integers(
        0, 256, size=(40, 40, 3), dtype=np.uint8
    )
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255

    protected = apply_effect(
        image,
        mask,
        effect=effect,
        strength=80,
        color="#9b204f",
    )

    assert np.array_equal(protected[0, 0], image[0, 0])
    assert not np.array_equal(protected[40, 40], image[40, 40])


def test_recovery_round_trip_restores_manual_region() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:] = (235, 235, 235)
    cv2.circle(image, (80, 60), 28, (20, 90, 210), -1)
    cv2.line(image, (0, 10), (159, 10), (10, 10, 10), 3)
    source = png_bytes(image)

    service = ImageMaskingService(
        person_detector=lambda _image, _confidence: [],
        face_detector=lambda _image: [],
    )
    result = service.mask_upload(
        source,
        filename="source.png",
        content_type="image/png",
        selected_region_ids=[],
        manual_regions=[
            ManualRegion(id="manual-1", x=50, y=30, width=60, height=60)
        ],
        effect="black",
        strength=100,
        color="#000000",
    )
    masked = base64.b64decode(result.image_base64)
    restored_png, _alignment = service.restore_upload(
        masked,
        edited_filename="edited.png",
        edited_content_type="image/png",
        recovery_data=result.recovery.model_dump_json().encode(),
    )
    restored = cv2.imdecode(
        np.frombuffer(restored_png, np.uint8), cv2.IMREAD_COLOR
    )

    assert np.array_equal(restored[60, 80], image[60, 80])
    assert np.array_equal(restored[5, 5], image[5, 5])
