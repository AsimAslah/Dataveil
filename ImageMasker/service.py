"""In-memory image detection, masking, and recovery orchestration."""

from __future__ import annotations

import base64
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Prevent Ultralytics from performing online availability checks or emitting
# anonymous usage events when the local model is loaded.
os.environ.setdefault("YOLO_OFFLINE", "true")

from .config import (
    ALLOWED_EFFECTS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_MIME_TYPES,
    DEFAULT_CONFIDENCE,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    YOLO_MODEL_PATH,
)
from .schemas import (
    BoundingBox,
    ImageDetectResponse,
    ImageMaskResponse,
    ImageRecoveryPackage,
    ImageRegion,
    ManualRegion,
)
from .unified_masker import (
    FaceInstance,
    PersonInstance,
    apply_effect,
    assign_faces_to_people,
    combine_masks,
    label_people_left_to_right,
    yolo_person_instances,
)
from .unmasker import map_source_to_return, unmask_only_region


PersonDetector = Callable[[np.ndarray, float], list[PersonInstance]]
FaceDetector = Callable[[np.ndarray], list[FaceInstance]]


class ImageProcessingError(ValueError):
    """Raised when an image request cannot be processed safely."""


@dataclass(slots=True)
class DetectionBundle:
    image: np.ndarray
    response: ImageDetectResponse
    masks_by_id: dict[str, np.ndarray]


_inference_lock = RLock()
_face_lock = RLock()


@lru_cache(maxsize=1)
def _load_yolo_model():
    if not YOLO_MODEL_PATH.is_file():
        raise ImageProcessingError("The local image model is missing.")
    from ultralytics import YOLO

    return YOLO(str(YOLO_MODEL_PATH))


@lru_cache(maxsize=1)
def _load_face_cascade():
    cascade_path = Path(cv2.data.haarcascades) / (
        "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise ImageProcessingError("The local face detector is unavailable.")
    return cascade


def _default_person_detector(
    image: np.ndarray, confidence: float
) -> list[PersonInstance]:
    with _inference_lock:
        return yolo_person_instances(
            _load_yolo_model(),
            image,
            conf=confidence,
            device="cpu",
        )


def _default_face_detector(image: np.ndarray) -> list[FaceInstance]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _load_face_cascade()
    min_side = max(24, min(image.shape[:2]) // 18)
    with _face_lock:
        rectangles = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
    height, width = image.shape[:2]
    faces: list[FaceInstance] = []
    for x, y, box_width, box_height in rectangles:
        padding_x = int(box_width * 0.15)
        padding_y = int(box_height * 0.2)
        x1 = max(0, int(x - padding_x))
        y1 = max(0, int(y - padding_y))
        x2 = min(width - 1, int(x + box_width + padding_x))
        y2 = min(height - 1, int(y + box_height + padding_y))
        mask = np.zeros((height, width), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        faces.append(
            FaceInstance(
                bbox_xyxy=(x1, y1, x2, y2),
                score=0.75,
                landmarks_106=None,
                mask=mask,
            )
        )
    return faces


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    if not ok:
        raise ImageProcessingError("The protected image could not be encoded.")
    return encoded.tobytes()


def _to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _from_base64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ImageProcessingError(
            f"The recovery {label} is invalid."
        ) from exc


def decode_uploaded_image(
    data: bytes,
    *,
    filename: str | None,
    content_type: str | None,
) -> np.ndarray:
    if not data:
        raise ImageProcessingError("The uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageProcessingError("The uploaded image is too large.")

    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ImageProcessingError("Upload a PNG, JPG, or JPEG image.")
    if content_type and content_type.lower() not in ALLOWED_IMAGE_MIME_TYPES:
        raise ImageProcessingError("Upload a PNG, JPG, or JPEG image.")

    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = data.startswith(b"\xff\xd8")
    if not (is_png or is_jpeg):
        raise ImageProcessingError("The image must contain PNG or JPEG data.")

    try:
        with Image.open(BytesIO(data)) as header:
            decoded_format = (header.format or "").upper()
            width, height = header.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageProcessingError(
            "The uploaded image could not be decoded."
        ) from exc
    if decoded_format not in {"PNG", "JPEG"}:
        raise ImageProcessingError("The image must contain PNG or JPEG data.")
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ImageProcessingError("The decoded image dimensions are too large.")

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ImageProcessingError("The uploaded image could not be decoded.")
    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise ImageProcessingError("The decoded image dimensions are too large.")
    return image


def _decode_recovery_image(
    value: str, mode: int, label: str
) -> np.ndarray:
    data = _from_base64(value, label)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), mode)
    if image is None:
        raise ImageProcessingError(f"The recovery {label} is invalid.")
    return image


class ImageMaskingService:
    def __init__(
        self,
        *,
        person_detector: PersonDetector | None = None,
        face_detector: FaceDetector | None = None,
    ) -> None:
        self._person_detector = person_detector or _default_person_detector
        self._face_detector = face_detector or _default_face_detector

    def _detect(
        self, image: np.ndarray, confidence: float
    ) -> DetectionBundle:
        if not 0.1 <= confidence <= 0.9:
            raise ImageProcessingError(
                "Detection confidence must be between 0.1 and 0.9."
            )
        people = label_people_left_to_right(
            self._person_detector(image, confidence)
        )
        faces = self._face_detector(image)
        face_by_person = assign_faces_to_people(
            people, faces, min_iou=0.008
        )

        regions: list[ImageRegion] = []
        masks_by_id: dict[str, np.ndarray] = {}
        assigned_face_ids: set[int] = set()

        for person in people:
            x1, y1, x2, y2 = person.bbox_xyxy
            body_id = f"person:{person.label}:body"
            regions.append(
                ImageRegion(
                    id=body_id,
                    label=f"{person.label.title()} person — body",
                    person_label=person.label,
                    kind="body",
                    confidence=max(0.0, min(1.0, person.score)),
                    bbox=BoundingBox(
                        x=x1,
                        y=y1,
                        width=max(1, x2 - x1),
                        height=max(1, y2 - y1),
                    ),
                )
            )
            masks_by_id[body_id] = person.mask

            face = face_by_person.get(person.label)
            if face is not None:
                assigned_face_ids.add(id(face))
                fx1, fy1, fx2, fy2 = face.bbox_xyxy
                face_id = f"person:{person.label}:face"
                regions.append(
                    ImageRegion(
                        id=face_id,
                        label=f"{person.label.title()} person — face",
                        person_label=person.label,
                        kind="face",
                        confidence=max(0.0, min(1.0, face.score)),
                        bbox=BoundingBox(
                            x=fx1,
                            y=fy1,
                            width=max(1, fx2 - fx1),
                            height=max(1, fy2 - fy1),
                        ),
                    )
                )
                masks_by_id[face_id] = face.mask

        # A face can still be protected when body segmentation misses it.
        standalone_index = 0
        for face in faces:
            if id(face) in assigned_face_ids:
                continue
            x1, y1, x2, y2 = face.bbox_xyxy
            face_id = f"face:{standalone_index}"
            standalone_index += 1
            regions.append(
                ImageRegion(
                    id=face_id,
                    label=f"Detected face {standalone_index}",
                    person_label=None,
                    kind="face",
                    confidence=max(0.0, min(1.0, face.score)),
                    bbox=BoundingBox(
                        x=x1,
                        y=y1,
                        width=max(1, x2 - x1),
                        height=max(1, y2 - y1),
                    ),
                )
            )
            masks_by_id[face_id] = face.mask

        height, width = image.shape[:2]
        response = ImageDetectResponse(
            width=width,
            height=height,
            person_count=len(people),
            regions=regions,
        )
        return DetectionBundle(
            image=image,
            response=response,
            masks_by_id=masks_by_id,
        )

    def detect_upload(
        self,
        data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> ImageDetectResponse:
        image = decode_uploaded_image(
            data, filename=filename, content_type=content_type
        )
        return self._detect(image, confidence).response

    def mask_upload(
        self,
        data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
        selected_region_ids: list[str] | None,
        manual_regions: list[ManualRegion],
        effect: str,
        strength: int,
        color: str,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> ImageMaskResponse:
        if effect not in ALLOWED_EFFECTS:
            raise ImageProcessingError("The selected masking effect is invalid.")
        if not 1 <= strength <= 100:
            raise ImageProcessingError("Masking strength must be 1 to 100.")

        image = decode_uploaded_image(
            data, filename=filename, content_type=content_type
        )
        bundle = self._detect(image, confidence)
        selected = (
            list(bundle.masks_by_id)
            if selected_region_ids is None
            else selected_region_ids
        )
        masks: list[np.ndarray] = []
        for region_id in selected:
            mask = bundle.masks_by_id.get(region_id)
            if mask is None:
                raise ImageProcessingError(
                    "A selected image region is no longer available."
                )
            if effect == "black":
                rows, columns = np.where(mask > 0)
                if rows.size and columns.size:
                    boxed_mask = np.zeros_like(mask)
                    boxed_mask[
                        rows.min() : rows.max() + 1,
                        columns.min() : columns.max() + 1,
                    ] = 255
                    mask = boxed_mask
            masks.append(mask)

        height, width = image.shape[:2]
        manual_ids: list[str] = []
        for region in manual_regions:
            x1 = min(width - 1, region.x)
            y1 = min(height - 1, region.y)
            x2 = min(width, region.x + region.width)
            y2 = min(height, region.y + region.height)
            if x2 <= x1 or y2 <= y1:
                continue
            manual_mask = np.zeros((height, width), dtype=np.uint8)
            manual_mask[y1:y2, x1:x2] = 255
            masks.append(manual_mask)
            manual_ids.append(region.id)

        if not masks:
            raise ImageProcessingError(
                "Select at least one detected or manual region."
            )

        combined_mask = combine_masks(masks)
        protected = apply_effect(
            image,
            combined_mask,
            effect=effect,
            strength=strength,
            color=color,
        )
        protected_png = _encode_png(protected)
        mask_png = _encode_png(combined_mask)

        protected_pixels = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        outside = combined_mask == 0
        protected_pixels[outside, :3] = 0
        protected_pixels[:, :, 3] = combined_mask
        protected_pixels_png = _encode_png(protected_pixels)

        all_selected_ids = selected + manual_ids
        recovery = ImageRecoveryPackage(
            format="dataveil-image-recovery",
            version=1,
            width=width,
            height=height,
            mask_png_base64=_to_base64(mask_png),
            reference_png_base64=_to_base64(protected_png),
            protected_pixels_png_base64=_to_base64(protected_pixels_png),
            selected_region_ids=all_selected_ids,
        )
        return ImageMaskResponse(
            width=width,
            height=height,
            media_type="image/png",
            image_base64=_to_base64(protected_png),
            effect=effect,
            strength=strength,
            selected_region_ids=all_selected_ids,
            recovery=recovery,
        )

    def restore_upload(
        self,
        edited_data: bytes,
        *,
        edited_filename: str | None,
        edited_content_type: str | None,
        recovery_data: bytes,
    ) -> tuple[bytes, dict[str, Any]]:
        edited = decode_uploaded_image(
            edited_data,
            filename=edited_filename,
            content_type=edited_content_type,
        )
        try:
            recovery = ImageRecoveryPackage.model_validate_json(
                recovery_data
            )
        except Exception as exc:
            raise ImageProcessingError(
                "The DataVeil image recovery file is invalid."
            ) from exc
        if recovery.width * recovery.height > MAX_IMAGE_PIXELS:
            raise ImageProcessingError(
                "The recovery image dimensions are too large."
            )

        reference = _decode_recovery_image(
            recovery.reference_png_base64,
            cv2.IMREAD_COLOR,
            "reference image",
        )
        mask = _decode_recovery_image(
            recovery.mask_png_base64,
            cv2.IMREAD_GRAYSCALE,
            "mask",
        )
        protected_pixels = _decode_recovery_image(
            recovery.protected_pixels_png_base64,
            cv2.IMREAD_UNCHANGED,
            "protected pixels",
        )
        expected_shape = (recovery.height, recovery.width)
        if (
            reference.shape[:2] != expected_shape
            or mask.shape[:2] != expected_shape
            or protected_pixels.shape[:2] != expected_shape
            or protected_pixels.ndim != 3
            or protected_pixels.shape[2] != 4
        ):
            raise ImageProcessingError(
                "The recovery images have inconsistent dimensions."
            )

        mask = (mask > 127).astype(np.uint8) * 255
        reconstructed = reference.copy()
        protected_bgr = protected_pixels[:, :, :3]
        restore_pixels = mask.astype(bool)
        reconstructed[restore_pixels] = protected_bgr[restore_pixels]

        edited_height, edited_width = edited.shape[:2]
        mapped_original, mapped_mask, alignment = map_source_to_return(
            reconstructed,
            mask,
            (edited_height, edited_width),
            edited,
            source_space="dataveil_recovery",
        )
        restored = unmask_only_region(
            edited, mapped_original, mapped_mask
        )
        return _encode_png(restored), alignment
