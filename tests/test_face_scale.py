import cv2
import numpy as np

from ImageMasker.service import scale_binary_mask


def _center(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.where(mask > 0)
    return float(columns.mean()), float(rows.mean())


def test_face_mask_scale_changes_area_without_moving_center() -> None:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 60), (30, 38), 0, 0, 360, 255, -1)

    smaller = scale_binary_mask(mask, 70)
    larger = scale_binary_mask(mask, 130)

    assert np.count_nonzero(smaller) < np.count_nonzero(mask)
    assert np.count_nonzero(larger) > np.count_nonzero(mask)
    for scaled in (smaller, larger):
        center_x, center_y = _center(scaled)
        assert abs(center_x - 80) <= 1
        assert abs(center_y - 60) <= 1


def test_face_mask_scale_clips_safely_at_image_edge() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(mask, (4, 4), 8, 255, -1)

    scaled = scale_binary_mask(mask, 140)

    assert scaled.shape == mask.shape
    assert scaled.dtype == np.uint8
    assert set(np.unique(scaled)).issubset({0, 255})
