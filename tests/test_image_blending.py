from __future__ import annotations

import cv2
import numpy as np

from ImageMasker.unmasker import build_feather_alpha, unmask_only_region


def test_restoration_feathers_a_circular_mask_boundary() -> None:
    edited = np.full((80, 80, 3), 30, dtype=np.uint8)
    original = np.full((80, 80, 3), 220, dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 25, 255, -1)

    alpha = build_feather_alpha(mask)
    restored = unmask_only_region(edited, original, mask)

    assert 0 < alpha[40, 14] < 1
    assert alpha[40, 15] == 1
    assert alpha[40, 40] == 1
    assert np.all(restored[40, 14] > edited[40, 14])
    assert np.all(restored[40, 14] < original[40, 14])
    assert np.array_equal(restored[40, 15], original[40, 15])
    assert np.array_equal(restored[40, 40], original[40, 40])


def test_restoration_does_not_mix_mask_color_inside_boundary() -> None:
    edited = np.zeros((80, 80, 3), dtype=np.uint8)
    edited[:, :] = (20, 170, 30)
    original = np.full((80, 80, 3), 220, dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 25, 255, -1)

    restored = unmask_only_region(edited, original, mask)

    assert np.array_equal(restored[40, 15], original[40, 15])
    assert np.array_equal(restored[40, 40], original[40, 40])
    assert np.array_equal(restored[0, 0], edited[0, 0])


def test_restoration_with_an_empty_mask_leaves_image_unchanged() -> None:
    edited = np.full((20, 24, 3), 90, dtype=np.uint8)
    original = np.full((20, 24, 3), 210, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)

    restored = unmask_only_region(edited, original, mask)

    assert np.array_equal(restored, edited)
