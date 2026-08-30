from pathlib import Path

import cv2
import numpy as np

from ImageMasker.config import (
    FACE_PARSING_MODEL_PATH,
    FACE_PARSING_MODEL_SHA256,
)
from ImageMasker.face_parser import (
    SemanticFaceParser,
    semantic_privacy_mask,
)


def test_semantic_privacy_mask_uses_face_classes_not_the_crop_box() -> None:
    labels = np.zeros((80, 100), dtype=np.uint8)
    cv2.ellipse(labels, (50, 43), (24, 29), 0, 0, 360, 1, -1)
    labels[9:25, 26:75] = 17
    labels[70:79, 40:61] = 14

    mask = semantic_privacy_mask(labels)

    assert mask[43, 50] == 255
    assert mask[12, 50] == 255
    assert mask[75, 50] == 0
    assert mask[0, 0] == 0
    assert np.count_nonzero(mask) < mask.size // 2


def test_bundled_face_parser_checkpoint_loads_offline() -> None:
    assert Path(FACE_PARSING_MODEL_PATH).is_file()
    parser = SemanticFaceParser(
        FACE_PARSING_MODEL_PATH,
        expected_sha256=FACE_PARSING_MODEL_SHA256,
        input_size=64,
    )
    labels = parser.parse_labels(np.zeros((36, 48, 3), dtype=np.uint8))

    assert labels.shape == (36, 48)
    assert labels.dtype == np.uint8
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 18
