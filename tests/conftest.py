from __future__ import annotations

from collections.abc import Iterable

import pytest

from DocMasker.detector import DetectionCandidate


class StubDetector:
    def __init__(self, candidates: Iterable[DetectionCandidate]) -> None:
        self._candidates = list(candidates)

    def detect(self, _text: str) -> list[DetectionCandidate]:
        return list(self._candidates)


@pytest.fixture
def stub_detector_type():
    return StubDetector

