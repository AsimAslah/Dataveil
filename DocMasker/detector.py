"""Presidio and spaCy-backed sensitive entity detection."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .config import DETECTION_SCORE_THRESHOLD, LANGUAGE, SPACY_MODEL
from .recognizers import build_custom_recognizers


PRESIDIO_TO_OUTPUT_TYPE: dict[str, str] = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
    "CREDIT_CARD": "CREDIT_CARD",
    "AADHAAR_NUMBER": "AADHAAR",
    "IN_PIN_CODE": "INDIAN_PIN",
}

SUPPORTED_PRESIDIO_ENTITIES = tuple(PRESIDIO_TO_OUTPUT_TYPE)


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    entity_type: str
    start: int
    end: int
    score: float


class EntityDetector(Protocol):
    def detect(self, text: str) -> list[DetectionCandidate]:
        """Detect sensitive spans in text."""


class PresidioDetector:
    """Local detector with a CPU-optimized spaCy pipeline."""

    def __init__(
        self,
        *,
        model_name: str = SPACY_MODEL,
        score_threshold: float = DETECTION_SCORE_THRESHOLD,
    ) -> None:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": model_name}],
        }
        nlp_engine = NlpEngineProvider(
            nlp_configuration=configuration
        ).create_engine()

        registry = RecognizerRegistry(supported_languages=[LANGUAGE])
        registry.load_predefined_recognizers(languages=[LANGUAGE])
        # Presidio's built-in email validator uses tldextract, whose default
        # suffix-list behavior can perform a network lookup. Replace it with
        # our deterministic regex recognizer to guarantee local-only runtime.
        registry.remove_recognizer("EmailRecognizer", language=LANGUAGE)
        for recognizer in build_custom_recognizers():
            registry.add_recognizer(recognizer)

        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=[LANGUAGE],
        )
        self._score_threshold = score_threshold
        self._analysis_lock = RLock()

    def detect(self, text: str) -> list[DetectionCandidate]:
        if not text:
            return []

        # Some NLP components maintain reusable internal buffers. Serializing
        # analysis avoids cross-request races without retaining request data.
        with self._analysis_lock:
            results = self._analyzer.analyze(
                text=text,
                language=LANGUAGE,
                entities=list(SUPPORTED_PRESIDIO_ENTITIES),
                score_threshold=self._score_threshold,
                return_decision_process=False,
            )

        candidates: list[DetectionCandidate] = []
        for result in results:
            output_type = PRESIDIO_TO_OUTPUT_TYPE.get(result.entity_type)
            if output_type is None:
                continue
            start = result.start
            end = result.end
            if output_type == "URL":
                # Presidio's URL recognizer may include sentence punctuation.
                # Trimming it here preserves the surrounding prose exactly.
                while (
                    end > start
                    and text[end - 1] in ".,;:!?)]}>\"'"
                ):
                    end -= 1
            if end <= start:
                continue
            candidates.append(
                DetectionCandidate(
                    entity_type=output_type,
                    start=start,
                    end=end,
                    score=float(result.score),
                )
            )
        return candidates
