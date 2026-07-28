from __future__ import annotations

import re

from DocMasker.detector import DetectionCandidate
from DocMasker.masking import mask_detected_text
from DocMasker.restoration import restore_text


def candidate(
    text: str,
    value: str,
    entity_type: str,
    *,
    occurrence: int = 0,
    score: float = 0.9,
    flags: int = 0,
) -> DetectionCandidate:
    matches = list(re.finditer(re.escape(value), text, flags))
    match = matches[occurrence]
    return DetectionCandidate(
        entity_type=entity_type,
        start=match.start(),
        end=match.end(),
        score=score,
    )


def test_basic_masking_preserves_sentence_shape() -> None:
    text = "Alice emailed alice@example.com."
    result = mask_detected_text(
        text,
        [
            candidate(text, "Alice", "PERSON"),
            candidate(text, "alice@example.com", "EMAIL"),
        ],
    )

    assert result.masked_text == "<PERSON_1> emailed <EMAIL_1>."
    assert result.detection_count == 2
    assert result.original_text == text


def test_repeated_entity_uses_one_placeholder_case_insensitively() -> None:
    text = "Alice met ALICE after work."
    result = mask_detected_text(
        text,
        [
            candidate(text, "Alice", "PERSON"),
            candidate(text, "ALICE", "PERSON"),
        ],
    )

    assert result.masked_text == (
        "<PERSON_1> met <PERSON_1> after work."
    )
    assert list(result.placeholder_mapping) == ["<PERSON_1>"]
    assert result.placeholder_mapping["<PERSON_1>"]["original_values"] == [
        "Alice",
        "ALICE",
    ]


def test_multiple_values_of_same_type_get_separate_numbers() -> None:
    text = "Alice introduced Bob."
    result = mask_detected_text(
        text,
        [
            candidate(text, "Alice", "PERSON"),
            candidate(text, "Bob", "PERSON"),
        ],
    )

    assert result.masked_text == "<PERSON_1> introduced <PERSON_2>."


def test_specific_long_url_wins_over_overlapping_ip() -> None:
    text = "Visit https://192.168.1.5/private."
    url = candidate(text, "https://192.168.1.5/private", "URL", score=0.7)
    ip = candidate(text, "192.168.1.5", "IP_ADDRESS", score=0.95)

    result = mask_detected_text(text, [ip, url])

    assert result.masked_text == "Visit <URL_1>."
    assert result.detection_count == 1
    assert result.detected_entities[0].entity_type == "URL"


def test_empty_input_returns_empty_result() -> None:
    result = mask_detected_text("", [])

    assert result.masked_text == ""
    assert result.detected_entities == []
    assert result.placeholder_mapping == {}
    assert result.detection_count == 0


def test_punctuation_and_newlines_are_preserved() -> None:
    text = 'Call Priya, then email "p@example.com"!\nThanks.'
    result = mask_detected_text(
        text,
        [
            candidate(text, "Priya", "PERSON"),
            candidate(text, "p@example.com", "EMAIL"),
        ],
    )

    assert result.masked_text == (
        'Call <PERSON_1>, then email "<EMAIL_1>"!\nThanks.'
    )


def test_restoration_reconstructs_case_variants_exactly() -> None:
    text = "Alice met ALICE."
    result = mask_detected_text(
        text,
        [
            candidate(text, "Alice", "PERSON"),
            candidate(text, "ALICE", "PERSON"),
        ],
    )

    restored = restore_text(
        result.masked_text, result.placeholder_mapping
    )

    assert restored == text


def test_invalid_and_duplicate_candidates_are_handled_safely() -> None:
    text = "Alice"
    exact = candidate(text, "Alice", "PERSON", score=0.8)
    stronger_duplicate = candidate(text, "Alice", "PERSON", score=0.95)
    invalid = DetectionCandidate("EMAIL", -1, 100, 1.0)

    result = mask_detected_text(
        text, [exact, stronger_duplicate, invalid]
    )

    assert result.masked_text == "<PERSON_1>"
    assert result.detection_count == 1
    assert result.detected_entities[0].confidence_score == 0.95


def test_pseudonym_style_is_exactly_restorable() -> None:
    text = "Alice met Bob."
    result = mask_detected_text(
        text,
        [
            candidate(text, "Alice", "PERSON"),
            candidate(text, "Bob", "PERSON"),
        ],
        replacement_style="pseudonym",
    )

    assert result.masked_text == (
        "[PrivatePerson-1] met [PrivatePerson-2]."
    )
    assert restore_text(
        result.masked_text, result.placeholder_mapping
    ) == text
