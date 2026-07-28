from __future__ import annotations

import pytest

from DocMasker.detector import PresidioDetector
from DocMasker.recognizers import build_custom_recognizers
from DocMasker.service import MaskingService


def custom_result_types(text: str) -> set[str]:
    result_types: set[str] = set()
    for recognizer in build_custom_recognizers():
        results = recognizer.analyze(
            text=text,
            entities=[recognizer.supported_entities[0]],
        )
        result_types.update(result.entity_type for result in results)
    return result_types


def test_indian_phone_number_custom_recognizer() -> None:
    assert "PHONE_NUMBER" in custom_result_types(
        "Call me on +91 98765 43210."
    )


def test_local_only_email_custom_recognizer() -> None:
    assert "EMAIL_ADDRESS" in custom_result_types(
        "Email me at riya.sharma+private@example.co.in."
    )


def test_aadhaar_like_number_custom_recognizer() -> None:
    assert "AADHAAR_NUMBER" in custom_result_types(
        "Aadhaar: 2345 6789 0123"
    )


def test_aadhaar_recognizer_does_not_claim_part_of_credit_card() -> None:
    assert "AADHAAR_NUMBER" not in custom_result_types(
        "Card: 4111 1111 1111 1111"
    )


def test_indian_pin_code_custom_recognizer() -> None:
    assert "IN_PIN_CODE" in custom_result_types(
        "The postal PIN is 560001."
    )


@pytest.fixture(scope="module")
def detector() -> PresidioDetector:
    return PresidioDetector()


def detected_types(detector: PresidioDetector, text: str) -> set[str]:
    return {candidate.entity_type for candidate in detector.detect(text)}


def test_email_address_detection(detector: PresidioDetector) -> None:
    assert "EMAIL" in detected_types(
        detector, "Email Riya at riya.sharma@example.com."
    )


def test_ip_address_detection(detector: PresidioDetector) -> None:
    assert "IP_ADDRESS" in detected_types(
        detector, "The private server is 192.168.1.10."
    )


def test_url_detection(detector: PresidioDetector) -> None:
    assert "URL" in detected_types(
        detector, "Open https://example.com/private?id=42."
    )


def test_credit_card_like_number_detection(
    detector: PresidioDetector,
) -> None:
    assert "CREDIT_CARD" in detected_types(
        detector, "Card: 4111 1111 1111 1111."
    )


def test_names_locations_organizations_and_dates(
    detector: PresidioDetector,
) -> None:
    types = detected_types(
        detector,
        "Riya Sharma joined Microsoft in Bengaluru on July 28, 2026.",
    )

    assert {"PERSON", "ORGANIZATION", "LOCATION", "DATE"} <= types


def test_india_specific_entities_through_full_detector(
    detector: PresidioDetector,
) -> None:
    types = detected_types(
        detector,
        "Aadhaar 2345 6789 0123, mobile +91 98765 43210, PIN 560001.",
    )

    assert {"AADHAAR", "PHONE", "INDIAN_PIN"} <= types


def test_credit_card_and_url_punctuation_mask_cleanly(
    detector: PresidioDetector,
) -> None:
    text = (
        "Card 4111 1111 1111 1111, then open "
        "https://example.com/private?id=42, please."
    )
    result = MaskingService(detector=detector).mask(text)

    assert result.masked_text == (
        "Card <CREDIT_CARD_1>, then open <URL_1>, please."
    )
