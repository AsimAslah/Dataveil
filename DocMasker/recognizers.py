"""Custom Presidio recognizers for India-specific identifiers."""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


def build_custom_recognizers() -> list[PatternRecognizer]:
    """Return deterministic recognizers which require no network access."""

    local_email = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        name="Local-only email address recognizer",
        supported_language="en",
        patterns=[
            Pattern(
                name="RFC-style email address with dotted domain",
                regex=(
                    r"(?i)\b[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
                    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
                    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+\b"
                ),
                score=0.65,
            )
        ],
        context=["email", "mail", "contact"],
    )

    aadhaar = PatternRecognizer(
        supported_entity="AADHAAR_NUMBER",
        name="Indian Aadhaar-like number recognizer",
        supported_language="en",
        patterns=[
            Pattern(
                name="12 digits, optionally grouped in fours",
                regex=(
                    r"(?<!\d)(?<!\d\s)(?<!\d-)"
                    r"[2-9]\d{3}(?:[\s-]?\d{4}){2}"
                    r"(?![\s-]?\d)"
                ),
                score=0.55,
            )
        ],
        context=["aadhaar", "aadhar", "uidai", "uid number", "identity number"],
    )

    indian_pin = PatternRecognizer(
        supported_entity="IN_PIN_CODE",
        name="Indian PIN code recognizer",
        supported_language="en",
        patterns=[
            Pattern(
                name="six digit Indian postal index number",
                regex=r"(?<!\d)[1-9]\d{5}(?!\d)",
                score=0.45,
            )
        ],
        context=["pin", "pincode", "pin code", "postal", "postcode", "address"],
    )

    indian_phone = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        name="Indian mobile number recognizer",
        supported_language="en",
        patterns=[
            Pattern(
                name="Indian mobile with optional country or trunk prefix",
                regex=r"(?<!\d)(?:(?:\+91|0091)[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)",
                score=0.65,
            )
        ],
        context=["phone", "mobile", "call", "contact", "whatsapp", "tel"],
    )

    return [local_email, aadhaar, indian_pin, indian_phone]
