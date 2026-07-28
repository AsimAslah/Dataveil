"""Overlap resolution and deterministic, request-scoped masking."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

from .detector import DetectionCandidate


ENTITY_PRIORITY: dict[str, int] = {
    "AADHAAR": 120,
    "EMAIL": 115,
    "URL": 110,
    "PHONE": 105,
    "CREDIT_CARD": 100,
    "IP_ADDRESS": 95,
    "INDIAN_PIN": 90,
    "DATE": 70,
    "PERSON": 60,
    "ORGANIZATION": 55,
    "LOCATION": 50,
}


@dataclass(frozen=True, slots=True)
class MaskedOccurrence:
    entity_type: str
    original_value: str
    placeholder: str
    start: int
    end: int
    confidence_score: float


@dataclass(slots=True)
class MaskingResult:
    original_text: str
    masked_text: str
    detected_entities: list[MaskedOccurrence]
    placeholder_mapping: dict[str, dict[str, object]]

    @property
    def detection_count(self) -> int:
        return len(self.detected_entities)


def _spans_overlap(left: DetectionCandidate, right: DetectionCandidate) -> bool:
    return left.start < right.end and right.start < left.end


def resolve_overlaps(
    text: str, candidates: list[DetectionCandidate]
) -> list[DetectionCandidate]:
    """Select non-overlapping spans using specificity, score, and length."""

    valid: list[DetectionCandidate] = []
    best_exact_match: dict[tuple[str, int, int], DetectionCandidate] = {}
    for candidate in candidates:
        if not (0 <= candidate.start < candidate.end <= len(text)):
            continue
        key = (candidate.entity_type, candidate.start, candidate.end)
        previous = best_exact_match.get(key)
        if previous is None or candidate.score > previous.score:
            best_exact_match[key] = candidate

    ranked = sorted(
        best_exact_match.values(),
        key=lambda item: (
            -ENTITY_PRIORITY.get(item.entity_type, 0),
            -item.score,
            -(item.end - item.start),
            item.start,
            item.end,
        ),
    )

    for candidate in ranked:
        if any(_spans_overlap(candidate, selected) for selected in valid):
            continue
        valid.append(candidate)

    return sorted(valid, key=lambda item: (item.start, item.end))


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def canonicalize_value(entity_type: str, value: str) -> str:
    """Return a stable key used only for the lifetime of one request."""

    collapsed = " ".join(value.split())
    if entity_type in {
        "PHONE",
        "CREDIT_CARD",
        "AADHAAR",
        "INDIAN_PIN",
    }:
        return _digits_only(collapsed)
    if entity_type == "IP_ADDRESS":
        try:
            return ipaddress.ip_address(collapsed).compressed.casefold()
        except ValueError:
            return collapsed.casefold()
    return collapsed.casefold()


def mask_detected_text(
    text: str,
    candidates: list[DetectionCandidate],
    *,
    replacement_style: str = "placeholder",
) -> MaskingResult:
    selected = resolve_overlaps(text, candidates)
    if not selected:
        return MaskingResult(
            original_text=text,
            masked_text=text,
            detected_entities=[],
            placeholder_mapping={},
        )

    counters: dict[str, int] = {}
    placeholders_by_value: dict[tuple[str, str], str] = {}
    placeholder_mapping: dict[str, dict[str, object]] = {}
    occurrences: list[MaskedOccurrence] = []

    for candidate in selected:
        original_value = text[candidate.start : candidate.end]
        canonical_value = canonicalize_value(
            candidate.entity_type, original_value
        )
        value_key = (candidate.entity_type, canonical_value)
        placeholder = placeholders_by_value.get(value_key)
        if placeholder is None:
            counters[candidate.entity_type] = (
                counters.get(candidate.entity_type, 0) + 1
            )
            number = counters[candidate.entity_type]
            placeholder = _replacement_for_style(
                candidate.entity_type,
                number,
                replacement_style,
            )
            placeholders_by_value[value_key] = placeholder
            if placeholder:
                placeholder_mapping.setdefault(
                    placeholder,
                    {
                        "entity_type": candidate.entity_type,
                        "original_values": [],
                    },
                )

        if placeholder:
            original_values = placeholder_mapping[placeholder][
                "original_values"
            ]
            if isinstance(original_values, list):
                original_values.append(original_value)

        occurrences.append(
            MaskedOccurrence(
                entity_type=candidate.entity_type,
                original_value=original_value,
                placeholder=placeholder,
                start=candidate.start,
                end=candidate.end,
                confidence_score=round(
                    max(0.0, min(1.0, candidate.score)), 4
                ),
            )
        )

    masked_text = text
    for occurrence in reversed(occurrences):
        masked_text = (
            masked_text[: occurrence.start]
            + occurrence.placeholder
            + masked_text[occurrence.end :]
        )

    return MaskingResult(
        original_text=text,
        masked_text=masked_text,
        detected_entities=occurrences,
        placeholder_mapping=placeholder_mapping,
    )


def _replacement_for_style(
    entity_type: str,
    number: int,
    replacement_style: str,
) -> str:
    if replacement_style == "remove":
        return ""
    if replacement_style == "redact":
        return f"<REDACTED_{entity_type}_{number}>"
    if replacement_style == "pseudonym":
        readable_type = entity_type.replace("_", " ").title().replace(" ", "")
        return f"[Private{readable_type}-{number}]"
    if replacement_style == "generalize":
        return f"[{entity_type.replace('_', ' ')}]"
    return f"<{entity_type}_{number}>"
