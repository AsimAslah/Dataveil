"""Exact local restoration from an occurrence-aware placeholder mapping."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class InvalidRestorationMapping(ValueError):
    """Raised when a mapping cannot safely reconstruct the masked text."""


def _entry_values(entry: Any) -> Sequence[str]:
    if hasattr(entry, "original_values"):
        values = entry.original_values
    elif isinstance(entry, Mapping):
        values = entry.get("original_values")
    else:
        values = None

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise InvalidRestorationMapping("A mapping entry is malformed.")
    if not all(isinstance(value, str) for value in values):
        raise InvalidRestorationMapping("A mapping entry is malformed.")
    return values


def restore_text(
    masked_text: str, placeholder_mapping: Mapping[str, Any]
) -> str:
    """Restore placeholders in occurrence order without persisting the map."""

    occurrence_indexes: dict[str, int] = defaultdict(int)
    replacements = [
        key
        for key in placeholder_mapping
        if isinstance(key, str) and key
    ]
    if not replacements:
        return masked_text
    replacement_re = re.compile(
        "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True))
    )

    def replace(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        entry = placeholder_mapping.get(placeholder)
        if entry is None:
            return placeholder

        values = _entry_values(entry)
        index = occurrence_indexes[placeholder]
        if index >= len(values):
            raise InvalidRestorationMapping(
                "A placeholder has fewer source values than occurrences."
            )
        occurrence_indexes[placeholder] += 1
        return values[index]

    return replacement_re.sub(replace, masked_text)
