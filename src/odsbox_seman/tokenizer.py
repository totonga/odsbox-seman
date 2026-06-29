"""Text tokenisation helpers for ODS model names."""

from __future__ import annotations

import re

# Matches the boundary between a lowercase letter and an uppercase letter, or
# between two uppercase letters where the second starts a new word (e.g. "XMLParser").
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Strip well-known AO/MDM prefixes that add no semantic value in search.
_AO_PREFIX_RE = re.compile(r"^[Aa][Oo](?=[A-Z])")


def tokenize(name: str) -> str:
    """Split snake_case and CamelCase identifiers, lowercase, and join with spaces.

    Examples::

        >>> tokenize("tyre_pressure")
        'tyre pressure'
        >>> tokenize("MeaQuantity")
        'mea quantity'
        >>> tokenize("AoTestEquipment")
        'ao test equipment'
        >>> tokenize("vehicle_manufacturer")
        'vehicle manufacturer'
    """
    if not name:
        return ""
    # Strip AO/ao prefix so "AoTestEquipment" → "ao test equipment" (prefix kept for context)
    # but split it as a separate token via CamelCase splitting below.
    # Insert spaces at CamelCase boundaries.
    spaced = _CAMEL_RE.sub(" ", name)
    # Replace underscores and hyphens with spaces.
    spaced = spaced.replace("_", " ").replace("-", " ")
    # Collapse runs of whitespace and lowercase everything.
    return " ".join(spaced.lower().split())
