"""Public data types and exceptions for odsbox-seman."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class ModelMatch:
    """A single search result referencing an ODS model element."""

    kind: Literal["attribute", "relation", "enumeration"]
    entity_name: str  # empty for enumerations
    entity_base_name: str  # empty for enumerations
    item_name: str  # attribute / relation / enumeration name
    item_base_name: str
    data_type: int  # ods.DataTypeEnum value; 0 for relations & enumerations
    score: float  # cosine similarity in [0, 1]

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (for JSON caching)."""
        return asdict(self)


class SemanticSearchUnavailableError(RuntimeError):
    """Raised when semantic search cannot run because an optional dependency is missing."""

    def __init__(self, hint: str) -> None:
        super().__init__(hint)
        self.hint = hint
