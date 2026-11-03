"""Public data types and exceptions for odsbox-seman."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class ModelMatch:
    """A single search result referencing an ODS model element."""

    kind: Literal["entity", "attribute", "relation", "enumeration"]
    entity_name: str  # empty for enumerations
    entity_base_name: str  # empty for enumerations
    item_name: str  # entity / attribute / relation / enumeration name
    item_base_name: str
    data_type: int  # ods.DataTypeEnum value; 0 for entities, relations & enumerations
    score: float  # cosine similarity in [0, 1]

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (for JSON caching)."""
        return asdict(self)


SearchMode = Literal["entity", "attribute", "relation", "enumeration", "all"]


@dataclass
class SearchSection:
    """One mode's matches and selection time in seconds, excluding shared scoring."""

    mode: SearchMode
    matches: list[ModelMatch]
    elapsed_seconds: float


@dataclass
class MultiSearchResult:
    """Ordered mode sections, their ranked union, and processing times in seconds."""

    sections: list[SearchSection]
    merged: list[ModelMatch]
    shared_seconds: float
    merge_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class CorpusStats:
    """How much of the ODS model remains searchable after the blacklists were applied."""

    entities_total: int = 0
    entities_kept: int = 0
    attributes_total: int = 0
    attributes_kept: int = 0
    relations_total: int = 0
    relations_kept: int = 0
    enumerations: int = 0
    excluded_entities: tuple[str, ...] = ()
    # Attributes dropped by attribute patterns in entities that are themselves kept.
    excluded_attributes: tuple[str, ...] = ()


class ConfigError(ValueError):
    """Raised when the configuration file is invalid; carries every problem found."""

    def __init__(self, path: object, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        lines = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Invalid configuration {path} ({len(errors)} error(s)):\n{lines}")


class SemanticSearchUnavailableError(RuntimeError):
    """Raised when semantic search cannot run because an optional dependency is missing."""

    def __init__(self, hint: str) -> None:
        super().__init__(hint)
        self.hint = hint
