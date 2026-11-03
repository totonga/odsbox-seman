"""odsbox-seman: Lightweight semantic search for ASAM ODS models using ONNX embeddings."""

from __future__ import annotations

import logging

from odsbox_seman.config import AliasConfig, BlacklistConfig, ModelConfig, SemanConfig, load_config
from odsbox_seman.model_searcher import ModelSearcher
from odsbox_seman.types import (
    ConfigError,
    CorpusStats,
    ModelMatch,
    MultiSearchResult,
    SearchSection,
    SemanticSearchUnavailableError,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.2.0"


def cli_main(argv: list[str] | None = None) -> int:
    """Run the optional CLI, importing its dependencies only when called."""
    from odsbox_seman.cli import main

    return main(argv)


__all__ = [
    "AliasConfig",
    "BlacklistConfig",
    "ConfigError",
    "CorpusStats",
    "ModelConfig",
    "ModelMatch",
    "ModelSearcher",
    "MultiSearchResult",
    "SearchSection",
    "SemanConfig",
    "SemanticSearchUnavailableError",
    "__version__",
    "cli_main",
    "load_config",
]
