"""odsbox-seman: Lightweight semantic search for ASAM ODS models using ONNX embeddings."""

from __future__ import annotations

from odsbox_seman.cli import main as cli_main
from odsbox_seman.model_searcher import ModelSearcher
from odsbox_seman.types import ModelMatch, SemanticSearchUnavailableError

__version__ = "0.2.0"

__all__ = [
    "ModelMatch",
    "ModelSearcher",
    "SemanticSearchUnavailableError",
    "__version__",
    "cli_main",
]
