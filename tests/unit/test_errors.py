"""Unit tests for error handling — missing dependencies, unavailable search."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from odsbox_seman.types import SemanticSearchUnavailableError


class TestMissingOnnxRuntime:
    def test_search_raises_when_onnxruntime_missing(
        self, nvh_model: Any, tmp_path: Any, mocker: Any
    ) -> None:
        """When onnxruntime is not installed, search() raises SemanticSearchUnavailableError."""
        from odsbox_seman.model_searcher import ModelSearcher

        idx = ModelSearcher(nvh_model, cache_dir=tmp_path)
        # Simulate no cache — force _ensure_loaded to run the embedder
        assert idx._embeddings is None

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = SemanticSearchUnavailableError(
            "Semantic search requires 'onnxruntime'. Install it with: uv add onnxruntime"
        )
        idx._embedder = mock_embedder

        with pytest.raises(SemanticSearchUnavailableError, match="onnxruntime"):
            idx._ensure_loaded()


class TestMissingTransformers:
    def test_unavailable_error_hint(self) -> None:
        hint = "Semantic search requires 'transformers'. Install it with: uv add transformers"
        err = SemanticSearchUnavailableError(hint)
        assert "transformers" in err.hint


class TestSearcherRaisesUnavailable:
    def test_raises_unavailable_factory(self) -> None:
        from odsbox_seman.model_searcher import ModelSearcher

        err = ModelSearcher.raises_unavailable("test hint")
        assert isinstance(err, SemanticSearchUnavailableError)
        assert err.hint == "test hint"


class TestWarmUp:
    def test_warm_up_calls_ensure_loaded(self, nvh_model: Any, tmp_path: Any) -> None:
        import numpy as np

        from odsbox_seman.model_searcher import ModelSearcher

        idx = ModelSearcher(nvh_model, cache_dir=tmp_path)
        n = len(idx._corpus)
        fake_emb = np.ones((n, 4), dtype=np.float32)

        # Patch the embedder so no real ONNX is loaded
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = fake_emb
        mock_embedder._ensure_loaded = MagicMock()
        idx._embedder = mock_embedder

        idx.warm_up()

        assert idx._embeddings is not None


class TestInjectEmbeddings:
    def test_inject_sets_embeddings(self, nvh_model: Any) -> None:
        import numpy as np

        from odsbox_seman.model_searcher import ModelSearcher

        idx = ModelSearcher(nvh_model)
        n = len(idx._corpus)
        fake = np.zeros((n, 8), dtype=np.float32)
        idx._inject_embeddings(fake)
        assert idx.embeddings is not None
        assert idx.embeddings.shape == (n, 8)

    def test_inject_with_custom_corpus(self, nvh_model: Any) -> None:
        import numpy as np

        from odsbox_seman.model_searcher import ModelSearcher
        from odsbox_seman.types import ModelMatch

        idx = ModelSearcher(nvh_model)
        small_corpus = [
            (
                "test text",
                ModelMatch(
                    kind="attribute",
                    entity_name="E",
                    entity_base_name="AoE",
                    item_name="A",
                    item_base_name="ao_a",
                    data_type=1,
                    score=0.0,
                ),
            )
        ]
        fake = np.ones((1, 4), dtype=np.float32)
        idx._inject_embeddings(fake, small_corpus)
        assert len(idx._corpus) == 1
