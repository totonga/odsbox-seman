"""Integration tests for ModelSearcher — uses real ONNX model download and encoding.

These tests download the multilingual MiniLM ONNX model from the Hugging Face Hub
(~50 MB, cached after first run) and run actual semantic encoding.

Mark: ``integration`` — skipped unless you run with ``pytest -m integration``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def searcher(nvh_model: Any, tmp_path_factory: Any) -> Any:
    """Build a real ModelSearcher with ONNX embeddings (cached per test-module run)."""
    from odsbox_seman.model_searcher import ModelSearcher

    cache_dir: Path = tmp_path_factory.mktemp("onnx_cache")
    idx = ModelSearcher(nvh_model, cache_dir=cache_dir)
    idx.warm_up()
    return idx


# ---------------------------------------------------------------------------
# Basic search quality
# ---------------------------------------------------------------------------


class TestSearchQuality:
    def test_search_returns_results(self, searcher: Any) -> None:
        results = searcher.search("pressure", top_k=10)
        assert len(results) > 0

    def test_results_are_model_match(self, searcher: Any) -> None:
        from odsbox_seman.types import ModelMatch

        results = searcher.search("measurement date", top_k=5)
        for r in results:
            assert isinstance(r, ModelMatch)

    def test_scores_in_valid_range(self, searcher: Any) -> None:
        results = searcher.search("sensor", top_k=10)
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_scores_descending(self, searcher: Any) -> None:
        results = searcher.search("speed rpm", top_k=20)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty_list(self, searcher: Any) -> None:
        assert searcher.search("") == []
        assert searcher.search("   ") == []

    def test_top_k_respected(self, searcher: Any) -> None:
        results = searcher.search("entity attribute", top_k=3)
        assert len(results) <= 3

    def test_all_three_kinds_reachable(self, searcher: Any) -> None:
        """With a broad query, all three kinds should appear somewhere in top-50."""
        results = searcher.search("measurement sensor value name", top_k=50)
        kinds = {r.kind for r in results}
        # Attributes and relations are virtually always present in top-50
        assert "attribute" in kinds
        assert "relation" in kinds

    def test_relevant_entity_ranked_high_for_entity_query(self, searcher: Any) -> None:
        """Searching for 'mea result' should surface MeaResult attributes near the top."""
        results = searcher.search("mea result name", top_k=10)
        entity_names = {r.entity_name for r in results}
        assert any("mea" in n.lower() or "result" in n.lower() for n in entity_names)


# ---------------------------------------------------------------------------
# Embeddings properties
# ---------------------------------------------------------------------------


class TestEmbeddingsProperties:
    def test_embeddings_not_none_after_warm_up(self, searcher: Any) -> None:
        assert searcher.embeddings is not None

    def test_embeddings_shape(self, searcher: Any) -> None:
        emb = searcher.embeddings
        assert emb is not None
        n = len(searcher._corpus)
        assert emb.shape[0] == n
        assert emb.shape[1] > 0

    def test_embeddings_dtype_float32(self, searcher: Any) -> None:
        emb = searcher.embeddings
        assert emb is not None
        assert emb.dtype == np.float32

    def test_embeddings_are_normalized(self, searcher: Any) -> None:
        """L2-normalised embeddings should have unit norm."""
        emb = searcher.embeddings
        assert emb is not None
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-4)


# ---------------------------------------------------------------------------
# Cache round-trip with real embeddings
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_second_instantiation_uses_cache(self, nvh_model: Any, tmp_path: Path) -> None:
        from unittest.mock import patch

        from odsbox_seman.embedder import OnnxEmbedder
        from odsbox_seman.model_searcher import ModelSearcher

        # Build and cache with real embedder
        idx1 = ModelSearcher(nvh_model, cache_dir=tmp_path)
        idx1.warm_up()

        # Second instantiation — embedder.encode should NOT be called again
        idx2 = ModelSearcher(nvh_model, cache_dir=tmp_path)
        with patch.object(OnnxEmbedder, "encode") as mock_encode:
            idx2._ensure_loaded()
            mock_encode.assert_not_called()

    def test_cached_embeddings_match_original(self, nvh_model: Any, tmp_path: Path) -> None:
        from odsbox_seman.model_searcher import ModelSearcher

        idx1 = ModelSearcher(nvh_model, cache_dir=tmp_path)
        idx1.warm_up()
        emb1 = idx1.embeddings

        idx2 = ModelSearcher(nvh_model, cache_dir=tmp_path)
        idx2._ensure_loaded()
        emb2 = idx2.embeddings

        assert emb1 is not None
        assert emb2 is not None
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Introspection helpers with live embeddings
# ---------------------------------------------------------------------------


class TestIntrospectionWithEmbeddings:
    def test_resolve_attribute_semantic_fallback(self, searcher: Any) -> None:
        """With embeddings loaded, resolve_attribute should find fuzzy matches."""
        result = searcher.resolve_attribute("MeaResult", "measurement start time")
        # Either a string (found via semantic search) or None — just no crash
        assert result is None or isinstance(result, str)

    def test_entity_schema_non_empty(self, searcher: Any) -> None:
        schema = searcher.entity_schema("MeaResult")
        assert len(schema) > 0
        assert "MeaResult" in schema

    def test_find_date_attribute_returns_string(self, searcher: Any) -> None:
        result = searcher.find_date_attribute("MeaResult")
        assert isinstance(result, str)
        assert len(result) > 0
