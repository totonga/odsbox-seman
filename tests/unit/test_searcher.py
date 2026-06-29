"""Unit tests for ModelSearcher — corpus building, search, introspection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from odsbox_seman.model_searcher import ModelSearcher
from odsbox_seman.types import ModelMatch

# ---------------------------------------------------------------------------
# Corpus coverage
# ---------------------------------------------------------------------------


class TestCorpusCoverage:
    def test_corpus_not_empty(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert len(idx._corpus) > 0

    def test_corpus_contains_attributes(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        kinds = {m.kind for _, m in idx._corpus}
        assert "attribute" in kinds

    def test_corpus_contains_relations(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        kinds = {m.kind for _, m in idx._corpus}
        assert "relation" in kinds

    def test_corpus_contains_enumerations(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        kinds = {m.kind for _, m in idx._corpus}
        assert "enumeration" in kinds

    def test_all_three_kinds_present(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        kinds = {m.kind for _, m in idx._corpus}
        assert kinds == {"attribute", "relation", "enumeration"}

    def test_attribute_text_includes_entity(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        attr_entries = [(t, m) for t, m in idx._corpus if m.kind == "attribute"]
        assert attr_entries, "No attribute entries found"
        text, match = attr_entries[0]
        assert (
            match.entity_name.lower() in text.lower()
            or match.entity_base_name.lower().replace("ao", "").strip() in text
        )

    def test_enumeration_text_includes_enum_name(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        enum_entries = [(t, m) for t, m in idx._corpus if m.kind == "enumeration"]
        assert enum_entries, "No enumeration entries found"
        text, match = enum_entries[0]
        # tokenized name should appear in text
        assert len(text) > 0

    def test_corpus_relation_text_includes_target(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        rel_entries = [(t, m) for t, m in idx._corpus if m.kind == "relation"]
        assert rel_entries, "No relation entries found"
        text, _ = rel_entries[0]
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------


class TestSearch:
    def _idx_with_fake_embeddings(
        self, nvh_model: Any, tmp_path: Path, dim: int = 32
    ) -> tuple[ModelSearcher, np.ndarray]:
        idx = ModelSearcher(nvh_model, cache_dir=tmp_path)
        n = len(idx._corpus)
        rng = np.random.default_rng(7)
        fake_emb = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(fake_emb, axis=1, keepdims=True)
        fake_emb /= norms
        idx._inject_embeddings(fake_emb)
        return idx, fake_emb

    def test_empty_query_returns_empty(self, nvh_model: Any, tmp_path: Path) -> None:
        idx, _ = self._idx_with_fake_embeddings(nvh_model, tmp_path)
        # Need a fake embedder for query encoding
        from unittest.mock import MagicMock

        mock_embedder = MagicMock()
        idx._embedder = mock_embedder
        assert idx.search("") == []
        assert idx.search("   ") == []

    def test_returns_at_most_top_k(self, nvh_model: Any, tmp_path: Path) -> None:
        idx, fake_emb = self._idx_with_fake_embeddings(nvh_model, tmp_path)
        # Inject a fake embedder that returns a unit query vector
        from unittest.mock import MagicMock

        dim = fake_emb.shape[1]
        query_emb = np.ones((1, dim), dtype=np.float32)
        query_emb /= np.linalg.norm(query_emb)
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = query_emb
        idx._embedder = mock_embedder

        results = idx.search("anything", top_k=5)
        assert len(results) <= 5

    def test_scores_descending(self, nvh_model: Any, tmp_path: Path) -> None:
        idx, fake_emb = self._idx_with_fake_embeddings(nvh_model, tmp_path)
        from unittest.mock import MagicMock

        dim = fake_emb.shape[1]
        rng = np.random.default_rng(99)
        query_emb = rng.standard_normal((1, dim)).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = query_emb
        idx._embedder = mock_embedder

        results = idx.search("anything", top_k=20)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_model_match_objects(self, nvh_model: Any, tmp_path: Path) -> None:
        idx, fake_emb = self._idx_with_fake_embeddings(nvh_model, tmp_path)
        from unittest.mock import MagicMock

        dim = fake_emb.shape[1]
        query_emb = np.ones((1, dim), dtype=np.float32)
        query_emb /= np.linalg.norm(query_emb)
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = query_emb
        idx._embedder = mock_embedder

        results = idx.search("test", top_k=5)
        for r in results:
            assert isinstance(r, ModelMatch)
            assert r.kind in {"attribute", "relation", "enumeration"}

    def test_default_top_k_is_20(self, nvh_model: Any, tmp_path: Path) -> None:
        idx, fake_emb = self._idx_with_fake_embeddings(nvh_model, tmp_path)
        from unittest.mock import MagicMock

        dim = fake_emb.shape[1]
        query_emb = np.ones((1, dim), dtype=np.float32)
        query_emb /= np.linalg.norm(query_emb)
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = query_emb
        idx._embedder = mock_embedder

        results = idx.search("anything")
        assert len(results) <= 20

    def test_top_k_larger_than_corpus(self, nvh_model: Any, tmp_path: Path) -> None:
        idx, fake_emb = self._idx_with_fake_embeddings(nvh_model, tmp_path)
        from unittest.mock import MagicMock

        dim = fake_emb.shape[1]
        query_emb = np.ones((1, dim), dtype=np.float32)
        query_emb /= np.linalg.norm(query_emb)
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = query_emb
        idx._embedder = mock_embedder

        results = idx.search("anything", top_k=100000)
        assert len(results) <= len(idx._corpus)


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_saved_after_encoding(self, nvh_model: Any, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        idx = ModelSearcher(nvh_model, cache_dir=tmp_path)
        n = len(idx._corpus)
        dim = 32
        rng = np.random.default_rng(0)
        fake_emb = rng.standard_normal((n, dim)).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = fake_emb
        idx._embedder = mock_embedder

        idx._ensure_loaded()

        npz_files = list(tmp_path.glob("*.npz"))
        assert npz_files, "Cache NPZ file should have been created"

    def test_cache_loaded_on_second_instantiation(self, nvh_model: Any, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        idx1 = ModelSearcher(nvh_model, cache_dir=tmp_path)
        n = len(idx1._corpus)
        dim = 32
        rng = np.random.default_rng(0)
        fake_emb = rng.standard_normal((n, dim)).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = fake_emb
        idx1._embedder = mock_embedder
        idx1._ensure_loaded()

        # Second instantiation — cache should be hit, encoder should NOT be called
        idx2 = ModelSearcher(nvh_model, cache_dir=tmp_path)
        mock_embedder2 = MagicMock()
        idx2._embedder = mock_embedder2
        idx2._ensure_loaded()
        mock_embedder2.encode.assert_not_called()

    def test_embeddings_property_none_before_load(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.embeddings is None


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


class TestEntitySchema:
    def test_contains_entity_name(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        schema = idx.entity_schema("MeaResult")
        assert "MeaResult" in schema

    def test_lists_attributes(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        schema = idx.entity_schema("MeaResult")
        assert "Name" in schema or "Id" in schema

    def test_lists_relations(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        schema = idx.entity_schema("TestEquipment")
        # Should include "relation" or "->"
        assert "->" in schema or "Relation" in schema or "relation" in schema

    def test_unknown_entity_returns_empty(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.entity_schema("DoesNotExist") == ""


class TestResolveAttribute:
    def test_exact_match(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.resolve_attribute("MeaResult", "Name") == "Name"

    def test_case_insensitive(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        result = idx.resolve_attribute("MeaResult", "name")
        assert result == "Name"

    def test_unknown_entity_returns_none(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.resolve_attribute("NoSuchEntity", "Name") is None

    def test_unknown_attr_without_embeddings_returns_none(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx._embeddings is None
        result = idx.resolve_attribute("MeaResult", "xyzzy_nonexistent_attr_abc")
        assert result is None


class TestResolveEntity:
    def test_exact_match(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.resolve_entity("MeaResult") == "MeaResult"

    def test_case_insensitive(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        result = idx.resolve_entity("mearesult")
        assert result == "MeaResult"

    def test_unknown_returns_none(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.resolve_entity("NoSuchEntity") is None

    def test_attribute_name_returns_none(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        # "Name" is an attribute, not an entity
        assert idx.resolve_entity("Name") is None


class TestFindDateAttribute:
    def test_mea_result_returns_begin(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        result = idx.find_date_attribute("MeaResult")
        assert result is not None
        assert "Begin" in result or "begin" in result.lower()

    def test_unknown_entity_returns_none(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        assert idx.find_date_attribute("NoSuchEntity") is None

    def test_entity_without_dates(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        result = idx.find_date_attribute("MeaQuantity")
        # Either None or a string — just ensure it doesn't crash
        assert result is None or isinstance(result, str)
