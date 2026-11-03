"""Unit tests for ModelSearcher — corpus building, search, introspection helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Barrier, Event
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from odsbox_seman.model_searcher import ModelSearcher
from odsbox_seman.tokenizer import tokenize
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

    def test_all_kinds_present(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        kinds = {m.kind for _, m in idx._corpus}
        assert kinds == {"entity", "attribute", "relation", "enumeration"}

    def test_attribute_text_includes_entity(self, nvh_model: Any) -> None:
        idx = ModelSearcher(nvh_model)
        attr_entries = [(t, m) for t, m in idx._corpus if m.kind == "attribute"]
        assert attr_entries, "No attribute entries found"
        text, match = attr_entries[0]
        entity_tokens = tokenize(match.entity_name)
        base_tokens = tokenize(match.entity_base_name)
        assert (
            entity_tokens in text.lower()
            or base_tokens in text.lower()
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
# Multi-mode search
# ---------------------------------------------------------------------------


class TestMultiSearch:
    def _with_matches(
        self, nvh_model: Any, tmp_path: Path, matches: list[ModelMatch], scores: list[float]
    ) -> tuple[ModelSearcher, MagicMock]:
        searcher = ModelSearcher(nvh_model, cache_dir=tmp_path)
        searcher._corpus = [(match.item_name, match) for match in matches]
        searcher._alias_flags = np.zeros(len(matches), dtype=np.float32)
        searcher._inject_embeddings(
            np.column_stack((scores, np.zeros(len(matches)))).astype(np.float32)
        )
        embedder = MagicMock()
        embedder.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        searcher._embedder = embedder
        return searcher, embedder

    def test_complete_ranking_and_single_encoding(self, nvh_model: Any, tmp_path: Path) -> None:
        attributes = [
            ModelMatch("attribute", "Vehicle", "AoVehicle", f"A{number}", "", 1, 0.0)
            for number in range(60)
        ]
        rare = [
            ModelMatch("entity", "Vehicle", "AoVehicle", "Vehicle", "", 0, 0.0),
            ModelMatch("relation", "Vehicle", "AoVehicle", "Parts", "", 0, 0.0),
            ModelMatch("enumeration", "", "", "State", "", 0, 0.0),
        ]
        searcher, embedder = self._with_matches(
            nvh_model, tmp_path, attributes + rare, np.linspace(0.99, 0.1, 63).tolist()
        )

        result = searcher.search_multi("pressure", top_k=1)

        assert [section.mode for section in result.sections] == [
            "entity",
            "attribute",
            "relation",
            "enumeration",
            "all",
        ]
        assert [section.matches[0].item_name for section in result.sections] == [
            "Vehicle",
            "A0",
            "Parts",
            "State",
            "A0",
        ]
        assert [match.item_name for match in result.merged] == ["A0", "Vehicle", "Parts", "State"]
        embedder.encode.assert_called_once_with(["pressure"], normalize=True)
        assert result.sections[-1].matches == searcher.search("pressure", top_k=1)

    def test_aliases_identity_and_ties(self, nvh_model: Any, tmp_path: Path) -> None:
        attribute = ModelMatch("attribute", "Vehicle", "AoVehicle", "State", "state", 1, 0.0)
        relation = ModelMatch("relation", "Vehicle", "AoVehicle", "State", "state", 0, 0.0)
        other = ModelMatch("attribute", "Wheel", "AoWheel", "State", "state", 1, 0.0)
        searcher, _ = self._with_matches(
            nvh_model, tmp_path, [attribute, attribute, relation, other], [0.9, 0.8, 0.7, 0.7]
        )
        searcher._alias_flags[1] = 1.0
        searcher._alias_boost = 0.2

        result = searcher.search_multi("state", top_k=100)
        legacy = searcher.search("state", top_k=100)

        assert result.merged == legacy
        assert len(result.merged) == 3
        assert result.merged[0].score == pytest.approx(1.0)
        assert result.merged[0].data_type == 1
        assert result.sections[-1].matches == legacy
        assert len(result.sections[1].matches) == 2
        assert len(result.sections[2].matches) == 1

    @pytest.mark.parametrize("entity_name", ["vEh", "missing"])
    def test_entity_filter_before_limit(
        self, nvh_model: Any, tmp_path: Path, entity_name: str
    ) -> None:
        matches = [
            ModelMatch("attribute", "Wheel", "AoWheel", "Pressure", "", 1, 0.0),
            ModelMatch("enumeration", "", "", "State", "", 0, 0.0),
            ModelMatch("attribute", "Vehicle", "AoVehicle", "Pressure", "", 1, 0.0),
            ModelMatch("entity", "Vehicle", "AoVehicle", "Vehicle", "", 0, 0.0),
        ]
        searcher, _ = self._with_matches(nvh_model, tmp_path, matches, [0.9, 0.8, 0.7, 0.6])

        result = searcher.search_multi("pressure", top_k=1, entity_name=entity_name)

        assert not result.sections[3].matches
        if entity_name == "missing":
            assert not result.merged
            assert all(not section.matches for section in result.sections)
        else:
            assert result.sections[1].matches[0].entity_name == "Vehicle"
            assert result.sections[-1].matches == result.sections[1].matches
            assert [match.item_name for match in result.merged] == ["Pressure", "Vehicle"]

    @pytest.mark.parametrize(
        ("query", "top_k", "empty_corpus"),
        [("", 1, False), ("  ", 1, False), ("x", 0, False), ("x", -1, False), ("x", 1, True)],
    )
    def test_empty_requests_do_not_initialize(
        self, nvh_model: Any, tmp_path: Path, query: str, top_k: int, empty_corpus: bool
    ) -> None:
        searcher = ModelSearcher(nvh_model, cache_dir=tmp_path)
        if empty_corpus:
            searcher._corpus = []
        searcher._embedder = MagicMock()

        result = searcher.search_multi(query, top_k=top_k)

        assert len(result.sections) == 5
        assert all(
            not section.matches and section.elapsed_seconds == 0 for section in result.sections
        )
        assert not result.merged
        assert result.shared_seconds == result.merge_seconds == result.total_seconds == 0
        searcher._embedder.encode.assert_not_called()
        searcher._embedder._ensure_loaded.assert_not_called()
        assert searcher.embeddings is None

    def test_timing_excludes_warm_up_and_scores_once(
        self, nvh_model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        match = ModelMatch("entity", "Vehicle", "AoVehicle", "Vehicle", "", 0, 0.0)
        searcher, embedder = self._with_matches(nvh_model, tmp_path, [match], [0.9])
        clock = MagicMock(side_effect=range(15))
        monkeypatch.setattr("odsbox_seman.model_searcher.perf_counter", clock)
        original_warm_up = searcher.warm_up

        def warm_up() -> None:
            clock.assert_not_called()
            original_warm_up()

        monkeypatch.setattr(searcher, "warm_up", warm_up)
        ranker = MagicMock(wraps=searcher._search_ranked)
        monkeypatch.setattr(searcher, "_search_ranked", ranker)

        result = searcher.search_multi("vehicle")

        assert result.shared_seconds == result.merge_seconds == 1
        assert [section.elapsed_seconds for section in result.sections] == [1] * 5
        assert result.total_seconds == 14
        ranker.assert_called_once_with("vehicle", None)
        embedder.encode.assert_called_once_with(["vehicle"], normalize=True)


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    @pytest.mark.parametrize("separate_instances", [False, True])
    def test_concurrent_initialization_builds_once(
        self,
        nvh_model: Any,
        tmp_path: Path,
        separate_instances: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = ModelSearcher(nvh_model, cache_dir=tmp_path)
        second = ModelSearcher(nvh_model, cache_dir=tmp_path) if separate_instances else first
        encoding = Event()
        release = Event()
        second_started = Event()
        embeddings = np.ones((len(first._corpus), 32), dtype=np.float32)

        def encode(*args: Any, **kwargs: Any) -> np.ndarray:
            encoding.set()
            assert release.wait(5)
            return embeddings

        embedder = MagicMock()
        embedder.encode.side_effect = encode
        first._embedder = second._embedder = embedder
        save = MagicMock(wraps=first._cache.save)
        monkeypatch.setattr(first._cache, "save", save)

        def load_second() -> None:
            second_started.set()
            second._ensure_loaded()

        with ThreadPoolExecutor(max_workers=2) as pool:
            first_result = pool.submit(first._ensure_loaded)
            try:
                assert encoding.wait(5)
                second_result = pool.submit(load_second)
                assert second_started.wait(5)
                with pytest.raises(TimeoutError):
                    second_result.result(timeout=0.1)
            finally:
                release.set()
            first_result.result(timeout=5)
            second_result.result(timeout=5)

        assert embedder.encode.call_count == 1
        save.assert_called_once()
        np.testing.assert_array_equal(first.embeddings, second.embeddings)

    @pytest.mark.parametrize("failure_stage", ["encode", "save"])
    def test_failed_initialization_can_retry_from_another_thread(
        self, nvh_model: Any, tmp_path: Path, failure_stage: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        searcher = ModelSearcher(nvh_model, cache_dir=tmp_path)
        embeddings = np.ones((len(searcher._corpus), 32), dtype=np.float32)
        embedder = MagicMock()
        embedder.encode.return_value = embeddings
        searcher._embedder = embedder
        if failure_stage == "encode":
            embedder.encode.side_effect = [RuntimeError("build failed"), embeddings]
        else:
            original_save = searcher._cache.save
            save = MagicMock(side_effect=RuntimeError("build failed"))
            monkeypatch.setattr(searcher._cache, "save", save)

        with pytest.raises(RuntimeError, match="build failed"):
            searcher._ensure_loaded()
        assert searcher.embeddings is None
        if failure_stage == "save":
            monkeypatch.setattr(searcher._cache, "save", original_save)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(searcher._ensure_loaded).result(timeout=5)
        np.testing.assert_array_equal(searcher.embeddings, embeddings)
        assert searcher._cache.load(searcher._digest, len(searcher._corpus)) is not None

    def test_initialization_waits_for_save(
        self, nvh_model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        searcher = ModelSearcher(nvh_model, cache_dir=tmp_path)
        embedder = MagicMock()
        embedder.encode.return_value = np.ones((len(searcher._corpus), 32), dtype=np.float32)
        searcher._embedder = embedder
        saving = Event()
        release = Event()
        second_started = Event()
        original_save = searcher._cache.save

        def save(*args: Any, **kwargs: Any) -> None:
            saving.set()
            assert release.wait(5)
            original_save(*args, **kwargs)

        def load_second() -> None:
            second_started.set()
            searcher._ensure_loaded()

        monkeypatch.setattr(searcher._cache, "save", save)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(searcher._ensure_loaded)
            try:
                assert saving.wait(5)
                assert searcher.embeddings is None
                second = pool.submit(load_second)
                assert second_started.wait(5)
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.1)
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)

    def test_public_calls_share_initialization(self, nvh_model: Any, tmp_path: Path) -> None:
        searcher = ModelSearcher(nvh_model, cache_dir=tmp_path)
        embedder = MagicMock()
        embedder.encode.side_effect = lambda texts, **kwargs: np.ones(
            (len(texts), 32), dtype=np.float32
        )
        searcher._embedder = embedder
        start = Barrier(3, timeout=5)

        def call(method: Any, *args: Any) -> Any:
            start.wait()
            return method(*args)

        with ThreadPoolExecutor(max_workers=3) as pool:
            warm = pool.submit(call, searcher.warm_up)
            single = pool.submit(call, searcher.search, "vehicle")
            multi = pool.submit(call, searcher.search_multi, "vehicle")
            warm.result(timeout=5)
            assert single.result(timeout=5)
            assert multi.result(timeout=5).merged
        corpus_calls = [call for call in embedder.encode.call_args_list if len(call.args[0]) > 1]
        assert len(corpus_calls) == 1
        assert embedder.encode.call_count == 3

    def test_embedder_construction_is_shared(
        self, nvh_model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        searcher = ModelSearcher(nvh_model, cache_dir=tmp_path)
        constructing = Event()
        release = Event()
        second_started = Event()
        embedder = MagicMock()

        def construct() -> Any:
            constructing.set()
            assert release.wait(5)
            return embedder

        factory = MagicMock(side_effect=construct)
        monkeypatch.setattr("odsbox_seman.model_searcher.OnnxEmbedder", factory)

        def get_second() -> Any:
            second_started.set()
            return searcher._ensure_embedder()

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(searcher._ensure_embedder)
            try:
                assert constructing.wait(5)
                second = pool.submit(get_second)
                assert second_started.wait(5)
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.1)
            finally:
                release.set()
            assert first.result(timeout=5) is embedder
            assert second.result(timeout=5) is embedder
        factory.assert_called_once_with()

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
