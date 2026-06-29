"""Unit tests for CacheManager (disk cache)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from odsbox_seman.cache import CacheManager, model_hash
from odsbox_seman.types import ModelMatch


def _make_corpus(n: int) -> list[tuple[str, ModelMatch]]:
    return [
        (
            f"entity attr text {i}",
            ModelMatch(
                kind="attribute",
                entity_name=f"Entity{i}",
                entity_base_name=f"AoEntity{i}",
                item_name=f"Attr{i}",
                item_base_name=f"ao_attr_{i}",
                data_type=1,
                score=0.0,
            ),
        )
        for i in range(n)
    ]


def _make_embeddings(n: int, dim: int = 32) -> Any:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestModelHash:
    def test_same_model_same_hash(self, nvh_model: Any) -> None:
        h1 = model_hash(nvh_model)
        h2 = model_hash(nvh_model)
        assert h1 == h2

    def test_hash_is_hex_string(self, nvh_model: Any) -> None:
        h = model_hash(nvh_model)
        assert len(h) == 64  # SHA-256 hex = 64 chars
        int(h, 16)  # must be valid hex


class TestCacheManager:
    def test_save_creates_files(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(5)
        emb = _make_embeddings(5)
        cache.save("deadbeef", emb, corpus)
        assert (tmp_path / "deadbeef.npz").exists()
        assert (tmp_path / "deadbeef.json").exists()

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        assert cache.load("nosuchkey", 5) is None

    def test_roundtrip_embeddings(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(10)
        emb = _make_embeddings(10)
        cache.save("abc123", emb, corpus)
        result = cache.load("abc123", 10)
        assert result is not None
        loaded_emb, loaded_corpus = result
        np.testing.assert_allclose(loaded_emb, emb, rtol=1e-5)

    def test_roundtrip_corpus_text(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(3)
        emb = _make_embeddings(3)
        cache.save("abc", emb, corpus)
        result = cache.load("abc", 3)
        assert result is not None
        _, loaded_corpus = result
        assert len(loaded_corpus) == 3
        for i, (text, match) in enumerate(loaded_corpus):
            assert f"text {i}" in text
            assert match.entity_name == f"Entity{i}"

    def test_roundtrip_corpus_kinds(self, tmp_path: Path) -> None:
        corpus: list[tuple[str, ModelMatch]] = [
            (
                "rel text",
                ModelMatch(
                    kind="relation",
                    entity_name="E",
                    entity_base_name="AoE",
                    item_name="R",
                    item_base_name="ao_r",
                    data_type=0,
                    score=0.0,
                ),
            ),
            (
                "enum text",
                ModelMatch(
                    kind="enumeration",
                    entity_name="",
                    entity_base_name="",
                    item_name="MyEnum",
                    item_base_name="",
                    data_type=0,
                    score=0.0,
                ),
            ),
        ]
        cache = CacheManager(tmp_path)
        emb = _make_embeddings(2)
        cache.save("kinds", emb, corpus)
        result = cache.load("kinds", 2)
        assert result is not None
        _, loaded_corpus = result
        assert loaded_corpus[0][1].kind == "relation"
        assert loaded_corpus[1][1].kind == "enumeration"

    def test_length_mismatch_returns_none(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(5)
        emb = _make_embeddings(5)
        cache.save("mismatch", emb, corpus)
        # Ask for 10 entries but only 5 were saved → mismatch
        assert cache.load("mismatch", 10) is None

    def test_npz_json_length_mismatch_returns_none(self, tmp_path: Path) -> None:
        """Corrupt the JSON to have fewer entries than the NPZ."""
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(5)
        emb = _make_embeddings(5)
        cache.save("corrupt", emb, corpus)
        json_path = tmp_path / "corrupt.json"
        raw = json.loads(json_path.read_text())
        # Remove two entries to create mismatch
        json_path.write_text(json.dumps(raw[:3]))
        assert cache.load("corrupt", 5) is None

    def test_corrupted_npz_returns_none(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(3)
        emb = _make_embeddings(3)
        cache.save("bad_npz", emb, corpus)
        # Overwrite the NPZ with garbage
        (tmp_path / "bad_npz.npz").write_bytes(b"not a numpy file")
        assert cache.load("bad_npz", 3) is None

    def test_missing_json_returns_none(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(3)
        emb = _make_embeddings(3)
        cache.save("nojson", emb, corpus)
        (tmp_path / "nojson.json").unlink()
        assert cache.load("nojson", 3) is None

    def test_cache_dir_created_on_save(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "cache"
        cache = CacheManager(new_dir)
        corpus = _make_corpus(2)
        emb = _make_embeddings(2)
        cache.save("new_dir_test", emb, corpus)
        assert new_dir.exists()

    def test_default_cache_dir(self) -> None:
        """CacheManager uses ~/.ods-seman/search_cache/ by default."""
        cache = CacheManager()
        expected = Path.home() / ".ods-seman" / "search_cache"
        assert cache._dir == expected

    def test_score_reset_on_load(self, tmp_path: Path) -> None:
        """Loaded ModelMatch entries always have score=0.0 (scores are transient)."""
        corpus = _make_corpus(2)
        emb = _make_embeddings(2)
        cache = CacheManager(tmp_path)
        cache.save("score_reset", emb, corpus)
        result = cache.load("score_reset", 2)
        assert result is not None
        _, loaded = result
        for _, match in loaded:
            assert match.score == 0.0

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus1 = _make_corpus(2)
        corpus2 = _make_corpus(3)
        emb1 = _make_embeddings(2)
        emb2 = _make_embeddings(3)
        cache.save("over", emb1, corpus1)
        cache.save("over", emb2, corpus2)
        # Should now load the newer (3-entry) corpus
        result = cache.load("over", 3)
        assert result is not None
        _, loaded = result
        assert len(loaded) == 3

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_various_sizes(self, tmp_path: Path, n: int) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(n)
        emb = _make_embeddings(n)
        cache.save(f"size_{n}", emb, corpus)
        result = cache.load(f"size_{n}", n)
        assert result is not None
        _, loaded = result
        assert len(loaded) == n
