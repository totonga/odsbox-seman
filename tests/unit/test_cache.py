"""Unit tests for CacheManager (disk cache)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pytest

from odsbox_seman.cache import CacheManager, model_hash
from odsbox_seman.config import AliasConfig, BlacklistConfig, ModelConfig
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
    def test_lock_is_shared_for_normalized_paths(self, tmp_path: Path) -> None:
        first = CacheManager(tmp_path)
        second = CacheManager(tmp_path / "nested" / "..")
        started = Event()

        def acquire() -> None:
            started.set()
            with second.lock("shared"):
                pass

        with ThreadPoolExecutor(max_workers=1) as pool:
            with first.lock("shared"), first.lock("shared"):
                result = pool.submit(acquire)
                assert started.wait(5)
                with pytest.raises(TimeoutError):
                    result.result(timeout=0.1)
            result.result(timeout=5)

    @pytest.mark.parametrize("different_directory", [False, True])
    def test_different_keys_do_not_block(self, tmp_path: Path, different_directory: bool) -> None:
        first = CacheManager(tmp_path)
        second = CacheManager(tmp_path / "other" if different_directory else tmp_path)
        digest = "same" if different_directory else "other"

        def save() -> None:
            second.save(digest, _make_embeddings(2), _make_corpus(2))

        with ThreadPoolExecutor(max_workers=1) as pool, first.lock("same"):
            pool.submit(save).result(timeout=5)

    def test_load_waits_for_complete_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = CacheManager(tmp_path)
        reader = CacheManager(tmp_path)
        writer.save("shared", _make_embeddings(2), _make_corpus(2))
        writing = Event()
        release = Event()
        reading = Event()
        original_write = Path.write_text

        def write(path: Path, *args: Any, **kwargs: Any) -> int:
            writing.set()
            assert release.wait(5)
            return original_write(path, *args, **kwargs)

        def load() -> Any:
            reading.set()
            return reader.load("shared", 3)

        monkeypatch.setattr(Path, "write_text", write)
        with ThreadPoolExecutor(max_workers=2) as pool:
            saved = pool.submit(writer.save, "shared", _make_embeddings(3), _make_corpus(3))
            try:
                assert writing.wait(5)
                loaded = pool.submit(load)
                assert reading.wait(5)
                with pytest.raises(TimeoutError):
                    loaded.result(timeout=0.1)
            finally:
                release.set()
            saved.result(timeout=5)
            result = loaded.result(timeout=5)
        assert result is not None
        assert len(result[1]) == 3

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
        loaded_emb, loaded_corpus, _ = result
        np.testing.assert_allclose(loaded_emb, emb, rtol=1e-5)

    def test_roundtrip_corpus_text(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(3)
        emb = _make_embeddings(3)
        cache.save("abc", emb, corpus)
        result = cache.load("abc", 3)
        assert result is not None
        _, loaded_corpus, _ = result
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
        _, loaded_corpus, _ = result
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

    def test_env_cache_dir_created_on_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "nested" / "search_cache"
        monkeypatch.setenv("ODSBOX_SEMAN_CACHE_DIR", str(cache_dir))

        cache = CacheManager()
        cache.save("env_cache", _make_embeddings(2), _make_corpus(2))

        assert cache._dir == cache_dir
        assert (cache_dir / "env_cache.npz").exists()
        assert (cache_dir / "env_cache.json").exists()

    def test_explicit_cache_dir_overrides_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_cache_dir = tmp_path / "environment"
        explicit_cache_dir = tmp_path / "explicit"
        monkeypatch.setenv("ODSBOX_SEMAN_CACHE_DIR", str(env_cache_dir))

        cache = CacheManager(explicit_cache_dir)

        assert cache._dir == explicit_cache_dir

    def test_default_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CacheManager uses ~/.ods-seman/search_cache/ by default."""
        monkeypatch.delenv("ODSBOX_SEMAN_CACHE_DIR", raising=False)
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
        _, loaded, _ = result
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
        _, loaded, _ = result
        assert len(loaded) == 3

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_various_sizes(self, tmp_path: Path, n: int) -> None:
        cache = CacheManager(tmp_path)
        corpus = _make_corpus(n)
        emb = _make_embeddings(n)
        cache.save(f"size_{n}", emb, corpus)
        result = cache.load(f"size_{n}", n)
        assert result is not None
        _, loaded, _ = result
        assert len(loaded) == n

    def test_alias_flags_roundtrip(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        flags = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        cache.save("flags", _make_embeddings(3), _make_corpus(3), flags)
        result = cache.load("flags", 3)
        assert result is not None
        np.testing.assert_array_equal(result[2], flags)

    def test_old_cache_without_alias_flags_is_a_miss(self, tmp_path: Path) -> None:
        cache = CacheManager(tmp_path)
        cache.save("old", _make_embeddings(2), _make_corpus(2))
        np.savez_compressed(str(tmp_path / "old.npz"), embeddings=_make_embeddings(2))
        assert cache.load("old", 2) is None


class TestModelHashWithConfig:
    def test_config_changes_hash(self, nvh_model: Any) -> None:
        base = ModelConfig(name="m", model_file=Path("m.json"))
        with_alias = ModelConfig(
            name="m",
            model_file=Path("m.json"),
            aliases=AliasConfig(entities={"Test": ("Versuch",)}),
        )
        assert model_hash(nvh_model, base) != model_hash(nvh_model)
        assert model_hash(nvh_model, base) != model_hash(nvh_model, with_alias)

    def test_alias_boost_does_not_change_hash(self, nvh_model: Any) -> None:
        a = ModelConfig(name="m", model_file=Path("m.json"), alias_boost=0.1)
        b = ModelConfig(name="m", model_file=Path("m.json"), alias_boost=0.3)
        assert model_hash(nvh_model, a) == model_hash(nvh_model, b)

    def test_blacklist_changes_hash(self, nvh_model: Any) -> None:
        a = ModelConfig(name="m", model_file=Path("m.json"))
        b = ModelConfig(
            name="m", model_file=Path("m.json"), blacklist=BlacklistConfig(entities=("Tpl*",))
        )
        assert model_hash(nvh_model, a) != model_hash(nvh_model, b)
