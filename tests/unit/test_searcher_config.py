"""Tests for config-driven blacklists and aliases in ModelSearcher."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from odsbox_seman.config import (
    DEFAULT_ALIAS_BOOST,
    AliasConfig,
    BlacklistConfig,
    ModelConfig,
    load_config,
)
from odsbox_seman.model_searcher import ModelSearcher, _lookup, _lookup_nested

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "odsbox-seman-config.toml"


class _ConstantEmbedder:
    """Maps every text to the same unit vector so scores are fully controlled by boosts."""

    def encode(self, texts: list[str], normalize: bool = True) -> Any:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _searcher(
    model: Any,
    *,
    aliases: AliasConfig | None = None,
    blacklist: BlacklistConfig | None = None,
    alias_boost: float = DEFAULT_ALIAS_BOOST,
) -> ModelSearcher:
    cfg = ModelConfig(
        name="m",
        model_file=Path("m.json"),
        aliases=aliases if aliases is not None else AliasConfig(),
        blacklist=blacklist if blacklist is not None else BlacklistConfig(),
        alias_boost=alias_boost,
    )
    return ModelSearcher(model, config=cfg)


def _with_flat_embeddings(searcher: ModelSearcher) -> ModelSearcher:
    n = len(searcher._corpus)
    searcher._embeddings = np.tile(np.array([[0.5, 0.0]], dtype=np.float32), (n, 1))
    searcher._embedder = _ConstantEmbedder()  # type: ignore[assignment]
    return searcher


class TestBlacklist:
    def test_entity_pattern_matches_name_case_insensitively(self, nvh_model: Any) -> None:
        searcher = _searcher(nvh_model, blacklist=BlacklistConfig(entities=("tpl*",)))
        names = {m.entity_name for _, m in searcher._corpus}
        assert not any(n.lower().startswith("tpl") for n in names)
        assert any(m.entity_name for _, m in searcher._corpus)

    def test_entity_pattern_matches_base_name(self, nvh_model: Any) -> None:
        entity = next(e for e in nvh_model.entities.values() if e.name != e.base_name)
        searcher = _searcher(nvh_model, blacklist=BlacklistConfig(entities=(entity.base_name,)))
        assert entity.name not in {m.entity_name for _, m in searcher._corpus}

    def test_attribute_pattern(self, nvh_model: Any) -> None:
        searcher = _searcher(nvh_model, blacklist=BlacklistConfig(attributes=("version",)))
        attrs = [m for _, m in searcher._corpus if m.kind == "attribute"]
        assert attrs
        assert not any("version" in (m.item_name.lower(), m.item_base_name.lower()) for m in attrs)

    def test_no_config_keeps_everything(self, nvh_model: Any) -> None:
        plain = ModelSearcher(nvh_model)
        filtered = _searcher(nvh_model, blacklist=BlacklistConfig(entities=("Tpl*",)))
        assert len(filtered._corpus) < len(plain._corpus)


class TestAliases:
    def test_fileless_config_applies_to_external_model(
        self, nvh_model: Any, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.toml"
        path.write_text(
            '[models.NVH.aliases.entities]\nTest = ["Versuch"]\n'
            '[models.NVH.blacklist]\nentities = ["Tpl*"]\n',
            encoding="utf-8",
        )
        config = load_config(path).models["NVH"]
        searcher = _with_flat_embeddings(
            ModelSearcher(nvh_model, config=config, cache_dir=tmp_path)
        )
        assert config.model_file is None
        assert searcher.search("Versuch", top_k=1)[0].item_name == "Test"
        assert not any(match.entity_name.startswith("Tpl") for _, match in searcher._corpus)

    def test_entity_alias_adds_flagged_entries(self, nvh_model: Any) -> None:
        searcher = _searcher(nvh_model, aliases=AliasConfig(entities={"mearesult": ("Messung",)}))
        flagged = [
            m for (_, m), f in zip(searcher._corpus, searcher._alias_flags, strict=True) if f
        ]
        assert [m.kind for m in flagged] == ["entity"]
        assert flagged[0].item_name == "MeaResult"
        assert any("messung" in t.lower() for t, _ in searcher._corpus)

    def test_entity_alias_enriches_attribute_text(self, nvh_model: Any) -> None:
        searcher = _searcher(nvh_model, aliases=AliasConfig(entities={"Test": ("Versuch",)}))
        texts = [
            t for t, m in searcher._corpus if m.kind == "attribute" and m.entity_name == "Test"
        ]
        assert texts
        assert all("versuch" in t.lower() for t in texts)

    def test_attribute_alias(self, nvh_model: Any) -> None:
        searcher = _searcher(
            nvh_model,
            aliases=AliasConfig(attributes={"vehicle": {"MANUFACTURER": ("Hersteller",)}}),
        )
        flagged = [
            m for (_, m), f in zip(searcher._corpus, searcher._alias_flags, strict=True) if f
        ]
        assert flagged
        assert all(m.kind == "attribute" and m.entity_name == "vehicle" for m in flagged)

    def test_enum_alias(self, nvh_model: Any) -> None:
        enum = next(e for e in nvh_model.enumerations.values() if len(e.items))
        item = next(iter(enum.items))
        searcher = _searcher(
            nvh_model, aliases=AliasConfig(enumerations={enum.name: {item: ("Fremdwort",)}})
        )
        flagged = [
            m for (_, m), f in zip(searcher._corpus, searcher._alias_flags, strict=True) if f
        ]
        assert len(flagged) == 1
        assert flagged[0].kind == "enumeration"
        assert flagged[0].item_name == enum.name
        assert flagged[0].item_base_name == item

    def test_alias_entries_rank_above_plain_entries(self, nvh_model: Any) -> None:
        searcher = _with_flat_embeddings(
            _searcher(
                nvh_model, aliases=AliasConfig(entities={"Test": ("Versuch",)}), alias_boost=0.2
            )
        )
        results = searcher.search("Versuch", top_k=5)
        assert results[0].kind == "entity"
        assert results[0].item_name == "Test"
        assert results[0].score > results[1].score

    def test_search_deduplicates_items(self, nvh_model: Any) -> None:
        searcher = _with_flat_embeddings(
            _searcher(nvh_model, aliases=AliasConfig(entities={"Test": ("Versuch", "Experiment")}))
        )
        results = searcher.search("x", top_k=len(searcher._corpus))
        keys = [(m.kind, m.entity_name, m.item_name, m.item_base_name) for m in results]
        assert len(keys) == len(set(keys))

    def test_boost_is_clamped_to_one(self, nvh_model: Any) -> None:
        searcher = _with_flat_embeddings(
            _searcher(
                nvh_model, aliases=AliasConfig(entities={"Test": ("Versuch",)}), alias_boost=1.0
            )
        )
        searcher._embeddings = np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(searcher._corpus), 1)
        )
        assert searcher.search("Versuch", top_k=1)[0].score <= 1.0


class TestLookupUnion:
    def test_name_and_base_name_aliases_are_combined(self) -> None:
        folded = {"test": ("Versuch",), "aotest": ("Run", "versuch")}
        assert _lookup(folded, "Test", "AoTest") == ("Versuch", "Run", "versuch")

    def test_nested_lookup_merges_keys_case_insensitively(self) -> None:
        folded: dict[str, dict[str, tuple[str, ...]]] = {
            "test": {"Name": ("A",)},
            "aotest": {"name": ("B",), "id": ("C",)},
        }
        assert _lookup_nested(folded, "Test", "AoTest") == {"name": ("A", "B"), "id": ("C",)}


class TestStats:
    def test_counts_reflect_blacklists(self, nvh_model: Any) -> None:
        plain = ModelSearcher(nvh_model).stats
        filtered = _searcher(
            nvh_model, blacklist=BlacklistConfig(entities=("Tpl*",), attributes=("version",))
        ).stats
        assert plain.entities_kept == plain.entities_total == len(nvh_model.entities)
        assert plain.excluded_entities == ()
        assert filtered.entities_total == plain.entities_total
        assert filtered.entities_kept == filtered.entities_total - len(filtered.excluded_entities)
        assert filtered.excluded_entities
        assert all(n.lower().startswith("tpl") for n in filtered.excluded_entities)
        assert filtered.attributes_kept < filtered.attributes_total
        assert all(a.lower().endswith(".version") for a in filtered.excluded_attributes)

    def test_unmatched_blacklist_pattern_is_warned(
        self, nvh_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="odsbox_seman"):
            _searcher(nvh_model, blacklist=BlacklistConfig(entities=("NoSuchEntity*",)))
        assert "NoSuchEntity*" in caplog.text

    def test_unmatched_attribute_pattern_is_warned(
        self, nvh_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="odsbox_seman"):
            _searcher(nvh_model, blacklist=BlacklistConfig(attributes=("NoSuchAttribute*",)))
        assert "attribute blacklist 'NoSuchAttribute*' matches nothing" in caplog.text

    def test_unmatched_alias_keys_are_logged_at_info(
        self, nvh_model: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="odsbox_seman"):
            _searcher(
                nvh_model,
                aliases=AliasConfig(
                    entities={"NoSuchEntity": ("Alias",)},
                    attributes={"NoSuchOwner": {"Name": ("Title",)}},
                ),
            )
        messages = [record for record in caplog.records if "matches no entity" in record.message]
        assert len(messages) == 2
        assert all(record.levelno == logging.INFO for record in messages)
        assert "NoSuchEntity" in caplog.text
        assert "NoSuchOwner" in caplog.text


class TestSampleConfig:
    def test_sample_config_builds_corpus(self, nvh_model: Any) -> None:
        model_cfg = load_config(_CONFIG_PATH).models["NVH"]
        searcher = ModelSearcher(nvh_model, config=model_cfg)
        entities = {m.entity_name for _, m in searcher._corpus}
        assert not any(n.lower().startswith(("tpl", "cat", "rights")) for n in entities)
        assert "MeaResult" in entities
        assert searcher._alias_flags.sum() > 0
