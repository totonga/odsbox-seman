"""Unit tests for ModelMatch and SemanticSearchUnavailableError."""

from __future__ import annotations

from odsbox_seman.types import ModelMatch, SemanticSearchUnavailableError


class TestModelMatch:
    def test_attribute_kind(self) -> None:
        m = ModelMatch(
            kind="attribute",
            entity_name="MeaResult",
            entity_base_name="AoMeasurement",
            item_name="Name",
            item_base_name="ao_name",
            data_type=1,
            score=0.9,
        )
        assert m.kind == "attribute"
        assert m.entity_name == "MeaResult"
        assert m.score == 0.9

    def test_relation_kind(self) -> None:
        m = ModelMatch(
            kind="relation",
            entity_name="MeaResult",
            entity_base_name="AoMeasurement",
            item_name="Channels",
            item_base_name="ao_channels",
            data_type=0,
            score=0.5,
        )
        assert m.kind == "relation"
        assert m.data_type == 0

    def test_enumeration_kind(self) -> None:
        m = ModelMatch(
            kind="enumeration",
            entity_name="",
            entity_base_name="",
            item_name="DataTypeEnum",
            item_base_name="",
            data_type=0,
            score=0.7,
        )
        assert m.kind == "enumeration"
        assert m.entity_name == ""

    def test_to_dict_roundtrip(self) -> None:
        m = ModelMatch(
            kind="attribute",
            entity_name="E",
            entity_base_name="AoE",
            item_name="A",
            item_base_name="ao_a",
            data_type=2,
            score=0.8,
        )
        d = m.to_dict()
        assert d["kind"] == "attribute"
        assert d["entity_name"] == "E"
        assert d["score"] == 0.8
        assert d["data_type"] == 2

    def test_to_dict_has_all_fields(self) -> None:
        m = ModelMatch(
            kind="relation",
            entity_name="A",
            entity_base_name="B",
            item_name="C",
            item_base_name="D",
            data_type=0,
            score=0.1,
        )
        d = m.to_dict()
        expected_keys = {
            "kind",
            "entity_name",
            "entity_base_name",
            "item_name",
            "item_base_name",
            "data_type",
            "score",
        }
        assert set(d.keys()) == expected_keys


class TestSemanticSearchUnavailableError:
    def test_is_runtime_error(self) -> None:
        err = SemanticSearchUnavailableError("install with: uv add onnxruntime")
        assert isinstance(err, RuntimeError)

    def test_hint_attribute(self) -> None:
        hint = "some helpful message"
        err = SemanticSearchUnavailableError(hint)
        assert err.hint == hint

    def test_str_representation(self) -> None:
        hint = "install onnxruntime"
        err = SemanticSearchUnavailableError(hint)
        assert hint in str(err)
