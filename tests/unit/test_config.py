"""Tests for config loading, schema validation and fingerprinting."""

from __future__ import annotations

import re
import tomllib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from odsbox_seman import AliasConfig, BlacklistConfig, ModelConfig
from odsbox_seman.config import _load_schema, load_config, validate_config_data
from odsbox_seman.types import ConfigError

_SAMPLE = Path(__file__).parent.parent / "data" / "odsbox-seman-config.toml"


def _write(tmp_path: Path, body: str, *, with_model: bool = True) -> Path:
    if with_model:
        (tmp_path / "m.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "odsbox-seman-config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_schema_is_valid() -> None:
    Draft202012Validator.check_schema(_load_schema())


def test_alias_defaults_are_independent() -> None:
    first, second = AliasConfig(), AliasConfig()
    first.entities["Test"] = ("Versuch",)
    first.attributes["Test"] = {"Name": ("Titel",)}
    first.enumerations["Valid"] = {"Yes": ("Ja",)}
    assert second.entities == {}
    assert second.attributes == {}
    assert second.enumerations == {}


@pytest.mark.parametrize("field_name", ["entities", "attributes"])
def test_section_fields_are_frozen(field_name: str) -> None:
    aliases = AliasConfig()
    blacklist = BlacklistConfig()
    with pytest.raises(FrozenInstanceError):
        setattr(aliases, field_name, {})
    with pytest.raises(FrozenInstanceError):
        setattr(blacklist, field_name, ("Tpl*",))
    assert blacklist.entities == blacklist.attributes == ()


def test_model_section_defaults_are_independent() -> None:
    first = ModelConfig(name="A", model_file=Path("a.json"))
    second = ModelConfig(name="B", model_file=Path("b.json"))
    assert first.aliases == second.aliases == AliasConfig()
    assert first.blacklist == second.blacklist == BlacklistConfig()
    assert first.aliases is not second.aliases
    assert first.blacklist is not second.blacklist
    first.aliases.entities["Test"] = ("Versuch",)
    assert second.aliases.entities == {}


def test_loaded_models_do_not_share_alias_maps(tmp_path: Path) -> None:
    body = (
        '[settings.aliases.entities]\nTest = ["Versuch"]\n'
        '[settings.aliases.attributes.Test]\nName = ["Titel"]\n'
        '[settings.aliases.enumerations.Valid]\nYes = ["Ja"]\n'
        '[models.A]\nmodel_file = "m.json"\n'
        '[models.B]\nmodel_file = "m.json"\n'
    )
    cfg = load_config(_write(tmp_path, body))
    first, second = cfg.models["A"], cfg.models["B"]
    first.aliases.entities["Test"] = ("Changed",)
    first.aliases.attributes["Test"]["Name"] = ("Changed",)
    first.aliases.enumerations["Valid"]["Yes"] = ("Changed",)
    assert second.aliases == AliasConfig(
        entities={"Test": ("Versuch",)},
        attributes={"Test": {"Name": ("Titel",)}},
        enumerations={"Valid": {"Yes": ("Ja",)}},
    )


def test_missing_sections_load_as_dataclasses(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, '[models.A]\nmodel_file = "m.json"\n'))
    assert cfg.models["A"].aliases == AliasConfig()
    assert cfg.models["A"].blacklist == BlacklistConfig()


def test_fingerprint_normalization_and_excluded_fields() -> None:
    model = ModelConfig(
        name="A",
        model_file=Path("a.json"),
        blacklist=BlacklistConfig(entities=("Tpl*", "Cat*"), attributes=("ID", "version")),
    )
    equivalent = replace(
        model,
        name="B",
        model_file=Path("b.json"),
        alias_boost=0.5,
        blacklist=BlacklistConfig(entities=("cat*", "tpl*"), attributes=("VERSION", "id")),
    )
    assert model.fingerprint() == equivalent.fingerprint()


def test_sample_config_loads() -> None:
    cfg = load_config(_SAMPLE)
    model = cfg.models["NVH"]
    assert cfg.default_model == "NVH"
    assert model.model_file is not None
    assert model.model_file.is_file()
    assert "Tpl*" in model.blacklist.entities
    assert model.aliases.entities["MeaQuantity"] == ("Verlauf", "Signal")
    assert model.aliases.attributes["Vehicle"]["description"] == ("Comment",)
    assert model.alias_boost == 0.1


def test_model_file_is_relative_to_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, '[models.A]\nmodel_file = "m.json"\n'))
    assert cfg.models["A"].model_file == (tmp_path / "m.json").resolve()


def test_model_file_defaults_to_none() -> None:
    model = ModelConfig(name="A")
    assert model.model_file is None
    assert model.fingerprint() == replace(model, model_file=Path("a.json")).fingerprint()


def test_config_without_model_file_loads(tmp_path: Path) -> None:
    body = (
        '[settings.aliases.entities]\nTest = ["Versuch"]\n'
        "[models.A]\nalias_boost = 0.2\n"
        '[models.A.blacklist]\nentities = ["Tpl*"]\n'
    )
    cfg = load_config(_write(tmp_path, body, with_model=False))
    model = cfg.models["A"]
    assert model.model_file is None
    assert cfg.default_model == "A"
    assert model.alias_boost == 0.2
    assert model.aliases.entities == {"Test": ("Versuch",)}
    assert model.blacklist.entities == ("Tpl*",)


@pytest.mark.parametrize("value", [None, "", 1])
def test_explicit_model_file_must_be_nonempty_string(value: object) -> None:
    errors = validate_config_data({"models": {"A": {"model_file": value}}})
    assert any("models.A.model_file" in error for error in errors)


def test_default_model_is_first_and_boost_precedence(tmp_path: Path) -> None:
    body = (
        "[settings]\nalias_boost = 0.3\n"
        '[models.A]\nmodel_file = "m.json"\n'
        '[models.B]\nmodel_file = "m.json"\nalias_boost = 0.5\n'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.default_model == "A"
    assert cfg.models["A"].alias_boost == 0.3
    assert cfg.models["B"].alias_boost == 0.5


def test_all_schema_errors_are_reported_sorted() -> None:
    errors = validate_config_data(
        {
            "typo": 1,
            "models": {
                "B": {"model_file": 1, "aliases": {"entities": {"E": []}}},
                "A": {"model_file": "x", "blacklist": {"entities": ["ok", ""], "nope": 1}},
            },
        }
    )
    assert len(errors) == 5
    assert errors.index(next(e for e in errors if e.startswith("models.A"))) < errors.index(
        next(e for e in errors if e.startswith("models.B"))
    )
    joined = "\n".join(errors)
    for expected in ("typo", "models.B.model_file", "models.B.aliases.entities.E", "nope"):
        assert expected in joined
    assert any(e.startswith("models.A.blacklist.entities[1]") for e in errors)


def test_global_aliases_merge_case_insensitively_without_duplicates(tmp_path: Path) -> None:
    body = (
        "[settings.aliases.entities]\n"
        'Test = ["Versuch", "Run"]\n'
        "[settings.aliases.attributes.Test]\n"
        'Name = ["Titel"]\n'
        '[models.A]\nmodel_file = "m.json"\n'
        "[models.A.aliases.entities]\n"
        'test = ["versuch", "Experiment"]\n'
        "[models.A.aliases.attributes.test]\n"
        'name = ["Bezeichnung"]\n'
        '[models.B]\nmodel_file = "m.json"\n'
    )
    cfg = load_config(_write(tmp_path, body))
    a, b = cfg.models["A"], cfg.models["B"]
    assert a.aliases.entities == {"Test": ("Versuch", "Run", "Experiment")}
    assert a.aliases.attributes == {"Test": {"Name": ("Titel", "Bezeichnung")}}
    assert b.aliases.entities == {"Test": ("Versuch", "Run")}
    assert a.fingerprint() != b.fingerprint()


def test_config_doc_examples_are_valid() -> None:
    doc = Path(__file__).parents[2] / "docs" / "CONFIG.md"
    blocks = re.findall(r"```toml\n(.*?)```", doc.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks
    for block in blocks:
        assert validate_config_data(tomllib.loads(block)) == []


def test_toml_syntax_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot parse TOML"):
        load_config(_write(tmp_path, "[models.A\n"))


def test_missing_models_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, ""))
    assert "models" in exc.value.errors[0]


def test_semantic_errors_are_collected(tmp_path: Path) -> None:
    body = '[settings]\ndefault_model = "Z"\n[models.A]\nmodel_file = "missing.json"\n'
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body, with_model=False))
    assert len(exc.value.errors) == 2
    assert any("unknown model 'Z'" in e for e in exc.value.errors)
    assert any("file not found" in e for e in exc.value.errors)


def test_fingerprint_ignores_boost_and_path_but_not_content(tmp_path: Path) -> None:
    base = '[models.A]\nmodel_file = "m.json"\n'
    a = load_config(_write(tmp_path, base)).models["A"]
    b = load_config(_write(tmp_path, base + "alias_boost = 0.4\n")).models["A"]
    c = load_config(_write(tmp_path, base + '[models.A.aliases.entities]\nTest = ["x"]\n')).models[
        "A"
    ]
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_blacklist_matching_is_case_insensitive(tmp_path: Path) -> None:
    body = '[models.A]\nmodel_file = "m.json"\n[models.A.blacklist]\nentities = ["Tpl*"]\n'
    model = load_config(_write(tmp_path, body)).models["A"]
    assert model.is_entity_blacklisted("TPLGroup", "")
    assert model.is_entity_blacklisted("x", "tplBase")
    assert not model.is_entity_blacklisted("Test", "AoTest")


@pytest.mark.parametrize(
    ("sections", "expected"),
    [
        ("", "1b9052056b02968f2569622c1490f4f92dffcc7395f192813ec5ca2864f56948"),
        (
            '[models.A.blacklist]\nentities = ["Tpl*", "Cat*"]\n'
            'attributes = ["version", "ID"]\n'
            '[models.A.aliases.entities]\nTest = ["Versuch", "Run"]\n'
            '[models.A.aliases.attributes.Test]\nName = ["Titel"]\n'
            '[models.A.aliases.enumerations.valid_enum]\nvalid = ["gueltig"]\n',
            "76a4569bb55badacbdae065e45365e07370cd3344d53a023a1517cfb61666e29",
        ),
    ],
)
def test_fingerprint_preserves_cache_compatibility(
    tmp_path: Path, sections: str, expected: str
) -> None:
    model = load_config(_write(tmp_path, '[models.A]\nmodel_file = "m.json"\n' + sections))
    assert model.models["A"].fingerprint() == expected


def test_global_enumeration_aliases_merge_without_duplicates(tmp_path: Path) -> None:
    body = (
        '[settings.aliases.enumerations.Valid]\nYes = ["Ja", "True"]\n'
        '[models.A]\nmodel_file = "m.json"\n'
        '[models.A.aliases.enumerations.valid]\nyes = ["ja", "Oui"]\n'
        '[models.B]\nmodel_file = "m.json"\n'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.models["A"].aliases.enumerations == {"Valid": {"Yes": ("Ja", "True", "Oui")}}
    assert cfg.models["B"].aliases.enumerations == {"Valid": {"Yes": ("Ja", "True")}}
