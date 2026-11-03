"""Configuration file (TOML) for odsbox-seman.

The file is validated against ``odsbox-seman-config.schema.json``; all problems are
reported at once in a single :class:`~odsbox_seman.types.ConfigError`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from odsbox_seman.types import ConfigError

log = logging.getLogger(__name__)

DEFAULT_ALIAS_BOOST = 0.1
_SCHEMA_NAME = "odsbox-seman-config.schema.json"

AliasMap = dict[str, tuple[str, ...]]
NestedAliasMap = dict[str, dict[str, tuple[str, ...]]]


def _matches_any(patterns: tuple[str, ...], *names: str) -> bool:
    lowered = [n.lower() for n in names if n]
    return any(fnmatch.fnmatchcase(n, p.lower()) for p in patterns for n in lowered)


@dataclass(frozen=True)
class AliasConfig:
    """Search aliases grouped by kind; contained dictionaries remain mutable."""

    entities: AliasMap = field(default_factory=dict)
    attributes: NestedAliasMap = field(default_factory=dict)
    enumerations: NestedAliasMap = field(default_factory=dict)


@dataclass(frozen=True)
class BlacklistConfig:
    """Case-insensitive glob patterns for excluded model elements."""

    entities: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    """Settings for one named ODS model."""

    name: str
    model_file: Path | None = None
    alias_boost: float = DEFAULT_ALIAS_BOOST
    aliases: AliasConfig = field(default_factory=AliasConfig)
    blacklist: BlacklistConfig = field(default_factory=BlacklistConfig)

    def is_entity_blacklisted(self, name: str, base_name: str) -> bool:
        return _matches_any(self.blacklist.entities, name, base_name)

    def is_attribute_blacklisted(self, name: str, base_name: str) -> bool:
        return _matches_any(self.blacklist.attributes, name, base_name)

    def fingerprint(self) -> str:
        """SHA-256 over everything that changes the embedded texts.

        ``alias_boost`` and ``model_file`` are excluded: the boost is applied at
        search time and the model content is hashed separately.
        """
        payload = {
            "entity_blacklist": sorted(p.lower() for p in self.blacklist.entities),
            "attribute_blacklist": sorted(p.lower() for p in self.blacklist.attributes),
            "entity_aliases": self.aliases.entities,
            "attribute_aliases": self.aliases.attributes,
            "enum_aliases": self.aliases.enumerations,
        }
        raw = json.dumps(payload, sort_keys=True, default=list)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanConfig:
    """All configured models of a config file."""

    path: Path
    models: dict[str, ModelConfig]
    default_model: str


def _format_path(parts: Any) -> str:
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "<root>"


def _load_schema() -> dict[str, Any]:
    text = resources.files("odsbox_seman").joinpath(_SCHEMA_NAME).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def validate_config_data(data: dict[str, Any]) -> list[str]:
    """Return every schema violation in *data*, ordered by location."""
    validator = Draft202012Validator(_load_schema())
    errors = sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.absolute_path])
    return [f"{_format_path(e.absolute_path)}: {e.message}" for e in errors]


def _alias_map(raw: dict[str, Any]) -> AliasMap:
    return {k: tuple(v) for k, v in raw.items()}


def _nested_alias_map(raw: dict[str, Any]) -> NestedAliasMap:
    return {k: _alias_map(v) for k, v in raw.items()}


def _alias_config(raw: dict[str, Any]) -> AliasConfig:
    return AliasConfig(
        entities=_alias_map(raw.get("entities", {})),
        attributes=_nested_alias_map(raw.get("attributes", {})),
        enumerations=_nested_alias_map(raw.get("enumerations", {})),
    )


def _blacklist_config(raw: dict[str, Any]) -> BlacklistConfig:
    return BlacklistConfig(
        entities=tuple(raw.get("entities", ())),
        attributes=tuple(raw.get("attributes", ())),
    )


def _merge_aliases(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for alias in (a for group in groups for a in group):
        if alias.lower() not in seen:
            seen.add(alias.lower())
            merged.append(alias)
    return tuple(merged)


def _merge_alias_maps(*maps: AliasMap) -> AliasMap:
    """Merge maps with case-insensitive keys; the first spelling of a key wins."""
    keys: dict[str, str] = {}
    groups: dict[str, list[tuple[str, ...]]] = {}
    for mapping in maps:
        for key, aliases in mapping.items():
            canonical = keys.setdefault(key.lower(), key)
            groups.setdefault(canonical, []).append(aliases)
    return {k: _merge_aliases(*v) for k, v in groups.items()}


def _merge_nested_alias_maps(*maps: NestedAliasMap) -> NestedAliasMap:
    keys: dict[str, str] = {}
    groups: dict[str, list[AliasMap]] = {}
    for mapping in maps:
        for key, inner in mapping.items():
            canonical = keys.setdefault(key.lower(), key)
            groups.setdefault(canonical, []).append(inner)
    return {k: _merge_alias_maps(*v) for k, v in groups.items()}


def load_config(path: Path | str) -> SemanConfig:
    """Load and validate a config file; raise :class:`ConfigError` listing all problems."""
    config_path = Path(path)
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(config_path, ["file not found"]) from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(config_path, [f"cannot parse TOML: {exc}"]) from None

    errors = validate_config_data(data)
    if errors:
        raise ConfigError(config_path, errors)

    settings: dict[str, Any] = data.get("settings", {})
    global_boost = float(settings.get("alias_boost", DEFAULT_ALIAS_BOOST))
    global_aliases = _alias_config(settings.get("aliases", {}))
    models: dict[str, ModelConfig] = {}
    for name, raw in data["models"].items():
        aliases = _alias_config(raw.get("aliases", {}))
        models[name] = ModelConfig(
            name=name,
            model_file=(
                (config_path.parent / raw["model_file"]).resolve() if "model_file" in raw else None
            ),
            alias_boost=float(raw.get("alias_boost", global_boost)),
            blacklist=_blacklist_config(raw.get("blacklist", {})),
            aliases=AliasConfig(
                entities=_merge_alias_maps(global_aliases.entities, aliases.entities),
                attributes=_merge_nested_alias_maps(global_aliases.attributes, aliases.attributes),
                enumerations=_merge_nested_alias_maps(
                    global_aliases.enumerations, aliases.enumerations
                ),
            ),
        )
        log.info(
            "Config model '%s': file=%s alias_boost=%.2f",
            name,
            models[name].model_file,
            models[name].alias_boost,
        )

    semantic_errors: list[str] = []
    default_model = settings.get("default_model", next(iter(models)))
    if default_model not in models:
        semantic_errors.append(f"settings.default_model: unknown model '{default_model}'")
    semantic_errors.extend(
        f"models.{m.name}.model_file: file not found: {m.model_file}"
        for m in models.values()
        if m.model_file is not None and not m.model_file.is_file()
    )
    if semantic_errors:
        raise ConfigError(config_path, semantic_errors)

    return SemanConfig(path=config_path, models=models, default_model=default_model)
