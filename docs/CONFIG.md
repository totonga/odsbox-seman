# Configuration file

odsbox-seman reads a TOML file that describes one or more ODS models, which elements to hide
(blacklists) and which extra search terms to add (aliases).

The file is validated against
[`odsbox-seman-config.schema.json`](../src/odsbox_seman/odsbox-seman-config.schema.json).
All problems are reported at once in a single `ConfigError`; the CLI exits with code 2 before
loading any model. With the Even Better TOML extension, VS Code validates the file while you edit
it (see `.vscode/settings.json`); other files can start with the `#:schema` line shown below.

## Example

```toml
#:schema ../../src/odsbox_seman/odsbox-seman-config.schema.json

[settings]
default_model = "NVH"
alias_boost = 0.1

[settings.aliases.entities]          # applied to every model
Test = ["Versuch"]

[models.NVH]
model_file = "mdm_nvh_model.json"    # relative to this file
alias_boost = 0.15                   # optional, overrides settings.alias_boost

[models.NVH.blacklist]
entities = ["Tpl*", "Cat*", "MDM*"]
attributes = ["version"]

[models.NVH.aliases.entities]
MeaResult = ["Messung"]
MeaQuantity = ["Verlauf", "Signal"]

[models.NVH.aliases.attributes.Vehicle]
description = ["Comment"]

[models.NVH.aliases.enumerations.valid_enum]
valid = ["gueltig"]
```

## Python API

`load_config(path)` returns a `SemanConfig` whose `models` dictionary contains `ModelConfig`
instances. Each model has an `AliasConfig` in `aliases` and a `BlacklistConfig` in `blacklist`.
All four types are available from the package root. Model names and alias keys remain dynamic
dictionary keys; alias values and blacklist patterns are tuples.

To configure a `ModelSearcher` directly, construct a per-model config:

```python
from odsbox_seman import AliasConfig, BlacklistConfig, ModelConfig

model_config = ModelConfig(
    name="NVH",
    alias_boost=0.15,
    aliases=AliasConfig(
        entities={"MeaResult": ("Messung",)},
        attributes={"Vehicle": {"description": ("Comment",)}},
        enumerations={"valid_enum": {"valid": ("gueltig",)}},
    ),
    blacklist=BlacklistConfig(
        entities=("Tpl*", "Cat*"),
        attributes=("version",),
    ),
)
```

Pass this object as `config=model_config` when constructing `ModelSearcher` with a parsed ODS
model. `model_file` defaults to `None` and is not needed when the model comes from another
source, such as an ODS connection. It may also be omitted from TOML:

```toml
[models.NVH.aliases.entities]
MeaResult = ["Messung"]
```

Load the config and pass `config.models["NVH"]` to `ModelSearcher` together with the externally
supplied model. Named `[models.<NAME>]` sections are still required. The CLI needs a
`model_file` for each model it loads and reports an error if the selected model omits it.

Omitted sections receive independent, empty defaults. These are frozen dataclasses:
fields cannot be reassigned, but their contained dictionaries remain mutable. Avoid mutating
configuration after constructing a searcher, which has already built its corpus and cache key.
Direct dataclass construction does not perform schema validation; use `load_config()` for
validated TOML configuration, including global alias merging and relative model-file resolution.

### Migrating flat model fields

The flat `ModelConfig` fields and constructor keywords have been removed:

| Previous field | Nested field |
| --- | --- |
| `entity_aliases` | `aliases.entities` |
| `attribute_aliases` | `aliases.attributes` |
| `enum_aliases` | `aliases.enumerations` |
| `entity_blacklist` | `blacklist.entities` |
| `attribute_blacklist` | `blacklist.attributes` |

For example, replace `ModelConfig(..., entity_aliases={"Test": ("Versuch",)})` with
`ModelConfig(..., aliases=AliasConfig(entities={"Test": ("Versuch",)}))`.
`is_entity_blacklisted()` and `is_attribute_blacklisted()` remain available on `ModelConfig`.
`SemanConfig`, TOML syntax, schema validation, merge behavior, and existing cache keys are unchanged.

## `[settings]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `default_model` | string | first model | Model used at start. Must exist in `[models]`. |
| `alias_boost` | number 0..1 | `0.1` | Score bonus for matches found via an alias. |
| `aliases` | table | empty | Global aliases, see [Aliases](#aliases). |

## `[models.<NAME>]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `model_file` | string | omitted (`None`) | Optional ODS model JSON file, relative to the config file. Required when loading the model through the CLI. |
| `alias_boost` | number 0..1 | `settings.alias_boost` | Overrides the global boost for this model. |
| `blacklist` | table | empty | See [Blacklist](#blacklist). |
| `aliases` | table | empty | See [Aliases](#aliases). |

## Blacklist

Patterns are case-insensitive globs (`*`, `?`, `[...]`) matched against both `name` and
`base_name`.

- `blacklist.entities`: matching entities are removed together with their attributes and
  relations. Relations of other entities that point to a removed entity are removed as well.
- `blacklist.attributes`: matching attributes (and relations with a matching name) are removed from
  all entities.

Patterns that match nothing in the model are logged as warnings (see
[Checking the result](#checking-the-result)).

## Aliases

Aliases are extra search terms, for example translations or project jargon. Each alias is indexed
as its own entry and gets `alias_boost` added to its score, so it ranks above plain model entries.

| Table | Structure |
| --- | --- |
| `aliases.entities` | `EntityName = ["alias", ...]` |
| `aliases.attributes.<Entity>` | `attributeName = ["alias", ...]` |
| `aliases.enumerations.<Enum>` | `itemName = ["alias", ...]` |

- Keys match the `name` or the `base_name` of the model element, case-insensitively. If both are
  configured, their aliases are combined.
- Entity aliases also enrich the search text of the entity's attributes and relations.
- Aliases must be non-empty strings and unique within a list.

### Global and model aliases

`[settings.aliases]` has the same structure as `[models.<NAME>.aliases]` and applies to every
model. Both are merged per key (case-insensitive), global aliases first, with duplicates removed.
Keys that do not exist in a model are ignored, so one global table can serve models of different
shapes. Using base names (for example `AoTest`) makes global aliases portable between models.

### Alias boost

The boost fades in with the raw similarity of the match: it is 0 below a similarity of 0.3 and
applied in full from 0.6 on. Weak matches therefore do not profit from a boost, and unrelated
aliases do not outrank real hits. The boost is configurable globally and per model; the fade
range is fixed.

## Cache

The embedding cache key contains the model content and a fingerprint of the blacklists and the
merged aliases. Changing those rebuilds the index on the next start; changing only `alias_boost`
does not, because the boost is applied at search time.

## Checking the result

- `info` in the REPL shows how much of the model remains, for example
  `Model coverage: 42/180 entities, 310/1200 attributes, ...`.
- `info -v` additionally lists the blacklisted entities and attributes.
- `--verbose` (`-v`, `-vv`) writes logs to stderr, including unmatched blacklist patterns, alias
  keys that match no entity, and each blacklisted element at debug level.

## Validation errors

```text
Invalid configuration odsbox-seman-config.toml (2 error(s)):
  - models.A.model_file: 1 is not of type 'string'
  - models.A: Additional properties are not allowed ('bogus' was unexpected)
```
