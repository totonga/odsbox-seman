# CLI mode

The CLI lets you search an ODS model interactively from the terminal or process queries in batch mode.
Install it with `pip install "odsbox-seman[cli]"` to include its optional command-line dependencies.

## Interactive mode

When the CLI is started with a terminal attached, it enters an interactive REPL.

```bash
uv run odsbox-seman ./tests/data/odsbox-seman-config.toml
```

You will see a prompt like this:

```text
odsbox-seman CLI
Type 'help' for commands.
seman[NVH]>
```

### Supported commands

- `search QUERY` — search all kinds
- `search all QUERY` — search all kinds
- `search entity QUERY` — search entities only
- `search attribute QUERY` — search attributes only
- `search relation QUERY` — search relations only
- `search enumeration QUERY` — search enumerations only
- `search multi QUERY` — show every mode separately, merged results and processing times
- `models` — list the configured models (the active one is marked with `*`)
- `use NAME` — switch to another configured model
- `info` — show the active model, its search index summary and how much of the model remains after the blacklists
- `info -v` — additionally list the blacklisted entities and attributes
- `help` — show available commands
- `quit` or `exit` — leave the REPL

### Interactive tips

- Press the up and down arrow keys to browse previous inputs.
- Use `help` to see the command list at any time.
- Use `info` to inspect how many indexed entries were loaded.

### Example session

```text
seman> search pressure
seman> search attribute pressure
seman> info
seman> quit
```

## Multi-mode search

```text
search multi pressure
search multi entity:Vehicle pressure
```

`multi` prints sections in this order: `entity`, `attribute`, `relation`,
`enumeration`, `all`, then `merged`. Each of the first five sections contains up
to three results (or fewer if `--top-k` is lower). The `all` section shows the
globally ranked top three across all kinds. Empty sections explicitly
show `No results.`.

The `merged` section combines the five sections, removes duplicate model items
and shows the top ten by descending score, so it can contain more results than
an individual section. Aliases of the same model item appear
only once; items belonging to different kinds or entities remain distinct.

An optional `entity:NAME` filter uses a case-insensitive substring match and
applies before each section's top-k limit. Standalone enumerations have no entity
name and are excluded when a nonempty entity filter is supplied.

The query is encoded and scored only once. Output includes shared scoring/ranking
time, each mode's selection time, merge time and total processing time, all in
milliseconds. Per-mode selection times measure filtering and result collection,
not five independent searches; shared scoring is reported separately. Total time
excludes model/index initialization, output rendering and time waiting for input.
Blank queries, empty models and nonpositive top-k limits produce empty sections
with zero timings without running inference. Existing search commands keep their
original output format.

The same result structure is available in Python. The API returns the complete
merged union; the CLI displays only its top ten:

```python
result = searcher.search_multi("pressure", top_k=3, entity_name="Vehicle")
for section in result.sections:
    print(section.mode, section.matches, section.elapsed_seconds)
print(result.merged, result.total_seconds)
```

## Batch mode

If the CLI is used with piped input or non-interactive stdin, it runs in batch mode.

```bash
cat queries.txt | uv run odsbox-seman ./tests/data/odsbox-seman-config.toml --model NVH
```

Each input line is treated as a search request.

For example, in PowerShell:

```powershell
'search multi pressure' | uv run odsbox-seman ./tests/data/odsbox-seman-config.toml --top-k 3
```

If no valid cached search index is available, the CLI reports tokenization and embedding
on stderr before starting that work, followed by a completion message when the index is
cached. These notices appear without `-v` and leave batch results on stdout unchanged.

## Configuration file

The CLI reads a TOML file that describes one or more models, their blacklists and aliases.
See [CONFIG.md](CONFIG.md) for the full reference.

```toml
#:schema ../../src/odsbox_seman/odsbox-seman-config.schema.json

[models.NVH]
model_file = "mdm_nvh_model.json"

[models.NVH.blacklist]
entities = ["Tpl*", "Cat*"]

[models.NVH.aliases.entities]
MeaResult = ["Messung"]
```

## Command-line options

```bash
uv run odsbox-seman --help
```

Available options:

- `config` — path to the TOML config file
- `--model NAME` — model to start with (default: `settings.default_model`)
- `--top-k` — maximum number of results per query; `multi` is capped at three per mode and ten merged
- `-v`, `--verbose` — log to stderr; `-v` shows progress (INFO), `-vv` shows details (DEBUG)
