# CLI mode

The CLI lets you search an ODS model interactively from the terminal or process queries in batch mode.

## Interactive mode

When the CLI is started with a terminal attached, it enters an interactive REPL.

```bash
uv run odsbox-seman ./tests/data/mdm_nvh_model.json
```

You will see a prompt like this:

```text
odsbox-seman CLI
Type 'help' for commands.
seman>
```

### Supported commands

- `search QUERY` — search all kinds
- `search all QUERY` — search all kinds
- `search attribute QUERY` — search attributes only
- `search relation QUERY` — search relations only
- `search enumeration QUERY` — search enumerations only
- `info` — show the loaded search index summary
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

## Batch mode

If the CLI is used with piped input or non-interactive stdin, it runs in batch mode.

```bash
cat queries.txt | uv run odsbox-seman ./tests/data/mdm_nvh_model.json
```

Each input line is treated as a search request.

## Command-line options

```bash
uv run odsbox-seman --help
```

Available options:

- `model` — path to the ODS model JSON file
- `--top-k` — maximum number of results to print per query
