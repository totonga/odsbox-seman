"""Interactive CLI for semantic search over ODS models."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from google.protobuf.json_format import Parse
from odsbox.proto.ods_pb2 import Model
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text

from odsbox_seman.config import ModelConfig, SemanConfig, load_config
from odsbox_seman.model_searcher import ModelSearcher
from odsbox_seman.types import (
    ConfigError,
    CorpusStats,
    ModelMatch,
    MultiSearchResult,
    SemanticSearchUnavailableError,
)

_SUPPORTED_MODES = {"all", "entity", "attribute", "relation", "enumeration", "multi"}


@dataclass(slots=True)
class SearchRequest:
    mode: str
    entity_name: str | None
    query: str

    def __str__(self) -> str:
        return (
            f"SearchRequest(mode={self.mode!r}, entity_name={self.entity_name!r}, "
            f"query={self.query!r})"
        )


def parse_search_request(text: str) -> SearchRequest:
    """Parse a search request from CLI input.

    Supported forms:
    - search pressure
    - search attribute pressure
    - search relation entity:Vehicle pressure
    - search all temperature
    - search multi temperature
    """
    stripped = text.strip()
    if not stripped:
        return SearchRequest(mode="all", entity_name=None, query="")

    parts = stripped.split()
    remaining = parts[1:] if len(parts) >= 2 and parts[0].lower() == "search" else parts

    if not remaining:
        return SearchRequest(mode="all", entity_name=None, query="")

    first = remaining[0].lower()
    if first in _SUPPORTED_MODES:
        mode = first
        tokens = remaining[1:]
    else:
        mode = "all"
        tokens = remaining

    entity_name: str | None = None
    query_tokens: list[str] = []
    for token in tokens:
        if token.lower().startswith("entity:"):
            entity_name = token.split(":", 1)[1].strip()
            if not entity_name:
                entity_name = None
        else:
            query_tokens.append(token)

    query = " ".join(query_tokens).strip()
    return SearchRequest(mode=mode, entity_name=entity_name, query=query)


def _load_model(model_path: Path) -> Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = Model()
    with model_path.open(encoding="utf-8") as handle:
        Parse(handle.read(), model)
    return model


def _format_result(match: ModelMatch) -> str:
    if match.kind == "entity":
        return f"[entity] {match.item_name} (score={match.score:.3f})"
    entity = match.entity_name or "-"
    return f"[{match.kind}] {entity} :: {match.item_name} (score={match.score:.3f})"


def _format_result_text(match: ModelMatch) -> Text:
    kind_styles = {
        "entity": "bold cyan",
        "attribute": "bold green",
        "relation": "bold magenta",
        "enumeration": "bold yellow",
    }
    result = Text("[")
    result.append(match.kind, style=kind_styles.get(match.kind, "bold cyan"))
    result.append("] ")
    if match.kind != "entity":
        result.append(match.entity_name or "-", style="bold")
        result.append(" :: ", style="dim")
    result.append(match.item_name, style="bold white")
    result.append(" (score=", style="dim")
    result.append(f"{match.score:.3f}", style="bold bright_blue")
    result.append(")", style="dim")
    return result


def _print_multi(
    result: MultiSearchResult, stream: TextIO, *, console: Console | None = None
) -> None:
    groups: list[tuple[str, list[ModelMatch], str, float]] = [
        (section.mode, section.matches, "selection", section.elapsed_seconds)
        for section in result.sections
    ]
    groups.append(("merged", result.merged[:10], "merge", result.merge_seconds))
    for mode, matches, timing_label, seconds in groups:
        stream.write(
            f"=== {mode} ({len(matches)} results; {timing_label}: {seconds * 1000:.3f} ms) ===\n"
        )
        if not matches:
            stream.write("No results.\n")
        for match in matches:
            if console is not None:
                console.print(_format_result_text(match), soft_wrap=True)
            else:
                stream.write(f"{_format_result(match)}\n")
    stream.write(f"Shared scoring/ranking: {result.shared_seconds * 1000:.3f} ms\n")
    stream.write(f"Total processing time: {result.total_seconds * 1000:.3f} ms\n")


def _build_searcher(model_config: ModelConfig) -> ModelSearcher:
    if model_config.model_file is None:
        raise ValueError(f"Model '{model_config.name}' requires model_file for CLI use.")
    searcher = ModelSearcher(_load_model(model_config.model_file), config=model_config)
    searcher.warm_up()
    return searcher


class SearcherRegistry:
    """Builds one searcher per configured model on first use and tracks the active one."""

    def __init__(
        self,
        config: SemanConfig,
        active: str | None = None,
        factory: Callable[[ModelConfig], Any] | None = None,
    ) -> None:
        self._config = config
        self._factory = factory or _build_searcher
        self._searchers: dict[str, Any] = {}
        self.active_name: str = active or config.default_model
        if self.active_name not in config.models:
            raise KeyError(self.active_name)

    @property
    def names(self) -> list[str]:
        return list(self._config.models)

    @property
    def active(self) -> Any:
        if self.active_name not in self._searchers:
            self._searchers[self.active_name] = self._factory(self._config.models[self.active_name])
        return self._searchers[self.active_name]

    def use(self, name: str) -> None:
        """Switch the active model; raises ``KeyError`` for unknown names."""
        if name not in self._config.models:
            raise KeyError(name)
        self.active_name = name


def _run_search(searcher: Any, request: SearchRequest, *, top_k: int = 10) -> list[ModelMatch]:
    if not request.query.strip():
        return []

    filtered_search = request.mode != "all" or request.entity_name is not None
    # Filters apply after ranking, so fetch a larger pool to still fill top_k.
    matches = searcher.search(request.query, top_k=top_k * 50 if filtered_search else top_k)
    filtered: list[ModelMatch] = []
    for match in matches:
        if request.mode != "all" and match.kind != request.mode:
            continue
        if request.entity_name is not None:
            entity = request.entity_name.lower()
            if entity not in (match.entity_name or "").lower():
                continue
        filtered.append(match)
    return filtered[:top_k]


def _print_help(stream: TextIO) -> None:
    stream.write(
        "Commands:\n"
        "  search QUERY             Search all kinds\n"
        "  search all QUERY        Search all kinds\n"
        "  search entity QUERY     Search entities only\n"
        "  search attribute QUERY  Search attributes only\n"
        "  search relation QUERY   Search relations only\n"
        "  search enumeration QUERY\n"
        "                          Search enumerations only\n"
        "  search multi QUERY      Search every mode, with merged results and timings\n"
        "  models                  List configured models\n"
        "  use NAME                Switch to model NAME\n"
        "  info                    Show model summary\n"
        "  info -v                 Also list blacklisted entities and attributes\n"
        "  help                    Show this help\n"
        "  quit / exit             Exit the CLI\n"
    )


def _print_info(searcher: Any, stream: TextIO, *, verbose: bool = False) -> None:
    corpus = getattr(searcher, "_corpus", [])
    entry_count = len(corpus)
    kinds = Counter(
        getattr(match, "kind", "unknown") for _, match in corpus if getattr(match, "kind", None)
    )
    summary = ", ".join(
        f"{count} {name}{'' if count == 1 else 's'}" for name, count in sorted(kinds.items())
    )
    if summary:
        stream.write(f"Loaded search index with {entry_count} entries ({summary}).\n")
    else:
        stream.write(f"Loaded search index with {entry_count} entries.\n")

    stats = getattr(searcher, "stats", None)
    if not isinstance(stats, CorpusStats):
        return
    stream.write(
        f"Model coverage: {stats.entities_kept}/{stats.entities_total} entities, "
        f"{stats.attributes_kept}/{stats.attributes_total} attributes, "
        f"{stats.relations_kept}/{stats.relations_total} relations, "
        f"{stats.enumerations} enumerations.\n"
    )
    if verbose:
        if stats.excluded_entities:
            stream.write(f"Blacklisted entities: {', '.join(stats.excluded_entities)}\n")
        if stats.excluded_attributes:
            stream.write(f"Blacklisted attributes: {', '.join(stats.excluded_attributes)}\n")


def run_repl(registry: SearcherRegistry, *, stream: TextIO | None = None, top_k: int = 10) -> int:
    stream = sys.stdout if stream is None else stream
    stream.write("odsbox-seman CLI\n")
    stream.write("Type 'help' for commands.\n")

    stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    color_output = stdin_is_tty and bool(getattr(stream, "isatty", lambda: False)())
    console = Console(
        file=stream,
        force_terminal=color_output,
        no_color=not color_output,
        highlight=False,
    )

    # Set up history file in user's home directory
    history_file = Path.home() / ".odsbox-seman-history"
    session: PromptSession[str] | None = None
    use_prompt_session = False
    if stdin_is_tty and stdout_is_tty:
        try:
            session = PromptSession(history=FileHistory(str(history_file)))
            use_prompt_session = True
        except Exception:
            # Fall back to standard input if PromptSession can't be created
            # (e.g., during testing or in non-interactive environments)
            session = None
            use_prompt_session = False

    while True:
        try:
            prompt = f"seman[{registry.active_name}]> "
            if use_prompt_session and session is not None:
                line = session.prompt(prompt)
            else:
                stream.write(prompt)
                stream.flush()
                line = input()
        except EOFError:
            stream.write("\n")
            return 0
        except KeyboardInterrupt:
            stream.write("\n")
            return 0

        command = line.strip()
        if not command:
            continue
        if command.lower() in {"quit", "exit"}:
            return 0
        if command.lower() == "help":
            _print_help(stream)
            continue
        if command.lower() == "models":
            for name in registry.names:
                marker = "*" if name == registry.active_name else " "
                stream.write(f"{marker} {name}\n")
            continue
        if command.lower().split(maxsplit=1)[0] == "use":
            parts = command.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else ""
            previous = registry.active_name
            try:
                registry.use(target)
                registry.active  # noqa: B018 - build and warm up now so errors show here
            except KeyError:
                stream.write(f"Unknown model '{target}'. Available: {', '.join(registry.names)}\n")
            except ValueError as exc:
                registry.use(previous)
                stream.write(f"{exc}\n")
            except SemanticSearchUnavailableError as exc:
                stream.write(f"{exc}\n")
            else:
                stream.write(f"Using model '{target}'.\n")
            continue
        if command.lower().split(maxsplit=1)[0] == "info":
            flags = command.lower().split()[1:]
            stream.write(f"Model: {registry.active_name}\n")
            _print_info(registry.active, stream, verbose="-v" in flags)
            continue
        if command.lower().startswith("search"):
            request = parse_search_request(command)
            if request.mode == "multi":
                multi_result = registry.active.search_multi(
                    request.query, top_k=min(top_k, 3), entity_name=request.entity_name
                )
                _print_multi(multi_result, stream, console=console)
                continue
            results = _run_search(registry.active, request, top_k=top_k)
            if not results:
                stream.write("No results.\n")
            else:
                for result in results:
                    console.print(_format_result_text(result), soft_wrap=True)
            continue

        stream.write("Unknown command. Type 'help' for available commands.\n")


def run_batch(searcher: Any, *, stream: TextIO | None = None, top_k: int = 10) -> int:
    stream = sys.stdout if stream is None else stream
    for line in sys.stdin:
        request = parse_search_request(line)
        if request.mode == "multi":
            multi_result = searcher.search_multi(
                request.query, top_k=min(top_k, 3), entity_name=request.entity_name
            )
            _print_multi(multi_result, stream)
            continue
        results = _run_search(searcher, request, top_k=top_k)
        if not results:
            stream.write("No results.\n")
            continue
        for result in results:
            stream.write(f"{_format_result(result)}\n")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive semantic search over an ODS model")
    parser.add_argument("config", help="Path to the odsbox-seman TOML config file")
    parser.add_argument("--model", help="Name of the model to start with (default: from config)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum results (multi: at most 3 per mode and 10 merged)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Log to stderr: -v for progress (INFO), -vv for details (DEBUG)",
    )
    return parser


def configure_logging(verbosity: int) -> None:
    """Send ``odsbox_seman`` logs to stderr; other libraries keep their defaults."""
    logger = logging.getLogger("odsbox_seman")
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[min(verbosity, 2)])
    logging.getLogger("odsbox_seman.model_searcher.progress").setLevel(logging.INFO)
    if not any(getattr(h, "_seman_cli", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._seman_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        registry = SearcherRegistry(config, args.model)
    except KeyError:
        print(
            f"Unknown model '{args.model}'. Available: {', '.join(config.models)}",
            file=sys.stderr,
        )
        return 2

    try:
        searcher = registry.active
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except SemanticSearchUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 2

    if sys.stdin.isatty():
        return run_repl(registry, top_k=args.top_k)
    return run_batch(searcher, top_k=args.top_k)
