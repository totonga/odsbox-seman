"""Interactive CLI for semantic search over ODS models."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from google.protobuf.json_format import Parse
from odsbox.proto.ods_pb2 import Model
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from odsbox_seman.model_searcher import ModelSearcher
from odsbox_seman.types import ModelMatch, SemanticSearchUnavailableError

_SUPPORTED_MODES = {"all", "attribute", "relation", "enumeration"}


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
    entity = match.entity_name or "-"
    return f"[{match.kind}] {entity} :: {match.item_name} (score={match.score:.3f})"


def _run_search(searcher: Any, request: SearchRequest, *, top_k: int = 10) -> list[ModelMatch]:
    if not request.query.strip():
        return []

    matches = searcher.search(request.query, top_k=top_k)
    filtered: list[ModelMatch] = []
    for match in matches:
        if request.mode != "all" and match.kind != request.mode:
            continue
        if request.entity_name is not None:
            entity = request.entity_name.lower()
            if entity not in (match.entity_name or "").lower():
                continue
        filtered.append(match)
    return filtered


def _print_help(stream: TextIO) -> None:
    stream.write(
        "Commands:\n"
        "  search QUERY             Search all kinds\n"
        "  search all QUERY        Search all kinds\n"
        "  search attribute QUERY  Search attributes only\n"
        "  search relation QUERY   Search relations only\n"
        "  search enumeration QUERY\n"
        "                          Search enumerations only\n"
        "  info                    Show model summary\n"
        "  help                    Show this help\n"
        "  quit / exit             Exit the CLI\n"
    )


def _print_info(searcher: Any, stream: TextIO) -> None:
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


def run_repl(searcher: Any, *, stream: TextIO | None = None, top_k: int = 10) -> int:
    stream = sys.stdout if stream is None else stream
    stream.write("odsbox-seman CLI\n")
    stream.write("Type 'help' for commands.\n")

    stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

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
            if use_prompt_session and session is not None:
                line = session.prompt("seman> ")
            else:
                stream.write("seman> ")
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
        if command.lower() == "info":
            _print_info(searcher, stream)
            continue
        if command.lower().startswith("search"):
            request = parse_search_request(command)
            results = _run_search(searcher, request, top_k=top_k)
            if not results:
                stream.write("No results.\n")
            else:
                for result in results:
                    stream.write(f"{_format_result(result)}\n")
            continue

        stream.write("Unknown command. Type 'help' for available commands.\n")


def run_batch(searcher: Any, *, stream: TextIO | None = None, top_k: int = 10) -> int:
    stream = sys.stdout if stream is None else stream
    for line in sys.stdin:
        request = parse_search_request(line)
        results = _run_search(searcher, request, top_k=top_k)
        if not results:
            stream.write("No results.\n")
            continue
        for result in results:
            stream.write(f"{_format_result(result)}\n")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive semantic search over an ODS model")
    parser.add_argument("model", help="Path to the ODS model JSON file")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    try:
        model = _load_model(model_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 2

    try:
        searcher = ModelSearcher(model)
        searcher.warm_up()
    except SemanticSearchUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    if sys.stdin.isatty():
        return run_repl(searcher, top_k=args.top_k)
    return run_batch(searcher, top_k=args.top_k)
