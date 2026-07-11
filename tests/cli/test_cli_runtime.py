from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import odsbox_seman.cli.repl as repl
from odsbox_seman.cli.repl import (
    SearchRequest,
    _format_result,
    _print_help,
    _print_info,
    _run_search,
    run_batch,
    run_repl,
)
from odsbox_seman.types import ModelMatch


class FakeSearcher:
    def __init__(self, results: list[ModelMatch]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10) -> list[ModelMatch]:
        self.calls.append((query, top_k))
        return self.results


def test_format_result_contains_kind_entity_and_score() -> None:
    match = ModelMatch(
        kind="attribute",
        entity_name="Vehicle",
        entity_base_name="AoVehicle",
        item_name="Pressure",
        item_base_name="pressure",
        data_type=1,
        score=0.84567,
    )

    assert _format_result(match) == "[attribute] Vehicle :: Pressure (score=0.846)"


def test_run_search_filters_by_mode_and_entity() -> None:
    searcher = FakeSearcher(
        [
            ModelMatch(
                kind="attribute",
                entity_name="Vehicle",
                entity_base_name="AoVehicle",
                item_name="Pressure",
                item_base_name="pressure",
                data_type=1,
                score=0.8,
            ),
            ModelMatch(
                kind="relation",
                entity_name="Vehicle",
                entity_base_name="AoVehicle",
                item_name="hasPart",
                item_base_name="has_part",
                data_type=0,
                score=0.9,
            ),
            ModelMatch(
                kind="attribute",
                entity_name="Wheel",
                entity_base_name="AoWheel",
                item_name="Diameter",
                item_base_name="diameter",
                data_type=1,
                score=0.7,
            ),
        ]
    )

    results = _run_search(
        searcher,
        SearchRequest(mode="attribute", entity_name="Vehicle", query="pressure"),
        top_k=3,
    )

    assert [match.item_name for match in results] == ["Pressure"]
    assert searcher.calls == [("pressure", 3)]


def test_run_repl_prints_info_and_quits(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    fake_searcher = SimpleNamespace(_corpus=[("dummy", SimpleNamespace(kind="attribute"))])

    responses = iter(["info", "quit"])
    monkeypatch.setattr("builtins.input", lambda: next(responses))

    exit_code = run_repl(fake_searcher, stream=stream)

    assert exit_code == 0
    assert "odsbox-seman CLI" in stream.getvalue()
    assert "Loaded search index" in stream.getvalue()
    assert "attribute" in stream.getvalue()


def test_run_repl_skips_prompt_session_outside_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    fake_searcher = SimpleNamespace(_corpus=[])

    created_sessions: list[str] = []

    class DummyPromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            created_sessions.append("created")

    monkeypatch.setattr(repl, "PromptSession", DummyPromptSession)
    monkeypatch.setattr(repl.sys, "stdin", io.StringIO())
    monkeypatch.setattr(repl.sys, "stdout", stream)
    monkeypatch.setattr("builtins.input", lambda: "quit")

    exit_code = run_repl(fake_searcher, stream=stream)

    assert exit_code == 0
    assert created_sessions == []


def test_run_repl_handles_help_search_and_unknown_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    fake_searcher = FakeSearcher(
        [
            ModelMatch(
                kind="attribute",
                entity_name="Vehicle",
                entity_base_name="AoVehicle",
                item_name="Pressure",
                item_base_name="pressure",
                data_type=1,
                score=0.9,
            )
        ]
    )

    responses = iter(["help", "search pressure", "unknown", "quit"])
    monkeypatch.setattr("builtins.input", lambda: next(responses))

    exit_code = run_repl(fake_searcher, stream=stream)

    assert exit_code == 0
    assert "Commands:" in stream.getvalue()
    assert "[attribute] Vehicle :: Pressure" in stream.getvalue()
    assert "Unknown command." in stream.getvalue()


def test_run_batch_processes_multiple_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    fake_searcher = FakeSearcher(
        [
            ModelMatch(
                kind="relation",
                entity_name="Vehicle",
                entity_base_name="AoVehicle",
                item_name="hasPart",
                item_base_name="has_part",
                data_type=0,
                score=0.8,
            )
        ]
    )

    monkeypatch.setattr(sys, "stdin", io.StringIO("search pressure\nsearch unknown\n"))

    exit_code = run_batch(fake_searcher, stream=stream)

    assert exit_code == 0
    assert stream.getvalue().count("[relation] Vehicle :: hasPart") == 2


def test_print_help_and_info_write_expected_output() -> None:
    stream = io.StringIO()
    searcher = SimpleNamespace(_corpus=[("dummy", SimpleNamespace(kind="attribute"))])

    _print_help(stream)
    _print_info(searcher, stream)

    assert "search attribute QUERY" in stream.getvalue()
    assert "Loaded search index" in stream.getvalue()
    assert "1 attribute" in stream.getvalue()


def test_build_arg_parser_and_main_use_repl_for_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = Path("tests/data/mdm_nvh_model.json")

    class FakeSearcher:
        def __init__(self, model: object) -> None:
            self.model = model

        def warm_up(self) -> None:
            return None

    monkeypatch.setattr(repl, "_load_model", lambda path: object())
    monkeypatch.setattr(repl, "ModelSearcher", FakeSearcher)
    monkeypatch.setattr(repl, "run_repl", lambda searcher, *, stream=None, top_k=10: 7)
    monkeypatch.setattr(repl.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    exit_code = repl.main([str(model_path)])

    assert exit_code == 7


def test_main_returns_2_for_missing_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        repl, "_load_model", lambda path: (_ for _ in ()).throw(FileNotFoundError("missing"))
    )

    exit_code = repl.main(["missing.json"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing" in captured.err
