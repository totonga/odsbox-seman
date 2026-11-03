from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

import odsbox_seman.cli.repl as repl
from odsbox_seman.cache import CacheManager
from odsbox_seman.cli.repl import (
    SearcherRegistry,
    SearchRequest,
    _format_result,
    _print_help,
    _print_info,
    _run_search,
    run_batch,
    run_repl,
)
from odsbox_seman.config import ModelConfig, SemanConfig
from odsbox_seman.types import CorpusStats, ModelMatch, MultiSearchResult, SearchSection

_SAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "odsbox-seman-config.toml"


def _registry(*searchers: object) -> SearcherRegistry:
    models = {
        f"M{i}": ModelConfig(name=f"M{i}", model_file=Path(f"m{i}.json"))
        for i in range(len(searchers))
    }
    by_name = {f"M{i}": s for i, s in enumerate(searchers)}
    config = SemanConfig(path=Path("c.toml"), models=models, default_model="M0")
    return SearcherRegistry(config, factory=lambda cfg: by_name[cfg.name])


class FakeSearcher:
    def __init__(self, results: list[ModelMatch]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10) -> list[ModelMatch]:
        self.calls.append((query, top_k))
        return self.results


@pytest.mark.parametrize("interactive", [False, True])
def test_multi_output_sections_timings_and_forwarding(
    interactive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    match = ModelMatch("attribute", "Vehicle", "AoVehicle", "Pressure", "pressure", 1, 0.9)
    searcher = MagicMock()
    searcher.search_multi.return_value = MultiSearchResult(
        [
            SearchSection("entity", [], 0.001),
            SearchSection("attribute", [match], 0.002),
            SearchSection("relation", [], 0.003),
            SearchSection("enumeration", [], 0.004),
            SearchSection("all", [match], 0.005),
        ],
        [match],
        0.010,
        0.006,
        0.031,
    )
    stream = io.StringIO()
    command = "search multi entity:Vehicle pressure"
    monkeypatch.setattr(sys, "stdin", io.StringIO(command + "\n" + command + "\n"))
    if interactive:
        responses = iter([command, "quit"])
        monkeypatch.setattr("builtins.input", lambda: next(responses))
        assert run_repl(_registry(searcher), stream=stream) == 0
    else:
        assert run_batch(searcher, stream=stream) == 0

    output = stream.getvalue()
    headings = [
        f"=== {mode} ("
        for mode in ("entity", "attribute", "relation", "enumeration", "all", "merged")
    ]
    assert [output.index(heading) for heading in headings] == sorted(
        output.index(heading) for heading in headings
    )
    assert "selection: 2.000 ms" in output
    assert "merge: 6.000 ms" in output
    assert "Shared scoring/ranking: 10.000 ms" in output
    assert "Total processing time: 31.000 ms" in output
    assert "No results." in output
    assert "[attribute] Vehicle :: Pressure (score=0.900)" in output
    assert "\x1b[" not in output
    assert output.count("Total processing time:") == (1 if interactive else 2)
    searcher.search_multi.assert_called_with("pressure", top_k=3, entity_name="Vehicle")
    searcher.search.assert_not_called()


def test_multi_repl_colors_results_and_uses_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    responses = iter(["use M1", "search multi pressure", "quit"])

    class DummyPromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def prompt(self, prompt: str) -> str:
            return next(responses)

    match = ModelMatch("attribute", "Vehicle", "AoVehicle", "Pressure", "", 1, 0.9)
    first, second = MagicMock(), MagicMock()
    second.search_multi.return_value = MultiSearchResult(
        [SearchSection("attribute", [match], 0.0)], [match], 0.0, 0.0, 0.0
    )
    stream = TTYStream()
    monkeypatch.setattr(repl, "PromptSession", DummyPromptSession)
    monkeypatch.setattr(sys, "stdin", TTYStream())
    monkeypatch.setattr(sys, "stdout", stream)

    assert run_repl(_registry(first, second), stream=stream) == 0
    assert "\x1b[" in stream.getvalue()
    first.search_multi.assert_not_called()
    second.search_multi.assert_called_once_with("pressure", top_k=3, entity_name=None)


@pytest.mark.parametrize("interactive", [False, True])
def test_multi_limits_each_mode_and_merged(
    interactive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from odsbox.proto.ods_pb2 import Model

    from odsbox_seman.model_searcher import ModelSearcher

    kinds: tuple[Literal["entity", "attribute", "relation", "enumeration"], ...] = (
        "entity",
        "attribute",
        "relation",
        "enumeration",
    )
    matches = [
        ModelMatch(kind, "Vehicle", "AoVehicle", f"{kind}{number}", "", 0, 1 - number * 0.1)
        for kind in kinds
        for number in range(4)
    ]
    ranked = sorted(matches, key=lambda match: match.score, reverse=True)
    searcher = ModelSearcher(Model())
    searcher._corpus = [(match.item_name, match) for match in ranked]
    monkeypatch.setattr(searcher, "warm_up", lambda: None)
    monkeypatch.setattr(searcher, "_search_ranked", lambda query, top_k: ranked)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("search multi pressure\n"))
    if interactive:
        responses = iter(["search multi pressure", "quit"])
        monkeypatch.setattr("builtins.input", lambda: next(responses))
        assert run_repl(_registry(searcher), stream=stream) == 0
    else:
        assert run_batch(searcher, stream=stream) == 0

    output = stream.getvalue()
    for mode in ("entity", "attribute", "relation", "enumeration", "all"):
        assert f"=== {mode} (3 results;" in output
    assert "=== merged (10 results;" in output
    merged_output = output.split("=== merged", 1)[1]
    assert merged_output.count("(score=") == 10
    assert "(score=0.700)" not in output


def test_multi_batch_empty_query_prints_empty_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    from odsbox.proto.ods_pb2 import Model

    from odsbox_seman.model_searcher import ModelSearcher

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("search multi\n"))
    assert run_batch(ModelSearcher(Model()), stream=stream) == 0
    assert stream.getvalue().count("No results.") == 6
    assert "Total processing time: 0.000 ms" in stream.getvalue()


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
    assert searcher.calls == [("pressure", 150)]


def test_run_repl_prints_info_and_quits(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    fake_searcher = SimpleNamespace(_corpus=[("dummy", SimpleNamespace(kind="attribute"))])

    responses = iter(["info", "quit"])
    monkeypatch.setattr("builtins.input", lambda: next(responses))

    exit_code = run_repl(_registry(fake_searcher), stream=stream)

    assert exit_code == 0
    assert "odsbox-seman CLI" in stream.getvalue()
    assert "Model: M0" in stream.getvalue()
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
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr("builtins.input", lambda: "quit")

    exit_code = run_repl(_registry(fake_searcher), stream=stream)

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

    exit_code = run_repl(_registry(fake_searcher), stream=stream)

    assert exit_code == 0
    assert "Commands:" in stream.getvalue()
    assert "[attribute] Vehicle :: Pressure" in stream.getvalue()
    assert "Unknown command." in stream.getvalue()


def test_run_repl_highlights_search_results_in_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    class DummyPromptSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def prompt(self, prompt: str) -> str:
            return next(responses)

    match = ModelMatch("attribute", "Vehicle", "AoVehicle", "Pressure", "pressure", 1, 0.9)
    stream = TTYStream()
    responses = iter(["search pressure", "quit"])
    monkeypatch.setattr(repl, "PromptSession", DummyPromptSession)
    monkeypatch.setattr(sys, "stdin", TTYStream())
    monkeypatch.setattr(sys, "stdout", stream)

    assert run_repl(_registry(FakeSearcher([match])), stream=stream) == 0
    assert "\x1b[" in stream.getvalue()


def test_run_repl_models_and_use_switch_searcher(monkeypatch: pytest.MonkeyPatch) -> None:
    def match(name: str) -> ModelMatch:
        return ModelMatch("attribute", "E", "AoE", name, name.lower(), 1, 0.5)

    stream = io.StringIO()
    registry = _registry(FakeSearcher([match("First")]), FakeSearcher([match("Second")]))
    responses = iter(["models", "search x", "use M1", "search x", "use nope", "quit"])
    monkeypatch.setattr("builtins.input", lambda: next(responses))

    assert run_repl(registry, stream=stream) == 0

    out = stream.getvalue()
    assert "* M0" in out
    assert "  M1" in out
    assert out.index(":: First") < out.index("Using model 'M1'") < out.index(":: Second")
    assert "Unknown model 'nope'" in out
    assert registry.active_name == "M1"


def test_registry_builds_searchers_lazily() -> None:
    built: list[str] = []
    config = SemanConfig(
        path=Path("c.toml"),
        models={n: ModelConfig(name=n, model_file=Path(f"{n}.json")) for n in ("A", "B")},
        default_model="A",
    )

    def factory(cfg: ModelConfig) -> str:
        built.append(cfg.name)
        return cfg.name

    registry = SearcherRegistry(config, factory=factory)

    assert built == []
    assert registry.active == "A"
    assert registry.active == "A"
    registry.use("B")
    assert registry.active == "B"
    assert built == ["A", "B"]
    with pytest.raises(KeyError):
        registry.use("C")


def test_build_searcher_requires_model_file(monkeypatch: pytest.MonkeyPatch) -> None:
    load_model = MagicMock()
    monkeypatch.setattr(repl, "_load_model", load_model)
    with pytest.raises(ValueError, match="Model 'NVH'.*model_file.*CLI"):
        repl._build_searcher(ModelConfig(name="NVH"))
    load_model.assert_not_called()


def test_main_reports_missing_model_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "odsbox-seman-config.toml"
    config.write_text("[models.NVH]\n", encoding="utf-8")
    assert repl.main([str(config)]) == 2
    assert "Model 'NVH' requires model_file for CLI use" in capsys.readouterr().err


def test_repl_keeps_active_model_when_fileless_model_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SemanConfig(
        path=Path("c.toml"),
        models={"A": ModelConfig(name="A"), "B": ModelConfig(name="B")},
        default_model="A",
    )
    searcher = FakeSearcher([])
    registry = SearcherRegistry(
        config, factory=lambda cfg: searcher if cfg.name == "A" else repl._build_searcher(cfg)
    )
    commands = iter(["use B", "search vehicle", "quit"])
    monkeypatch.setattr("builtins.input", lambda: next(commands))
    stream = io.StringIO()
    assert run_repl(registry, stream=stream) == 0
    assert "Model 'B' requires model_file for CLI use" in stream.getvalue()
    assert registry.active_name == "A"
    assert searcher.calls == [("vehicle", 10)]


@pytest.mark.parametrize("cached", [False, True])
def test_build_searcher_reports_index_build_before_encoding(
    cached: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = ModelConfig(name="NVH", model_file=_SAMPLE_CONFIG.parent / "mdm_nvh_model.json")
    cached_index: tuple[
        npt.NDArray[np.float32],
        list[tuple[str, ModelMatch]],
        npt.NDArray[np.float32],
    ] = (np.zeros((0, 32), dtype=np.float32), [], np.zeros(0, dtype=np.float32))
    monkeypatch.setattr(CacheManager, "load", lambda *args: cached_index if cached else None)
    monkeypatch.setattr(CacheManager, "save", lambda *args: None)
    embedder = MagicMock()

    def encode(texts: list[str], *, normalize: bool) -> np.ndarray:
        err = capsys.readouterr().err
        assert "Tokenizing and embedding model" in err
        assert "this may take a while" in err
        assert "Search index built and cached" not in err
        return np.zeros((len(texts), 32), dtype=np.float32)

    embedder.encode.side_effect = encode
    monkeypatch.setattr("odsbox_seman.model_searcher.OnnxEmbedder", lambda: embedder)
    logger = logging.getLogger("odsbox_seman")
    progress_logger = logging.getLogger("odsbox_seman.model_searcher.progress")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(progress_logger, "level", progress_logger.level)
    repl.configure_logging(0)

    repl._build_searcher(config)

    err = capsys.readouterr().err
    if cached:
        embedder.encode.assert_not_called()
        assert err == ""
    else:
        embedder.encode.assert_called_once()
        assert "Search index built and cached" in err
    assert capsys.readouterr().out == ""


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
    assert "search entity QUERY" in stream.getvalue()
    assert "Loaded search index" in stream.getvalue()
    assert "1 attribute" in stream.getvalue()


def test_print_info_shows_coverage_and_blacklist_when_verbose() -> None:
    stats = CorpusStats(
        entities_total=5,
        entities_kept=3,
        attributes_total=20,
        attributes_kept=15,
        relations_total=4,
        relations_kept=2,
        enumerations=1,
        excluded_entities=("TplA", "TplB"),
        excluded_attributes=("Test.version",),
    )
    searcher = SimpleNamespace(_corpus=[], stats=stats)

    plain = io.StringIO()
    _print_info(searcher, plain)
    assert "3/5 entities, 15/20 attributes, 2/4 relations" in plain.getvalue()
    assert "TplA" not in plain.getvalue()

    verbose = io.StringIO()
    _print_info(searcher, verbose, verbose=True)
    assert "Blacklisted entities: TplA, TplB" in verbose.getvalue()
    assert "Blacklisted attributes: Test.version" in verbose.getvalue()


def test_main_uses_repl_for_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []

    def build_searcher(cfg: ModelConfig) -> object:
        built.append(cfg.name)
        return object()

    monkeypatch.setattr(repl, "_build_searcher", build_searcher)
    monkeypatch.setattr(repl, "run_repl", lambda registry, *, stream=None, top_k=10: 7)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))

    assert repl.main([str(_SAMPLE_CONFIG)]) == 7
    assert built == ["NVH"]


def test_main_returns_2_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert repl.main([str(_SAMPLE_CONFIG), "--model", "nope"]) == 2
    assert "Unknown model 'nope'" in capsys.readouterr().err


def test_main_returns_2_for_missing_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert repl.main(["missing.toml"]) == 2
    assert "file not found" in capsys.readouterr().err


def test_main_reports_all_config_errors_at_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "odsbox-seman-config.toml"
    config.write_text(
        '[models.A]\nmodel_file = 1\nbogus = true\n[models.A.blacklist]\nentities = [""]\n',
        encoding="utf-8",
    )

    assert repl.main([str(config)]) == 2

    err = capsys.readouterr().err
    assert "3 error(s)" in err
    assert "models.A.model_file" in err
    assert "models.A.blacklist.entities[0]" in err
    assert "bogus" in err
