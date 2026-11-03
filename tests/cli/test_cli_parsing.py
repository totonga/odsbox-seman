from __future__ import annotations

import logging

from odsbox_seman.cli.repl import (
    SearchRequest,
    build_arg_parser,
    configure_logging,
    parse_search_request,
)


def test_verbose_flag_counts() -> None:
    parser = build_arg_parser()
    assert parser.parse_args(["c.toml"]).verbose == 0
    assert parser.parse_args(["c.toml", "-v"]).verbose == 1
    assert parser.parse_args(["c.toml", "-vv"]).verbose == 2
    assert parser.parse_args(["c.toml", "--verbose", "-v"]).verbose == 2


def test_configure_logging_sets_level_and_adds_one_handler() -> None:
    logger = logging.getLogger("odsbox_seman")
    before = list(logger.handlers)
    try:
        configure_logging(2)
        configure_logging(1)
        assert logger.level == logging.INFO
        assert len(logger.handlers) == len(before) + 1
    finally:
        logger.handlers[:] = before
        logger.setLevel(logging.NOTSET)


def test_parse_search_request_with_mode_and_entity() -> None:
    request = parse_search_request("search attribute entity:Vehicle pressure")
    assert request.mode == "attribute"
    assert request.entity_name == "Vehicle"
    assert request.query == "pressure"


def test_parse_search_request_without_mode() -> None:
    request = parse_search_request("search pressure")
    assert request.mode == "all"
    assert request.entity_name is None
    assert request.query == "pressure"


def test_parse_search_request_with_all_mode() -> None:
    request = parse_search_request("search all temperature")
    assert request.mode == "all"
    assert request.query == "temperature"


def test_parse_search_request_with_multi_mode() -> None:
    request = parse_search_request("search MULTI entity:Vehicle pressure")
    assert request.mode == "multi"
    assert request.entity_name == "Vehicle"
    assert request.query == "pressure"
    assert parse_search_request("multi pressure").mode == "multi"
    assert parse_search_request("search multi").query == ""


def test_parse_search_request_returns_empty_query_for_blank_input() -> None:
    request = parse_search_request("   ")
    assert request.mode == "all"
    assert request.entity_name is None
    assert request.query == ""


def test_search_request_to_string() -> None:
    request = SearchRequest(mode="relation", entity_name="Wheel", query="pressure")
    assert str(request) == "SearchRequest(mode='relation', entity_name='Wheel', query='pressure')"
