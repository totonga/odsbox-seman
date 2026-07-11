from __future__ import annotations

from odsbox_seman.cli.repl import SearchRequest, parse_search_request


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


def test_parse_search_request_returns_empty_query_for_blank_input() -> None:
    request = parse_search_request("   ")
    assert request.mode == "all"
    assert request.entity_name is None
    assert request.query == ""


def test_search_request_to_string() -> None:
    request = SearchRequest(mode="relation", entity_name="Wheel", query="pressure")
    assert str(request) == "SearchRequest(mode='relation', entity_name='Wheel', query='pressure')"
