"""Unit tests for the tokenize() helper."""

from __future__ import annotations

from odsbox_seman.tokenizer import tokenize


class TestTokenize:
    def test_snake_case(self) -> None:
        assert tokenize("tyre_pressure") == "tyre pressure"

    def test_camel_case(self) -> None:
        assert tokenize("MeaQuantity") == "mea quantity"

    def test_ao_prefix(self) -> None:
        assert tokenize("AoTestEquipment") == "ao test equipment"

    def test_mixed(self) -> None:
        assert tokenize("vehicle_manufacturer") == "vehicle manufacturer"

    def test_empty(self) -> None:
        assert tokenize("") == ""

    def test_all_caps_word(self) -> None:
        result = tokenize("DT_STRING")
        assert "string" in result

    def test_hyphen_separator(self) -> None:
        assert tokenize("some-attr") == "some attr"

    def test_multiple_underscores(self) -> None:
        assert tokenize("a_b_c") == "a b c"

    def test_leading_trailing_whitespace_cleaned(self) -> None:
        result = tokenize("MyEntity")
        assert result == result.strip()

    def test_lowercase_output(self) -> None:
        result = tokenize("AbcDef")
        assert result == result.lower()

    def test_consecutive_uppercase(self) -> None:
        # "XMLParser" should include "parser" in the result
        result = tokenize("XMLParser")
        assert "parser" in result

    def test_single_word(self) -> None:
        assert tokenize("name") == "name"

    def test_numbers_in_name(self) -> None:
        result = tokenize("sensor3Value")
        assert "sensor" in result
        assert "value" in result
