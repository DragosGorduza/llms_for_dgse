"""Parsing numbers out of free-form LLM text.

Expect: the last JSON object wins, nested and single-quoted objects work, percent signs and commas are stripped,
and missing keys or no JSON raise ParseError.
"""

import pytest

from dgse_llm.llms import ParseError, build_answer_model, extract_json, parse_values


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_reasoning_then_answer_takes_last_object():
    text = 'First guess {"a": 1}. After thinking more, final answer:\n{"a": -30, "b": 2}'
    assert extract_json(text) == {"a": -30, "b": 2}


def test_nested_object_is_returned_whole():
    assert extract_json('x {"a": {"b": 1}, "c": 2} y') == {"a": {"b": 1}, "c": 2}


def test_single_quotes_accepted():
    assert extract_json("{'social change': -12.5}") == {"social change": -12.5}


def test_braces_inside_text_do_not_break_parsing():
    assert extract_json('set {x} notation, answer {"a": 3}') == {"a": 3}


def test_no_json_raises():
    with pytest.raises(ParseError):
        extract_json("no braces here")


def test_values_converted_to_float():
    out = parse_values('{"a": "-30%", "b": "1,250.5", "c": 4, "d": "about 2e-3"}', ["a", "b", "c", "d"])
    assert out == {"a": -30.0, "b": 1250.5, "c": 4.0, "d": 0.002}


def test_missing_key_raises():
    with pytest.raises(ParseError, match="Missing keys"):
        parse_values('{"a": 1}', ["a", "b"])


def test_non_numeric_value_raises():
    with pytest.raises(ParseError):
        parse_values('{"a": "many"}', ["a"])
    with pytest.raises(ParseError):
        parse_values('{"a": true}', ["a"])


def test_extra_keys_are_ignored_and_only_requested_keys_returned():
    out = parse_values('{"a": 1, "b": 2, "comment": "text"}', ["a"])
    assert out == {"a": 1.0}


def test_keys_that_are_not_python_names_work():
    assert parse_values("{'social change': -12.5}", ["social change"]) == {"social change": -12.5}


def test_unicode_minus_is_read():
    assert parse_values('{"a": "\u221230%"}', ["a"]) == {"a": -30.0}


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_rejected(raw):
    with pytest.raises(ParseError, match="'a'"):
        parse_values('{"a": %s}' % raw, ["a"])


def test_bounds_accept_inside_and_reject_outside():
    assert parse_values('{"a": 50}', ["a"], {"a": (-100, 100)}) == {"a": 50.0}
    assert parse_values('{"a": 100}', ["a"], {"a": (-100, 100)}) == {"a": 100.0}
    with pytest.raises(ParseError, match="'a'.*100"):
        parse_values('{"a": 250}', ["a"], {"a": (-100, 100)})


def test_one_bad_value_among_several_names_the_bad_key():
    with pytest.raises(ParseError, match="'b'") as err:
        parse_values('{"a": 1, "b": "lots"}', ["a", "b"])
    assert "'a'" not in str(err.value)


def test_answer_model_is_a_pydantic_model_with_float_fields():
    model = build_answer_model(["a", "b"], {"a": (0, 1)})
    answer = model.model_validate({"a": "0.5", "b": 3})
    assert answer.model_dump(by_alias=True) == {"a": 0.5, "b": 3.0}
    assert all(isinstance(v, float) for v in answer.model_dump().values())
