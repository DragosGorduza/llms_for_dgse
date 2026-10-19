"""Extract and validate numeric values from free-form LLM text. Types are enforced with pydantic."""

from __future__ import annotations

import ast
import json
import re
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, create_model


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """Return the LAST top-level JSON-like object in the text (models often reason first, answer last)."""
    decoder = json.JSONDecoder()
    found: dict | None = None
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            obj, end = None, None
        if isinstance(obj, dict):
            found = obj
            i += end
            continue
        # Fallback for single-quoted, Python-style dicts.
        depth, stop = 0, None
        for j in range(i, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                stop = j + 1
                break
        if stop is not None:
            try:
                obj = ast.literal_eval(text[i:stop])
            except (ValueError, SyntaxError):
                obj = None
            if isinstance(obj, dict):
                found = obj
                i = stop
                continue
        i += 1
    if found is None:
        raise ParseError("No JSON object found in LLM response")
    return found


def _to_float(value) -> float:
    """Lenient pre-conversion: numbers pass, strings such as '-30%' or '1,250.5' are read, anything else is rejected."""
    if isinstance(value, bool):
        raise ParseError(f"Boolean is not numeric: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.replace(",", "").replace("\u2212", "-"))
        if match:
            return float(match.group(0))
    raise ParseError(f"Cannot convert {value!r} to a number")


def build_answer_model(
    keys: list[str], bounds: dict[str, tuple[float, float]] | None = None
) -> type[BaseModel]:
    """Pydantic model for an LLM answer: one finite float per requested key, optionally inside (low, high).

    Extra keys in the answer are ignored. Fields are named v0, v1, ... with the answer key as alias, so keys that are
    not valid Python names (e.g. 'social change') work.
    """
    bounds = bounds or {}
    fields = {}
    for i, key in enumerate(keys):
        lo, hi = bounds.get(key, (None, None))
        fields[f"v{i}"] = (
            Annotated[float, BeforeValidator(_to_float)],
            Field(alias=key, ge=lo, le=hi, allow_inf_nan=False),
        )
    return create_model("LLMAnswer", __config__=ConfigDict(extra="ignore"), **fields)


def parse_values(
    text: str, keys: list[str], bounds: dict[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    """Extract the last JSON object from `text` and validate it with pydantic.

    Parameters
    ----------
    text : str                     the raw LLM answer.
    keys : list[str]               keys that must be present, each convertible to a finite float.
    bounds : dict[str, (float, float)] | None   optional (low, high) plausibility range per key, inclusive.

    Returns
    -------
    dict[str, float] with exactly `keys` (extra keys in the answer are dropped).

    Raises ParseError if there is no JSON object, a key is missing, a value is not a finite number, or a value is
    outside its bounds.
    """
    data = extract_json(text)
    missing = [k for k in keys if k not in data]
    if missing:
        raise ParseError(f"Missing keys {missing} in LLM response; got {sorted(data)}")
    try:
        answer = build_answer_model(keys, bounds).model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"'{e['loc'][0]}': {e['msg']}" for e in exc.errors())
        raise ParseError(f"Invalid LLM answer: {problems}") from exc
    dumped = answer.model_dump(by_alias=True)
    return {k: dumped[k] for k in keys}
