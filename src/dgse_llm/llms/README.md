# llms/

Run any LLM on any prompt. Nothing here knows about macro models.

| Piece | File | Purpose |
| --- | --- | --- |
| `LLMClient`, `LLMRequest`, `LLMResponse`, registry, disk cache | `base.py` | Provider-independent interface. `complete()` checks the cache, then calls the provider's `_complete()`. |
| Providers | `providers/` | `openai` (Responses API) and `mock` (offline, rule-driven). |
| `PromptRenderer` | `prompts.py` | Jinja templates from `prompts/`; undefined variables raise. |
| `parse_values`, `extract_json`, `build_answer_model` | `parsing.py` | Takes the LAST JSON object in the text, accepts single quotes, strips `%`, commas and unicode minus, then validates it with a pydantic model: every requested key must be present, convertible to a finite float (no NaN or infinity, no booleans) and inside its optional `answer_bounds`. Extra keys are dropped. Raises `ParseError` naming the bad key. |
| `get_secret` | `secrets.py` | Environment variable, then `secrets/secrets.env`. Raises `MissingSecretError`. |

## Validated types

Pydantic checks both directions of an LLM call:

- **What the provider returns:** `LLMResponse` is a pydantic model (`text` must be a string, token counts integers,
  latency non-negative, no unknown fields), so a provider returning the wrong type fails immediately and nothing
  invalid is cached or logged.
- **What the LLM says:** `parse_values` builds a pydantic model from the shock's `llm.keys` (floats, finite) and
  `llm.answer_bounds` (inclusive plausibility range per key, in the units the prompt asks for) and validates the
  extracted JSON against it. A failed validation is recorded in `llm_calls.jsonl` with the error and stops that run
  (the sweep continues); the bad number never reaches the model.

## Cache

Only answers at `temperature: 0` are cached. With a temperature above 0, or no temperature at all (set
`temperature: null` for models that reject it, such as reasoning models), nothing is cached: every call is a fresh draw
from the model's response distribution, which is what repeated runs are meant to sample. Cached key = hash of
provider, model id, temperature, seed, max tokens, prompt and `cache_salt`; stored in `.llm_cache/` (gitignored).
The mock provider never caches.

## Add an LLM

1. Copy `providers/_template.py` to `providers/<name>.py`; implement `_complete` and decorate with `@register_llm("<name>")`. It is auto-discovered.
2. Read the key with `get_secret(self.config.api_key_env or "<NAME>_API_KEY")` inside `_complete`, never at import time.
3. Add `configs/llms/<name>.yaml` and the key name to `secrets/secrets.example.env`.

## Mock provider

`options.rules` is a list of `{pattern, response}`; the first regex matching the prompt wins; otherwise
`options.default_response`. Use patterns on the narrative text, since the persona and question text are in every prompt.

## Tests (`tests/llms`)

Mock rules and default, cache hit and salt, parser edge cases (nested JSON, single quotes, `%`, missing keys,
last object wins, NaN/infinity, bounds, extra keys), `LLMResponse` type checks (`test_response.py`), secrets lookup and
missing-key error, prompt rendering. Live OpenAI tests are marked `live`
and skipped unless `--run-live` is given and a key exists.
