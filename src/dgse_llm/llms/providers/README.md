# llms/providers/

One file per LLM provider. Every non-underscore module here is imported automatically and registers itself with `@register_llm("<name>")`.

| Provider | Notes |
| --- | --- |
| `openai` | Responses API. Needs `pip install openai` and `OPENAI_API_KEY`. The key is read when the first call is made, never at import. |
| `mock` | Offline and deterministic: regex `rules` over the prompt, else `default_response`. Never cached. Used by all default tests. |

To add one, copy `_template.py`; see `../README.md`.
