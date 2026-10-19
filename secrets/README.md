# secrets/

Holds API keys. Nothing here except this file and `secrets.example.env` is tracked by git.

## Setup

1. Copy `secrets.example.env` to `secrets.env`.
2. Fill in the keys you need (e.g. `OPENAI_API_KEY=...`).

Environment variables with the same names override the file.

## Rules

- Never paste a key into a notebook, config, prompt or test.
- An `LLMConfig` only names the secret (`api_key_env`), it never contains it.
- Keys are never written to `runs/`, `llm_calls.jsonl` or the manifest.
- `tests/test_repo_hygiene.py` fails if a key-like string (`sk-...`) appears in tracked source.

## Expect

If a key is missing, the loader raises `MissingSecretError` naming the variable and this file.
Tests with the mock LLM never need a key.
