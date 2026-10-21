# configs/llms/

One `LLMConfig` per file: `name`, `provider` (a registered provider), `model_id`, `temperature`, `seed`,
`max_tokens`, `api_key_env` (the NAME of the secret in `secrets/secrets.env`), `cache`, and provider `options`.
The `mock_*` files answer from regex rules and need no key; `openai_gpt4o_mini` needs `OPENAI_API_KEY`. The OpenAI
provider uses the operating system certificate store. For a managed HTTPS proxy, set `options.ca_bundle` to its PEM
CA bundle path.
