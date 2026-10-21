# configs/

System-wide configuration only: `plotting.yaml` (one colour per macro model, line styles, font sizes, figure size, used by
every report figure; see `src/dgse_llm/reporting/README.md`). Add a colour there when you add a model.

Everything about a particular test (model and its parameters, LLMs with temperature and key reference, prompts, personas,
scenarios, tracked variables) lives in [experiments/](../experiments/README.md). Prompts live in
[prompts/](../prompts/README.md).

Never put a key in any YAML; `api_key_env` in an experiment is the name of the secret in `secrets/secrets.env`.
