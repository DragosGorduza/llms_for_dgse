# configs/shocks/

One `ShockSpec` per file. Fields: `id`, `variable` (an exogenous, shockable input), optional `model`, `mode`
(`additive` or `replace`), `profile` (`one_time`, `multi_period`, `geometric_decay`, `from_series`,
`narrative_timeline`), `intensity`, `source` (`manual` or `llm`) and, for LLM shocks, an `llm` block
(`template`, `keys`, `mapping`, `mapping_params`, `events`, `tail_decay`). See `src/dgse_llm/shocks/README.md`.
Shocks that name a `model` are only applied to that model.
