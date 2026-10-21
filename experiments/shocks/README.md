# Shocks

One file per shock. A shock is only a description of an event; it has **no timing**. When it happens, and in what
order, is set in the `scenarios` of an experiment file.

```yaml
# experiments/shocks/pandemic_rumours.yaml   (the file name is the shock's name)
text: "Rumours are spreading that a dangerous new virus will reach the country within weeks."
description: Optional note for humans; not shown to the LLM.
```

In an experiment file, a scenario maps periods to shock names:

```yaml
scenarios:
  one_time:   {shocks: {4: pandemic_rumours}}
  mixed:      {shocks: {5: energy_price_spike, 12: sovereign_default, 20: technology_breakthrough}}
  repeated:   {shocks: {5: energy_price_spike, 12: energy_price_spike}}   # the same shock twice
  long:       {shocks: {4: pandemic_rumours, "8-11": pandemic_peak, 12: pandemic_recedes}}
  together:   {shocks: {4: [bank_collapse, energy_price_spike]}}          # two shocks in one period
```

- Each event is shown to the LLM in its own call; the answer (a relative % change) is applied in that period (or range)
  only. Nothing is held between events; periods that are not listed are not shocked.
- Shocks in the same period add up. The same shock at two periods reuses one LLM answer.
- Periods must be below the experiment's `T`.
- File names must be unique, which they are by construction. A file with any key other than `text` and `description`
  (for example a period) is rejected.
