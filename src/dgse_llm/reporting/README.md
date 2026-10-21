# reporting/

`report.py` builds markdown from the files in a run or experiment directory only, so any past run can be re-reported.

- `write_run_report(run_dir)` -> `report.md` and `figures/`: setup, modified variables, shock-path figure, every LLM
  prompt and answer (truncated; full text in `llm_calls.jsonl`), model-path comparison, IRF figure, metrics, solver
  diagnostics, provenance. Trajectory and IRF panels mark each shock's onset with a dashed vertical line and a legend
  label; persistent shocks get one marker at their start. When a scenario mixes classic and LLM shocks the figures show four economies per variable
  (base-model, + classic shock, + LLM shock, + both) and the report adds peak and cumulative tables with an
  `interaction` column.
- `write_experiment_report(exp_dir)` -> `experiment_report.md`: table of all runs (with prompt, narrative, persona,
  timing, model info and raw answer) and failures, the distribution of answers per cell (n draws, mean, std, min,
  median, max), the effect of showing the model to the LLM (answer with minus without model info), peak response per
  run, overlay of IRFs across runs. Needs `index.parquet`.
- `md_table(rows, columns)` renders a markdown table.

`figures/trajectories.png` compares the base-model (the stored unshocked baseline) with each shocked economy, all in the
model's colour, told apart by line style and shade. With only one kind of shock there is one shocked line (`shocked`,
everything applied).
With both classic and LLM shocks there are three: `classic` (without the LLM shock), `llm` (without the classic shock)
and `shocked` (both), so the LLM contribution can be separated from the classic one. Baselines are solved and cached
per model, solver/options and horizon during an experiment, and saved with every successful run. IRF figures show each
economy minus the baseline, so their grey dashed `base-model` reference line is zero. Shock-path figures show injected
shocks, not model outcomes.

CLI: `dgse-llm report <dir>` picks the experiment report if the directory has `index.parquet`, else the run report.

## Colours

`configs/plotting.yaml` is the single source of truth, read through `style.py`:

| Item | Colour use |
| --- | --- |
| `model_colors` | One colour per macro model (`rbc` blue, `nk3` vermillion, `krusell_smith` green; colour-blind-safe Okabe-Ito). Every line of that model in every figure uses it. |
| `arms` | The economies inside one run, all in the model colour, told apart by line style and shade (alpha), same line width: base-model (dashed, faintest), + classic shock (dash-dot), + LLM shock (dotted), + both (solid, fully opaque). |
| `run_linestyles`, `run_alphas` | In the experiment overlay, runs of the same model (other scenario, intensity, mode, repeat) keep the model colour; the first is solid and opaque, later ones get another line style and a fainter shade. Both cycle. |
| `neutral_color` | Grey dashed zero line (`base-model` reference) in impulse-response figures. Never used for a model. |
| `event_color` | Dark grey dashed vertical shock-event lines in every figure (run, experiment and evaluation), whatever the model. |
| `fonts` | Font sizes in points for every figure: `suptitle` (whole figure), `title` (each panel), `axis_label`, `tick`, `legend`. No font size is hardcoded in `report.py`; a test enforces it. |
| `figure` | Figure size: `panel_width` and `panel_height` in inches per panel, `max_columns` panels per row, and `dpi` of the saved png. Total figure size = panel size x columns/rows, so figures with more variables grow. Also not hardcoded. |
| `fallback_palette` | Deterministic colour (by model name) for a model not yet listed. |

When you add a model, add its colour to `model_colors`; `tests/reporting/test_style.py` fails otherwise, and also
fails if two models share a colour.
Re-report a run directory to generate the comparison from saved trajectories without solving again or calling an LLM.

## Tests (`tests/reporting`)

Report files exist and contain the expected sections, LLM prompts appear, failed runs show their error, the
experiment report lists every run, and comparison plots show the stored baseline values with an orange legend entry.
