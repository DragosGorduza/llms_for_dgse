# evaluation/

Like-with-like comparison of the specifications inside one experiment run. Reads only the files of
`runs/<experiment_id>/` (`index.parquet`, each ok run's `metrics.json` and `trajectories.parquet`) and writes
`runs/<experiment_id>/evaluations/`.

```python
from dgse_llm.evaluation import evaluate_experiment
evaluate_experiment("ks_mock")            # name under runs/, or a path to the run directory
```

Command line: `dgse-llm evaluate ks_mock` (optionally `--axes model_info llm`), or `dgse-llm run <experiment> --evaluate`
to evaluate right after the run.

## Logic

- **Axes**: `llm`, `prompt`, `persona`, `scenario`, `model_info`. Default: `model_info`, `llm`, `persona`, `scenario`
  (`prompt` changes the shocked input, so it is opt-in). Axes with one level are skipped.
- **Like with like**: along an axis every other axis is held fixed (a "cell"); only runs inside the same cell are
  paired. Repeats (draws) of a cell are averaged first; their spread gives a band in the figures and a `z` score.
- **Pairs**: every pair of levels (a, b), ordered `without -> with` for `model_info` and alphabetically otherwise;
  differences are always b - a.
- **Statistics** per pair, cell, variable and metric (`change_pct` = the raw LLM answer, stored as variable
  `llm_answer`; `peak`, `peak_t`, `cumulative`, `half_life` of each impulse response), and trajectory distances
  (RMSE, relative RMSE, correlation, largest gap and its period, summed difference) between mean impulse responses.
- **Axis importance**: per cell, largest minus smallest mean across the axis's levels, averaged over cells, to see
  which axis moves answers and outcomes most.

## Outputs (`evaluations/`)

| File | Content |
| --- | --- |
| `evaluation.md` | Axis importance, then per axis: summary of answers and outcomes, trajectory distances, answers per cell, figures |
| `cells.parquet`, `irf_cells.parquet` | Mean / std / n per cell, and the mean impulse response per cell with lowest and highest draw |
| `axis_importance.parquet` | Range across levels per axis, variable and metric |
| `<axis>/pairs.parquet`, `pairs_summary.parquet` | Paired differences per cell, and their summary over cells |
| `<axis>/trajectory_pairs.parquet`, `trajectory_pairs_summary.parquet`, `mean_difference.parquet` | Path distances per cell, summary, and mean b - a path over cells |
| `<axis>/figures/` | `cell__*.png` (one per cell, one line per level), `mean_difference__a__b.png`, `scatter_peak__a__b.png`, `scatter_cumulative__a__b.png` |

Figures follow `configs/plotting.yaml`: the model colour throughout, levels told apart by line style and shade,
repeat or cross-cell ranges as bands with `band_alpha`, grey only for zero and 45-degree reference lines.
