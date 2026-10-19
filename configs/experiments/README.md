# configs/experiments/

One `ExperimentConfig` per file. `models`, `llms` and each scenario's `shocks` are lists of names of files in the
sibling folders (or inline mappings). `sweep` loops over intensities, modes and repeats. Outputs go to
`runs/<id>/`. `ks_pandemic_openai` is the only file that calls a paid API.
