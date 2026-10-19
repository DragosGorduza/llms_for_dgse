# configs/models/

One `ModelConfig` per file: `name` (registered model), `params` (overrides of the model's `default_params`),
`solver` and `solver_options`. Unknown parameter names raise an error when the model is built.
Add a file when you add a model, or to create a variant (for example `krusell_smith_500.yaml` with `params: {nA: 500}`).
