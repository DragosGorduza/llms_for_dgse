"""Load experiment YAML; string entries are resolved to files under configs/{models,llms,shocks}/."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from dgse_llm.paths import configs_dir
from dgse_llm.schemas import ExperimentConfig, load_yaml


def _resolve(entry: Any, folder: Path) -> Any:
    if isinstance(entry, str):
        return load_yaml(folder / f"{entry}.yaml")
    return entry


def resolve_experiment(raw: dict[str, Any], configs_root: Path | None = None) -> ExperimentConfig:
    root = Path(configs_root) if configs_root else configs_dir()
    raw = deepcopy(raw)
    raw["models"] = [_resolve(m, root / "models") for m in raw.get("models", [])]
    raw["llms"] = [_resolve(m, root / "llms") for m in raw.get("llms", [])]
    for scenario in raw.get("scenarios", []):
        scenario["shocks"] = [_resolve(s, root / "shocks") for s in scenario.get("shocks", [])]
    return ExperimentConfig.model_validate(raw)


def load_experiment(path: str | Path, configs_root: Path | None = None) -> ExperimentConfig:
    return resolve_experiment(load_yaml(path), configs_root)
