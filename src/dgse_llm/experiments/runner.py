"""Expand an experiment into cells and run each one, writing a run directory per cell."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from dgse_llm.llms import LLMClient, PromptRenderer, build_llm
from dgse_llm.models import build_model
from dgse_llm.paths import runs_dir
from dgse_llm.schemas import (
    ExperimentConfig,
    LLMCall,
    LLMConfig,
    ModelConfig,
    RunManifest,
    Scenario,
    ShockSpec,
    stable_hash,
)
from dgse_llm.shocks import apply_shocks, build_shock_path
from dgse_llm.solvers import Solution, build_solver

from .metrics import compute_metrics
from .store import RunStore, write_index


@dataclass
class Cell:
    run_id: str
    model: ModelConfig
    llm: LLMConfig | None
    scenario: Scenario
    intensity: float | None
    mode: str | None
    repeat: int


def _slug(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "__".join(str(p) for p in parts))


def expand_cells(cfg: ExperimentConfig) -> list[Cell]:
    """Models x scenarios x LLMs (only if the scenario uses an LLM) x intensities x modes x repeats."""
    intensities = cfg.sweep.intensities or [None]
    modes = cfg.sweep.modes or [None]
    cells: list[Cell] = []
    for model in cfg.models:
        for scenario in cfg.scenarios:
            shocks = [s for s in scenario.shocks if s.model in (None, model.name)]
            if not shocks:
                continue
            scenario_for_model = scenario.model_copy(update={"shocks": shocks})
            needs_llm = any(s.source == "llm" for s in shocks)
            if needs_llm and not cfg.llms:
                raise ValueError(f"Scenario '{scenario.id}' uses an LLM but the experiment lists no llms")
            for llm in cfg.llms if needs_llm else [None]:
                for intensity in intensities:
                    for mode in modes:
                        for repeat in range(cfg.sweep.repeats):
                            run_id = _slug(
                                model.name, llm.name if llm else "nollm", scenario.id,
                                f"i{'-' if intensity is None else intensity}",
                                f"m{mode or '-'}", f"r{repeat}",
                            )
                            cells.append(Cell(run_id, model, llm, scenario_for_model, intensity, mode, repeat))
    return cells


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_config(cfg: ExperimentConfig, cell: Cell) -> dict:
    overrides = {}
    if cell.intensity is not None:
        overrides["intensity"] = cell.intensity
    if cell.mode is not None:
        overrides["mode"] = cell.mode
    scenario = cell.scenario.model_copy(
        update={"shocks": [s.model_copy(update=overrides) for s in cell.scenario.shocks]}
    )
    return {
        "experiment_id": cfg.id,
        "T": cfg.T,
        "model": cell.model.model_dump(mode="json"),
        "llm": cell.llm.model_dump(mode="json") if cell.llm else None,
        "scenario": scenario.model_dump(mode="json"),
        "intensity": cell.intensity,
        "mode": cell.mode,
        "repeat": cell.repeat,
    }


def run_cell(
    cfg: ExperimentConfig,
    cell: Cell,
    out_dir: Path,
    renderer: PromptRenderer | None = None,
    llms: dict[str, LLMClient] | None = None,
    baselines: dict | None = None,
    llm_factory: Callable[[LLMConfig], LLMClient] = build_llm,
) -> RunManifest:
    llms = llms if llms is not None else {}
    baselines = baselines if baselines is not None else {}
    resolved = _resolved_config(cfg, cell)
    store = RunStore(out_dir / cell.run_id)
    store.write_config(resolved)
    manifest = RunManifest(
        run_id=cell.run_id,
        experiment_id=cfg.id,
        config_hash=stable_hash(resolved),
        model=cell.model.name,
        llm=cell.llm.name if cell.llm else None,
        scenario=cell.scenario.id,
        intensity=cell.intensity,
        mode=cell.mode,
        repeat=cell.repeat,
        python=sys.version.split()[0],
        numpy=np.__version__,
        started=_now(),
    )
    calls: list[LLMCall] = []
    try:
        T = cfg.T
        model = build_model(cell.model)
        solver = build_solver(cell.model.solver, cell.model.solver_options)
        base_key = (model.cache_key(), cell.model.solver, json.dumps(cell.model.solver_options, sort_keys=True), T)
        base_exog = model.baseline_exog(T)
        if base_key not in baselines:
            baselines[base_key] = solver.solve(model, base_exog, T)
        baseline: Solution = baselines[base_key]

        llm = None
        if cell.llm is not None:
            if cell.llm.name not in llms:
                llms[cell.llm.name] = llm_factory(cell.llm)
            llm = llms[cell.llm.name]

        specs = [ShockSpec.model_validate(s) for s in resolved["scenario"]["shocks"]]
        paths = [
            build_shock_path(s, model, T, llm=llm, renderer=renderer, recorder=calls.append)
            for s in specs
        ]
        shocked = solver.solve(model, apply_shocks(model, base_exog, paths), T)

        # Classic/manual and LLM shocks are also solved separately when a scenario has both, so the report can
        # compare baseline, baseline + classic, baseline + LLM and baseline + classic + LLM.
        llm_paths = [p for p in paths if p.provenance.get("source") == "llm"]
        classic_paths = [p for p in paths if p.provenance.get("source") != "llm"]
        solved: dict[str, Solution] = {"baseline": baseline, "shocked": shocked}
        if llm_paths and classic_paths:
            solved["classic"] = solver.solve(model, apply_shocks(model, base_exog, classic_paths), T)
            solved["llm"] = solver.solve(model, apply_shocks(model, base_exog, llm_paths), T)

        def diff(sol: Solution) -> dict[str, np.ndarray]:
            return {k: sol.paths[k] - baseline.paths[k] for k in sol.paths}

        irf = diff(shocked)
        tracked = [v for v in (cfg.tracked or []) if v in irf] or sorted(irf)
        scenarios = {"baseline": baseline.paths, "shocked": shocked.paths, "irf": irf}
        if "classic" in solved:
            irf_classic, irf_llm = diff(solved["classic"]), diff(solved["llm"])
            # Effect of combining the two beyond the sum of their separate effects (zero in a linear model).
            interaction = {k: irf[k] - irf_classic[k] - irf_llm[k] for k in irf}
            scenarios.update(
                classic=solved["classic"].paths, llm=solved["llm"].paths,
                irf_classic=irf_classic, irf_llm=irf_llm, irf_interaction=interaction,
            )
            store.write_arm_metrics({
                "classic": compute_metrics(irf_classic, tracked),
                "llm": compute_metrics(irf_llm, tracked),
                "classic_llm": compute_metrics(irf, tracked),
                "interaction": compute_metrics(interaction, tracked),
            })
        store.write_shock_paths(paths)
        store.write_trajectories(scenarios, tracked)
        store.write_metrics(compute_metrics(irf, tracked))
        manifest.modified = [
            {"variable": p.variable, "mode": p.mode, "source": p.provenance.get("source"), "shock_id": p.shock_id}
            for p in paths
        ]
        manifest.arms = [a for a in ("baseline", "classic", "llm", "shocked") if a in solved]
        manifest.diagnostics = {name: sol.diagnostics for name, sol in solved.items()}
        manifest.status = "ok"
    except Exception as exc:  # one failing cell must not abort a sweep
        manifest.status = "failed"
        manifest.error = f"{type(exc).__name__}: {exc}"
    finally:
        manifest.n_llm_calls = len(calls)
        manifest.finished = _now()
        store.write_llm_calls(calls)
        store.write_manifest(manifest)
    return manifest


def run_experiment(
    cfg: ExperimentConfig,
    out_root: str | Path | None = None,
    renderer: PromptRenderer | None = None,
    llm_factory: Callable[[LLMConfig], LLMClient] = build_llm,
    on_cell: Callable[[RunManifest], None] | None = None,
) -> list[RunManifest]:
    root = Path(out_root or cfg.out_dir or runs_dir())
    exp_dir = root / cfg.id
    renderer = renderer or PromptRenderer()
    llms: dict[str, LLMClient] = {}
    baselines: dict = {}
    manifests: list[RunManifest] = []
    for cell in expand_cells(cfg):
        manifest = run_cell(cfg, cell, exp_dir, renderer, llms, baselines, llm_factory)
        manifests.append(manifest)
        if on_cell:
            on_cell(manifest)
    write_index(exp_dir, manifests)
    return manifests
