"""Locations of the repo folders; each can be overridden by an environment variable."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("DGSE_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return Path(os.environ.get("DGSE_CONFIGS_DIR", repo_root() / "configs"))


def experiments_dir() -> Path:
    return Path(os.environ.get("DGSE_EXPERIMENTS_DIR", repo_root() / "experiments"))


def prompts_dir() -> Path:
    return Path(os.environ.get("DGSE_PROMPTS_DIR", repo_root() / "prompts"))


def secrets_dir() -> Path:
    return Path(os.environ.get("DGSE_SECRETS_DIR", repo_root() / "secrets"))


def runs_dir() -> Path:
    return Path(os.environ.get("DGSE_RUNS_DIR", repo_root() / "runs"))


def llm_cache_dir() -> Path:
    return Path(os.environ.get("DGSE_LLM_CACHE_DIR", repo_root() / ".llm_cache"))
