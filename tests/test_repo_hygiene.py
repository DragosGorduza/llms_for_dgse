"""Repository hygiene.

Expect: every code and config folder has a README, every model has a .md, no key-like strings are committed, and the
git ignore rules keep secrets and run outputs out of version control.
"""

import re
from pathlib import Path

import pytest

from dgse_llm.paths import repo_root

ROOT = repo_root()
SRC = ROOT / "src" / "dgse_llm"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".llm_cache", "runs", "node_modules"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".env", ".ipynb", ".j2", ".txt", ".json", ".cfg", ".ini"}
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


def package_dirs():
    return sorted(
        p for p in SRC.rglob("*") if p.is_dir() and p.name != "__pycache__" and not p.name.startswith("_")
    )


@pytest.mark.parametrize("folder", package_dirs(), ids=lambda p: str(p.relative_to(ROOT)))
def test_source_folders_have_readmes(folder):
    assert (folder / "README.md").exists()


@pytest.mark.parametrize(
    "folder",
    ["configs", "experiments", "experiments/shocks", "prompts", "secrets", "runs", "tests"],
)
def test_top_level_folders_have_readmes(folder):
    assert (ROOT / folder / "README.md").exists()


def test_every_model_folder_in_experiments_has_a_readme_and_prompts_folder():
    from dgse_llm.models import MODEL_REGISTRY

    for name in MODEL_REGISTRY.names():
        assert (ROOT / "experiments" / name / "README.md").exists(), name
        assert (ROOT / "prompts" / name / "prompts.yaml").exists(), name


def test_root_readme_exists():
    assert (ROOT / "README.md").exists()


def test_every_model_has_a_description_file():
    from dgse_llm.models import MODEL_REGISTRY

    for name in MODEL_REGISTRY.names():
        assert (SRC / "models" / f"{name}.py").exists() and (SRC / "models" / f"{name}.md").exists(), name


def test_no_api_keys_in_repository():
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix not in TEXT_SUFFIXES or path == ROOT / "secrets" / "secrets.env":
            continue
        if KEY_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    # Test files may contain obviously fake keys; real ones must not appear anywhere else.
    offenders = [o for o in offenders if not o.startswith("tests")]
    assert not offenders, f"key-like strings found in: {offenders}. Remove and rotate them."


def test_gitignore_protects_secrets_and_outputs():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for rule in ("secrets/*", "!secrets/README.md", "!secrets/secrets.example.env", "runs/*", ".llm_cache/"):
        assert rule in lines, rule


def test_example_secrets_file_has_no_values():
    for line in (ROOT / "secrets" / "secrets.example.env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            assert line.split("=", 1)[1].strip() == "", "example file must contain blank placeholders only"


def test_configs_never_contain_inline_keys():
    for folder in ("configs", "experiments"):
        for path in (ROOT / folder).rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            assert "api_key:" not in text, f"{path.name}: use api_key_env, never a key"
