"""Read API keys from the environment or secrets/secrets.env. Keys are never logged."""

from __future__ import annotations

import os
from pathlib import Path

from dgse_llm.paths import secrets_dir


class MissingSecretError(RuntimeError):
    pass


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values


def get_secret(name: str, secrets_file: Path | None = None) -> str:
    """Environment variable first, then secrets/secrets.env. Empty values count as missing."""
    value = os.environ.get(name, "")
    if value:
        return value
    path = Path(secrets_file) if secrets_file else secrets_dir() / "secrets.env"
    if path.exists():
        value = _parse_env_file(path).get(name, "")
        if value:
            return value
    raise MissingSecretError(
        f"Secret '{name}' not found. Set the environment variable or add it to {path} "
        f"(see secrets/README.md)."
    )
