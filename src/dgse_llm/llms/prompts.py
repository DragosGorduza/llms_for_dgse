"""Render Jinja prompt templates from the prompts/ folder."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dgse_llm.paths import prompts_dir


class PromptRenderer:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else prompts_dir()
        self._env = Environment(
            loader=FileSystemLoader(str(self.directory)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def render(self, template: str, **variables) -> str:
        return self._env.get_template(template).render(**variables)
