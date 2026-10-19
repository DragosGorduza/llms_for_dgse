"""Tiny name -> object registry plus auto-import of every module in a package."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            self.add(name, obj)
            return obj

        return decorator

    def add(self, name: str, obj: Any) -> None:
        if name in self._items and self._items[name] is not obj:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._items[name] = obj

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def import_submodules(package_name: str) -> None:
    """Import every non-underscore module of a package so its @register decorators run."""
    package = importlib.import_module(package_name)
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{info.name}")
