"""Config file I/O for kernelopt — reads/writes ~/.config/kernelopt/config.toml."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


def config_path() -> Path:
    return Path.home() / ".config" / "kernelopt" / "config.toml"


def load_config() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def save_config(project_id: str, region: str) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[vertex]\n"
        f'project_id = "{project_id}"\n'
        f'region = "{region}"\n'
    )
    path.write_text(content, encoding="utf-8")
    return path
