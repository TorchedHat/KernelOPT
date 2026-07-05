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


def save_config(
    project_id: str = "",
    region: str = "us-east5",
    provider: str = "vertex",
    base_url: str = "",
    model: str = "",
) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'[provider]\nname = "{provider}"',
    ]
    if model:
        lines.append(f'model = "{model}"')
    if base_url:
        lines.append(f'base_url = "{base_url}"')
    lines.append("")
    if project_id:
        lines.append(f"[vertex]\nproject_id = \"{project_id}\"\nregion = \"{region}\"")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
