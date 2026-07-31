"""Parse XLA autotuning results.

XLA's autotuner produces structured results via protos containing
per-fusion configs (block_m/n/k, num_stages, num_warps) and timings.

Access via:
  --xla_gpu_dump_autotune_results_to=<path>
  --xla_gpu_per_fusion_autotune_cache_dir=<dir>

References:
    - AutotuneResult proto: xla/autotuning.proto
    - AutotunerCache: xla/service/gpu/autotuning/autotuner_cache.h
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def parse_autotune_results(results_path: str) -> list[dict]:
    """Parse XLA autotune results from a dump file.

    Supports text proto format (.textproto / .txt).

    Args:
        results_path: Path to the autotune results file.

    Returns:
        List of dicts compatible with XLAAutotuneResult.
    """
    path = Path(results_path)
    if not path.exists():
        return []

    try:
        content = path.read_text()
    except Exception:
        return []

    if path.suffix in (".textproto", ".txt", ".pbtxt"):
        return _parse_text_proto(content)

    return []


def _parse_text_proto(content: str) -> list[dict]:
    """Parse text proto format autotune results."""
    results = []
    current: dict[str, Any] = {}

    for line in content.split("\n"):
        line = line.strip()

        if line.startswith("results {"):
            current = {
                "fusion_name": "",
                "block_m": 0, "block_n": 0, "block_k": 0,
                "num_stages": 0, "num_warps": 0,
                "run_time_us": 0.0,
            }
        elif line == "}" and current:
            results.append(current)
            current = {}
        elif ":" in line and current is not None:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"')

            if key == "device":
                current["fusion_name"] = val
            elif key == "block_m":
                current["block_m"] = int(val)
            elif key == "block_n":
                current["block_n"] = int(val)
            elif key == "block_k":
                current["block_k"] = int(val)
            elif key == "num_stages":
                current["num_stages"] = int(val)
            elif key == "num_warps":
                current["num_warps"] = int(val)
            elif key == "run_time" or key == "nanos":
                try:
                    current["run_time_us"] = float(val) / 1000.0
                except ValueError:
                    pass

    return results


def parse_per_fusion_cache(cache_dir: str) -> list[dict]:
    """Parse per-fusion autotune cache directory.

    Each file in the directory contains autotune results for one fusion.

    Args:
        cache_dir: Path to the per-fusion cache directory.

    Returns:
        List of dicts compatible with XLAAutotuneResult.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    results = []
    for f in sorted(cache_path.iterdir()):
        if not f.is_file():
            continue
        try:
            content = f.read_text()
            parsed = _parse_text_proto(content)
            for entry in parsed:
                entry["fusion_name"] = entry.get("fusion_name") or f.stem
            results.extend(parsed)
        except Exception:
            continue

    return results


def setup_autotune_dump_flags(dump_path: str) -> None:
    """Set XLA flags for autotune result dumping.

    Must be called BEFORE importing JAX.
    """
    existing = os.environ.get("XLA_FLAGS", "")
    flag = f"--xla_gpu_dump_autotune_results_to={dump_path}"
    if flag not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()
