"""Pure function: parse NCU --csv output into structured data."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def _parse_number(raw: str) -> float | str:
    """Try to convert a string to float, stripping comma thousands separators."""
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return raw


def _parse_block_grid(raw: str) -> int:
    """Extract total threads/blocks from NCU format like '(256, 1, 1)'."""
    m = re.findall(r"\d+", raw)
    if not m:
        return 0
    nums = [int(x) for x in m]
    result = 1
    for n in nums:
        result *= n
    return result


def list_kernels(filepath: str) -> list[tuple[int, str]]:
    """Return all (kernel_id, kernel_name) pairs found in the CSV."""
    path = Path(filepath)
    kernels: dict[int, str] = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kid = int(row.get("ID", "0"))
            kname = row.get("Kernel Name", "")
            if kid not in kernels and kname:
                kernels[kid] = kname
    return sorted(kernels.items())


def find_bottleneck_kernel(filepath: str) -> tuple[int, list[dict]]:
    """Discover the most expensive kernel in a multi-kernel NCU CSV.

    Scans all kernel_ids, extracts duration for each, and returns the
    kernel_id with the highest duration (the bottleneck).

    Args:
        filepath: Path to NCU CSV with potentially multiple kernel_ids.

    Returns:
        (bottleneck_kernel_id, all_kernels) where all_kernels is a list of
        dicts: [{"kernel_id": int, "kernel_name": str, "duration_ns": float}]
        sorted by duration descending.
    """
    path = Path(filepath)
    kernel_durations: dict[int, dict] = {}

    _DURATION_METRICS = {
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.avg.per_second",
    }

    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kid = int(row.get("ID", "0"))
            kname = row.get("Kernel Name", "")
            metric_name = row.get("Metric Name", "").strip()

            if kid not in kernel_durations and kname:
                kernel_durations[kid] = {
                    "kernel_id": kid,
                    "kernel_name": kname,
                    "duration_ns": 0.0,
                }

            if kid in kernel_durations and metric_name in ("Duration", "gpu__time_duration.sum"):
                raw = row.get("Metric Value", "0")
                val = _parse_number(raw)
                if isinstance(val, (int, float)):
                    unit = row.get("Metric Unit", "").strip().lower()
                    ns = val
                    if unit in ("us", "µs") or "usecond" in unit:
                        ns = val * 1000
                    elif unit == "ms" or "msecond" in unit:
                        ns = val * 1_000_000
                    elif unit == "s" or ("second" in unit and "nsecond" not in unit and "usecond" not in unit and "msecond" not in unit):
                        ns = val * 1_000_000_000
                    kernel_durations[kid]["duration_ns"] = ns

    all_kernels = sorted(
        kernel_durations.values(),
        key=lambda k: k["duration_ns"],
        reverse=True,
    )

    if not all_kernels:
        return 0, []

    # Classify kernels for the planner's context (but do NOT filter).
    # The planner decides the optimization strategy -- it may choose to
    # rewrite a library call as Triton, fuse it with adjacent kernels, etc.
    _LIBRARY_PATTERNS = ("cublas", "xmma_gemm", "cutlass", "cudnn", "cufft")
    _RUNTIME_PATTERNS = ("at::", "void at::", "distribution_elementwise")
    for k in all_kernels:
        name_lower = k["kernel_name"].lower()
        if any(p in name_lower for p in _RUNTIME_PATTERNS):
            k["kind"] = "runtime"
        elif any(p in name_lower for p in _LIBRARY_PATTERNS):
            k["kind"] = "library"
        else:
            k["kind"] = "user"

    # Return the most expensive kernel overall as the bottleneck.
    return all_kernels[0]["kernel_id"], all_kernels


def parse_ncu_csv(
    filepath: str, kernel_id: int = 0
) -> tuple[dict, list, dict]:
    """
    Parse an NCU CSV file for a specific kernel_id.

    Returns:
        sections: { "Section Name": { "Metric Name": {"value": v, "unit": u} } }
        rules:    [ {"name", "type", "description", "speedup_type", "speedup_pct"} ]
        launch:   flat dict of launch/kernel metadata
    """
    path = Path(filepath)
    sections: dict[str, dict[str, dict]] = {}
    rules: list[dict] = []
    launch: dict[str, str | int | float] = {}

    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = int(row.get("ID", "0"))
            if row_id != kernel_id:
                continue

            # Capture launch metadata from the first matching row
            if not launch:
                launch = {
                    "kernel_name": row.get("Kernel Name", ""),
                    "block_size": _parse_block_grid(row.get("Block Size", "")),
                    "grid_size": _parse_block_grid(row.get("Grid Size", "")),
                    "device": row.get("Device", ""),
                    "cc": row.get("CC", ""),
                    "process_name": row.get("Process Name", ""),
                }

            section_name = row.get("Section Name", "").strip()
            metric_name = row.get("Metric Name", "").strip()
            rule_name = row.get("Rule Name", "").strip()

            # Metric row: has Section Name + Metric Name, no Rule Name
            if section_name and metric_name and not rule_name:
                if section_name not in sections:
                    sections[section_name] = {}
                raw_value = row.get("Metric Value", "")
                unit = row.get("Metric Unit", "")
                sections[section_name][metric_name] = {
                    "value": _parse_number(raw_value),
                    "unit": unit,
                }

            # Rule row: has Rule Name, no Metric Name
            elif rule_name and not metric_name:
                rule_type = row.get("Rule Type", "")
                description = row.get("Rule Description", "")
                speedup_raw = row.get("Estimated Speedup", "").strip()
                speedup_type_raw = row.get("Estimated Speedup Type", "").strip()

                speedup_pct = None
                if speedup_raw:
                    try:
                        speedup_pct = float(speedup_raw.replace("%", "").replace(",", ""))
                    except ValueError:
                        pass

                rules.append({
                    "name": rule_name,
                    "type": rule_type,
                    "description": description,
                    "speedup_type": speedup_type_raw or None,
                    "speedup_pct": speedup_pct,
                })

    return sections, rules, launch
