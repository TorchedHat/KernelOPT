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
