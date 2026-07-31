"""Parse XLA FusionProcessDump for fusion decision analysis.

XLA's PriorityFusion pass logs every fusion decision as a protobuf.
Each step is one of: Fusion, ProducerIneligible, UpdatePriority.

ProducerIneligible records the producer name + rejection reason,
which is XLA's equivalent of Inductor's WhyNoFuse.

Enable via: XLA_FLAGS='--xla_dump_to=/path --xla_dump_hlo_as_proto'

References:
    - FusionProcessDump proto: xla/service/gpu/fusion_process_dump.proto
    - PriorityFusion: xla/backends/gpu/transforms/priority_fusion.cc
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def parse_fusion_dump_text(dump_dir: str) -> list[dict]:
    """Parse text-format fusion process dumps from XLA dump directory.

    Falls back to parsing the optimized HLO text if no proto dump exists.

    Args:
        dump_dir: Path to XLA dump directory (from --xla_dump_to).

    Returns:
        List of fusion decision dicts compatible with XLAFusionDecision.
    """
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return []

    decisions = []

    for f in sorted(dump_path.glob("*.fusion_process_dump*")):
        try:
            content = f.read_text(errors="replace")
            decisions.extend(_parse_fusion_text(content))
        except Exception:
            continue

    if not decisions:
        for f in sorted(dump_path.glob("*after_optimizations*.txt")):
            try:
                content = f.read_text(errors="replace")
                decisions.extend(_extract_fusions_from_hlo(content))
            except Exception:
                continue

    return decisions


def _parse_fusion_text(content: str) -> list[dict]:
    """Parse a text-format fusion process dump."""
    decisions = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        if "fusing" in line.lower() and "into" in line.lower():
            parts = re.findall(r"(\w+)", line)
            if len(parts) >= 3:
                decisions.append({
                    "step_type": "Fusion",
                    "producer_name": parts[0] if parts else "",
                    "consumer_name": parts[-1] if parts else "",
                    "reason": line[:200],
                })
        elif "cannot" in line.lower() or "ineligible" in line.lower():
            decisions.append({
                "step_type": "ProducerIneligible",
                "producer_name": "",
                "consumer_name": "",
                "reason": line[:200],
            })

    return decisions


def _extract_fusions_from_hlo(hlo_text: str) -> list[dict]:
    """Extract fusion decisions from optimized HLO text.

    The optimized HLO contains fused computations named like
    %fused_reduce, %fused_computation.1, etc.
    """
    decisions = []

    for match in re.finditer(r"%(\w*fused\w*)\s*\(", hlo_text):
        decisions.append({
            "step_type": "Fusion",
            "producer_name": match.group(1),
            "consumer_name": "",
            "reason": f"Fused computation: {match.group(1)}",
        })

    return decisions


def setup_xla_dump_flags(dump_dir: str) -> None:
    """Set XLA environment flags for fusion dump capture.

    Must be called BEFORE importing JAX.
    """
    flags = [
        f"--xla_dump_to={dump_dir}",
        "--xla_dump_hlo_as_text",
    ]
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = " ".join(filter(None, [existing] + flags))
