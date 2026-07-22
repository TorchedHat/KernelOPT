"""Optimization memory: bounded queue of experience items.

Iterative experience curation inspired by AccelOpt
(Zhang et al., "AccelOpt: A Self-Improving LLM Agentic System for
AI Accelerator Kernel Optimization", MLSys 2026).
Extended with cross-run strategy tracking, semantic retrieval,
and pre-seeded benchmark guidelines.
"""

from __future__ import annotations

import json
from pathlib import Path

Q_CAP = 8        # queue capacity (default)
SPE_GATE = 1.05  # min speedup to store positive item
SLO_GATE = 1.20  # min slowdown factor to store negative item
KEEP_N = 4       # max items added per iteration

# Seeds for reproducible correctness testing.  Three random seeds plus
# input diversity; used by both the agentic correctness agent and the
# compiler-first correctness path.
CORRECTNESS_SEEDS = [0, 42, 1337]


def should_store(speedup: float, spe_gate: float = SPE_GATE, slo_gate: float = SLO_GATE) -> bool:
    """Return True if the speedup (or slowdown) is significant enough."""
    if speedup >= spe_gate:
        return True
    if speedup > 0 and (1.0 / speedup) >= slo_gate:
        return True
    return False


def select_top_k(
    items: list[dict], keep_n: int = KEEP_N,
    spe_gate: float = SPE_GATE, slo_gate: float = SLO_GATE,
) -> list[dict]:
    """
    From a list of candidate experience items, select the keep_n most
    significant ones (by |speedup - 1.0|) that pass curation thresholds.
    """
    filtered = [
        item for item in items
        if should_store(item.get("speedup", 1.0), spe_gate, slo_gate)
    ]
    filtered.sort(key=lambda x: abs(x.get("speedup", 1.0) - 1.0), reverse=True)
    return filtered[:keep_n]


def append_item(memory: list[dict], item: dict, capacity: int = Q_CAP) -> list[dict]:
    """Append an experience item, evicting the oldest if over capacity."""
    memory = list(memory)
    memory.append(item)
    if len(memory) > capacity:
        memory = memory[-capacity:]
    return memory


def append_items(
    memory: list[dict], items: list[dict], capacity: int = Q_CAP,
) -> list[dict]:
    """Append multiple items, respecting capacity."""
    memory = list(memory)
    for item in items:
        memory.append(item)
    if len(memory) > capacity:
        memory = memory[-capacity:]
    return memory


def load_memory(path: str | Path | None) -> list[dict]:
    """Load optimization memory from a JSON file."""
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_memory(memory: list[dict], path: str | Path) -> None:
    """Save optimization memory to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(memory, f, indent=2)
