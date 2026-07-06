"""Optimization memory: bounded queue of experience items."""

from __future__ import annotations

import json
from pathlib import Path

EXPN = 8        # queue capacity (default)
T_POS = 1.04    # min speedup to store positive item (empirically tuned)
T_NEG = 1.15    # min slowdown factor to store negative item (empirically tuned)
TOP_K = 4       # max items added per iteration

# Seeds for reproducible correctness testing.  Three random seeds plus
# input diversity; used by both the agentic correctness agent and the
# compiler-first correctness path.
CORRECTNESS_SEEDS = [0, 42, 1337]


def should_store(speedup: float, t_pos: float = T_POS, t_neg: float = T_NEG) -> bool:
    """Return True if the speedup (or slowdown) is significant enough."""
    if speedup >= t_pos:
        return True
    if speedup > 0 and (1.0 / speedup) >= t_neg:
        return True
    return False


def select_top_k(
    items: list[dict], top_k: int = TOP_K,
    t_pos: float = T_POS, t_neg: float = T_NEG,
) -> list[dict]:
    """
    From a list of candidate experience items, select the top_k most
    significant ones (by |speedup - 1.0|) that pass curation thresholds.
    """
    filtered = [
        item for item in items
        if should_store(item.get("speedup", 1.0), t_pos, t_neg)
    ]
    filtered.sort(key=lambda x: abs(x.get("speedup", 1.0) - 1.0), reverse=True)
    return filtered[:top_k]


def append_item(memory: list[dict], item: dict, capacity: int = EXPN) -> list[dict]:
    """Append an experience item, evicting the oldest if over capacity."""
    memory = list(memory)
    memory.append(item)
    if len(memory) > capacity:
        memory = memory[-capacity:]
    return memory


def append_items(
    memory: list[dict], items: list[dict], capacity: int = EXPN,
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
