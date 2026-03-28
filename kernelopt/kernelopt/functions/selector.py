"""Pure function: compare candidates and select the best by latency."""

from __future__ import annotations

from kernelopt.state import PlanCandidate


def select_best(candidates: list[PlanCandidate]) -> PlanCandidate | None:
    """Return the candidate with the lowest latency, or None if empty."""
    valid = [c for c in candidates if c.latency_ns < float("inf")]
    if not valid:
        return None
    return min(valid, key=lambda c: c.latency_ns)
