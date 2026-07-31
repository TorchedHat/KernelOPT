"""Read Inductor's autotuning results after compilation.

The AlgorithmSelectorCache stores benchmarked timings for template
operations (GEMM, conv). After compilation completes, we read the
local (in-memory) cache to extract what was tried and what won.

References:
    - AlgorithmSelectorCache: select_algorithm.py line 3949
    - get_algorithm_selector_cache(): select_algorithm.py line 6087
"""

from __future__ import annotations


def extract_autotune_state() -> list[dict]:
    """Read AlgorithmSelectorCache after compilation.

    Returns list of autotune entry dicts. Empty if no template ops
    were autotuned or the cache is not accessible.
    """
    try:
        from torch._inductor.select_algorithm import get_algorithm_selector_cache
    except ImportError:
        return []

    try:
        cache = get_algorithm_selector_cache()
    except Exception:
        return []

    local = getattr(cache, "local_cache", None)
    if local is None:
        local = getattr(cache, "_local_cache", None)
    if local is None:
        return []

    results = []
    for key, value in local.items():
        if not isinstance(value, dict):
            continue

        timings = []
        for choice_key, timing in value.items():
            if isinstance(timing, (int, float)) and timing < float("inf"):
                timings.append({
                    "config": str(choice_key)[:200],
                    "timing_ms": float(timing),
                })

        if not timings:
            continue

        timings.sort(key=lambda x: x["timing_ms"])
        winner = timings[0]
        runner_up = timings[1] if len(timings) > 1 else None

        results.append({
            "operation": str(key)[:300],
            "num_configs_tried": len(timings),
            "winner_type": "",
            "winner_timing_ms": winner["timing_ms"],
            "runner_up_type": "",
            "runner_up_timing_ms": runner_up["timing_ms"] if runner_up else 0.0,
            "all_timings": tuple(timings[:20]),
        })

    return results
