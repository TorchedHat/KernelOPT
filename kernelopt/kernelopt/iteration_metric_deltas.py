"""NCU metric formatting and baseline-vs-candidate deltas (optimization_log.md + HUD).

Keep metric rows aligned with ``_write_final_outputs`` in ``langgraph_pipeline``.
"""

from __future__ import annotations

from typing import Any

HUD_STACK_METRIC_NAME_W = 22

# NCU CSV places some metrics under different sections/names across tool versions.
LOG_METRIC_ALTERNATIVES: dict[str, list[tuple[str, str]]] = {
    "DRAM Throughput (%)": [
        ("GPU Speed Of Light Throughput", "DRAM Throughput"),
        ("Memory Workload Analysis", "DRAM Throughput"),
    ],
    "IPC Active": [
        ("Compute Workload Analysis", "Executed Ipc Active"),
        ("Compute Workload Analysis", "Issued Ipc Active"),
        ("Instruction Statistics", "IPC Active"),
    ],
}

# Same rows as optimization_log.md "Metric Comparison" table.
LOG_METRIC_COMPARISON_ROWS: list[tuple[str, str, str]] = [
    ("Duration (ns)", "GPU Speed Of Light Throughput", "Duration"),
    ("SM Throughput (%)", "GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
    ("Memory Throughput (%)", "GPU Speed Of Light Throughput", "Memory Throughput"),
    ("DRAM Throughput (%)", "Memory Workload Analysis", "DRAM Throughput"),
    ("Achieved Occupancy (%)", "Occupancy", "Achieved Occupancy"),
    ("Registers/Thread", "Launch Statistics", "Registers Per Thread"),
    ("Waves/SM", "Launch Statistics", "Waves Per SM"),
    ("No Eligible (%)", "Scheduler Statistics", "No Eligible"),
    ("IPC Active", "Instruction Statistics", "IPC Active"),
]


def fmt_metric(entry: dict) -> str | None:
    """Format a single NCU metric dict, or None if unusable."""
    if not isinstance(entry, dict):
        return None
    val = entry.get("value")
    unit = (entry.get("unit") or "").strip()
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return None
    if isinstance(val, float):
        return f"{val:.2f} {unit}".strip() if unit else f"{val:.2f}"
    return f"{val} {unit}".strip() if unit else str(val)


def metric_value_str(ctx: dict, section: str, metric: str, default: str = "—") -> str:
    """Safely extract a metric value from a profiling context."""
    try:
        entry = ctx["sections"][section][metric]
        out = fmt_metric(entry)
        return out if out is not None else default
    except (KeyError, TypeError):
        return default


def metric_value_str_any(
    ctx: dict, alternatives: list[tuple[str, str]], default: str = "—"
) -> str:
    """Like metric_value_str but try several (section, metric) pairs (first hit wins)."""
    for section, metric in alternatives:
        try:
            entry = ctx["sections"][section][metric]
            out = fmt_metric(entry)
            if out is not None:
                return out
        except (KeyError, TypeError):
            continue
    return default


def duration_ns_display(ctx: dict) -> str:
    """Format kernel duration from profiling context as integer nanoseconds."""
    from kernelopt.agents.profiler.context_builder import get_latency_ns

    v = get_latency_ns(ctx)
    if v == float("inf") or v != v:
        return "—"
    return f"{v:.0f} ns"


def metric_numeric(
    ctx: dict,
    label: str,
    section: str,
    metric: str,
    alts: list[tuple[str, str]] | None,
) -> float | None:
    """Parse a comparable float for delta computation; None if missing."""
    if label == "Duration (ns)":
        from kernelopt.agents.profiler.context_builder import get_latency_ns

        v = get_latency_ns(ctx)
        if v == float("inf") or v != v:
            return None
        return float(v)
    if alts:
        for s, m in alts:
            try:
                raw = ctx["sections"][s][m].get("value")
                if raw is None or (isinstance(raw, float) and raw != raw):
                    continue
                return float(raw)
            except (KeyError, TypeError, ValueError):
                continue
        return None
    try:
        raw = ctx["sections"][section][metric]["value"]
        if raw is None or (isinstance(raw, float) and raw != raw):
            return None
        return float(raw)
    except (KeyError, TypeError, ValueError):
        return None


def delta_cell(label: str, base_v: float | None, cand_v: float | None) -> str:
    """Human-readable delta vs baseline for one metric row."""
    if base_v is None or cand_v is None:
        return "—"
    d = cand_v - base_v
    if label == "Duration (ns)":
        rel_pct = (cand_v / base_v - 1.0) * 100.0 if base_v else 0.0
        return f"{d:+.0f} ns ({rel_pct:+.1f}%)"
    if label.endswith("(%)") or "Occupancy" in label:
        return f"{d:+.2f} pp"
    if label == "Registers/Thread":
        return f"{d:+.0f}"
    if label == "Waves/SM":
        return f"{d:+.2f}"
    if label == "IPC Active":
        return f"{d:+.2f}"
    return f"{d:+.2f}"


def _candidate_ctx(c: Any) -> dict:
    if isinstance(c, dict):
        ctx = c.get("profiling_context") or {}
        return ctx if isinstance(ctx, dict) else {}
    ctx = getattr(c, "profiling_context", None) or {}
    return ctx if isinstance(ctx, dict) else {}


def _candidate_plan_letter(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("plan_letter", "?"))
    return str(getattr(c, "plan_letter", "?"))


def _candidate_direction(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("direction", "") or "").strip()
    return str(getattr(c, "direction", "") or "").strip()


def _latency_ns_value(c: Any) -> float:
    if isinstance(c, dict):
        try:
            return float(c.get("latency_ns", float("inf")))
        except (TypeError, ValueError):
            return float("inf")
    try:
        return float(getattr(c, "latency_ns", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def hud_profiled_best_speedup_only(profiled: list[Any]) -> list[Any]:
    """Lowest latency among profiled candidates (highest speedup this profile round)."""
    finite = [c for c in profiled if _latency_ns_value(c) < float("inf")]
    if not finite:
        return []
    return [min(finite, key=_latency_ns_value)]


def resolve_baseline_latency_ns(
    baseline_ctx: dict,
    explicit_ns: float | None = None,
) -> float:
    """Finite baseline kernel latency (ns), or inf if unknown."""
    if explicit_ns is not None:
        try:
            v = float(explicit_ns)
        except (TypeError, ValueError):
            v = float("inf")
        if v == v and 0 < v < float("inf"):
            return v
    ctx = baseline_ctx if isinstance(baseline_ctx, dict) else {}
    from kernelopt.agents.profiler.context_builder import get_latency_ns

    v = get_latency_ns(ctx)
    if v == float("inf") or v != v or v <= 0:
        return float("inf")
    return float(v)


def plan_beats_baseline_latency(candidate: Any, baseline_latency_ns: float) -> bool:
    """True iff candidate latency is strictly below run baseline (faster)."""
    if baseline_latency_ns >= float("inf") or baseline_latency_ns != baseline_latency_ns:
        return False
    if baseline_latency_ns <= 0:
        return False
    lat = _latency_ns_value(candidate)
    if lat >= float("inf") or lat != lat or lat <= 0:
        return False
    return lat < baseline_latency_ns


def has_profiling_context(c: Any) -> bool:
    """True if candidate has a non-empty profiling_context dict."""
    return bool(_candidate_ctx(c))


_ELL = "..."  # ASCII — predictable width vs Unicode … in TTY/HUD


def _truncate_cell(s: str, width: int) -> str:
    s = s.strip()
    if len(s) <= width:
        return s
    if width <= len(_ELL):
        return s[:width]
    return s[: width - len(_ELL)] + _ELL


def hud_delta_section_lines(
    baseline_ctx: dict,
    profiled: list[Any],
    *,
    max_metrics: int = 9,
    max_plans: int | None = None,
    terminal_columns: int | None = None,
) -> list[str]:
    """HUD: one or more plan blocks (metrics under each plan), same Δ as optimization_log."""
    if not profiled:
        return []
    n_total = len(profiled)
    if max_plans is None:
        mp = n_total
    else:
        mp = max(1, min(int(max_plans), n_total))
    profiled = profiled[:mp]
    iter_baseline_ctx = baseline_ctx if isinstance(baseline_ctx, dict) else {}
    rows = LOG_METRIC_COMPARISON_ROWS[: max(1, max_metrics)]

    cols = max(48, int(terminal_columns or 100))
    dir_budget = max(24, cols - 18)
    delta_budget = max(28, cols - 6 - HUD_STACK_METRIC_NAME_W)

    out: list[str] = []

    mw = HUD_STACK_METRIC_NAME_W
    for c in profiled:
        letter = _candidate_plan_letter(c)
        direction = _candidate_direction(c)
        direction = _truncate_cell(direction, dir_budget) if direction else "—"
        out.append(f"  ── Plan {letter} · {direction} ──")
        ctx = _candidate_ctx(c)
        for label, section, metric in rows:
            alts = LOG_METRIC_ALTERNATIVES.get(label)
            base_num = metric_numeric(iter_baseline_ctx, label, section, metric, alts)
            cn = metric_numeric(ctx, label, section, metric, alts)
            d = delta_cell(label, base_num, cn)
            lbl = label if len(label) <= mw else _truncate_cell(label, mw)
            lbl = f"{lbl:<{mw}}"
            d_disp = _truncate_cell(d, delta_budget)
            out.append(f"    {lbl}  {d_disp}")
        out.append("")

    if out and out[-1] == "":
        out.pop()

    if n_total > mp:
        out.append(
            f"  (+{n_total - mp} plans omitted — [t] log tail or optimization_log.md)"
        )
    return out
