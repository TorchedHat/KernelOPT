"""Animated progress display for non-HUD mode.

When --hud is off, wraps long-running pipeline phases with Rich spinners
so the user always sees something moving during LLM calls, NCU profiling,
and other blocking operations.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

_console = Console()

_BAR_WIDTH = 8
_BAR_COLORS = ["cyan", "green", "yellow", "magenta", "blue", "red"]

PHASE_CONFIG = {
    "baseline":    ("🔬", "cyan",    "Profiling baseline"),
    "planning":    ("🧠", "yellow",  "Generating plans"),
    "executing":   ("⚡", "green",   "Executing"),
    "correctness": ("🔍", "blue",    "Checking correctness"),
    "profiling":   ("📊", "magenta", "Profiling with NCU"),
    "summarize":   ("📝", "yellow",  "Summarizing results"),
    "selecting":   ("🏆", "green",   "Selecting best"),
}


def _bouncing_bar(tick: int, width: int = _BAR_WIDTH) -> Text:
    """Render a colorful bouncing block bar: [■□□□□□□□] with the filled
    block sliding left-to-right then right-to-left."""
    cycle = 2 * (width - 1)
    pos = tick % cycle
    if pos >= width:
        pos = cycle - pos

    bar = Text("[")
    for i in range(width):
        if i == pos:
            color = _BAR_COLORS[tick % len(_BAR_COLORS)]
            bar.append("■", style=f"bold {color}")
        else:
            bar.append("□", style="dim")
    bar.append("]")
    return bar


class _BouncingBarDisplay:
    """Renderable: emoji + colorful bouncing block bar + message + elapsed time."""

    def __init__(self, message: str, emoji: str, color: str):
        self._message = message
        self._emoji = emoji
        self._color = color
        self._start = time.monotonic()
        self._tick = 0

    def __rich_console__(self, console, options):
        elapsed = time.monotonic() - self._start
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        bar = _bouncing_bar(self._tick)
        self._tick += 1

        table = Table.grid(padding=(0, 1))
        table.add_column(width=2)
        table.add_column(width=_BAR_WIDTH + 2)
        table.add_column()
        table.add_column(justify="right", style="dim")
        table.add_row(
            Text(self._emoji),
            bar,
            Text(self._message, style=self._color),
            Text(f"[{time_str}]"),
        )
        yield from table.__rich_console__(console, options)


@contextmanager
def phase_spinner(
    phase: str,
    detail: str = "",
    *,
    console: Optional[Console] = None,
) -> Iterator[None]:
    """Context manager that shows an animated bouncing bar for a pipeline phase.

    Usage::

        with phase_spinner("planning", "4 optimization plans"):
            plans = run_planner_agent(...)
    """
    c = console or _console
    emoji, color, label = PHASE_CONFIG.get(
        phase, ("⏳", "white", phase.capitalize())
    )
    message = f"{label} {detail}".strip() if detail else label

    display = _BouncingBarDisplay(message, emoji, color)

    with Live(display, console=c, refresh_per_second=8, transient=True):
        yield

    elapsed = time.monotonic() - display._start
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
    c.print(f"  [green]✓[/] {message}  [dim]\\[{time_str}][/]")


def print_run_header(
    kernel_name: str,
    gpu_name: str,
    T: int,
    N: int,
    K: int,
    output_dir: str,
    model: str,
    framework: str = "auto",
    *,
    console: Optional[Console] = None,
) -> None:
    """Print a structured run header at pipeline start."""
    c = console or _console
    c.print()
    c.print("[bold]kernelopt[/] [dim]v0.1.0[/]")
    c.print(f"  Kernel:    [cyan]{kernel_name}[/]")
    c.print(f"  GPU:       [cyan]{gpu_name}[/]")
    c.print(f"  Config:    T={T} iterations, N={N} plans, K={K} attempts")
    c.print(f"  Framework: [green]{framework}[/]")
    c.print(f"  Model:     [dim]{model}[/]")
    c.print(f"  Output:    [dim]{output_dir}/[/]")
    c.print()


def print_run_summary(
    baseline_ns: float,
    best_ns: float,
    best_direction: str,
    iterations_completed: int,
    iterations_total: int,
    iteration_results: list,
    output_dir: str,
    total_elapsed: float,
    *,
    console: Optional[Console] = None,
) -> None:
    """Print a structured summary when the pipeline completes."""
    c = console or _console

    mins, secs = divmod(int(total_elapsed), 60)
    time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
    speedup = baseline_ns / best_ns if best_ns > 0 else 0.0

    c.print()
    c.print("[bold]━━ Run Complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    c.print(f"  Total time:   [bold]{time_str}[/]")

    if speedup > 1.005:
        c.print(
            f"  Best speedup: [bold green]{speedup:.2f}x[/] "
            f"([dim]{best_direction}[/])"
        )
    else:
        c.print(f"  Best speedup: [yellow]{speedup:.2f}x[/] (no improvement)")

    c.print(
        f"  Baseline:     {baseline_ns / 1000:.2f} µs  →  "
        f"Best: [green]{best_ns / 1000:.2f} µs[/]"
    )
    c.print()

    for i, result in enumerate(iteration_results):
        candidates = getattr(result, "candidates", [])
        kept = [c for c in candidates if getattr(c, "kept", False)]
        if kept:
            k = kept[0]
            c.print(
                f"    Iter {i + 1}: [green]{k.speedup:.2f}x[/] — "
                f"{k.direction}"
            )
        else:
            best_c = max(candidates, key=lambda x: x.speedup, default=None)
            if best_c and best_c.speedup > 0:
                c.print(
                    f"    Iter {i + 1}: [dim]no improvement[/] "
                    f"(best candidate {best_c.speedup:.2f}x)"
                )
            else:
                c.print(f"    Iter {i + 1}: [dim]no valid candidates[/]")

    c.print()
    c.print("  [bold]Files:[/]")
    c.print(f"    Best kernel: [cyan]{output_dir}/best_kernel.*[/]")
    c.print(f"    Memory:      [dim]{output_dir}/optimization_memory.json[/]")
    c.print(f"    Full log:    [dim]{output_dir}/optimization_log.md[/]")
    c.print()
    c.print("  [bold]Next steps:[/]")
    c.print(f"    Continue: [cyan]kernelopt run ... --checkpoint {output_dir}/[/]")
    c.print(f"    Inspect:  [cyan]kernelopt show {output_dir}/baseline.csv[/]")
    c.print()


def print_iteration_header(
    iteration: int,
    total: int,
    best_speedup: float,
    estimated_remaining: Optional[float] = None,
    *,
    console: Optional[Console] = None,
) -> None:
    """Print the iteration header with optional ETA."""
    c = console or _console
    eta = ""
    if estimated_remaining is not None and estimated_remaining > 0:
        mins, secs = divmod(int(estimated_remaining), 60)
        eta = f"  [dim](~{mins}m {secs:02d}s remaining)[/]" if mins > 0 else f"  [dim](~{secs}s remaining)[/]"

    c.print()
    c.print(
        f"[bold]━━ Iteration {iteration}/{total}[/]  "
        f"Best: [green]{best_speedup:.2f}x[/]{eta}"
    )
