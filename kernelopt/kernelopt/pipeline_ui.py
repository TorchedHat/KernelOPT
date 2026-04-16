"""HUD for kernelopt: one stationary panel on a second TTY; main terminal = full NCU/log output."""

from __future__ import annotations

import io
import os
import re
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, IO, Iterator

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from kernelopt.state import GraphState

_default_console = Console()
_ctx: ContextVar["PipelineUI | None"] = ContextVar("pipeline_ui", default=None)

# Full redraw on foreign TTY at most this often (mirrored log lines can be very chatty).
_SPLIT_HUD_MIN_INTERVAL_S = 0.45
# Re-query nvidia-smi metrics on HUD (same idea as `watch -n1 nvidia-smi` for live VRAM/util/temp).
_HUD_LIVE_GPU_REFRESH_S = 1.0
# Rate-limit HUD write failure messages on the main terminal.
_HUD_ERR_LOG_INTERVAL_S = 8.0


def get_pipeline_ui() -> PipelineUI | None:
    return _ctx.get()


def plain_from_rich_string(s: str, max_len: int = 140) -> str:
    """Strip Rich markup for one-line event log."""
    try:
        from rich.text import Text as RText

        plain = RText.from_markup(s).plain
    except Exception:
        plain = re.sub(r"\[/?[^\]]*\]", "", s)
    plain = " ".join(plain.split())
    if len(plain) > max_len:
        plain = plain[: max_len - 3] + "..."
    return plain


class PipelineUI:
    """Mutable HUD state; secondary TTY gets a single full-screen repaint (no Rich Live)."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        is_split_hud: bool = False,
        hud_file: IO[str] | None = None,
    ) -> None:
        self.console = console or _default_console
        self.is_split_hud = is_split_hud
        self._hud_file = hud_file
        self.phase = "Starting"
        self.detail = ""
        self.iteration_0based = 0
        self.T = 5
        self.baseline_ns: float | None = None
        self.best_ns: float | None = None
        self.best_speedup = 1.0
        self.framework = "auto"
        self.resume_tag = ""
        self._events: deque[str] = deque(maxlen=14)
        self._last_split_draw = 0.0
        self._split_redraw_pending = False
        # Same structured fields as validate_gpu_for_profiling return (+ live updates)
        self._sys_snap: dict[str, Any] = {}
        self._last_gpu_live_refresh = 0.0
        self._last_hud_err_log = 0.0

    def set_pre_run_system_snapshot(self, snap: dict) -> None:
        """Copy CLI pre-run metrics so HUD matches main-terminal System Status."""
        if not snap:
            return
        self._sys_snap = {k: snap[k] for k in snap}
        self._last_gpu_live_refresh = 0.0  # allow immediate live read on first paint
        self._refresh(force=True)

    def ingest_post_baseline(
        self,
        *,
        baseline_latency_ns: float,
        best_latency_ns: float,
        framework: str,
        iteration_0based: int,
    ) -> None:
        """After baseline NCU: sync numeric HUD fields from authoritative pipeline state."""
        self.iteration_0based = iteration_0based
        self.framework = str(framework)
        if baseline_latency_ns == baseline_latency_ns and baseline_latency_ns != float("inf"):
            self.baseline_ns = float(baseline_latency_ns)
        if best_latency_ns == best_latency_ns and best_latency_ns != float("inf"):
            self.best_ns = float(best_latency_ns)
        self._recalc_speedup()
        self._refresh(force=True)

    def seed_from_graph(self, gs: GraphState) -> None:
        """Initialize HUD labels from LangGraph state (fresh or --memory resume)."""
        cfg = gs["config"]
        self.T = cfg.T
        self.iteration_0based = gs.get("iteration", 0)
        self.framework = str(gs.get("framework", "auto"))
        bl = gs.get("baseline_latency_ns")
        if bl is not None and bl == bl and bl != float("inf"):
            self.baseline_ns = float(bl)
        best = gs.get("best_latency_ns")
        if best is not None and best == best and best != float("inf"):
            self.best_ns = float(best)
        entry = gs.get("_entry", "fresh")
        if entry != "fresh":
            self.resume_tag = f"resume → {entry}"
        self._recalc_speedup()
        self._refresh(force=True)

    def _recalc_speedup(self) -> None:
        if (
            self.baseline_ns is not None
            and self.best_ns is not None
            and self.best_ns > 0
            and self.baseline_ns != float("inf")
            and self.best_ns != float("inf")
        ):
            self.best_speedup = round(self.baseline_ns / self.best_ns, 2)
        else:
            self.best_speedup = 1.0

    def set_phase(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = (detail or "")[:120]
        self._refresh(force=True)

    def sync_iteration_header(
        self,
        iteration_0based: int,
        baseline_ns: float,
        best_ns: float,
    ) -> None:
        self.iteration_0based = iteration_0based
        self.baseline_ns = baseline_ns if baseline_ns == baseline_ns else None
        self.best_ns = best_ns if best_ns == best_ns else None
        self._recalc_speedup()
        self._refresh(force=True)

    def add_event(self, plain_line: str) -> None:
        line = plain_line.strip()
        if not line:
            return
        self._events.append(line)
        self._refresh(force=False)

    def _refresh(self, *, force: bool = False) -> None:
        if not self.is_split_hud or self._hud_file is None:
            return
        now = time.monotonic()
        if not force:
            if now - self._last_split_draw < _SPLIT_HUD_MIN_INTERVAL_S:
                self._split_redraw_pending = True
                return
        self._split_redraw_pending = False
        self._last_split_draw = now
        self._paint_split_hud()

    def flush_pending_split_redraw(self) -> None:
        """If throttling skipped draws, paint once with latest state (call on phase boundaries)."""
        if self._split_redraw_pending and self.is_split_hud and self._hud_file is not None:
            self._split_redraw_pending = False
            self._last_split_draw = time.monotonic()
            self._paint_split_hud()

    def _maybe_refresh_live_gpu_metrics(self) -> None:
        """Refresh VRAM/util/temp from nvidia-smi so HUD matches current GPU (same source as monitoring)."""
        if not self._sys_snap:
            return
        now = time.monotonic()
        if now - self._last_gpu_live_refresh < _HUD_LIVE_GPU_REFRESH_S:
            return
        self._last_gpu_live_refresh = now
        try:
            from kernelopt.functions.gpu_monitoring import get_gpu_metrics, select_gpu

            gid = int(self._sys_snap.get("gpu_id", select_gpu()))
            m = get_gpu_metrics(gid)
            for k in (
                "name",
                "memory_total_mb",
                "memory_free_mb",
                "memory_used_mb",
                "utilization_pct",
                "temperature_c",
            ):
                if k in m:
                    self._sys_snap[k] = m[k]
        except Exception:
            pass

    def _render_system_block(self) -> RenderableType:
        """Lines aligned with validate_gpu_for_profiling 'System Status' output."""
        s = self._sys_snap
        if not s:
            return Text("(No pre-run system snapshot — run from kernelopt CLI)", style="dim italic")

        lines: list[RenderableType] = []
        hdr = Text()
        hdr.append("System", style="bold")
        lines.append(hdr)

        tot = s.get("sys_mem_total_mb")
        av = s.get("sys_mem_available_mb")
        if isinstance(tot, (int, float)) and isinstance(av, (int, float)) and tot > 0:
            pct = 100.0 * float(av) / float(tot)
            lines.append(
                Text.from_markup(
                    f"  CPU RAM: [cyan]{float(av):.0f}[/] MB avail / "
                    f"[cyan]{float(tot):.0f}[/] MB total ([cyan]{pct:.1f}%[/] avail)"
                )
            )

        gid = s.get("gpu_id", "?")
        name = s.get("name", "?")
        mtot = s.get("memory_total_mb")
        mfree = s.get("memory_free_mb")
        util = s.get("utilization_pct")
        temp = s.get("temperature_c")
        nproc = int(s.get("active_process_count", 0) or 0)

        if isinstance(mtot, (int, float)) and float(mtot) > 0 and isinstance(mfree, (int, float)):
            gpct = 100.0 * float(mfree) / float(mtot)
            lines.append(
                Text.from_markup(
                    f"  GPU: [cyan]GPU {gid}[/] ({name})\n"
                    f"  VRAM: [cyan]{float(mfree):.0f}[/] MB free / "
                    f"[cyan]{float(mtot):.0f}[/] MB total ([cyan]{gpct:.1f}%[/] free)"
                )
            )
        if isinstance(util, (int, float)):
            lines.append(Text.from_markup(f"  GPU util: [cyan]{float(util):.1f}%[/]"))
        if isinstance(temp, (int, float)) and float(temp) > 0:
            lines.append(Text(f"  GPU temp: {float(temp):.0f}°C"))
        if nproc > 0:
            lines.append(Text.from_markup(f"  Active GPU procs: [yellow]{nproc}[/]"))

        return Group(*lines)

    def _paint_split_hud(self) -> None:
        """One stationary frame: clear the secondary screen, then draw a single panel."""
        f = self._hud_file
        try:
            self._maybe_refresh_live_gpu_metrics()
            f.write("\x1b[2J\x1b[H")
            f.flush()
            self.console.print(self.render())
            f.flush()
        except Exception as e:
            now = time.monotonic()
            if now - self._last_hud_err_log >= _HUD_ERR_LOG_INTERVAL_S:
                self._last_hud_err_log = now
                _default_console.print(
                    f"[yellow]HUD write failed[/] ({type(e).__name__}): {e}"
                )

    def render(self) -> RenderableType:
        sys_block = self._render_system_block()
        sub = Text()
        sub.append("Phase: ", style="bold")
        sub.append(self.phase, style="cyan")
        if self.detail:
            sub.append(" — ")
            sub.append(self.detail, style="dim")

        iter_line = Text()
        iter_line.append("Iteration: ", style="bold")
        iter_line.append(
            f"{self.iteration_0based + 1}/{self.T}",
            style="green",
        )
        iter_line.append("   Framework: ", style="dim")
        iter_line.append(self.framework, style="white")
        if self.resume_tag:
            iter_line.append(f"   ({self.resume_tag})", style="dim yellow")

        metrics = Text()
        metrics.append("Best speedup: ", style="bold")
        metrics.append(f"{self.best_speedup}×", style="green")
        metrics.append("   Best: ", style="dim")
        if (
            self.best_ns is not None
            and self.best_ns == self.best_ns
            and self.best_ns > 0
            and self.best_ns != float("inf")
        ):
            metrics.append(f"{self.best_ns:.0f} ns", style="white")
        else:
            metrics.append("—", style="dim")
        metrics.append("   Baseline: ", style="dim")
        if self.baseline_ns is not None and self.baseline_ns != float("inf"):
            metrics.append(f"{self.baseline_ns:.0f} ns", style="dim")
        else:
            metrics.append("—", style="dim")

        hint = Text(
            "(Main terminal: full NCU + logs)",
            style="dim italic",
        )

        events_header = Text("Recent:", style="bold dim")
        event_lines: list[RenderableType] = [events_header]
        if not self._events:
            event_lines.append(Text("  (no events yet)", style="dim italic"))
        else:
            for ev in self._events:
                event_lines.append(Text(f"  • {ev}", style="dim"))

        top: list[RenderableType] = []
        if sys_block is not None:
            top.append(sys_block)
            top.append(Text(""))
        top.extend([sub, iter_line, metrics, hint, Text(""), *event_lines])
        body = Group(*top)
        return Panel(
            body,
            title="[bold]kernelopt[/] [dim]status[/]",
            border_style="blue",
            padding=(0, 1),
        )


def pipe_print(*objects: Any, **kwargs: Any) -> None:
    """
    Always print full lines to the main console (NCU, kernelopt, etc.).
    If split HUD is active, mirror a plain one-line summary to the secondary TTY panel.
    """
    ui = get_pipeline_ui()
    _default_console.print(*objects, **kwargs)
    if ui is None or not ui.is_split_hud:
        return
    sep: str = kwargs.get("sep", " ")
    pieces: list[str] = []
    for obj in objects:
        if isinstance(obj, str):
            pieces.append(obj)
        else:
            pieces.append(str(obj))
    line = sep.join(pieces)
    ui.add_event(plain_from_rich_string(line))


def silent_console() -> Console:
    """Console that discards output."""
    return Console(file=io.StringIO())


def _console_for_hud_tty(f: IO[str]) -> Console:
    try:
        w = os.get_terminal_size(f.fileno()).columns
    except OSError:
        w = 100
    return Console(file=f, force_terminal=True, width=max(40, min(w, 120)))


# Alternate buffer: redraws do not flood scrollback. Hidden cursor: less flicker while repainting.
_SEQ_HUD_ALT_ENTER = "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H"
_SEQ_HUD_ALT_LEAVE = "\x1b[?25h\x1b[?1049l"


@contextmanager
def run_hud_session(hud_tty: str) -> Iterator[PipelineUI]:
    """
    Stationary HUD on a second terminal (path from `tty` in that window).
    Main stdout/stderr are unchanged so NCU and subprocess output stay readable.
    Uses the alternate screen buffer on the HUD tty so rapid redraws stay smooth
    and do not fill scrollback; restores the prior display when the session ends.
    """
    path = Path(hud_tty).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--hud-tty path does not exist: {path}")
    f: IO[str] | None = None
    try:
        f = open(path, "w", buffering=1)  # noqa: SIM115 — closed in finally
    except OSError as e:
        raise OSError(
            f"Cannot open HUD terminal {path!r} (permissions or wrong path). "
            "Run `tty` in the second window and pass that path."
        ) from e

    ui: PipelineUI | None = None
    token = None
    hud_alt_screen = False
    try:
        f.write(_SEQ_HUD_ALT_ENTER)
        f.flush()
        hud_alt_screen = True
        c = _console_for_hud_tty(f)
        ui = PipelineUI(console=c, is_split_hud=True, hud_file=f)
        token = _ctx.set(ui)
        yield ui
        ui.flush_pending_split_redraw()
    finally:
        if token is not None:
            _ctx.reset(token)
        if f is not None:
            try:
                if hud_alt_screen:
                    f.write(_SEQ_HUD_ALT_LEAVE + "\n")
                    f.flush()
            except OSError:
                pass
            try:
                f.close()
            except Exception:
                pass
