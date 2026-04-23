"""HUD for kernelopt: same-terminal alternate-screen panel; stream log to file when --hud."""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, IO, Iterator

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from kernelopt.state import GraphState

# Filename under output_dir for all stdout/stderr + NCU dumps when --hud is active.
HUD_TERMINAL_LOG_NAME = "kernelopt_terminal.log"

# Alternate buffer on the real TTY; hidden cursor while repainting.
_SEQ_HUD_ALT_ENTER = "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H"
_SEQ_HUD_ALT_LEAVE = "\x1b[?25h\x1b[?1049l"

_default_console = Console()
_ctx: ContextVar["PipelineUI | None"] = ContextVar("pipeline_ui", default=None)
_hud_log_console: Console | None = None
_hud_log_console_lock = threading.Lock()
_hud_log_raw_fp: IO[str] | None = None
_hud_log_raw_lock = threading.Lock()

# Full redraw at most this often (mirrored log lines can be very chatty).
_HUD_MIN_INTERVAL_S = 0.45
_HUD_LIVE_GPU_REFRESH_S = 1.0
_HUD_ERR_LOG_INTERVAL_S = 8.0


def get_pipeline_ui() -> PipelineUI | None:
    return _ctx.get()


def get_hud_log_console() -> Console | None:
    """Rich console writing to the run log file (set only during --hud)."""
    with _hud_log_console_lock:
        return _hud_log_console


def _set_hud_log_console(c: Console | None) -> None:
    global _hud_log_console
    with _hud_log_console_lock:
        _hud_log_console = c


def _set_hud_log_raw_fp(fp: IO[str] | None) -> None:
    global _hud_log_raw_fp
    with _hud_log_raw_lock:
        _hud_log_raw_fp = fp


def emit_profiler_output(text: str) -> None:
    """Write NCU / profiler subprocess text to terminal or run log (when --hud)."""
    if not text:
        return
    with _hud_log_raw_lock:
        fp = _hud_log_raw_fp
    if fp is not None:
        fp.write(text)
        fp.flush()
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def _tail_log_file(path: Path, *, max_lines: int = 40, max_bytes: int = 48_000) -> str:
    try:
        with path.open("rb") as bf:
            bf.seek(0, 2)
            sz = bf.tell()
            bf.seek(max(0, sz - max_bytes))
            chunk = bf.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()
        return "\n".join(lines[-max_lines:]) if lines else "(empty log)"
    except OSError as e:
        return f"(could not read log: {e})"


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


class _StdoutToLog:
    """Minimal file-like object: append to log only."""

    def __init__(self, log_fp: IO[str]) -> None:
        self._fp = log_fp

    def write(self, s: str) -> int:
        if s:
            self._fp.write(s)
            self._fp.flush()
        return len(s) if s else 0

    def flush(self) -> None:
        self._fp.flush()


class PipelineUI:
    """Same-terminal HUD (alternate screen) when ``--hud`` is set."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        is_primary_hud: bool = False,
        log_path: Path | None = None,
        real_stdout: IO[str] | None = None,
    ) -> None:
        self.console = console or _default_console
        self.is_primary_hud = is_primary_hud
        self._log_path = log_path
        self._real_stdout = real_stdout
        self._hud_panel_visible = True
        self._lock = threading.Lock()

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
        self._last_draw = 0.0
        self._redraw_pending = False
        self._sys_snap: dict[str, Any] = {}
        self._last_gpu_live_refresh = 0.0
        self._last_err_log = 0.0
        # Per-plan NCU deltas vs baseline (same metrics as optimization_log.md); HUD-only.
        self._plan_delta_lines: list[str] = []

    @staticmethod
    def _hud_terminal_columns() -> int:
        try:
            return max(40, int(shutil.get_terminal_size().columns))
        except OSError:
            return 100

    @property
    def is_hud_sink(self) -> bool:
        """True when this UI receives mirrored pipe_print lines for the HUD panel."""
        return self.is_primary_hud

    def set_pre_run_system_snapshot(self, snap: dict) -> None:
        if not snap:
            return
        self._sys_snap = {k: snap[k] for k in snap}
        self._last_gpu_live_refresh = 0.0
        self._refresh(force=True)

    def ingest_post_baseline(
        self,
        *,
        baseline_latency_ns: float,
        best_latency_ns: float,
        framework: str,
        iteration_0based: int,
    ) -> None:
        self.iteration_0based = iteration_0based
        self.framework = str(framework)
        if baseline_latency_ns == baseline_latency_ns and baseline_latency_ns != float("inf"):
            self.baseline_ns = float(baseline_latency_ns)
        if best_latency_ns == best_latency_ns and best_latency_ns != float("inf"):
            self.best_ns = float(best_latency_ns)
        self._recalc_speedup()
        self._refresh(force=True)

    def seed_from_graph(self, gs: GraphState) -> None:
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
        self._sync_plan_deltas_from_graph(gs)
        self._refresh(force=True)

    def set_plan_delta_from_profile(
        self,
        baseline_ctx: dict[str, Any],
        candidates: list[Any],
        *,
        baseline_latency_ns: float | None = None,
        terminal_columns: int | None = None,
    ) -> None:
        """Update HUD delta block after NCU (best plan only if faster than baseline)."""
        try:
            from kernelopt.iteration_metric_deltas import (
                has_profiling_context,
                hud_delta_section_lines,
                hud_profiled_best_speedup_only,
                plan_beats_baseline_latency,
                resolve_baseline_latency_ns,
            )

            profiled = [c for c in candidates if has_profiling_context(c)]
            profiled = hud_profiled_best_speedup_only(profiled)
            if not profiled:
                self._plan_delta_lines = []
                self._refresh(force=True)
                return
            base_ns = resolve_baseline_latency_ns(
                baseline_ctx if isinstance(baseline_ctx, dict) else {},
                baseline_latency_ns,
            )
            if not plan_beats_baseline_latency(profiled[0], base_ns):
                self._plan_delta_lines = []
                self._refresh(force=True)
                return
            cols = (
                terminal_columns
                if terminal_columns is not None
                else self._hud_terminal_columns()
            )
            self._plan_delta_lines = hud_delta_section_lines(
                baseline_ctx,
                profiled,
                max_metrics=9,
                max_plans=1,
                terminal_columns=cols,
            )
        except Exception:
            self._plan_delta_lines = []
        self._refresh(force=True)

    def _sync_plan_deltas_from_graph(self, gs: GraphState) -> None:
        """On --checkpoint resume, rebuild Δ block from last checkpointed iteration."""
        try:
            from kernelopt.iteration_metric_deltas import (
                has_profiling_context,
                hud_delta_section_lines,
                hud_profiled_best_speedup_only,
                plan_beats_baseline_latency,
                resolve_baseline_latency_ns,
            )

            baseline = gs.get("baseline_profiling_context") or {}
            results = gs.get("iteration_results") or []
            if not results:
                self._plan_delta_lines = []
                return
            last = results[-1]
            if isinstance(last, dict):
                cand_list = last.get("candidates") or []
            else:
                cand_list = getattr(last, "candidates", None) or []
            profiled = [c for c in cand_list if has_profiling_context(c)]
            if not profiled:
                self._plan_delta_lines = []
                return
            chosen: list[Any] = []
            if not isinstance(last, dict):
                bc = getattr(last, "best_candidate", None)
                if bc is not None and has_profiling_context(bc):
                    chosen = [bc]
            else:
                bc = last.get("best_candidate")
                if isinstance(bc, dict) and has_profiling_context(bc):
                    chosen = [bc]
            if not chosen:
                chosen = hud_profiled_best_speedup_only(profiled)
            if not chosen:
                self._plan_delta_lines = []
                return
            bl = gs.get("baseline_latency_ns")
            bl_ns = (
                float(bl)
                if isinstance(bl, (int, float)) and bl == bl
                else None
            )
            base_ns = resolve_baseline_latency_ns(
                baseline if isinstance(baseline, dict) else {},
                bl_ns,
            )
            if not plan_beats_baseline_latency(chosen[0], base_ns):
                self._plan_delta_lines = []
                return
            self._plan_delta_lines = hud_delta_section_lines(
                baseline,
                chosen,
                max_metrics=9,
                max_plans=1,
                terminal_columns=self._hud_terminal_columns(),
            )
        except Exception:
            self._plan_delta_lines = []

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

    def toggle_panel_visible(self) -> None:
        """Switch between HUD panel (alternate screen) and a short log tail on the main screen."""
        with self._lock:
            self._hud_panel_visible = not self._hud_panel_visible
            out = self._real_stdout
            if out is None:
                self._hud_panel_visible = True
                return
            if not self._hud_panel_visible:
                out.write(_SEQ_HUD_ALT_LEAVE + "\n")
                out.flush()
                log_p = self._log_path
                if log_p and log_p.is_file():
                    tail = _tail_log_file(log_p)
                    banner = (
                        f"\n[--hud] Log tail ({log_p.name}) — press [t] again for HUD panel —\n\n"
                    )
                    out.write(banner)
                    out.write(tail + "\n")
                else:
                    out.write(f"(no log yet at {log_p})\n")
                out.write("\n[Press t in this terminal when the prompt returns to return to HUD]\n")
                out.flush()
            else:
                out.write(_SEQ_HUD_ALT_ENTER)
                out.write("\x1b[2J\x1b[H")
                out.flush()
                self._last_draw = 0.0
                self._paint_locked()

    def _refresh(self, *, force: bool = False) -> None:
        if not self.is_hud_sink or (self.is_primary_hud and not self._hud_panel_visible):
            return
        now = time.monotonic()
        if not force:
            if now - self._last_draw < _HUD_MIN_INTERVAL_S:
                self._redraw_pending = True
                return
        self._redraw_pending = False
        self._last_draw = now
        with self._lock:
            if self.is_primary_hud and not self._hud_panel_visible:
                return
            self._paint_locked()

    def flush_pending_split_redraw(self) -> None:
        if self._redraw_pending and self.is_hud_sink:
            with self._lock:
                if self.is_primary_hud and not self._hud_panel_visible:
                    return
                self._redraw_pending = False
                self._last_draw = time.monotonic()
                self._paint_locked()

    def _maybe_refresh_live_gpu_metrics(self) -> None:
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

    def _paint_locked(self) -> None:
        target = self._real_stdout
        if target is None:
            return
        try:
            self._maybe_refresh_live_gpu_metrics()
            target.write("\x1b[2J\x1b[H")
            target.flush()
            self.console.print(self.render())
            target.flush()
        except Exception as e:
            now = time.monotonic()
            if now - self._last_err_log >= _HUD_ERR_LOG_INTERVAL_S:
                self._last_err_log = now
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

        delta_block: list[RenderableType] = []
        if self._plan_delta_lines:
            delta_block.append(Text(""))
            delta_block.append(
                Text("Best plan vs baseline(NCU metrics)", style="bold dim")
            )
            for ln in self._plan_delta_lines:
                delta_block.append(
                    Text(f"  {ln}", style="dim", no_wrap=True, overflow="ignore")
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
        top.extend([sub, iter_line, metrics, *delta_block, Text(""), *event_lines])
        body = Group(*top)
        return Panel(
            body,
            title="[bold]kernelopt[/] [dim]status[/]",
            border_style="blue",
            padding=(0, 1),
        )


def pipe_print(*objects: Any, **kwargs: Any) -> None:
    """Print to log (when --hud) or main console; mirror one-line summary into HUD."""
    ui = get_pipeline_ui()
    lc = get_hud_log_console()
    if lc is not None:
        lc.print(*objects, **kwargs)
        if ui is not None and ui.is_primary_hud and not ui._hud_panel_visible:
            try:
                c2 = Console(file=sys.__stdout__, force_terminal=True, width=100)
                c2.print(*objects, **kwargs)
            except Exception:
                pass
    else:
        _default_console.print(*objects, **kwargs)

    if ui is None or not ui.is_hud_sink:
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
    return Console(file=io.StringIO())


def _stdin_toggle_worker(stop: threading.Event, ui_ref: Callable[[], PipelineUI | None]) -> None:
    """Press [t] to toggle HUD vs log tail (best-effort; may not work during blocking subprocess)."""
    if not sys.stdin.isatty():
        return
    try:
        import select
        import termios
        import tty as tty_mod
    except ImportError:
        return
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return
    try:
        tty_mod.setcbreak(fd)
        while not stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch.lower() == "t":
                ui = ui_ref()
                if ui is not None and ui.is_primary_hud:
                    ui.toggle_panel_visible()
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass


@contextmanager
def run_primary_hud_session(log_path: Path) -> Iterator[PipelineUI]:
    """
    Same TTY: alternate-screen HUD; stdout/stderr/NCU text go to ``log_path``.
    Press ``t`` (when stdin is a TTY) to leave the panel and print a log tail.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    real_out = sys.__stdout__
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        _tw = os.get_terminal_size(real_out.fileno()).columns
    except OSError:
        _tw = 120
    log_console = Console(file=log_f, no_color=True, width=min(160, max(80, _tw)))

    _set_hud_log_console(log_console)
    _set_hud_log_raw_fp(log_f)
    wrapper = _StdoutToLog(log_f)
    sys.stdout = wrapper  # type: ignore[assignment]
    sys.stderr = wrapper  # type: ignore[assignment]

    stop = threading.Event()

    def _get_ui() -> PipelineUI | None:
        return get_pipeline_ui()

    toggle_thread = threading.Thread(
        target=_stdin_toggle_worker,
        args=(stop, _get_ui),
        daemon=True,
        name="kernelopt-hud-toggle",
    )

    ui: PipelineUI | None = None
    token = None
    try:
        real_out.write(_SEQ_HUD_ALT_ENTER)
        real_out.flush()
        try:
            w = os.get_terminal_size(real_out.fileno()).columns
        except OSError:
            w = 100
        hw_console = Console(file=real_out, force_terminal=True, width=max(40, min(w, 120)))
        ui = PipelineUI(
            console=hw_console,
            is_primary_hud=True,
            log_path=log_path,
            real_stdout=real_out,
        )
        token = _ctx.set(ui)
        toggle_thread.start()
        yield ui
        if ui is not None:
            ui.flush_pending_split_redraw()
    finally:
        stop.set()
        toggle_thread.join(timeout=0.5)
        _set_hud_log_console(None)
        _set_hud_log_raw_fp(None)
        sys.stdout = orig_out
        sys.stderr = orig_err
        try:
            log_f.close()
        except Exception:
            pass
        try:
            real_out.write(_SEQ_HUD_ALT_LEAVE + "\n")
            real_out.flush()
        except OSError:
            pass
        if token is not None:
            _ctx.reset(token)
