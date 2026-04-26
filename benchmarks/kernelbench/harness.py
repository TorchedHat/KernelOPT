"""KernelBench benchmark harness.

Loads KernelBench problems, measures baselines (eager + torch.compile),
runs correctness checks, and benchmarks timing -- aligned with the
KernelBench paper protocol (arXiv 2502.10517).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent.parent / "tests_suite" / "benchmarks" / "kernelbench"

WARMUP_REPS = 25
TIMING_REPS = 100
CORRECTNESS_SEEDS = [0, 42, 1337, 7, 2024]
NUM_CORRECTNESS_RUNS = 5


@dataclass
class ProblemSpec:
    """A loaded KernelBench problem."""
    problem_id: str
    level: int
    name: str
    source_path: str
    model_cls: Any = field(repr=False)
    get_inputs_fn: Any = field(repr=False)
    get_init_inputs_fn: Any = field(repr=False)


@dataclass
class BaselineResult:
    """Timing and correctness baselines for a single problem."""
    problem_id: str
    level: int
    name: str
    eager_ns: float = float("inf")
    compile_ns: float = float("inf")
    eager_ok: bool = False
    compile_ok: bool = False
    error: str = ""


def list_problems(levels: list[int] | None = None, root: Path | None = None) -> list[dict]:
    """List all available KernelBench problems with metadata."""
    root = root or BENCHMARKS_ROOT
    if levels is None:
        levels = [1, 2, 3]
    problems = []
    for level in sorted(levels):
        level_dir = root / f"level_{level}"
        if not level_dir.is_dir():
            continue
        for json_path in sorted(level_dir.glob("kernel_*.json")):
            meta = json.loads(json_path.read_text())
            py_path = json_path.with_suffix(".py")
            if py_path.exists():
                problems.append({
                    "problem_id": f"level_{level}/{py_path.stem}",
                    "level": level,
                    "name": meta.get("name", py_path.stem),
                    "path": str(py_path),
                })
    return problems


def load_problem(py_path: str | Path) -> ProblemSpec:
    """Dynamically import a KernelBench .py file and extract Model, get_inputs, get_init_inputs."""
    py_path = Path(py_path)
    parts = py_path.parts
    level = int([p for p in parts if p.startswith("level_")][0].split("_")[1])

    json_path = py_path.with_suffix(".json")
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}

    mod_name = f"_kb_{py_path.stem}_{id(py_path)}"
    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    return ProblemSpec(
        problem_id=f"level_{level}/{py_path.stem}",
        level=level,
        name=meta.get("name", py_path.stem),
        source_path=str(py_path),
        model_cls=getattr(mod, "Model"),
        get_inputs_fn=getattr(mod, "get_inputs"),
        get_init_inputs_fn=getattr(mod, "get_init_inputs"),
    )


def _benchmark_fn(fn, *args, warmup: int = WARMUP_REPS, reps: int = TIMING_REPS) -> float:
    """Benchmark a callable, return median time in nanoseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e6)

    times.sort()
    return times[len(times) // 2]


def _check_correctness(
    ref_outputs: list[torch.Tensor],
    test_fn,
    test_args: list,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, str]:
    """Check correctness by comparing test_fn outputs against reference across multiple seeds."""
    for seed in CORRECTNESS_SEEDS[:NUM_CORRECTNESS_RUNS]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        test_out = test_fn(*test_args)
        if isinstance(test_out, torch.Tensor):
            test_out = [test_out]
        if isinstance(ref_outputs, torch.Tensor):
            ref_outputs = [ref_outputs]

        for i, (ref, test) in enumerate(zip(ref_outputs, test_out)):
            if not isinstance(ref, torch.Tensor) or not isinstance(test, torch.Tensor):
                continue
            ref_f = ref.float()
            test_f = test.float()
            if not torch.allclose(ref_f, test_f, atol=atol, rtol=rtol):
                max_diff = (ref_f - test_f).abs().max().item()
                return False, f"seed={seed} output[{i}]: max_diff={max_diff:.3e}"
    return True, "passed"


def measure_baselines(
    problem: ProblemSpec,
    device: str = "cuda",
) -> BaselineResult:
    """Measure eager and torch.compile baselines for a KernelBench problem."""
    result = BaselineResult(
        problem_id=problem.problem_id,
        level=problem.level,
        name=problem.name,
    )
    try:
        init_inputs = problem.get_init_inputs_fn()
        model = problem.model_cls(*init_inputs).to(device).eval()
        inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in problem.get_inputs_fn()]

        with torch.no_grad():
            ref_out = model(*inputs)

        with torch.no_grad():
            result.eager_ns = _benchmark_fn(model, *inputs)
        result.eager_ok = True

        try:
            compiled = torch.compile(model, mode="default")
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()
                result.compile_ns = _benchmark_fn(compiled, *inputs)

            compile_out = compiled(*inputs)
            ok, msg = _check_correctness(ref_out, compiled, inputs)
            result.compile_ok = ok
        except Exception as e:
            result.compile_ok = False
            result.error = f"torch.compile failed: {e}"

    except Exception as e:
        result.error = f"baseline measurement failed: {e}\n{traceback.format_exc()}"

    return result


def benchmark_kernel(
    kernel_fn,
    inputs: list,
    ref_outputs,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """Benchmark a kernel function and check correctness against reference outputs.

    Returns dict with keys: correct, latency_ns, error, max_diff.
    """
    try:
        out = kernel_fn(*inputs)
        ok, msg = _check_correctness(ref_outputs, kernel_fn, inputs, atol=atol, rtol=rtol)
        if not ok:
            return {"correct": False, "latency_ns": float("inf"), "error": msg, "max_diff": msg}

        latency_ns = _benchmark_fn(kernel_fn, *inputs)
        return {"correct": True, "latency_ns": latency_ns, "error": "", "max_diff": ""}

    except Exception as e:
        return {"correct": False, "latency_ns": float("inf"), "error": str(e), "max_diff": ""}


def save_baseline_results(results: list[BaselineResult], output_path: str | Path) -> None:
    """Save baseline results to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    out.write_text(json.dumps(data, indent=2))


def load_baseline_results(path: str | Path) -> list[BaselineResult]:
    """Load baseline results from JSON."""
    data = json.loads(Path(path).read_text())
    return [BaselineResult(**d) for d in data]
