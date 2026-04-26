"""SOL-ExecBench benchmark harness.

Parses JSON specs, resolves symbolic axes per workload, generates inputs
(random / scalar / safetensors), measures baselines, and checks correctness.
Aligned with the NVIDIA SOL-ExecBench evaluation protocol.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import textwrap
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import torch

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests_suite" / "benchmarks" / "sol_execbench"
SAFETENSORS_ROOT = Path(__file__).resolve().parent / "data"

WARMUP_REPS = 25
TIMING_REPS = 100
CORRECTNESS_SEEDS = [0, 42, 1337, 7, 2024]
NUM_CORRECTNESS_RUNS = 5

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "float8_e4m3fn": getattr(torch, "float8_e4m3fn", torch.float16),
    "float8_e5m2": getattr(torch, "float8_e5m2", torch.float16),
}

CATEGORY_PATTERNS = {
    "gemm": ["gemm"],
    "fused_rmsnorm": ["fused_add_rmsnorm"],
    "rmsnorm": ["rmsnorm"],
    "gqa": ["gqa"],
    "mla": ["mla"],
    "moe": ["moe"],
}


def _categorize(name: str) -> str:
    name_lower = name.lower()
    for cat, patterns in CATEGORY_PATTERNS.items():
        if any(p in name_lower for p in patterns):
            return cat
    return "other"


@dataclass
class SolProblem:
    """A parsed SOL-ExecBench problem."""
    problem_id: str
    name: str
    description: str
    category: str
    source_path: str
    axes: dict
    inputs_spec: dict
    outputs_spec: dict
    reference_code: str
    workloads: list[dict]
    num_workloads: int


@dataclass
class WorkloadInputs:
    """Resolved concrete inputs for a single workload."""
    workload_idx: int
    axes_resolved: dict[str, int]
    tensors: dict[str, Any]
    shapes: dict[str, tuple]
    dtypes: dict[str, str]


@dataclass
class BaselineResult:
    """Baseline timing for a single problem + workload."""
    problem_id: str
    problem_name: str
    category: str
    workload_idx: int
    axes: dict
    ref_ns: float = float("inf")
    compile_ns: float = float("inf")
    ref_ok: bool = False
    compile_ok: bool = False
    error: str = ""


def list_problems(root: Optional[Path] = None) -> list[dict]:
    """List all SOL-ExecBench problems with metadata."""
    root = root or DATA_ROOT
    problems = []
    train_dir = root / "train"
    if not train_dir.is_dir():
        return problems
    for json_path in sorted(train_dir.glob("kernel_*.json")):
        raw = json.loads(json_path.read_text())
        name = raw.get("name", json_path.stem)
        problems.append({
            "problem_id": json_path.stem,
            "name": name,
            "category": _categorize(name),
            "path": str(json_path),
        })
    return problems


def load_problem(json_path: str | Path) -> SolProblem:
    """Parse a SOL-ExecBench JSON spec into a SolProblem."""
    raw = json.loads(Path(json_path).read_text())
    axes = json.loads(raw["axes"]) if isinstance(raw["axes"], str) else raw["axes"]
    inputs_spec = json.loads(raw["inputs"]) if isinstance(raw["inputs"], str) else raw["inputs"]
    outputs_spec = json.loads(raw["outputs"]) if isinstance(raw["outputs"], str) else raw["outputs"]
    workloads = json.loads(raw["workloads"]) if isinstance(raw["workloads"], str) else raw["workloads"]
    name = raw.get("name", Path(json_path).stem)

    return SolProblem(
        problem_id=Path(json_path).stem,
        name=name,
        description=raw.get("description", ""),
        category=_categorize(name),
        source_path=str(json_path),
        axes=axes,
        inputs_spec=inputs_spec,
        outputs_spec=outputs_spec,
        reference_code=raw.get("reference", ""),
        workloads=workloads,
        num_workloads=len(workloads),
    )


def resolve_axes(problem: SolProblem, workload_idx: int) -> dict[str, int]:
    """Resolve all axes for a specific workload. Returns axis_name -> int value."""
    resolved = {}
    for axis_name, axis_spec in problem.axes.items():
        if axis_spec.get("type") == "const":
            resolved[axis_name] = axis_spec["value"]
    wl = problem.workloads[workload_idx]
    for axis_name, value in wl.get("axes", {}).items():
        resolved[axis_name] = value
    return resolved


def _resolve_shape(shape_spec: list[str] | None, axes: dict[str, int]) -> tuple | None:
    """Resolve a symbolic shape like ["M", "K"] to concrete (4096, 7168)."""
    if shape_spec is None:
        return None
    return tuple(axes[dim] for dim in shape_spec)


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    return DTYPE_MAP.get(dtype_str, torch.float32)


def generate_inputs(
    problem: SolProblem,
    workload_idx: int,
    device: str = "cuda",
) -> WorkloadInputs:
    """Generate concrete inputs for a specific workload."""
    axes = resolve_axes(problem, workload_idx)
    wl = problem.workloads[workload_idx]
    wl_inputs = wl.get("inputs", {})

    tensors = {}
    shapes = {}
    dtypes = {}

    for input_name, input_spec in problem.inputs_spec.items():
        dtype_str = input_spec.get("dtype", "float32")
        shape_spec = input_spec.get("shape")
        wl_input = wl_inputs.get(input_name, {"type": "random"})
        input_type = wl_input.get("type", "random")

        dtypes[input_name] = dtype_str

        if input_type == "scalar":
            tensors[input_name] = wl_input["value"]
            shapes[input_name] = ()
        elif input_type == "safetensors":
            st_path = wl_input.get("path", "")
            tensor_key = wl_input.get("tensor_key", "")
            local_path = SAFETENSORS_ROOT / st_path
            if local_path.exists():
                from safetensors.torch import load_file
                data = load_file(str(local_path))
                tensors[input_name] = data[tensor_key].to(device)
                shapes[input_name] = tuple(tensors[input_name].shape)
            else:
                concrete_shape = _resolve_shape(shape_spec, axes)
                if concrete_shape:
                    torch_dtype = _get_torch_dtype(dtype_str)
                    if "int" in dtype_str:
                        tensors[input_name] = torch.randint(0, 100, concrete_shape, dtype=torch_dtype, device=device)
                    else:
                        tensors[input_name] = torch.randn(concrete_shape, dtype=torch_dtype, device=device)
                    shapes[input_name] = concrete_shape
                else:
                    tensors[input_name] = torch.tensor(0.0, device=device)
                    shapes[input_name] = ()
        else:
            concrete_shape = _resolve_shape(shape_spec, axes)
            if concrete_shape is None:
                tensors[input_name] = torch.tensor(0.0, device=device)
                shapes[input_name] = ()
            else:
                torch_dtype = _get_torch_dtype(dtype_str)
                if "int" in dtype_str:
                    tensors[input_name] = torch.randint(0, 100, concrete_shape, dtype=torch_dtype, device=device)
                else:
                    tensors[input_name] = torch.randn(concrete_shape, dtype=torch_dtype, device=device)
                shapes[input_name] = concrete_shape

    return WorkloadInputs(
        workload_idx=workload_idx,
        axes_resolved=axes,
        tensors=tensors,
        shapes=shapes,
        dtypes=dtypes,
    )


def _compile_reference(reference_code: str) -> types.FunctionType:
    """Compile the reference code string into an executable run() function."""
    mod = types.ModuleType("_sol_ref")
    mod.__dict__["torch"] = torch
    mod.__dict__["math"] = math
    exec(compile(reference_code, "<reference>", "exec"), mod.__dict__)
    return mod.__dict__["run"]


def _benchmark_fn(fn, args: list, warmup: int = WARMUP_REPS, reps: int = TIMING_REPS) -> float:
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


def _get_atol(dtype_str: str) -> float:
    if dtype_str in ("float32", "float64"):
        return 1e-4
    return 1e-2


def measure_baseline(
    problem: SolProblem,
    workload_idx: int,
    device: str = "cuda",
) -> BaselineResult:
    """Measure reference and torch.compile baselines for one workload."""
    result = BaselineResult(
        problem_id=problem.problem_id,
        problem_name=problem.name,
        category=problem.category,
        workload_idx=workload_idx,
        axes=resolve_axes(problem, workload_idx),
    )
    try:
        wl_inputs = generate_inputs(problem, workload_idx, device=device)
        run_fn = _compile_reference(problem.reference_code)
        args = list(wl_inputs.tensors.values())

        with torch.no_grad():
            ref_out = run_fn(*args)
            result.ref_ns = _benchmark_fn(run_fn, args)
            result.ref_ok = True

        try:
            compiled_fn = torch.compile(run_fn, mode="default")
            with torch.no_grad():
                for _ in range(3):
                    compiled_fn(*args)
                torch.cuda.synchronize()
                result.compile_ns = _benchmark_fn(compiled_fn, args)
                result.compile_ok = True
        except Exception as e:
            result.error = f"torch.compile failed: {e}"

    except Exception as e:
        result.error = f"baseline failed: {e}"

    return result


def needs_safetensors(problem: SolProblem) -> bool:
    """Check if any workload requires safetensors data."""
    for wl in problem.workloads:
        for inp in wl.get("inputs", {}).values():
            if isinstance(inp, dict) and inp.get("type") == "safetensors":
                return True
    return False


def get_safetensors_paths(problem: SolProblem) -> list[str]:
    """Get all unique safetensors paths needed by this problem."""
    paths = set()
    for wl in problem.workloads:
        for inp in wl.get("inputs", {}).values():
            if isinstance(inp, dict) and inp.get("type") == "safetensors":
                paths.add(inp.get("path", ""))
    return sorted(p for p in paths if p)


def save_baseline_results(results: list[BaselineResult], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))


def load_baseline_results(path: str | Path) -> list[BaselineResult]:
    data = json.loads(Path(path).read_text())
    return [BaselineResult(**d) for d in data]
