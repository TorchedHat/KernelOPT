"""KernelAgent adapter for SOL-ExecBench problems.

Wraps SOL-ExecBench JSON specs into the problem.py / test.py / input.py
format that KernelAgent's run_opt_manager.py expects.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import textwrap
from pathlib import Path
from typing import Optional

from benchmarks.sol_execbench.harness import load_problem, resolve_axes

KERNELAGENT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "KernelAgent"
RUN_OPT_MANAGER = KERNELAGENT_ROOT / "examples" / "run_opt_manager.py"
EXAMPLE_TEST = KERNELAGENT_ROOT / "examples" / "optimize_01_matvec" / "test.py"


def _create_problem_py(problem, workload_idx: int, out_path: Path) -> None:
    """Create a KernelBench-style problem.py from a SOL-ExecBench spec."""
    axes = resolve_axes(problem, workload_idx)
    ref_code = problem.reference_code

    input_alloc_lines = []
    input_names = list(problem.inputs_spec.keys())

    for name in input_names:
        spec = problem.inputs_spec[name]
        dtype_str = spec.get("dtype", "float32")
        shape_spec = spec.get("shape")

        wl = problem.workloads[workload_idx]
        wl_input = wl.get("inputs", {}).get(name, {"type": "random"})

        torch_dt = {
            "float16": "torch.float16", "bfloat16": "torch.bfloat16",
            "float32": "torch.float32", "int32": "torch.int32",
            "int64": "torch.int64", "float8_e4m3fn": "torch.float16",
        }.get(dtype_str, "torch.float32")

        if wl_input.get("type") == "scalar":
            input_alloc_lines.append(f"        {name} = {wl_input['value']}")
        elif shape_spec is not None:
            concrete = tuple(axes[d] for d in shape_spec)
            if "int" in dtype_str:
                input_alloc_lines.append(f"        {name} = torch.randint(0, 100, {concrete}, dtype={torch_dt})")
            else:
                input_alloc_lines.append(f"        {name} = torch.randn({concrete}, dtype={torch_dt})")
        else:
            input_alloc_lines.append(f"        {name} = torch.tensor(0.0)")

    alloc_block = "\n".join(input_alloc_lines)
    returns = ", ".join(input_names)

    code = textwrap.dedent(f'''\
        import torch
        import torch.nn as nn

        {ref_code}

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, {", ".join(input_names)}):
                return run({", ".join(input_names)})

        def get_inputs():
        {alloc_block}
            return [{returns}]

        def get_init_inputs():
            return []
    ''')
    out_path.write_text(code)


def _create_input_py(naive_kernel_path: str, out_path: Path) -> None:
    """Create input.py with kernel_function from the naive kernel."""
    source = Path(naive_kernel_path).read_text()
    if "def kernel_function" not in source and "def run(" in source:
        source += "\nkernel_function = run\n"
    elif "def kernel_function" not in source:
        launchers = re.findall(r"^def\s+(\w+)\s*\(", source, re.MULTILINE)
        jit_kernels = re.findall(r"@triton\.jit\s*\ndef\s+(\w+)", source)
        candidates = [f for f in launchers if f not in jit_kernels and f != "main"]
        if candidates:
            source += f"\nkernel_function = {candidates[0]}\n"
    out_path.write_text(source)


def _write_minimal_test(out_path: Path) -> None:
    out_path.write_text(textwrap.dedent('''\
        import sys
        import torch
        from kernel import kernel_function
        from problem import get_inputs, Model

        def test_kernel():
            device = "cuda"
            model = Model().to(device).eval()
            inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in get_inputs()]
            with torch.no_grad():
                ref = model(*inputs)
            out = kernel_function(*inputs)
            if out is None:
                out = inputs[0]
            if isinstance(ref, tuple):
                ref = ref[0]
            if isinstance(out, tuple):
                out = out[0]
            if torch.allclose(ref.float(), out.float(), rtol=1e-2, atol=1e-2):
                print("PASS")
                return True
            else:
                print(f"FAIL: max diff = {(ref.float() - out.float()).abs().max().item()}")
                return False

        if __name__ == "__main__":
            success = test_kernel()
            sys.exit(0 if success else 1)
    '''))


def _parse_program_db(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    data = json.loads(db_path.read_text())
    programs = data if isinstance(data, list) else data.get("programs", [])
    return [
        {"direction": p.get("bottleneck", p.get("diagnosis", "unknown")),
         "summary": p.get("description", ""),
         "time_ms": p.get("time_ms", float("inf"))}
        for p in programs
    ]


def run_kernelagent_sol(
    naive_kernel_path: str,
    problem_path: str,
    workload_idx: int,
    work_dir: str,
    *,
    strategy: str = "beam_search",
    max_rounds: int = 5,
    gpu_id: int = 0,
    timeout: int = 900,
) -> dict:
    """Run KernelAgent on a SOL-ExecBench problem for one workload."""
    problem = load_problem(problem_path)
    work = Path(work_dir) / f"wl{workload_idx}"
    work.mkdir(parents=True, exist_ok=True)

    _create_problem_py(problem, workload_idx, work / "problem.py")
    _create_input_py(naive_kernel_path, work / "input.py")
    if EXAMPLE_TEST.exists():
        shutil.copy2(str(EXAMPLE_TEST), work / "test.py")
    else:
        _write_minimal_test(work / "test.py")

    cmd = [
        "python", str(RUN_OPT_MANAGER),
        "--kernel-dir", str(work),
        "--strategy", strategy,
        "--max-rounds", str(max_rounds),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=env, cwd=str(KERNELAGENT_ROOT))
        wall_clock = time.monotonic() - start
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        wall_clock = timeout
        success = False

    result = {
        "tool": "kernelagent",
        "workload_idx": workload_idx,
        "axes": resolve_axes(problem, workload_idx),
        "success": success,
        "wall_clock_s": round(wall_clock, 1),
    }

    best = work / f"optimized_kernel_{strategy}.py"
    if best.exists():
        result["best_kernel_path"] = str(best)

    db = work / "opt_manager_logs" / strategy / "program_db.json"
    result["iterations"] = _parse_program_db(db)

    return result


def run_kernelagent_track_a(
    naive_kernel_path: str,
    problem_path: str,
    problem_id: str,
    results_dir: str,
    *,
    max_rounds: int = 5,
    gpu_id: int = 0,
    workload_indices: Optional[list[int]] = None,
) -> dict:
    """Track A: optimize a naive kernel across all workloads."""
    problem = load_problem(problem_path)
    if workload_indices is None:
        workload_indices = list(range(problem.num_workloads))

    out_dir = Path(results_dir) / "kernelagent" / "track_a" / problem_id
    workload_results = []

    for wi in workload_indices:
        r = run_kernelagent_sol(
            naive_kernel_path, problem_path, wi, str(out_dir),
            max_rounds=max_rounds, gpu_id=gpu_id,
        )
        workload_results.append(r)

    return {
        "problem_id": problem_id,
        "problem_name": problem.name,
        "category": problem.category,
        "tool": "kernelagent",
        "track": "A",
        "workloads": workload_results,
        "total_wall_clock_s": sum(r["wall_clock_s"] for r in workload_results),
    }
