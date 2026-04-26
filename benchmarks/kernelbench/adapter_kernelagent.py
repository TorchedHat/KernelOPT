"""KernelAgent adapter for KernelBench problems.

Creates the directory structure KernelAgent's run_opt_manager.py expects
(problem.py, test.py, input.py), invokes optimization, and collects results.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

KERNELAGENT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "KernelAgent"
RUN_OPT_MANAGER = KERNELAGENT_ROOT / "examples" / "run_opt_manager.py"
EXAMPLE_TEST_PY = KERNELAGENT_ROOT / "examples" / "optimize_01_matvec" / "test.py"


def _create_problem_dir(
    problem_py_path: str,
    naive_kernel_path: str,
    work_dir: str,
) -> Path:
    """Create the kernel-dir structure KernelAgent expects.

    Layout:
        work_dir/
            problem.py  -- KernelBench Model + get_inputs (symlink or copy)
            test.py     -- correctness test (from KernelAgent examples, generic)
            input.py    -- naive Triton kernel (renamed to expose kernel_function)
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    shutil.copy2(problem_py_path, work / "problem.py")

    if EXAMPLE_TEST_PY.exists():
        shutil.copy2(str(EXAMPLE_TEST_PY), work / "test.py")
    else:
        _write_minimal_test(work / "test.py")

    naive_src = Path(naive_kernel_path).read_text()
    input_py = work / "input.py"
    if "def kernel_function" not in naive_src:
        naive_src = _wrap_as_kernel_function(naive_src)
    input_py.write_text(naive_src)

    return work


def _wrap_as_kernel_function(source: str) -> str:
    """If the naive kernel doesn't export kernel_function, add an alias.

    Looks for the main launcher function (not the @triton.jit kernel) and
    aliases it as kernel_function at the bottom of the file.
    """
    import re
    launchers = re.findall(r"^def\s+(\w+)\s*\(", source, re.MULTILINE)
    jit_kernels = re.findall(r"@triton\.jit\s*\ndef\s+(\w+)", source)
    candidates = [f for f in launchers if f not in jit_kernels and f != "main"]

    if candidates:
        return source + f"\n\nkernel_function = {candidates[0]}\n"
    return source + "\n\n# WARNING: could not detect launcher function for kernel_function alias\n"


def _write_minimal_test(path: Path) -> None:
    """Write a minimal test.py if the KernelAgent example test isn't available."""
    path.write_text('''\
import sys
import torch
from kernel import kernel_function
from problem import get_init_inputs, get_inputs, Model

def test_kernel():
    device = "cuda"
    model = Model(*get_init_inputs()).to(device).eval()
    inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in get_inputs()]
    with torch.no_grad():
        ref = model(*inputs)
    out = kernel_function(*inputs)
    if out is None:
        out = inputs[0]
    if torch.allclose(ref.float(), out.float(), rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    else:
        print(f"FAIL: max diff = {(ref.float() - out.float()).abs().max().item()}")
        return False

if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
''')


def _parse_program_db(db_path: Path) -> list[dict]:
    """Parse KernelAgent's program_db.json for optimization history."""
    if not db_path.exists():
        return []
    data = json.loads(db_path.read_text())
    iterations = []
    programs = data if isinstance(data, list) else data.get("programs", [])
    for prog in programs:
        iterations.append({
            "direction": prog.get("bottleneck", prog.get("diagnosis", "unknown")),
            "summary": prog.get("description", ""),
            "time_ms": prog.get("time_ms", float("inf")),
            "generation": prog.get("generation", 0),
        })
    return iterations


def run_kernelagent(
    problem_py_path: str,
    naive_kernel_path: str,
    work_dir: str,
    *,
    strategy: str = "beam_search",
    max_rounds: int = 5,
    gpu_id: int = 0,
    timeout: int | None = None,
) -> dict:
    """Run KernelAgent optimization on a problem and collect results."""
    work = _create_problem_dir(problem_py_path, naive_kernel_path, work_dir)

    cmd = [
        "python", str(RUN_OPT_MANAGER),
        "--kernel-dir", str(work),
        "--strategy", strategy,
        "--max-rounds", str(max_rounds),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.monotonic()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
        cwd=str(KERNELAGENT_ROOT),
    )
    wall_clock = time.monotonic() - start
    success = proc.returncode == 0
    stdout = proc.stdout
    stderr = proc.stderr

    result = {
        "tool": "kernelagent",
        "success": success,
        "wall_clock_s": round(wall_clock, 1),
        "stdout": stdout[-2000:] if stdout else "",
        "stderr": stderr[-1000:] if stderr else "",
    }

    best_kernel = work / f"optimized_kernel_{strategy}.py"
    if best_kernel.exists():
        result["best_kernel_path"] = str(best_kernel)
        result["best_kernel_source"] = best_kernel.read_text()

    db_path = work / "opt_manager_logs" / strategy / "program_db.json"
    result["iterations"] = _parse_program_db(db_path)

    _extract_timing_from_stdout(stdout, result)

    return result


def _extract_timing_from_stdout(stdout: str, result: dict) -> None:
    """Extract best_time_ms and baseline times from KernelAgent stdout."""
    import re
    for line in stdout.split("\n"):
        if "Best time:" in line:
            match = re.search(r"([\d.]+)\s*ms", line)
            if match:
                result["best_time_ms"] = float(match.group(1))
                result["best_ns"] = float(match.group(1)) * 1e6


def run_kernelagent_track_a(
    problem_py_path: str,
    naive_kernel_path: str,
    problem_id: str,
    results_dir: str,
    *,
    max_rounds: int = 5,
    gpu_id: int = 0,
) -> dict:
    """Track A: optimize a given naive kernel with KernelAgent."""
    work_dir = str(Path(results_dir) / "kernelagent" / "track_a" / problem_id)
    result = run_kernelagent(
        problem_py_path, naive_kernel_path, work_dir,
        max_rounds=max_rounds, gpu_id=gpu_id,
    )
    result["problem_id"] = problem_id
    result["track"] = "A"
    return result


def run_kernelagent_track_b(
    problem_py_path: str,
    problem_id: str,
    results_dir: str,
    *,
    max_rounds: int = 5,
    gpu_id: int = 0,
) -> dict:
    """Track B: end-to-end (KernelAgent generates + optimizes from PyTorch spec).

    Uses TritonKernelAgent.generate_kernel() for initial generation, then
    feeds result into OptimizationManager.
    """
    work_dir = Path(results_dir) / "kernelagent" / "track_b" / problem_id
    work_dir.mkdir(parents=True, exist_ok=True)

    source = Path(problem_py_path).read_text()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    gen_script = work_dir / "_generate.py"
    gen_script.write_text(f'''\
import sys
sys.path.insert(0, "{KERNELAGENT_ROOT}")
from triton_kernel_agent.agent import TritonKernelAgent

agent = TritonKernelAgent()
result = agent.generate_kernel(
    problem_description="""{source}""",
)
kernel_code = result.get("kernel_code", "")
(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout")
from pathlib import Path
Path(sys.argv[1]).write_text(kernel_code)
print("Generation OK")
''')

    gen_output = work_dir / "generated_kernel.py"
    try:
        subprocess.run(
            ["python", str(gen_script), str(gen_output)],
            capture_output=True, text=True, timeout=120, env=env,
            cwd=str(KERNELAGENT_ROOT),
        )
    except Exception:
        pass

    if gen_output.exists():
        naive_path = str(gen_output)
    else:
        naive_path = str(work_dir / "input.py")
        (work_dir / "input.py").write_text("# Generation failed\ndef kernel_function(*args): pass\n")

    result = run_kernelagent(
        problem_py_path, naive_path, str(work_dir),
        max_rounds=max_rounds, gpu_id=gpu_id,
    )
    result["problem_id"] = problem_id
    result["track"] = "B"
    return result
