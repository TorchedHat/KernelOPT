"""Decoupled autotuner for GPU kernels (Triton, CUDA, Helion).

Extracts tunable parameters from GPU kernels and sweeps a configuration grid:
  - Triton: `tl.constexpr` parameters (BLOCK_SIZE, num_warps, num_stages)
  - CUDA:   `#define` macros, template parameters, and dim3 configs
  - Helion: detects `hl.register_block_size()` -- defers to Helion's native autotuner

The LLM designs the kernel *algorithm*; this module tunes the *parameters*.
"""

from __future__ import annotations

import ast
import copy
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


def extract_constexpr_params(source: str) -> list[dict]:
    """Find all tl.constexpr parameters across @triton.jit kernels.

    Returns a list of dicts: [{"kernel": name, "param": param_name, "default": value_or_None}, ...]
    """
    tree = ast.parse(source)
    params = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_jit = any(
            (isinstance(d, ast.Attribute) and d.attr == "jit")
            or (isinstance(d, ast.Name) and d.id in ("jit",))
            for d in node.decorator_list
        )
        if not is_jit:
            continue

        for arg in node.args.args:
            annotation = arg.annotation
            if annotation is None:
                continue
            ann_src = ast.get_source_segment(source, annotation) or ""
            if "constexpr" in ann_src:
                default_val = _find_constexpr_default(source, node.name, arg.arg)
                params.append({
                    "kernel": node.name,
                    "param": arg.arg,
                    "default": default_val,
                })

    return params


def _find_constexpr_default(source: str, kernel_name: str, param_name: str) -> int | None:
    """Try to find the default value used when launching the kernel.

    Looks for patterns like: kernel[grid](... PARAM_NAME=128 ...)
    or:                      PARAM_NAME = 128; ... kernel[grid](... PARAM_NAME=PARAM_NAME ...)
    """
    # Pattern 1: keyword arg in kernel launch
    m = re.search(
        rf'{param_name}\s*=\s*(\d+)',
        source,
    )
    if m:
        return int(m.group(1))
    return None


def extract_cuda_defines(source: str) -> list[dict]:
    """Extract tunable parameters from CUDA source.

    Detects:
      - #define macros: #define TILE_SIZE 16
      - Template defaults: template<int BLOCK_SIZE = 256>
      - constexpr constants: constexpr int THREADS_PER_BLOCK = 1024

    Only extracts values with names suggesting GPU parameters.
    """
    tunable_patterns = re.compile(
        r'(?:TILE|BLOCK|THREAD|UNROLL|NUM_THREADS|SHARED)',
        re.IGNORECASE,
    )
    # Hardware constants that should never be tuned
    skip_names = {"WARP_SIZE", "warp_size"}
    params = []
    seen = set()

    # Pattern 1: #define macros
    for m in re.finditer(r'#define\s+(\w+)\s+(\d+)', source):
        name, value = m.group(1), int(m.group(2))
        if tunable_patterns.search(name) and name not in seen and name not in skip_names:
            params.append({"kernel": "__global__", "param": name, "default": value})
            seen.add(name)

    # Pattern 2: template defaults  e.g.  template<int BLOCK_SIZE = 256>
    for m in re.finditer(r'template\s*<[^>]*int\s+(\w+)\s*=\s*(\d+)', source):
        name, value = m.group(1), int(m.group(2))
        if tunable_patterns.search(name) and name not in seen and name not in skip_names:
            params.append({"kernel": "__global__", "param": name, "default": value})
            seen.add(name)

    # Pattern 3: constexpr int  e.g.  constexpr int THREADS_PER_BLOCK = 1024;
    for m in re.finditer(r'constexpr\s+int\s+(\w+)\s*=\s*(\d+)', source):
        name, value = m.group(1), int(m.group(2))
        if tunable_patterns.search(name) and name not in seen and name not in skip_names:
            params.append({"kernel": "__global__", "param": name, "default": value})
            seen.add(name)

    return params


def extract_helion_params(source: str) -> list[dict]:
    """Extract tunable parameters from Helion kernels.

    Helion uses hl.register_block_size() to declare tunable block sizes.
    These are resolved by Helion's compiler, not passed as Python constants.
    We extract them so the autotuner can sweep via Helion's config API.

    Also detects explicit integer block_size assignments.
    """
    params = []

    # Pattern 1: hl.register_block_size(dim) -- Helion's native tuning API
    for m in re.finditer(r'(\w+)\s*=\s*hl\.register_block_size\((\w+)\)', source):
        var_name = m.group(1)  # e.g. "block_size_m"
        dim_name = m.group(2)  # e.g. "m"
        params.append({
            "kernel": "@helion.kernel",
            "param": var_name,
            "default": None,
            "helion_dim": dim_name,
            "type": "register_block_size",
        })

    # Pattern 2: explicit integer block_size assignments
    for m in re.finditer(r'(\w*block_size\w*)\s*=\s*(\d+)', source, re.IGNORECASE):
        name, value = m.group(1), int(m.group(2))
        if not any(p["param"] == name for p in params):
            params.append({"kernel": "@helion.kernel", "param": name, "default": value})

    return params


def extract_python_gpu_params(source: str) -> list[dict]:
    """Extract tunable GPU parameters from Python variable assignments.

    Catches patterns like:
      TILE_SIZE = 16
      BLOCK_DIM = 256
      NUM_WARPS = 4

    These are common in inline CUDA (load_inline) and custom kernel wrappers.
    """
    tunable_patterns = re.compile(
        r'(?:TILE|BLOCK|THREAD|WARP|UNROLL|NUM_THREADS|SHARED|NUM_WARPS|NUM_STAGES)',
        re.IGNORECASE,
    )
    params = []
    # Match top-level assignments: NAME = integer
    for m in re.finditer(r'^(\w+)\s*=\s*(\d+)\s*$', source, re.MULTILINE):
        name, value = m.group(1), int(m.group(2))
        if tunable_patterns.search(name):
            params.append({"kernel": "module", "param": name, "default": value})
    return params


def extract_tunable_params(source: str, framework: str = "auto") -> list[dict]:
    """Extract tunable parameters from any supported framework.

    Auto-detects framework if not specified. Merges results from
    framework-specific extractors.
    """
    if framework == "auto":
        if "@triton.jit" in source or "tl.constexpr" in source:
            framework = "triton"
        elif "__global__" in source and "#define" in source:
            framework = "cuda"
        elif "@helion.kernel" in source or "@hl.kernel" in source:
            framework = "helion"
        elif "load_inline" in source or "cpp_extension" in source:
            framework = "inline_cuda"
        else:
            framework = "triton"

    if framework == "triton":
        return extract_constexpr_params(source)
    elif framework == "cuda":
        return extract_cuda_defines(source)
    elif framework == "inline_cuda":
        return extract_python_gpu_params(source)
    elif framework == "helion":
        return extract_helion_params(source)
    return []


def generate_config_grid(
    params: list[dict],
    *,
    block_range: tuple[int, int] = (32, 2048),
    warp_options: list[int] | None = None,
    stage_options: list[int] | None = None,
) -> list[dict[str, int]]:
    """Generate a grid of configurations to benchmark.

    Heuristics:
      - Parameters with "BLOCK" or "SIZE" in the name: powers of 2 in block_range
      - "num_warps": warp_options (default [1, 2, 4, 8])
      - "num_stages": stage_options (default [1, 2, 3, 4])
      - Others: try the default and +/- one power of 2
    """
    if warp_options is None:
        warp_options = [1, 2, 4, 8]
    if stage_options is None:
        stage_options = [1, 2, 3, 4]

    param_values: dict[str, list[int]] = {}

    for p in params:
        name = p["param"]
        default = p["default"]
        name_upper = name.upper()

        if "BLOCK" in name_upper or "SIZE" in name_upper:
            lo, hi = block_range
            vals = []
            v = lo
            while v <= hi:
                vals.append(v)
                v *= 2
            param_values[name] = vals

        elif "WARP" in name_upper or name_upper == "NUM_WARPS":
            param_values[name] = warp_options

        elif "STAGE" in name_upper or name_upper == "NUM_STAGES":
            param_values[name] = stage_options

        elif default is not None:
            vals = set()
            for shift in (-1, 0, 1):
                candidate = default * (2 ** shift)
                if candidate >= 1:
                    vals.add(int(candidate))
            param_values[name] = sorted(vals)
        else:
            continue

    if not param_values:
        return []

    # Cartesian product (capped at max_configs)
    names = list(param_values.keys())
    grids = [param_values[n] for n in names]

    configs = [{}]
    for name, vals in zip(names, grids):
        new_configs = []
        for cfg in configs:
            for v in vals:
                c = dict(cfg)
                c[name] = v
                new_configs.append(c)
        configs = new_configs

    max_configs = 256
    if len(configs) > max_configs:
        import random
        random.seed(42)
        configs = random.sample(configs, max_configs)

    return configs


def _build_benchmark_script(
    kernel_source: str,
    wrapper_call: str,
    param_configs: list[dict[str, int]],
    constexpr_params: list[dict],
) -> str:
    """Generate a standalone Python script that benchmarks all configs.

    The script:
    1. Defines the kernel (with parameterized constexpr)
    2. For each config: patches the launch call, benchmarks, prints timing
    """
    param_names = [p["param"] for p in constexpr_params]

    script = textwrap.dedent(f"""\
    import json, sys, time
    import torch
    import triton
    import triton.language as tl

    # --- Original kernel source (we'll patch constexpr values per config) ---
    KERNEL_SOURCE = {repr(kernel_source)}
    CONFIGS = {repr(param_configs)}

    results = []
    for i, cfg in enumerate(CONFIGS):
        try:
            # Build a patched source with specific constexpr values
            patched = KERNEL_SOURCE
            for pname, pval in cfg.items():
                # Replace the launch-site keyword arg
                import re
                patched = re.sub(
                    rf'{{pname}}\\s*=\\s*\\d+',
                    f'{{pname}}={{pval}}',
                    patched,
                )

            # Execute the patched source to define functions
            ns = {{}}
            exec(compile(patched, "<autotuner>", "exec"), ns)

            # Find the wrapper function (first non-underscored callable that's not a jit kernel)
            wrapper_fn = None
            for name, obj in ns.items():
                if name.startswith("_") or name.startswith("tl") or name == "triton":
                    continue
                if callable(obj) and not isinstance(obj, type):
                    if not hasattr(obj, 'fn'):  # skip JIT kernels
                        wrapper_fn = obj
                        break

            if wrapper_fn is None:
                results.append({{"config": cfg, "time_ms": float("inf"), "error": "no wrapper found"}})
                continue

            # Generate dummy inputs based on the wrapper's first argument
            # This is a best-effort approach; the actual benchmark should use real inputs
            torch.cuda.synchronize()

            # Benchmark using triton's do_bench
            ms = triton.testing.do_bench(
                lambda: exec(compile('{wrapper_call}', '<bench>', 'exec'), ns),
                warmup=25,
                rep=100,
            )
            results.append({{"config": cfg, "time_ms": ms}})
        except Exception as e:
            results.append({{"config": cfg, "time_ms": float("inf"), "error": str(e)[:200]}})

    # Output results as JSON
    print("AUTOTUNE_RESULTS:" + json.dumps(results))
    """)

    return script


def apply_best_config(source: str, best_config: dict[str, int]) -> str:
    """Patch the kernel source to use the best config's values.

    Handles:
      - CUDA #define macros: #define TILE_SIZE 16 -> #define TILE_SIZE 32
      - C++ template defaults: template<int BLOCK_SIZE = 256> -> = 512
      - C++ constexpr: constexpr int X = 1024 -> constexpr int X = 512
      - Python/Triton keyword args and assignments: BLOCK_SIZE=128 -> BLOCK_SIZE=256
    """
    patched = source
    for param_name, value in best_config.items():
        # CUDA #define macros
        patched = re.sub(
            rf'(#define\s+{param_name}\s+)\d+',
            rf'\g<1>{value}',
            patched,
        )
        # C++ template defaults: int PARAM = old -> int PARAM = new
        patched = re.sub(
            rf'(int\s+{param_name}\s*=\s*)\d+',
            rf'\g<1>{value}',
            patched,
        )
        # Python/Triton keyword arguments and variable assignments
        patched = re.sub(
            rf'({param_name}\s*=\s*)\d+',
            rf'\g<1>{value}',
            patched,
        )

    return patched


_BENCH_HARNESS = textwrap.dedent("""\

# --- Autotuner benchmark harness (appended) ---
if __name__ == "__main__":
    import json, sys, inspect, torch, triton

    # Strategy 1: use get_inputs() if available
    _inputs = None
    if "get_inputs" in dir():
        try:
            _inputs = get_inputs()
            _inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in _inputs]
        except Exception:
            _inputs = None

    # Find the wrapper function (non-jit, non-class, non-private callable)
    _wrapper = None
    _skip = {"get_inputs", "get_init_inputs", "torch", "triton", "tl", "benchmark",
             "module", "cuda_source", "cpp_source", "load_inline"}
    # Also skip any name that looks like a utility/config function (heuristic)
    _skip_prefixes = ("is_", "has_", "get_", "supports_", "check_", "test_",
                      "validate_", "setup_", "init_", "config_")
    import types as _types
    for _n, _obj in sorted(globals().items()):
        if _n.startswith("_") or _n in _skip or any(_n.startswith(p) for p in _skip_prefixes):
            continue
        if isinstance(_obj, type):
            continue
        if not isinstance(_obj, (_types.FunctionType, _types.BuiltinFunctionType)):
            continue
        if hasattr(_obj, "fn"):  # skip Triton JIT kernels
            continue
        _wrapper = _obj
        _wrapper_name = _n
        break

    if _wrapper is None:
        print("AUTOTUNE_RESULT:" + json.dumps({"error": "no wrapper found"}))
        sys.exit(0)

    # Strategy 2: generate inputs from wrapper signature if get_inputs unavailable
    if _inputs is None:
        try:
            _sig = inspect.signature(_wrapper)
            _inputs = []
            _size = 2**18  # 256K elements
            for _pname, _p in _sig.parameters.items():
                if _p.annotation != inspect.Parameter.empty:
                    _ann = str(_p.annotation)
                    if "Tensor" in _ann:
                        _inputs.append(torch.randn(_size, device="cuda"))
                    elif "float" in _ann:
                        _inputs.append(1.0)
                    elif "int" in _ann:
                        _inputs.append(_size)
                    elif "bool" in _ann:
                        _inputs.append(False)
                    elif "str" in _ann:
                        _inputs.append("")
                    else:
                        _inputs.append(torch.randn(_size, device="cuda"))
                elif _p.default != inspect.Parameter.empty:
                    _inputs.append(_p.default)
                else:
                    _inputs.append(torch.randn(_size, device="cuda"))
        except Exception as _e:
            print("AUTOTUNE_RESULT:" + json.dumps({"error": f"cannot generate inputs: {_e}"}))
            sys.exit(0)

    try:
        _ms = triton.testing.do_bench(lambda: _wrapper(*_inputs), warmup=25, rep=100)
        print("AUTOTUNE_RESULT:" + json.dumps({"time_ms": _ms}))
    except Exception as _e:
        print("AUTOTUNE_RESULT:" + json.dumps({"error": str(_e)[:200]}))
""")


def _build_cuda_bench_main(kernel_source: str) -> str:
    """Build a generic CUDA benchmark main() for kernels without one.

    Detects the extern "C" entry point signature and generates matching
    allocations + a benchmark loop. Falls back to a minimal stub if the
    signature can't be parsed.
    """
    # Find extern "C" wrapper: extern "C" void solve(const float* in, float* out, int N)
    extern_match = re.search(
        r'extern\s+"C"\s+void\s+(\w+)\s*\(([^)]*)\)',
        kernel_source,
    )

    if extern_match:
        fn_name = extern_match.group(1)
        params_str = extern_match.group(2)

        # Parse parameter list to generate allocations
        alloc_lines = []
        call_args = []
        free_lines = []
        param_idx = 0

        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue

            if "int" in param and "*" not in param:
                # Scalar int param (like N)
                var = param.split()[-1].strip()
                alloc_lines.append(f"    int {var} = 4096;")
                call_args.append(var)
            elif "float*" in param or "float *" in param:
                var = f"d_{param_idx}"
                alloc_lines.append(f"    float* {var}; cudaMalloc(&{var}, 4096 * sizeof(float));")
                free_lines.append(f"    cudaFree({var});")
                call_args.append(var)
                param_idx += 1
            elif "double*" in param or "double *" in param:
                var = f"d_{param_idx}"
                alloc_lines.append(f"    double* {var}; cudaMalloc(&{var}, 4096 * sizeof(double));")
                free_lines.append(f"    cudaFree({var});")
                call_args.append(var)
                param_idx += 1
            else:
                call_args.append("0")

        call_str = ", ".join(call_args)
        allocs = "\n".join(alloc_lines)
        frees = "\n".join(free_lines)

        return textwrap.dedent(f"""\

// --- Autotuner benchmark harness (auto-generated) ---
#include <stdio.h>
#include <stdlib.h>
int main() {{
{allocs}

    // Warmup
    for (int i = 0; i < 5; i++) {fn_name}({call_str});
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {fn_name}({call_str});
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("AUTOTUNE_RESULT:{{\\"time_ms\\": %.4f}}\\n", ms / 100.0f);

{frees}
    return 0;
}}
""")

    # Fallback: can't parse extern "C", skip benchmark harness
    return ""


def _parse_cuda_timing(stdout: str) -> float | None:
    """Extract timing from various CUDA benchmark output formats.

    Supports:
      - AUTOTUNE_RESULT:{"time_ms": 0.1234}
      - Avg latency: 12.345 us
      - avg_time=0.1234 ms
      - Time: 0.1234 ms
    """
    import json as _json

    for line in stdout.splitlines():
        if line.startswith("AUTOTUNE_RESULT:"):
            try:
                data = _json.loads(line[len("AUTOTUNE_RESULT:"):])
                if "time_ms" in data:
                    return data["time_ms"]
            except _json.JSONDecodeError:
                pass

        m = re.search(r'[Aa]vg[_ ](?:latency|time)[=:]\s*([\d.]+)\s*(us|ms)', line)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            return val / 1000.0 if unit == "us" else val

        m = re.search(r'[Tt]ime[=:]\s*([\d.]+)\s*ms', line)
        if m:
            return float(m.group(1))

    return None


def _run_autotune_cuda(
    kernel_source: str,
    params: list[dict],
    configs: list[dict],
    *,
    gpu_id: int = 0,
    timeout: int = 60,
    debug: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Autotune a CUDA .cu kernel by compiling with nvcc for each config."""
    import os

    best_config = None
    best_time = float("inf")
    default_time = None
    tested = 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Check if source has a main() -- if not, we need to append one
    has_main = "int main(" in kernel_source or "void main(" in kernel_source
    # Detect kernel function name for the harness
    kernel_names = re.findall(r'__global__\s+void\s+(\w+)', kernel_source)

    for i, cfg in enumerate(configs):
        patched = apply_best_config(kernel_source, cfg)

        # If no main(), generate a benchmark harness from the extern "C" signature
        if not has_main:
            harness = _build_cuda_bench_main(patched)
            if not harness:
                if debug:
                    print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> SKIP (no main and can't generate harness)")
                continue
            patched += harness

        cu_path = tempfile.mktemp(suffix=".cu", prefix="autotune_cuda_")
        bin_path = cu_path.replace(".cu", "")

        Path(cu_path).write_text(patched)

        try:
            # Compile
            comp = subprocess.run(
                ["nvcc", "-O2", "-Wno-deprecated-gpu-targets", "-o", bin_path, cu_path],
                capture_output=True, text=True, timeout=30, env=env,
            )
            if comp.returncode != 0:
                if debug:
                    print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> COMPILE FAILED ({comp.stderr[:100]})")
                continue

            # Run -- use --benchmark flag if the source supports it
            run_cmd = [bin_path]
            if "--benchmark" in kernel_source:
                run_cmd.append("--benchmark")
            result = subprocess.run(
                run_cmd, capture_output=True, text=True, timeout=timeout, env=env,
            )
            tested += 1

            ms = _parse_cuda_timing(result.stdout)
            if ms is not None:
                tested += 1
                is_default = all(
                    cfg.get(p["param"]) == p["default"]
                    for p in params
                    if p["default"] is not None and p["param"] in cfg
                )
                if is_default and default_time is None:
                    default_time = ms

                if ms < best_time:
                    best_time = ms
                    best_config = cfg

                if debug:
                    print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> {ms:.4f} ms")
            else:
                if debug:
                    print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> NO TIMING")

        except subprocess.TimeoutExpired:
            if debug:
                print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> TIMEOUT")
        finally:
            Path(cu_path).unlink(missing_ok=True)
            Path(bin_path).unlink(missing_ok=True)

    if best_config is None:
        return kernel_source, {
            "best_config": {},
            "best_time_ms": None,
            "configs_tested": tested,
            "speedup": 1.0,
            "skipped": "all configs failed",
        }

    best_source = apply_best_config(kernel_source, best_config)
    speedup = default_time / best_time if default_time and best_time > 0 else 1.0

    if debug:
        print(f"[autotuner] Best: {best_config} -> {best_time:.4f} ms"
              + (f" (default: {default_time:.4f} ms, speedup: {speedup:.2f}x)" if default_time else ""))

    return best_source, {
        "best_config": best_config,
        "best_time_ms": round(best_time, 4),
        "default_time_ms": round(default_time, 4) if default_time else None,
        "configs_tested": tested,
        "speedup": round(speedup, 4),
    }


def run_autotune(
    kernel_source: str,
    *,
    framework: str = "auto",
    gpu_id: int = 0,
    timeout: int = 60,
    debug: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run the full autotuning pipeline on a GPU kernel source.

    Supports Triton (tl.constexpr), CUDA (#define macros), and Helion kernels.
    Each config is written to a temp file with a benchmark harness appended,
    then executed as a subprocess.

    Returns (best_source, metadata).
    """
    import os

    params = extract_tunable_params(kernel_source, framework)
    if not params:
        return kernel_source, {
            "best_config": {},
            "best_time_ms": None,
            "configs_tested": 0,
            "speedup": 1.0,
            "skipped": "no tunable parameters found",
        }

    if debug:
        print(f"[autotuner] Found {len(params)} constexpr params: {[p['param'] for p in params]}")

    configs = generate_config_grid(params)
    if not configs:
        return kernel_source, {
            "best_config": {},
            "best_time_ms": None,
            "configs_tested": 0,
            "speedup": 1.0,
            "skipped": "could not generate config grid",
        }

    if debug:
        print(f"[autotuner] Generated {len(configs)} configs to benchmark")

    is_cuda_cu = framework == "cuda" or "#define" in kernel_source and "__global__" in kernel_source

    # For .cu files: compile with nvcc; for Python: subprocess with benchmark harness
    if is_cuda_cu:
        return _run_autotune_cuda(kernel_source, params, configs, gpu_id=gpu_id, timeout=timeout, debug=debug)

    # Python path (Triton / inline CUDA / Helion)
    has_get_inputs = "def get_inputs" in kernel_source
    if has_get_inputs:
        source_for_bench = _strip_main_block(kernel_source) + _BENCH_HARNESS
    else:
        source_for_bench = kernel_source + _BENCH_HARNESS

    best_config = None
    best_time = float("inf")
    default_time = None
    tested = 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for i, cfg in enumerate(configs):
        patched_with_bench = apply_best_config(source_for_bench, cfg)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="autotune_",
        ) as f:
            f.write(patched_with_bench)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            tested += 1

            for line in result.stdout.splitlines():
                if line.startswith("AUTOTUNE_RESULT:"):
                    import json
                    data = json.loads(line[len("AUTOTUNE_RESULT:"):])
                    if "time_ms" in data:
                        ms = data["time_ms"]

                        is_default = all(
                            cfg.get(p["param"]) == p["default"]
                            for p in params
                            if p["default"] is not None and p["param"] in cfg
                        )
                        if is_default and default_time is None:
                            default_time = ms

                        if ms < best_time:
                            best_time = ms
                            best_config = cfg

                        if debug:
                            print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> {ms:.4f} ms")
                    elif debug:
                        print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> FAILED ({data.get('error', '?')})")
                    break
            else:
                if debug:
                    err = result.stderr[:150] if result.returncode != 0 else "no AUTOTUNE_RESULT line"
                    print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> FAILED ({err})")

        except subprocess.TimeoutExpired:
            if debug:
                print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> TIMEOUT")
        except Exception as e:
            if debug:
                print(f"[autotuner] Config {i+1}/{len(configs)}: {cfg} -> ERROR ({e})")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if best_config is None:
        return kernel_source, {
            "best_config": {},
            "best_time_ms": None,
            "configs_tested": tested,
            "speedup": 1.0,
            "skipped": "all configs failed",
        }

    best_source = apply_best_config(kernel_source, best_config)
    speedup = default_time / best_time if default_time and best_time > 0 else 1.0

    if debug:
        print(f"[autotuner] Best: {best_config} -> {best_time:.4f} ms "
              f"(default: {default_time:.4f} ms, speedup: {speedup:.2f}x)" if default_time
              else f"[autotuner] Best: {best_config} -> {best_time:.4f} ms")

    return best_source, {
        "best_config": best_config,
        "best_time_ms": round(best_time, 4),
        "default_time_ms": round(default_time, 4) if default_time else None,
        "configs_tested": tested,
        "speedup": round(speedup, 4),
    }


def _strip_main_block(source: str) -> str:
    """Remove benchmarking / main blocks so only kernel + wrapper remain.

    Strips:
      - if __name__ == '__main__': ... block
      - benchmark.run(...) calls
      - @triton.testing.perf_report decorated functions
      - Module-level test/validation code (torch.testing.assert_close, print, etc.)
    """
    lines = source.splitlines(keepends=True)
    result = []
    skip_block = False
    block_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip if __name__ == '__main__': block
        if stripped.startswith("if __name__") and "__main__" in stripped:
            skip_block = True
            block_indent = len(line) - len(line.lstrip())
            i += 1
            continue

        # Skip @triton.testing.perf_report decorated functions
        if stripped.startswith("@triton.testing.perf_report"):
            skip_block = True
            block_indent = len(line) - len(line.lstrip())
            i += 1
            continue

        # Skip benchmark.run(...) calls
        if "benchmark.run(" in stripped:
            i += 1
            continue

        # Skip module-level test code
        if stripped.startswith("assert ") or stripped.startswith("torch.testing.assert"):
            i += 1
            continue

        # Skip lines starting with # %% (Jupyter cell markers)
        if stripped.startswith("# %%"):
            i += 1
            continue

        # Handle block skipping
        if skip_block:
            if stripped == "" or (len(line) - len(line.lstrip())) > block_indent:
                i += 1
                continue
            elif stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("@"):
                skip_block = False
                # Keep this line (it's a new definition)
            else:
                # Comment or other module-level code after the block
                skip_block = False

        if not skip_block:
            result.append(line)
        i += 1

    return "".join(result)
