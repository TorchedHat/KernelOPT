"""Wrap a Triton kernel + SOL-ExecBench spec into a standalone runnable .py.

Generates a self-contained script that kernelopt can profile with NCU.
Hardcodes input shapes from a resolved workload so the script is runnable
without the JSON spec at runtime.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from benchmarks.sol_execbench.harness import (
    load_problem,
    resolve_axes,
    SolProblem,
    DTYPE_MAP,
    SAFETENSORS_ROOT,
)


def _torch_dtype_str(dtype_name: str) -> str:
    """Convert dtype name to torch.dtype string for codegen."""
    mapping = {
        "float16": "torch.float16",
        "bfloat16": "torch.bfloat16",
        "float32": "torch.float32",
        "float64": "torch.float64",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "float8_e4m3fn": "torch.float8_e4m3fn",
        "float8_e5m2": "torch.float8_e5m2",
    }
    return mapping.get(dtype_name, "torch.float32")


def _gen_input_code(
    input_name: str,
    input_spec: dict,
    wl_input: dict,
    axes: dict[str, int],
) -> str:
    """Generate Python code to create one input tensor."""
    dtype_str = input_spec.get("dtype", "float32")
    shape_spec = input_spec.get("shape")
    input_type = wl_input.get("type", "random")

    if input_type == "scalar":
        return f"    {input_name} = {wl_input['value']}"

    if input_type == "safetensors":
        st_path = wl_input.get("path", "")
        tensor_key = wl_input.get("tensor_key", "")
        local = str(SAFETENSORS_ROOT / st_path)
        return (
            f"    from safetensors.torch import load_file as _load_sf\n"
            f"    {input_name} = _load_sf('{local}')['{tensor_key}'].to(device)"
        )

    if shape_spec is None:
        return f"    {input_name} = torch.tensor(0.0, device=device)"

    concrete = tuple(axes[dim] for dim in shape_spec)
    torch_dt = _torch_dtype_str(dtype_str)
    if "int" in dtype_str:
        return f"    {input_name} = torch.randint(0, 100, {concrete}, dtype={torch_dt}, device=device)"
    return f"    {input_name} = torch.randn({concrete}, dtype={torch_dt}, device=device)"


def wrap_kernel(
    kernel_source: str,
    problem: SolProblem,
    workload_idx: int,
    output_path: str,
) -> str:
    """Generate a standalone .py wrapping a Triton kernel for NCU profiling.

    Returns the path to the generated file.
    """
    axes = resolve_axes(problem, workload_idx)
    wl = problem.workloads[workload_idx]
    wl_inputs = wl.get("inputs", {})

    input_names = list(problem.inputs_spec.keys())
    input_lines = []
    for name in input_names:
        spec = problem.inputs_spec[name]
        wl_inp = wl_inputs.get(name, {"type": "random"})
        input_lines.append(_gen_input_code(name, spec, wl_inp, axes))

    inputs_block = "\n".join(input_lines)
    args_str = ", ".join(input_names)

    wrapper = textwrap.dedent(f'''\
        """Auto-generated wrapper for SOL-ExecBench: {problem.name} workload {workload_idx}"""
        import argparse
        import torch

        {kernel_source}

        def _allocate_inputs(device="cuda"):
        {inputs_block}
            return [{args_str}]

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--profile-only", action="store_true")
            parser.add_argument("--workload-idx", type=int, default={workload_idx})
            args = parser.parse_args()

            inputs = _allocate_inputs()
            if args.profile_only:
                run(*inputs)
                torch.cuda.synchronize()
                print("Profile-only run complete.")
            else:
                import time
                for _ in range(5):
                    run(*inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    run(*inputs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                print(f"Avg: {{(t1-t0)/20*1000:.3f}} ms")
    ''')

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(wrapper)
    return str(out)
