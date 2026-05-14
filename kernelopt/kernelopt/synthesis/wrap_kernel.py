"""Wrap Inductor-generated Triton code into a standalone kernelopt-compatible file.

Takes the raw output from ``compile_extract`` and produces a ``.py`` that:
  - has a ``@triton.jit`` kernel (cleaned of Inductor runtime deps)
  - has a ``kernel_function(*args)`` wrapper
  - has ``get_inputs()`` for the profiler
  - has ``__main__`` with ``--profile-only`` support
  - embeds the original PyTorch source for correctness reference
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def wrap_inductor_output(
    triton_kernel_source: str,
    call_source: str,
    benchmark_source: str,
    original_pytorch_source: str,
    inductor_source: str,
) -> str:
    """Produce a standalone .py from Inductor output.

    Args:
        triton_kernel_source: Raw kernel block from async_compile.triton(...)
        call_source: The ``call(args)`` function
        benchmark_source: The ``benchmark_compiled_module()`` function
        original_pytorch_source: The user's original PyTorch code (embedded for correctness)
        inductor_source: Full Inductor output (for fallback parsing)

    Returns:
        Complete standalone Python source string.
    """
    kernel_name, kernel_body, kernel_params = _clean_triton_kernel(triton_kernel_source)
    constexpr_params = _extract_constexpr_params(kernel_body)
    launch_code = _build_launch_wrapper(call_source, kernel_name, inductor_source, constexpr_params)
    get_inputs_code = _build_get_inputs(benchmark_source)

    escaped_ref = original_pytorch_source.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

    needs_extern = "extern_kernels" in launch_code
    needs_reinterpret = "reinterpret_tensor" in launch_code
    needs_triton_helpers = "triton_helpers" in kernel_body

    extra_imports = []
    if needs_triton_helpers:
        extra_imports.append("from torch._inductor.runtime import triton_helpers")
        extra_imports.append("from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math")
    if needs_extern:
        extra_imports.append("from torch._inductor.select_algorithm import extern_kernels")
    if needs_reinterpret:
        extra_imports.append("reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor")

    parts = [
        "import torch",
        "import triton",
        "import triton.language as tl",
    ]
    if extra_imports:
        parts.append("")
        parts.extend(extra_imports)
    parts.extend([
        "",
        f"_PYTORCH_REFERENCE = '''{escaped_ref}'''",
        "",
        kernel_body,
        "",
        launch_code,
        "",
        get_inputs_code,
        "",
        'if __name__ == "__main__":',
        "    import argparse",
        "    parser = argparse.ArgumentParser()",
        '    parser.add_argument("--profile-only", action="store_true")',
        "    args = parser.parse_args()",
        "    inputs = get_inputs()",
        "    if args.profile_only:",
        "        kernel_function(*inputs)",
        "        torch.cuda.synchronize()",
        "    else:",
        "        kernel_function(*inputs)",
        "        torch.cuda.synchronize()",
        '        print("Kernel executed successfully.")',
        "",
    ])
    return "\n".join(parts)


def _clean_triton_kernel(raw_kernel: str) -> tuple[str, str, list[str]]:
    """Strip Inductor-specific imports/decorators from a kernel block.

    Returns (kernel_name, cleaned_source, param_names).
    """
    lines = raw_kernel.strip().split("\n")
    cleaned = []
    skip_until_jit = True
    kernel_name = "triton_kernel_0"
    params: list[str] = []

    for line in lines:
        stripped = line.strip()

        if skip_until_jit:
            if stripped.startswith("@triton.jit"):
                skip_until_jit = False
                cleaned.append("@triton.jit")
                continue
            continue

        m = re.match(r"^def\s+(\w+)\s*\((.+?)(?:\)|,\s*$)", stripped)
        if m and not params:
            kernel_name = m.group(1)

        cleaned.append(line)

    body = "\n".join(cleaned)

    fn_match = re.search(r"def\s+(\w+)\s*\(([^)]+)\)", body)
    if fn_match:
        kernel_name = fn_match.group(1)
        params = [p.strip().split(":")[0].strip() for p in fn_match.group(2).split(",")]

    body = _replace_triton_helpers(body)

    return kernel_name, body, params


def _replace_triton_helpers(source: str) -> str:
    """Replace ``triton_helpers.*`` calls with standard Triton equivalents."""
    replacements = [
        (r'triton_helpers\.promote_to_tensor\(([^)]+)\)', r'\1'),
        (r'triton_helpers\.maximum\(([^,]+),\s*([^)]+)\)', r'tl.maximum(\1, \2)'),
        (r'triton_helpers\.minimum\(([^,]+),\s*([^)]+)\)', r'tl.minimum(\1, \2)'),
        (r'libdevice\.sqrt\(([^)]+)\)', r'tl.sqrt(\1)'),
        (r'libdevice\.exp\(([^)]+)\)', r'tl.exp(\1)'),
        (r'libdevice\.log\(([^)]+)\)', r'tl.log(\1)'),
        (r'libdevice\.abs\(([^)]+)\)', r'tl.abs(\1)'),
        (r'tl_math\.exp\(([^)]+)\)', r'tl.exp(\1)'),
        (r'tl_math\.log\(([^)]+)\)', r'tl.log(\1)'),
        (r'tl_math\.abs\(([^)]+)\)', r'tl.abs(\1)'),
    ]
    for pat, repl in replacements:
        source = re.sub(pat, repl, source)
    return source


def _replace_empty_strided(line: str) -> str:
    """Replace ``empty_strided_cuda(shape, stride, dtype)`` with torch.empty."""
    m = re.search(r'empty_strided_cuda\((\([^)]*\))\s*,\s*\([^)]*\)\s*,\s*([^)]+)\)', line)
    if m:
        shape = m.group(1)
        dtype = m.group(2).strip()
        line = line[:m.start()] + f"torch.empty({shape}, dtype={dtype}, device='cuda')" + line[m.end():]
    return line


def _build_launch_wrapper(
    call_source: str, kernel_name: str, inductor_source: str,
    constexpr_params: list[str] | None = None,
) -> str:
    """Build ``kernel_function(*args)`` from the Inductor ``call(args)`` function.

    The wrapper:
      - replaces ``empty_strided_cuda(...)`` with ``torch.empty_strided(..., device='cuda')``
      - replaces ``get_raw_stream(0)`` with ``torch.cuda.current_stream().cuda_stream``
      - strips ``assert_size_stride(...)`` calls
      - strips ``del`` statements
    """
    if not call_source:
        return _fallback_launch_wrapper(inductor_source, kernel_name)

    lines = call_source.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("def call("):
            cleaned.append("def kernel_function(*inputs):")
            cleaned.append("    args = list(inputs)")
            continue

        if re.match(r'\s*\w+(?:,\s*\w+)*\s*=\s*args\s*$', stripped):
            cleaned.append(f"    {stripped}")
            continue

        if stripped.startswith("args.clear()"):
            continue
        if stripped.startswith("assert_size_stride("):
            continue
        if stripped.startswith("assert_alignment("):
            continue
        if re.match(r'\s*del\s+\w+', stripped):
            continue

        line = _replace_empty_strided(line)

        line = re.sub(
            r'get_raw_stream\(\d+\)',
            'torch.cuda.current_stream().cuda_stream',
            line,
        )

        cleaned.append(line)

    result = "\n".join(cleaned)

    result = _convert_run_to_launch(result, kernel_name, constexpr_params)

    result = re.sub(
        r'return\s*\(([^,]+),?\s*\)',
        r'return \1',
        result,
    )

    return result


def _extract_constexpr_params(kernel_body: str) -> list[str]:
    """Extract tl.constexpr parameter names from a @triton.jit function SIGNATURE only.

    Only looks at the ``def func_name(...)`` line, not the body, to avoid
    picking up local aliases like ``RBLOCK: tl.constexpr = R0_BLOCK``.
    """
    sig_match = re.search(r'def\s+\w+\s*\(([^)]+)\)', kernel_body)
    if not sig_match:
        return []
    sig = sig_match.group(1)
    return re.findall(r'(\w+)\s*:\s*tl\.constexpr', sig)


_CONSTEXPR_DEFAULTS = {
    "XBLOCK": 128,
    "RBLOCK": 2048,
    "R0_BLOCK": 2048,
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": True,
    "USE_FAST_ACCUM": True,
    "ACC_TYPE": "tl.float32",
}


def _convert_run_to_launch(
    source: str, kernel_name: str, constexpr_params: list[str] | None = None
) -> str:
    """Convert ``kernel.run(args..., stream=s)`` to ``kernel[grid](args...)``."""
    if constexpr_params is None:
        constexpr_params = []

    run_pattern = re.compile(
        rf'\s*{re.escape(kernel_name)}\.run\(([^)]+)\)'
    )

    lines = source.split("\n")
    new_lines = []
    for line in lines:
        m = run_pattern.search(line)
        if m:
            indent = line[:len(line) - len(line.lstrip())]
            raw_args = m.group(1)
            args_list = [a.strip() for a in raw_args.split(",")]

            kernel_args = []
            for a in args_list:
                if a.startswith("stream=") or a.startswith("grid="):
                    continue
                kernel_args.append(a)

            ptr_args = []
            scalar_args = []
            for a in kernel_args:
                if a.startswith("buf") or a.startswith("arg") or a.startswith("in_") or a.startswith("out_"):
                    ptr_args.append(a)
                else:
                    scalar_args.append(a)

            numel_arg = scalar_args[0] if scalar_args else "1"

            constexpr_assigns = []
            constexpr_kwargs = []
            for p in constexpr_params:
                default = _CONSTEXPR_DEFAULTS.get(p, 128)
                if isinstance(default, bool):
                    constexpr_assigns.append(f"{indent}{p} = {default}")
                elif isinstance(default, str):
                    constexpr_assigns.append(f"{indent}{p} = {default}")
                else:
                    constexpr_assigns.append(f"{indent}{p} = {default}")
                constexpr_kwargs.append(f"{p}={p}")

            if not constexpr_params:
                constexpr_assigns.append(f"{indent}XBLOCK = 1024")
                constexpr_kwargs.append("XBLOCK=XBLOCK")

            grid_arg = numel_arg
            if "XBLOCK" in constexpr_params or not constexpr_params:
                xblock_name = "XBLOCK"
            elif "BLOCK_M" in constexpr_params:
                xblock_name = "BLOCK_M"
            else:
                xblock_name = constexpr_params[0] if constexpr_params else "XBLOCK"

            for assign in constexpr_assigns:
                new_lines.append(assign)

            new_lines.append(
                f"{indent}grid = (triton.cdiv({grid_arg}, {xblock_name}),)"
            )

            all_args = ", ".join(ptr_args + scalar_args + constexpr_kwargs)
            new_lines.append(
                f"{indent}{kernel_name}[grid]({all_args})"
            )
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def _fallback_launch_wrapper(inductor_source: str, kernel_name: str) -> str:
    """Minimal wrapper when call() couldn't be parsed."""
    return textwrap.dedent(f"""\
        def kernel_function(*inputs):
            # Fallback: run the kernel via the Inductor call() convention
            args = list(inputs)
            {kernel_name}.run(*args, stream=torch.cuda.current_stream().cuda_stream)
    """)


def _build_get_inputs(benchmark_source: str) -> str:
    """Extract input generation from ``benchmark_compiled_module()``.

    Parses ``rand_strided(shape, stride, device=..., dtype=...)`` calls
    and produces a standalone ``get_inputs()`` function.
    """
    if not benchmark_source:
        return textwrap.dedent("""\
            def get_inputs():
                return []
        """)

    pattern = re.compile(
        r"(\w+)\s*=\s*rand_strided\(\s*"
        r"\(([^)]*)\)\s*,\s*"
        r"\(([^)]*)\)\s*,\s*"
        r"device='([^']*)'\s*,\s*"
        r"dtype=([^\s,)]+)"
    )
    matches = pattern.findall(benchmark_source)

    if not matches:
        return textwrap.dedent("""\
            def get_inputs():
                return []
        """)

    input_lines = []
    var_names = []
    for var_name, shape, _stride, device, dtype in matches:
        shape_clean = shape.strip().rstrip(",").strip()
        if "," not in shape_clean:
            shape_str = f"({shape_clean},)"
        else:
            shape_str = f"({shape_clean})"
        input_lines.append(
            f"    {var_name} = torch.randn({shape_str}, device='{device}', dtype={dtype})"
        )
        var_names.append(var_name)

    return_list = ", ".join(var_names)
    body = "\n".join(input_lines)

    return f"def get_inputs():\n{body}\n    return [{return_list}]"
