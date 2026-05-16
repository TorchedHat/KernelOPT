"""Wrap Inductor output_code.py into a kernelopt-compatible file.

Design principle: keep the Inductor output INTACT as the execution engine.
It already has correct kernel launching, grid computation, constexprs,
multi-kernel orchestration, and extern_kernels support.  We only add
the kernelopt interface on top:

  - kernel_function(*inputs) wrapper
  - get_inputs() from benchmark_compiled_module
  - __main__ with --profile-only
  - _PYTORCH_REFERENCE embedding
  - _TRITON_KERNEL_SOURCES list (extracted @triton.jit text for LLM context)
"""

from __future__ import annotations

import re
from typing import Any


def wrap_inductor_output(
    inductor_source: str,
    original_pytorch_source: str,
    triton_kernel_sources: list[str],
) -> str:
    """Produce a standalone kernelopt-compatible .py from Inductor output.

    Instead of disassembling and rebuilding, we keep the full Inductor
    output as-is (it already works) and append the kernelopt interface.

    Args:
        inductor_source: Full Inductor output_code.py content.
        original_pytorch_source: The user's original PyTorch code.
        triton_kernel_sources: Extracted @triton.jit kernel source blocks
            (for LLM reading context, not for execution).

    Returns:
        Complete standalone Python source string.
    """
    escaped_ref = original_pytorch_source.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

    kernel_src_list = []
    for ks in triton_kernel_sources:
        cleaned = _clean_for_llm(ks)
        escaped_ks = cleaned.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
        kernel_src_list.append(f"    '''{escaped_ks}'''")
    kernel_src_joined = ",\n".join(kernel_src_list) if kernel_src_list else ""

    get_inputs_code = _extract_get_inputs(inductor_source)

    interface = f"""

# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''{escaped_ref}'''

_TRITON_KERNEL_SOURCES = [
{kernel_src_joined}
]

def kernel_function(*inputs):
    \"\"\"kernelopt entry point -- delegates to Inductor's call().\"\"\"
    return call(list(inputs))

{get_inputs_code}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    _args = parser.parse_args()
    _inputs = get_inputs()
    if _args.profile_only:
        kernel_function(*_inputs)
        torch.cuda.synchronize()
    else:
        kernel_function(*_inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
"""

    base = inductor_source.rstrip()

    if "if __name__" in base:
        base = re.sub(
            r'\nif __name__\s*==\s*["\']__main__["\'].*',
            '',
            base,
            flags=re.DOTALL,
        )

    return base.rstrip() + "\n" + interface


def _clean_for_llm(raw_kernel: str) -> str:
    """Strip Inductor-specific decorators from a kernel block for LLM reading.

    Keeps the @triton.jit function intact but removes triton_heuristics
    decorators and inductor-specific imports that would confuse the LLM.
    """
    lines = raw_kernel.strip().split("\n")
    cleaned = []
    skip = True

    for line in lines:
        stripped = line.strip()
        if skip:
            if stripped.startswith("@triton.jit"):
                skip = False
                cleaned.append("@triton.jit")
                continue
            continue
        cleaned.append(line)

    body = "\n".join(cleaned)

    replacements = [
        (r'triton_helpers\.promote_to_tensor\(([^)]+)\)', r'\1'),
        (r'triton_helpers\.maximum\(([^,]+),\s*([^)]+)\)', r'tl.maximum(\1, \2)'),
        (r'triton_helpers\.minimum\(([^,]+),\s*([^)]+)\)', r'tl.minimum(\1, \2)'),
        (r'libdevice\.sqrt\(([^)]+)\)', r'tl.sqrt(\1)'),
        (r'libdevice\.exp\(([^)]+)\)', r'tl.exp(\1)'),
        (r'libdevice\.log\(([^)]+)\)', r'tl.log(\1)'),
        (r'tl_math\.exp\(([^)]+)\)', r'tl.exp(\1)'),
        (r'tl_math\.log\(([^)]+)\)', r'tl.log(\1)'),
    ]
    for pat, repl in replacements:
        body = re.sub(pat, repl, body)

    return body


def build_standalone_kernel(
    triton_kernel_sources: list[str],
    inductor_source: str,
) -> str:
    """Build a complete standalone Triton file for the LLM to read and modify.

    This is what the planner/executor see as ``best_kernel``. It has:
      - proper imports
      - clean @triton.jit kernel(s)
      - kernel_function() wrapper
      - get_inputs()
      - __main__ with --profile-only

    Unlike the Inductor file (which uses async_compile), this is plain Triton
    that the LLM can read, modify, and submit as a complete file.
    """
    imports = [
        "import torch",
        "import triton",
        "import triton.language as tl",
        "from triton.language.extra import libdevice",
    ]

    kernel_parts = []
    for ks in triton_kernel_sources:
        kernel_parts.append(_clean_for_llm(ks))

    get_inputs_code = _extract_get_inputs(inductor_source)

    if kernel_parts:
        first_kernel = kernel_parts[0]
        wrapper_code, kernel_get_inputs = _build_standalone_wrapper(first_kernel)
        if kernel_get_inputs:
            get_inputs_code = kernel_get_inputs
    else:
        wrapper_code = "def kernel_function(*args):\n    pass"

    main_block = [
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
    ]

    parts = imports + [""] + kernel_parts + ["", wrapper_code, "", get_inputs_code, ""] + main_block + [""]
    return "\n".join(parts)


def _build_standalone_wrapper(kernel_source: str) -> tuple[str, str | None]:
    """Build kernel_function() wrapper and matching get_inputs() for a @triton.jit kernel.

    Analyses the kernel signature to determine:
      - Pointer params (in_ptr*, out_ptr*, in_out_ptr*) -> tensor arguments
      - Scalar params (xnumel, rnumel, etc.) -> integer arguments
      - Constexpr params (XBLOCK, RBLOCK, etc.) -> compile-time constants

    For in-place kernels (in_out_ptr*), returns the modified tensor.
    For output kernels (out_ptr*), allocates output tensor and returns it.

    Returns (wrapper_code, get_inputs_code_or_None).
    """
    fn_match = re.search(r'def\s+(\w+)\s*\(([^)]+)\)', kernel_source)
    if not fn_match:
        return "def kernel_function(*args):\n    pass", None

    kernel_name = fn_match.group(1)
    params_raw = fn_match.group(2)
    params = [p.strip().split(":")[0].strip() for p in params_raw.split(",")]

    ptr_params = []
    scalar_params = []
    constexpr_params = []

    for p in params:
        if p[0].isupper():
            constexpr_params.append(p)
        elif "ptr" in p or p.startswith("in_") or p.startswith("out_"):
            ptr_params.append(p)
        elif "numel" in p.lower() or p in ("n_elements", "n"):
            scalar_params.append(p)
        elif p[0].islower():
            scalar_params.append(p)
        else:
            scalar_params.append(p)

    in_ptrs = [p for p in ptr_params if p.startswith("in_ptr")]
    out_ptrs = [p for p in ptr_params if p.startswith("out_ptr")]
    inout_ptrs = [p for p in ptr_params if p.startswith("in_out_ptr")]
    all_ptrs = in_ptrs + out_ptrs + inout_ptrs

    # Build wrapper
    lines = ["def kernel_function(*args):", '    """Launch the Triton kernel."""']

    # Unpack pointer args from input
    input_idx = 0
    for p in inout_ptrs + in_ptrs:
        lines.append(f"    {p} = args[{input_idx}]")
        input_idx += 1

    # For output-only pointers, allocate based on the first input's shape
    if out_ptrs and (inout_ptrs or in_ptrs):
        ref = (inout_ptrs + in_ptrs)[0]
        for p in out_ptrs:
            lines.append(f"    {p} = torch.empty_like({ref})")

    # Scalar args (numel etc.) -- derive from first tensor
    for p in scalar_params:
        if "numel" in p.lower():
            ref = (inout_ptrs + in_ptrs + out_ptrs)[0] if ptr_params else "args[0]"
            lines.append(f"    {p} = {ref}.numel()")
        else:
            lines.append(f"    {p} = args[{input_idx}]")
            input_idx += 1

    # Grid and launch
    grid_numel = scalar_params[0] if scalar_params else f"({(inout_ptrs + in_ptrs)[0]}.numel() if {(inout_ptrs + in_ptrs)} else 1)"
    if not scalar_params and (inout_ptrs or in_ptrs):
        grid_numel = f"{(inout_ptrs + in_ptrs)[0]}.numel()"

    constexpr_defaults = {p: 1024 for p in constexpr_params}
    constexpr_kwargs = ", ".join(f"{p}={v}" for p, v in constexpr_defaults.items())
    all_args = ", ".join(ptr_params + scalar_params)

    lines.append(f"    grid = (triton.cdiv({grid_numel}, {constexpr_defaults.get(constexpr_params[0], 1024) if constexpr_params else 1024}),)")
    lines.append(f"    {kernel_name}[grid]({all_args}, {constexpr_kwargs})")

    # Return statement
    if inout_ptrs:
        lines.append(f"    return {inout_ptrs[0]}")
    elif out_ptrs:
        lines.append(f"    return {out_ptrs[0]}")

    wrapper_code = "\n".join(lines)

    # Build matching get_inputs() based on the kernel's actual params
    gi_lines = ["def get_inputs():"]
    shape = "(1024, 8192)"
    for p in inout_ptrs + in_ptrs:
        gi_lines.append(f"    {p} = torch.randn({shape}, device='cuda', dtype=torch.float32)")
    gi_lines.append(f"    return [{', '.join(inout_ptrs + in_ptrs)}]")
    get_inputs_code = "\n".join(gi_lines)

    return wrapper_code, get_inputs_code


def _extract_get_inputs(inductor_source: str) -> str:
    """Build get_inputs() from Inductor's benchmark_compiled_module().

    Parses rand_strided(...) calls and produces torch.randn/randint equivalents,
    handling non-float dtypes correctly.
    """
    pattern = re.compile(
        r"(\w+)\s*=\s*rand_strided\(\s*"
        r"\(([^)]*)\)\s*,\s*"
        r"\(([^)]*)\)\s*,\s*"
        r"device='([^']*)'\s*,\s*"
        r"dtype=([^\s,)]+)"
    )
    matches = pattern.findall(inductor_source)

    if not matches:
        return "def get_inputs():\n    return []"

    input_lines = []
    var_names = []
    for var_name, shape, _stride, device, dtype in matches:
        shape_clean = shape.strip().rstrip(",").strip()
        if not shape_clean:
            shape_str = "()"
        elif "," not in shape_clean:
            shape_str = f"({shape_clean},)"
        else:
            shape_str = f"({shape_clean})"

        dtype_clean = dtype.strip()

        if shape_str == "()":
            if any(t in dtype_clean for t in ["int", "uint", "bool"]):
                input_lines.append(
                    f"    {var_name} = torch.tensor(1, dtype={dtype_clean}, device='{device}')"
                )
            else:
                input_lines.append(
                    f"    {var_name} = torch.tensor(0.1, dtype={dtype_clean}, device='{device}')"
                )
        elif any(t in dtype_clean for t in ["int64", "int32", "int16", "int8", "uint8"]):
            input_lines.append(
                f"    {var_name} = torch.randint(0, 10, {shape_str}, dtype={dtype_clean}, device='{device}')"
            )
        elif "bool" in dtype_clean:
            input_lines.append(
                f"    {var_name} = torch.randint(0, 2, {shape_str}, device='{device}').bool()"
            )
        elif "float8" in dtype_clean:
            input_lines.append(
                f"    {var_name} = torch.randn({shape_str}, device='{device}').to({dtype_clean})"
            )
        else:
            input_lines.append(
                f"    {var_name} = torch.randn({shape_str}, device='{device}', dtype={dtype_clean})"
            )
        var_names.append(var_name)

    return_list = ", ".join(var_names)
    body = "\n".join(input_lines)
    return f"def get_inputs():\n{body}\n    return [{return_list}]"
