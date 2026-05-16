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
    """Build a complete standalone file for the LLM to read, modify, and submit.

    This is what the planner/executor see as ``best_kernel``. It includes the
    FULL computation (not just the Triton kernel in isolation):
      - imports
      - clean @triton.jit kernel(s)
      - kernel_function(*inputs) that does the ENTIRE computation
        (PyTorch ops for GEMM/conv + Triton for fused elementwise)
      - get_inputs() matching the full computation's args
      - __main__ with --profile-only

    The LLM can optimize any part: fuse GEMM+activation into one Triton kernel,
    rewrite library calls as Triton, or just tweak the existing Triton kernel.
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

    call_source = _extract_function_from_source(inductor_source, "call")
    wrapper_code = _translate_call_to_kernel_function(call_source, kernel_parts, inductor_source)
    get_inputs_code = _extract_get_inputs(inductor_source)

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


def _extract_function_from_source(source: str, func_name: str) -> str:
    """Extract a function definition from source by name."""
    lines = source.split("\n")
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(rf"def\s+{func_name}\s*\(", stripped):
            start = i
            break
    if start is None:
        return ""
    func_lines = [lines[start]]
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() == "":
            func_lines.append(line)
            continue
        if line[0] not in (" ", "\t") and line.strip() != "":
            break
        func_lines.append(line)
    while func_lines and func_lines[-1].strip() == "":
        func_lines.pop()
    return "\n".join(func_lines)


def _translate_call_to_kernel_function(
    call_source: str,
    kernel_parts: list[str],
    inductor_source: str,
) -> str:
    """Translate Inductor's call(args) to a clean kernel_function(*inputs).

    Converts:
      - extern_kernels.mm(a, b, out=c) -> c = torch.mm(a, b)
      - extern_kernels.addmm(bias, a, b, out=c) -> c = torch.addmm(bias, a, b)
      - extern_kernels.convolution(...) -> torch.nn.functional.conv2d(...)
      - empty_strided_cuda(shape, stride, dtype) -> torch.empty(shape, dtype=dtype, device='cuda')
      - kernel.run(ptrs, numel, stream=s) -> kernel[grid](ptrs, numel, XBLOCK=1024)
      - reinterpret_tensor(t, shape, stride, offset) -> t.view(shape) or t.as_strided(...)
      - assert_size_stride, del statements -> removed
      - get_raw_stream(0) -> removed (not needed for direct launch)
    """
    if not call_source:
        if kernel_parts:
            return _build_standalone_wrapper(kernel_parts[0])[0]
        return "def kernel_function(*inputs):\n    pass"

    lines = call_source.split("\n")
    out = []
    kernel_names = set()

    for kp in kernel_parts:
        m = re.search(r'def\s+(\w+)\s*\(', kp)
        if m:
            kernel_names.add(m.group(1))

    for line in lines:
        stripped = line.strip()

        if re.match(r'def\s+call\s*\(', stripped):
            out.append("def kernel_function(*inputs):")
            out.append("    args = list(inputs)")
            continue

        if stripped.startswith("args.clear()"):
            continue
        if stripped.startswith("assert_size_stride("):
            continue
        if stripped.startswith("assert_alignment("):
            continue
        if re.match(r'\s*del\s+\w+', stripped):
            continue
        if "torch.cuda._DeviceGuard" in stripped:
            continue
        if "torch.cuda.set_device" in stripped:
            continue
        if "get_raw_stream" in stripped:
            continue

        # Dedent lines that were inside 'with torch.cuda._DeviceGuard' block
        if line.startswith("            "):
            line = "    " + line.lstrip()
        elif line.startswith("        ") and not line.startswith("        #"):
            line = "    " + line.lstrip()

        indent = line[:len(line) - len(line.lstrip())]
        if not indent and stripped and not stripped.startswith("#"):
            indent = "    "

        # extern_kernels.mm(a, b, out=c) -> c = torch.mm(a, b)
        m = re.match(r'\s*extern_kernels\.mm\((.+?),\s*(.+?),\s*out=(\w+)\)', stripped)
        if m:
            a, b, c = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            b = re.sub(r'reinterpret_tensor\((\w+),\s*\([^)]+\),\s*\([^)]+\),\s*\d+\)', r'\1.T', b)
            out.append(f"{indent}{c} = torch.mm({a}, {b})")
            continue

        # extern_kernels.addmm(bias, a, b, out=c) -> c = torch.addmm(bias, a, b)
        m = re.match(r'\s*extern_kernels\.addmm\((.+?),\s*(.+?),\s*(.+?),\s*out=(\w+)\)', stripped)
        if m:
            bias, a, b, c = m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
            b = re.sub(r'reinterpret_tensor\((\w+),\s*\([^)]+\),\s*\([^)]+\),\s*\d+\)', r'\1.T', b)
            bias = re.sub(r'reinterpret_tensor\((\w+),\s*\([^)]+\),\s*\([^)]+\),\s*\d+\)', r'\1', bias)
            out.append(f"{indent}{c} = torch.addmm({bias}, {a}, {b})")
            continue

        # extern_kernels.convolution(...) -> torch.nn.functional.conv2d(...)
        m = re.match(r'\s*(\w+)\s*=\s*extern_kernels\.convolution\((.+)\)', stripped)
        if m:
            var = m.group(1)
            args_str = m.group(2)
            out.append(f"{indent}{var} = torch.ops.aten.convolution({args_str})")
            continue

        # empty_strided_cuda(shape, stride, dtype) -> torch.empty(shape, dtype, device='cuda')
        m = re.search(r'empty_strided_cuda\((\([^)]*\)),\s*\([^)]*\),\s*([^)]+)\)', stripped)
        if m:
            shape, dtype = m.group(1), m.group(2).strip()
            line = re.sub(
                r'empty_strided_cuda\((\([^)]*\)),\s*\([^)]*\),\s*([^)]+)\)',
                f"torch.empty({shape}, dtype={dtype}, device='cuda')",
                line,
            )

        # kernel.run(args, stream=s) -> kernel[grid](args, XBLOCK=1024)
        for kname in kernel_names:
            pattern = rf'{kname}\.run\(([^)]+)\)'
            m = re.search(pattern, stripped)
            if m:
                raw_args = m.group(1)
                args_list = [a.strip() for a in raw_args.split(",")]
                clean_args = [a for a in args_list if not a.startswith("stream=")]
                numel = clean_args[-1] if clean_args else "1"
                ptr_args = clean_args[:-1]
                all_a = ", ".join(ptr_args + [numel])
                line = f"{indent}grid = (triton.cdiv({numel}, 1024),)"
                out.append(line)
                out.append(f"{indent}{kname}[grid]({all_a}, XBLOCK=1024)")
                break
        else:
            # buf1 = buf0; del buf0  -- handle buffer reuse
            m = re.match(r'\s*(\w+)\s*=\s*(\w+)\s*;\s*del\s+\w+\s*#.*', stripped)
            if m:
                out.append(f"{indent}{m.group(1)} = {m.group(2)}")
                continue

            # reinterpret_tensor in non-extern context
            line = re.sub(
                r'reinterpret_tensor\((\w+),\s*(\([^)]+\)),\s*\([^)]+\),\s*\d+\)',
                r'\1.reshape(\2)',
                line,
            )

            out.append(line)
            continue

        # If we processed a kernel.run, skip the original line
        continue

    # Fix return to not be a tuple
    result = "\n".join(out)
    result = re.sub(r'return\s*\(([^,]+),?\s*\)', r'return \1', result)

    return result


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
