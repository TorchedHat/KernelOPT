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
