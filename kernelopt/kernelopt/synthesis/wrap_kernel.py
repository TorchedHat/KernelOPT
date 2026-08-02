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

    # Remove only the LAST top-level if __name__ block.
    # With TORCHINDUCTOR_BENCHMARK_KERNEL=1, async_compile.triton('''...''')
    # blocks may contain if __name__ inside triple-quoted strings — we must
    # not match those. Find the last occurrence at column 0 only.
    _main_matches = list(re.finditer(r'^if __name__\s*==\s*["\']__main__["\']', base, re.MULTILINE))
    if _main_matches:
        base = base[:_main_matches[-1].start()]

    # Guard Inductor boilerplate imports that may not exist in all PyTorch builds.
    # Source: torch/_inductor/codegen/wrapper.py lines 1406-1433 always emits these
    # assignments, but some symbols (cpu_pinned, xpu, mtia, p2p) may not exist
    # in every PyTorch build/version. Wrapping in try/except matches how PyTorch
    # itself guards empty_strided_p2p (wrapper.py line 1421-1433).
    _GUARD_PATTERNS = [
        # From torch._C._dynamo.guards (may be absent in older/partial builds)
        "empty_strided_cpu_pinned = ",
        "empty_strided_xpu = ",
        "empty_strided_mtia = ",
        # From torch._C._distributed_c10d._SymmetricMemory (requires distributed)
        "empty_strided_p2p = ",
    ]
    lines = base.split("\n")
    guarded = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(pat) for pat in _GUARD_PATTERNS):
            guarded.append(f"try:\n    {line}\nexcept (AttributeError, ImportError):\n    pass")
        else:
            guarded.append(line)
    base = "\n".join(guarded)

    return base.rstrip() + "\n" + interface


def _replace_with_balanced_parens(body: str, prefix: str, replacement_fn) -> str:
    """Replace ``prefix(...)`` occurrences using balanced-paren matching.

    Unlike a simple regex with ``[^)]+``, this correctly handles nested
    parentheses inside the argument list (e.g.
    ``triton_helpers.promote_to_tensor(foo(bar))``).

    Args:
        body: The source text to transform.
        prefix: The function-call prefix to match, e.g.
            ``"triton_helpers.promote_to_tensor"``.
        replacement_fn: Called with the inner text between the balanced
            parens; its return value replaces the entire ``prefix(...)``
            occurrence.

    Returns:
        The transformed source text.
    """
    result: list[str] = []
    search_token = prefix + "("
    i = 0
    while i < len(body):
        idx = body.find(search_token, i)
        if idx == -1:
            result.append(body[i:])
            break
        result.append(body[i:idx])
        # Start counting parens right after the prefix
        paren_start = idx + len(search_token)
        depth = 1
        j = paren_start
        while j < len(body) and depth > 0:
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
            j += 1
        inner = body[paren_start:j - 1]
        result.append(replacement_fn(inner))
        i = j
    return "".join(result)


def _clean_for_llm(raw_kernel: str) -> str:
    """Strip Inductor-specific decorators from a kernel block for LLM reading.

    Keeps the @triton.jit function intact but removes triton_heuristics
    decorators and inductor-specific imports that would confuse the LLM.
    """
    lines = raw_kernel.strip().split("\n")
    cleaned = []
    skip = True
    in_body = False

    for line in lines:
        stripped = line.strip()
        if skip:
            if stripped.startswith("@triton.jit"):
                skip = False
                cleaned.append("@triton.jit")
                continue
            continue
        # Stop at the next top-level definition after the kernel body
        # (benchmark harness: def get_args, def call, etc.)
        if in_body and stripped and not stripped.startswith("#") and not line[0:1].isspace():
            break
        if stripped.startswith("def "):
            in_body = True
        cleaned.append(line)

    body = "\n".join(cleaned)

    body = _replace_with_balanced_parens(body, "triton_helpers.promote_to_tensor", lambda inner: inner)
    body = _replace_with_balanced_parens(body, "triton_helpers.maximum", lambda inner: f"tl.maximum({inner})")
    body = _replace_with_balanced_parens(body, "triton_helpers.minimum", lambda inner: f"tl.minimum({inner})")
    body = _replace_with_balanced_parens(body, "libdevice.sqrt", lambda inner: f"tl.sqrt({inner})")
    body = _replace_with_balanced_parens(body, "libdevice.exp", lambda inner: f"tl.exp({inner})")
    body = _replace_with_balanced_parens(body, "libdevice.log", lambda inner: f"tl.log({inner})")
    body = _replace_with_balanced_parens(body, "tl_math.exp", lambda inner: f"tl.exp({inner})")
    body = _replace_with_balanced_parens(body, "tl_math.log", lambda inner: f"tl.log({inner})")

    return body


def _needs_strided_alloc(shape_str: str, stride_str: str) -> bool:
    """Check if Inductor's rand_strided strides differ from contiguous layout."""
    try:
        raw_shape = shape_str.strip().strip("()")
        raw_stride = stride_str.strip().strip("()")
        if not raw_shape or not raw_stride:
            return False
        dims = [int(x.strip()) for x in raw_shape.split(",") if x.strip()]
        strides = [int(x.strip()) for x in raw_stride.split(",") if x.strip()]
        if len(dims) != len(strides) or len(dims) == 0:
            return False
        # Compute expected contiguous strides
        expected = [1]
        for s in reversed(dims[1:]):
            expected.insert(0, expected[0] * s)
        return strides != expected
    except (ValueError, IndexError):
        return False


def _has_zero_stride(stride_str: str) -> bool:
    """Check if any stride dimension is 0 (broadcast / aliased memory)."""
    try:
        raw = stride_str.strip().strip("()")
        if not raw:
            return False
        strides = [int(x.strip()) for x in raw.split(",") if x.strip()]
        return 0 in strides
    except (ValueError, IndexError):
        return False


def _gen_expand_alloc(
    var_name: str, shape_str: str, stride_str: str,
    device: str, dtype_clean: str, fill: str,
) -> str:
    """Generate expand()-based allocation for stride-0 tensors.

    Creates a base tensor with size 1 in broadcast dimensions,
    fills it with data, then expands to the full shape preserving
    stride-0 semantics.
    """
    raw_shape = shape_str.strip().strip("()")
    raw_stride = stride_str.strip().strip("()")
    dims = [int(x.strip()) for x in raw_shape.split(",") if x.strip()]
    strides = [int(x.strip()) for x in raw_stride.split(",") if x.strip()]

    # Build base shape: 1 in broadcast dims, original in others
    base_dims = [1 if s == 0 else d for d, s in zip(dims, strides)]
    base_shape = f"({', '.join(str(d) for d in base_dims)}{',' if len(base_dims) == 1 else ''})"
    full_shape = f"({', '.join(str(d) for d in dims)}{',' if len(dims) == 1 else ''})"

    base_var = f"_{var_name}_base"

    if fill == "randint":
        return (
            f"    {base_var} = torch.randint(0, 10, {base_shape}, device='{device}', dtype={dtype_clean})\n"
            f"    {var_name} = {base_var}.expand({full_shape})"
        )
    elif fill == "bool":
        return (
            f"    {base_var} = torch.randint(0, 2, {base_shape}, device='{device}').bool()\n"
            f"    {var_name} = {base_var}.expand({full_shape})"
        )
    elif fill == "float8":
        return (
            f"    {base_var} = torch.randn({base_shape}, device='{device}').to({dtype_clean})\n"
            f"    {var_name} = {base_var}.expand({full_shape})"
        )
    else:  # normal float
        return (
            f"    {base_var} = torch.randn({base_shape}, device='{device}', dtype={dtype_clean})\n"
            f"    {var_name} = {base_var}.expand({full_shape})"
        )


def build_benchmark_standalone(
    benchmark_source: str,
    original_pytorch_source: str,
) -> str:
    """Build a clean standalone kernel from a TORCHINDUCTOR_BENCHMARK_KERNEL file.

    Extracts metadata from the benchmark file (grid_type, numels, constexprs,
    input shapes) and produces a clean ~80-line file the LLM can understand:

      - ``@triton.jit`` kernel (cleaned via ``_clean_for_llm()``)
      - ``kernel_function(*inputs)`` with correct ``kernel[grid](...)`` launch
      - ``get_inputs()`` returning properly shaped tensors
      - ``__main__`` with ``--profile-only``

    The grid computation is derived from ``inductor_meta['grid_type']`` and
    the numel values in ``get_args()`` — correct by construction.
    """
    # --- Extract metadata from the benchmark file ---

    # 1. Grid type: Grid1D, Grid2D, or Grid3D
    grid_match = re.search(r"""['"']grid_type['"']\s*:\s*['"']?(Grid\w+)['"']?""", benchmark_source)
    grid_type = grid_match.group(1) if grid_match else "Grid1D"

    # 2. Clean @triton.jit kernel body
    clean_kernel = _clean_for_llm(benchmark_source)
    if not clean_kernel.strip():
        clean_kernel = "@triton.jit\ndef kernel(*args): pass"

    # 3. Extract kernel name and signature
    sig_match = re.search(r'def\s+(\w+)\s*\(([^)]+)\)', clean_kernel)
    kernel_name = sig_match.group(1) if sig_match else "kernel"
    params_raw = sig_match.group(2) if sig_match else ""

    positional = []
    constexpr = []
    for param in params_raw.split(","):
        param = param.strip()
        pname = param.split(":")[0].strip()
        if not pname:
            continue
        if "tl.constexpr" in param or "constexpr" in param:
            constexpr.append(pname)
        else:
            positional.append(pname)

    # 4. Parse get_args() to find tensor inputs and scalar numels
    args_match = re.search(
        r'def get_args\(\):(.*?)(?=\ndef |\Z)', benchmark_source, re.DOTALL
    )
    tensor_lines = []
    numel_values = []
    if args_match:
        args_body = args_match.group(1)
        # Extract rand_strided calls (tensor inputs).
        # Use torch._dynamo.testing.rand_strided directly — it handles all
        # stride patterns (contiguous, padded, stride-0 broadcast, scalar)
        # correctly by construction. This replaces the previous if/else chain
        # that grew with each edge case (padded strides, zero strides, etc.).
        for m in re.finditer(
            r'(\w+)\s*=\s*rand_strided\(\s*(\([^)]*\))\s*,\s*(\([^)]*\))\s*,\s*'
            r"device='([^']*)'\s*,\s*dtype=([^\s,)]+)",
            args_body,
        ):
            var, shape, stride_raw, device, dtype = m.groups()
            dtype_clean = dtype.strip()
            tensor_lines.append(
                f"    {var} = rand_strided({shape}, {stride_raw}, device='{device}', dtype={dtype_clean})"
            )

        # Extract scalar numels from return statement
        ret_match = re.search(r'return\s+(.+)', args_body)
        if ret_match:
            ret_str = ret_match.group(1).strip().rstrip(",")
            for item in ret_str.split(","):
                item = item.strip().strip("[]")
                if re.match(r'^\d+$', item):
                    numel_values.append(int(item))

    # 5. Map numels to positional scalar params
    # Template kernels (triton_tem_*) use arg_A, arg_B naming for tensor pointers;
    # pointwise/reduction use in_ptr0, out_ptr0, in_out_ptr0 naming.
    _is_template = kernel_name.startswith("triton_tem")
    ptr_params = [
        p for p in positional
        if "ptr" in p or p.startswith(("in_", "out_"))
        or (_is_template and p.startswith("arg_"))
    ]
    scalar_params = [p for p in positional if p not in ptr_params]
    numel_map = {}
    for i, sp in enumerate(scalar_params):
        if i < len(numel_values):
            numel_map[sp] = numel_values[i]

    # --- Build the clean standalone file ---

    imports = [
        "import torch",
        "import triton",
        "import triton.language as tl",
        "from triton.language.extra import libdevice",
        "from torch._dynamo.testing import rand_strided",
    ]
    if "triton_helpers" in benchmark_source:
        imports.append("from torch._inductor.runtime import triton_helpers")

    # Constexpr defaults: derive from size_hints when available
    size_hints = {}
    sh_match = re.search(r"size_hints\s*=\s*\{([^}]+)\}", benchmark_source)
    if sh_match:
        for m in re.finditer(r"'(\w+)':\s*(\d+)", sh_match.group(1)):
            size_hints[m.group(1).lower().rstrip("_")] = int(m.group(2))

    is_reduction = any(k.startswith("r") for k in size_hints)
    r_hint = max((v for k, v in size_hints.items() if k.startswith("r")), default=0)

    constexpr_defaults = {}
    for cp in constexpr:
        cp_upper = cp.upper()
        if "RBLOCK" in cp_upper or "R0" in cp_upper:
            if r_hint >= 100_000:
                constexpr_defaults[cp] = 2048
            elif r_hint >= 10_000:
                constexpr_defaults[cp] = 1024
            else:
                constexpr_defaults[cp] = 512
        elif "YBLOCK" in cp_upper or cp_upper == "Y" or cp_upper.endswith("_Y"):
            constexpr_defaults[cp] = 8
        elif "XBLOCK" in cp_upper or cp_upper == "X" or cp_upper.endswith("_X"):
            if is_reduction and r_hint >= 10_000:
                constexpr_defaults[cp] = 1
            else:
                constexpr_defaults[cp] = 128
        elif "BLOCK_M" in cp_upper:
            constexpr_defaults[cp] = 128
        elif "BLOCK_N" in cp_upper:
            constexpr_defaults[cp] = 64
        elif "BLOCK_K" in cp_upper:
            constexpr_defaults[cp] = 32
        else:
            constexpr_defaults[cp] = 128

    # Pick grid dimension params: exclude r*numel (reduction dims) for grid
    def _pick_grid_param(params, index=-1):
        """Pick a scalar param for grid dim, preferring non-reduction params.
        Returns None when no params are available (caller must derive grid
        from tensor shapes instead).
        """
        non_r = [p for p in params if not re.match(r'^r\d*_?numel$|^r\d*_?n$|^rnumel$', p)]
        pool = non_r if non_r else params
        return pool[index] if pool else None

    # Build grid computation based on grid_type
    # Template kernels compute their grid internally from hardcoded M, N, K, BLOCK_*
    # values. Extract those to compute the external grid.
    if _is_template and not scalar_params:
        # Extract hardcoded dimensions and block sizes from kernel body
        _dim_vals = {}
        for _dm in re.finditer(r'^\s+([MNK])\s*=\s*(\d+)', clean_kernel, re.MULTILINE):
            _dim_vals[_dm.group(1)] = int(_dm.group(2))
        for _bm in re.finditer(r'^\s+BLOCK_([MNK])\s*:?\s*(?:tl\.)?constexpr\s*=\s*(\d+)', clean_kernel, re.MULTILINE):
            _dim_vals[f"BLOCK_{_bm.group(1)}"] = int(_bm.group(2))

        _M = _dim_vals.get("M", 1)
        _N = _dim_vals.get("N", 1)
        _BM = _dim_vals.get("BLOCK_M", 128)
        _BN = _dim_vals.get("BLOCK_N", 128)
        _grid_m = (_M + _BM - 1) // _BM
        _grid_n = (_N + _BN - 1) // _BN
        _total_programs = _grid_m * _grid_n
        grid_code = f"    grid = ({_total_programs}, 1, 1)"
    elif grid_type == "Grid2D" and len(constexpr) >= 2 and len(scalar_params) >= 2:
        x_numel = _pick_grid_param(scalar_params, -1)
        y_numel = _pick_grid_param([p for p in scalar_params if p != x_numel], -1)
        x_block_val = constexpr_defaults.get(constexpr[-1], 128)
        y_block_val = constexpr_defaults.get(constexpr[-2], 8)
        grid_code = f"    grid = (triton.cdiv({x_numel}, {x_block_val}), triton.cdiv({y_numel}, {y_block_val}))"
    elif grid_type == "Grid3D" and len(constexpr) >= 3 and len(scalar_params) >= 3:
        x_numel = _pick_grid_param(scalar_params, -1)
        remaining = [p for p in scalar_params if p != x_numel]
        y_numel = _pick_grid_param(remaining, -1)
        z_numel = _pick_grid_param([p for p in remaining if p != y_numel], -1)
        grid_code = (
            f"    grid = (triton.cdiv({x_numel}, {constexpr_defaults.get(constexpr[-1], 128)}), "
            f"triton.cdiv({y_numel}, {constexpr_defaults.get(constexpr[-2], 8)}), "
            f"triton.cdiv({z_numel}, {constexpr_defaults.get(constexpr[-3], 1)}))"
        )
    else:
        x_numel = _pick_grid_param(scalar_params) if scalar_params else None
        x_block_val = constexpr_defaults.get(constexpr[0], 128) if constexpr else 1024
        if x_numel is not None:
            grid_code = f"    grid = (triton.cdiv({x_numel}, {x_block_val}),)"
        else:
            # No scalar params — derive grid from the first tensor's element count
            grid_code = None  # will be set after tensor_var_names are known

    # Build kernel_function
    tensor_var_names = [re.match(r'\s*(\w+)', tl).group(1) for tl in tensor_lines] if tensor_lines else ["x"]

    # If get_args() has fewer tensors than the kernel's pointer params,
    # allocate output buffers for the missing out_ptr* parameters.
    out_ptrs_needed = [p for p in ptr_params if p not in tensor_var_names and ("out" in p)]
    alloc_lines = []
    alloc_gi_lines = []
    if len(tensor_var_names) < len(ptr_params) and out_ptrs_needed:
        xnumel_val = next(iter(numel_map.values()), 1024)
        first_input = tensor_var_names[0] if tensor_var_names else None
        for op in out_ptrs_needed:
            var = f"buf_{op}"
            if first_input:
                alloc_lines.append(
                    f"    {var} = torch.empty({xnumel_val}, dtype={first_input}.dtype, device={first_input}.device)"
                )
            else:
                alloc_lines.append(
                    f"    {var} = torch.empty({xnumel_val}, dtype=torch.float32, device='cuda')"
                )
            alloc_gi_lines.append(
                f"    {var} = torch.empty({xnumel_val}, dtype=torch.float32, device='cuda')"
            )
            tensor_var_names.append(var)

    scalar_args = ", ".join(
        str(numel_map.get(sp, 1024)) for sp in scalar_params
    )
    constexpr_args = ", ".join(
        f"{cp}={constexpr_defaults.get(cp, 1024)}" for cp in constexpr
    )
    all_launch_args = ", ".join(tensor_var_names)
    if scalar_args:
        all_launch_args += ", " + scalar_args
    if constexpr_args:
        all_launch_args += ", " + constexpr_args

    wrapper_lines = [
        "def kernel_function(*inputs):",
    ]
    for i, var in enumerate(tensor_var_names):
        if var.startswith("buf_"):
            continue
        wrapper_lines.append(f"    {var} = inputs[{i}]")
    # Set scalar numels
    for sp in scalar_params:
        val = numel_map.get(sp, 1024)
        wrapper_lines.append(f"    {sp} = {val}")
    # Allocate output buffers that weren't in get_args()
    wrapper_lines.extend(alloc_lines)
    if grid_code is None:
        # No scalar params to derive grid — use first tensor's element count
        first_tensor = tensor_var_names[0] if tensor_var_names else "inputs[0]"
        x_block_val = constexpr_defaults.get(constexpr[0], 128) if constexpr else 1024
        wrapper_lines.append(f"    grid = (triton.cdiv({first_tensor}.numel(), {x_block_val}),)")
    else:
        wrapper_lines.append(grid_code)
    wrapper_lines.append(f"    {kernel_name}[grid]({all_launch_args})")
    # Return the last output buffer if one was allocated, else last input
    ret_var = tensor_var_names[-1] if tensor_var_names else "None"
    wrapper_lines.append(f"    return {ret_var}")

    # Build get_inputs (only actual inputs, not dynamically allocated output buffers)
    input_var_names = [v for v in tensor_var_names if not v.startswith("buf_")]
    gi_lines = ["def get_inputs():"]
    if tensor_lines:
        gi_lines.extend(tensor_lines)
        gi_lines.append(f"    return [{', '.join(input_var_names)}]")
    else:
        gi_lines.append("    return [torch.randn(1024, device='cuda')]")

    main_block = [
        'if __name__ == "__main__":',
        "    import argparse",
        "    parser = argparse.ArgumentParser()",
        '    parser.add_argument("--profile-only", action="store_true")',
        "    args = parser.parse_args()",
        "    inputs = get_inputs()",
        "    kernel_function(*inputs)",
        "    torch.cuda.synchronize()",
        "    if not args.profile_only:",
        '        print("Kernel executed successfully.")',
    ]

    parts = (
        imports + [""]
        + [clean_kernel]
        + [""]
        + wrapper_lines
        + [""]
        + gi_lines
        + [""]
        + main_block
        + [""]
    )
    return "\n".join(parts)


def rebuild_standalone_from_body(
    kernel_body: str,
    benchmark_source: str,
) -> str:
    """Rebuild a standalone file by replacing the kernel body in the baseline.

    Uses the baseline standalone as a template — preserves its correct
    kernel_function(), get_inputs(), grid computation, and imports.
    Only swaps the @triton.jit kernel function (plus any @triton.autotune
    decorator the executor added).

    This approach works for ALL kernel types (pointwise, reduction, template)
    because the baseline already has a working wrapper for that specific kernel.

    Args:
        kernel_body: Modified @triton.jit function from the executor (may
            include @triton.autotune). Should be body-only but may contain
            imports, wrapper functions, or __main__ if the executor ignored
            the prompt instruction. We extract only the kernel.
        benchmark_source: Original baseline standalone file with working
            kernel_function(), get_inputs(), and __main__.
    """
    # --- Extract only the @triton.jit kernel from the executor's submission ---
    # The executor SHOULD submit only the kernel body, but often includes
    # imports, kernel_function(), get_inputs(), and __main__. Strip everything
    # except the @triton.autotune (optional) + @triton.jit + def kernel(...):
    kb_lines = kernel_body.split("\n")
    jit_start = None
    jit_end = None
    _kb_kernel_name = None
    _kb_in_body = False
    for i, line in enumerate(kb_lines):
        stripped = line.strip()
        # Find start: @triton.autotune or @triton.jit
        if jit_start is None:
            if stripped.startswith("@triton.autotune") or stripped.startswith("@triton.jit"):
                jit_start = i
            continue
        # Find the kernel's def line and name
        if _kb_kernel_name is None:
            name_m = re.match(r'^def\s+(\w+)\s*\(', stripped)
            if name_m:
                _kb_kernel_name = name_m.group(1)
            continue  # skip the def line
        # Track when we've entered indented body
        if not _kb_in_body and line[0:1].isspace() and stripped:
            _kb_in_body = True
        # Find end: next top-level def/class/if after the kernel body
        if _kb_in_body and stripped and not line[0:1].isspace() and not stripped.startswith("#") and not stripped.startswith("@"):
            if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("if __name__"):
                jit_end = i
                break
    if jit_start is not None:
        if jit_end is None:
            jit_end = len(kb_lines)
        # Strip trailing blank lines
        while jit_end > jit_start and not kb_lines[jit_end - 1].strip():
            jit_end -= 1
        kernel_body = "\n".join(kb_lines[jit_start:jit_end])

    lines = benchmark_source.split("\n")

    # --- Find the kernel block boundaries in the baseline ---
    # The kernel block starts at @triton.autotune or @triton.jit (whichever
    # comes first) and ends at the next top-level def/class/if-main.
    kernel_start = None
    kernel_end = None
    baseline_kernel_name = None

    in_body = False  # True after we've seen indented kernel body lines
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Find start: first @triton.autotune or @triton.jit at column 0
        if kernel_start is None:
            if stripped.startswith("@triton.autotune") or stripped.startswith("@triton.jit"):
                kernel_start = i
            continue
        # Find kernel name from the def line
        if baseline_kernel_name is None:
            name_match = re.match(r'^def\s+(\w+)\s*\(', stripped)
            if name_match:
                baseline_kernel_name = name_match.group(1)
            continue  # skip the def line itself — don't check for end
        # Track when we've entered the indented body
        if not in_body and line[0:1].isspace() and stripped:
            in_body = True
        # Find end: next top-level definition after the kernel body
        if in_body and stripped and not line[0:1].isspace() and not stripped.startswith("#") and not stripped.startswith("@"):
            if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("if __name__"):
                kernel_end = i
                break

    if kernel_start is None:
        # No kernel found in baseline — fall back to concatenation
        return benchmark_source + "\n\n" + kernel_body

    if kernel_end is None:
        kernel_end = len(lines)

    # --- Extract the new kernel name ---
    new_name_match = re.search(r'def\s+(\w+)\s*\(', kernel_body)
    new_kernel_name = new_name_match.group(1) if new_name_match else baseline_kernel_name

    # --- Build the new file ---
    # 1. Everything before the kernel block (imports, etc.)
    before = lines[:kernel_start]

    # 2. The executor's kernel body (with any @triton.autotune it added)
    new_kernel = kernel_body.rstrip().split("\n")

    # 3. Everything after the kernel block (kernel_function, get_inputs, __main__)
    after = lines[kernel_end:]

    # --- Handle kernel name change ---
    # If the executor renamed the kernel, update the launch call in kernel_function
    if baseline_kernel_name and new_kernel_name != baseline_kernel_name:
        after = [
            line.replace(baseline_kernel_name, new_kernel_name)
            for line in after
        ]

    # --- Handle @triton.autotune added by executor ---
    # If the executor added @triton.autotune but the baseline didn't have it,
    # the launch call in kernel_function() may pass constexpr kwargs explicitly
    # (e.g., XBLOCK=128). With autotune, these must NOT be passed — the autotuner
    # selects them. Strip constexpr kwargs from the launch call.
    baseline_had_autotune = any("@triton.autotune" in l for l in lines[kernel_start:kernel_end])
    executor_has_autotune = "@triton.autotune" in kernel_body

    if executor_has_autotune and not baseline_had_autotune:
        # Find constexpr param names from the new kernel body
        new_constexprs = set()
        for m in re.finditer(r'(\w+)\s*:\s*tl\.constexpr', kernel_body):
            new_constexprs.add(m.group(1))

        # Strip constexpr kwargs from launch call in the after section
        for i, line in enumerate(after):
            if new_kernel_name + "[" in line or (baseline_kernel_name and baseline_kernel_name + "[" in line):
                # Remove constexpr kwargs like XBLOCK=128, RBLOCK=512
                for cp in new_constexprs:
                    line = re.sub(rf',\s*{cp}\s*=\s*\d+', '', line)
                    line = re.sub(rf',\s*{cp}\s*=\s*\w+', '', line)
                after[i] = line

    # --- Add imports for triton_helpers if needed ---
    has_triton_helpers = "triton_helpers" in kernel_body
    baseline_has_helpers = any("triton_helpers" in l for l in before)
    if has_triton_helpers and not baseline_has_helpers:
        # Insert import after the last import line
        insert_idx = 0
        for i, line in enumerate(before):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                insert_idx = i + 1
        before.insert(insert_idx, "from torch._inductor.runtime import triton_helpers")

    # --- Assemble ---
    result_lines = before + [""] + new_kernel + [""] + after
    return "\n".join(result_lines)


def _extract_get_inputs(inductor_source: str) -> str:
    """Build get_inputs() from Inductor's module-level get_args() or benchmark.

    Parses rand_strided(...) calls and produces torch.randn/randint equivalents,
    handling non-float dtypes correctly.
    """
    # With TORCHINDUCTOR_BENCHMARK_KERNEL=1, the inductor source has
    # rand_strided calls inside async_compile.triton('''...''') blocks
    # (per-kernel benchmark harnesses).  Each per-kernel get_args() returns
    # that KERNEL's buffers (e.g. 2 tensors for a reduction), NOT the full
    # model inputs (e.g. 3 tensors: weight, bias, x).  We must use only
    # the MODULE-LEVEL get_args() — the one NOT inside a triple-quoted
    # string.  ^def with re.MULTILINE matches line-starts inside '''...'''
    # blocks too, so we filter by checking whether the match offset falls
    # inside an open triple-quote pair.
    last_module_get_args = ""
    for m in re.finditer(r'^def get_args\b.*', inductor_source, re.MULTILINE):
        # Count unmatched ''' before this position.  An odd count means
        # we are inside a triple-quoted string (per-kernel benchmark).
        preceding = inductor_source[:m.start()]
        if preceding.count("'''") % 2 == 1:
            continue  # skip — inside async_compile.triton('''...''')
        func_lines = []
        for line in inductor_source[m.start():].split("\n"):
            if func_lines and line and not line[0].isspace() and not line.startswith("#"):
                break
            func_lines.append(line)
        last_module_get_args = "\n".join(func_lines)

    # Fallback: if no module-level get_args() exists, search only the
    # module-level portion of the source (after async_compile.wait).
    # This avoids accidentally extracting per-kernel buffers from inside
    # the triple-quoted benchmark harnesses.
    if not last_module_get_args:
        # Everything after async_compile.wait(globals()) is module-level.
        wait_pos = inductor_source.find("async_compile.wait(globals())")
        if wait_pos != -1:
            last_module_get_args = inductor_source[wait_pos:]
        else:
            # No async_compile.wait found — strip all '''...''' blocks
            # so only module-level rand_strided calls remain.
            last_module_get_args = re.sub(
                r"'''.*?'''", "", inductor_source, flags=re.DOTALL
            )

    search_source = last_module_get_args if last_module_get_args else inductor_source

    pattern = re.compile(
        r"(\w+)\s*=\s*rand_strided\(\s*"
        r"\(([^)]*)\)\s*,\s*"
        r"\(([^)]*)\)\s*,\s*"
        r"device='([^']*)'\s*,\s*"
        r"dtype=([^\s,)]+)"
    )
    matches = pattern.findall(search_source)

    if not matches:
        return "def get_inputs():\n    return []"

    input_lines = []
    var_names = []
    for var_name, shape, stride_raw, device, dtype in matches:
        shape_clean = shape.strip().rstrip(",").strip()
        if not shape_clean:
            shape_str = "()"
        elif "," not in shape_clean:
            shape_str = f"({shape_clean},)"
        else:
            shape_str = f"({shape_clean})"

        dtype_clean = dtype.strip()

        # Use rand_strided directly — handles all stride patterns
        # (contiguous, padded, stride-0 broadcast, scalar) correctly.
        stride_clean = stride_raw.strip().rstrip(",").strip()
        if not stride_clean:
            stride_tuple = "()"
        elif "," not in stride_clean:
            stride_tuple = f"({stride_clean},)"
        else:
            stride_tuple = f"({stride_clean})"

        input_lines.append(
            f"    {var_name} = rand_strided({shape_str}, {stride_tuple}, device='{device}', dtype={dtype_clean})"
        )
        var_names.append(var_name)

    return_list = ", ".join(var_names)
    body = "\n".join(input_lines)
    return (
        f"from torch._dynamo.testing import rand_strided\n\n"
        f"def get_inputs():\n{body}\n    return [{return_list}]"
    )
