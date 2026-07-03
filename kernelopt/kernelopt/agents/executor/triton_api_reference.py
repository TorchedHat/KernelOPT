"""Triton API reference for the executor agent.

Introspects the installed Triton version at import time and builds a
categorized reference string with full signatures for core functions,
hardware constraints, and common mistakes to avoid.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Static signature database for the most-used Triton functions.
# Sourced from Triton 3.4.0 docs; the dynamic check below ensures only
# functions that actually exist in the installed version are emitted.
# ---------------------------------------------------------------------------

_CORE_SIGNATURES: dict[str, list[tuple[str, str]]] = {
    "Memory": [
        ("tl.load", "tl.load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False)"),
        ("tl.store", "tl.store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='')"),
        ("tl.atomic_add", "tl.atomic_add(pointer, val, mask=None, sem=None, scope=None)"),
        ("tl.atomic_max", "tl.atomic_max(pointer, val, mask=None, sem=None, scope=None)"),
        ("tl.atomic_min", "tl.atomic_min(pointer, val, mask=None, sem=None, scope=None)"),
        ("tl.atomic_cas", "tl.atomic_cas(pointer, cmp, val, sem=None, scope=None)"),
    ],
    "Math": [
        ("tl.dot", "tl.dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=tl.float32)"),
        ("tl.dot_scaled", "tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, out_dtype=tl.float32)"),
        ("tl.exp", "tl.exp(x)"),
        ("tl.exp2", "tl.exp2(x)"),
        ("tl.log", "tl.log(x)"),
        ("tl.log2", "tl.log2(x)"),
        ("tl.sqrt", "tl.sqrt(x)"),
        ("tl.rsqrt", "tl.rsqrt(x)"),
        ("tl.abs", "tl.abs(x)"),
        ("tl.fma", "tl.fma(x, y, z)"),
        ("tl.cos", "tl.cos(x)"),
        ("tl.sin", "tl.sin(x)"),
        ("tl.sigmoid", "tl.sigmoid(x)"),
        ("tl.softmax", "tl.softmax(x)"),
        ("tl.cast", "tl.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)"),
    ],
    "Reduction": [
        ("tl.sum", "tl.sum(input, axis=None, keep_dims=False)"),
        ("tl.max", "tl.max(input, axis=None, keep_dims=False)"),
        ("tl.min", "tl.min(input, axis=None, keep_dims=False)"),
        ("tl.reduce", "tl.reduce(input, axis, combine_fn, keep_dims=False)"),
        ("tl.cumsum", "tl.cumsum(input, axis)"),
        ("tl.argmax", "tl.argmax(input, axis)"),
        ("tl.argmin", "tl.argmin(input, axis)"),
    ],
    "Comparison": [
        ("tl.where", "tl.where(condition, x, y)"),
        ("tl.maximum", "tl.maximum(x, y, propagate_nan=NONE)"),
        ("tl.minimum", "tl.minimum(x, y, propagate_nan=NONE)"),
        ("tl.clamp", "tl.clamp(x, min, max, propagate_nan=NONE)"),
    ],
    "Indexing": [
        ("tl.program_id", "tl.program_id(axis)"),
        ("tl.num_programs", "tl.num_programs(axis)"),
        ("tl.arange", "tl.arange(start, end)"),
        ("tl.cdiv", "tl.cdiv(x, y)"),
        ("tl.expand_dims", "tl.expand_dims(input, axis)"),
        ("tl.broadcast_to", "tl.broadcast_to(input, *shape)"),
        ("tl.full", "tl.full(shape, value, dtype)"),
        ("tl.zeros", "tl.zeros(shape, dtype)"),
        ("tl.reshape", "tl.reshape(input, *shape, can_reorder=False)"),
        ("tl.permute", "tl.permute(input, *dims)"),
        ("tl.trans", "tl.trans(input, *dims)"),
        ("tl.cat", "tl.cat(input, other, can_reorder=False)"),
        ("tl.flip", "tl.flip(input)"),
        ("tl.max_contiguous", "tl.max_contiguous(input, values)"),
        ("tl.multiple_of", "tl.multiple_of(input, values)"),
    ],
    "Control Flow": [
        ("tl.range", "tl.range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, flatten=False, warp_specialize=False)"),
        ("tl.static_range", "tl.static_range(arg1, arg2=None, step=None)"),
        ("tl.debug_barrier", "tl.debug_barrier()"),
        ("tl.device_assert", "tl.device_assert(cond, msg='')"),
        ("tl.device_print", "tl.device_print(prefix, *args, hex=False)"),
    ],
    "Hopper+ (cc >= 9.0)": [
        ("tl.make_tensor_descriptor", "tl.make_tensor_descriptor(base, shape, strides, block_shape) -> tensor_descriptor"),
        ("tl.load_tensor_descriptor", "tl.load_tensor_descriptor(desc, offsets) -> tensor"),
        ("tl.store_tensor_descriptor", "tl.store_tensor_descriptor(desc, offsets, value)"),
        ("tl.make_block_ptr", "tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)"),
        ("tl.advance", "tl.advance(block_ptr, increments)"),
    ],
}

# Functions whose name (without the tl. prefix) we look up in the module
_CORE_NAMES = set()
for _sigs in _CORE_SIGNATURES.values():
    for _display_name, _ in _sigs:
        _CORE_NAMES.add(_display_name.split(".")[-1])


def build_triton_api_reference() -> str:
    """Build a categorized Triton API reference from the installed version.

    Returns a string (~1200 tokens) suitable for appending to the executor
    system prompt. Core functions get full signatures; remaining functions
    are listed by name only. Built by introspecting the actual installed
    Triton, so hallucinated APIs are excluded.
    """
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    tl_names = sorted(
        x for x in dir(tl)
        if not x.startswith("_") and callable(getattr(tl, x, None))
    )
    tl_name_set = set(tl_names)

    lib_names = sorted(
        x for x in dir(libdevice)
        if not x.startswith("_") and callable(getattr(libdevice, x, None))
    )

    # --- Build categorized core section ---
    core_lines: list[str] = []
    for category, sigs in _CORE_SIGNATURES.items():
        entries = []
        for display_name, signature in sigs:
            func_name = display_name.split(".")[-1]
            if func_name in tl_name_set:
                entries.append(f"  - {signature}")
        if entries:
            core_lines.append(f"\n### {category}")
            core_lines.extend(entries)

    # --- Additional tl.* functions not in core ---
    remaining = sorted(n for n in tl_names if n not in _CORE_NAMES)
    remaining_str = ", ".join(f"tl.{n}" for n in remaining) if remaining else "(none)"

    # --- libdevice most-used subset ---
    lib_most_used = [
        "tanh", "erf", "erfinv", "pow", "cbrt", "exp", "exp2", "log",
        "log2", "log1p", "cos", "sin", "tan", "acos", "asin", "atan",
        "atan2", "cosh", "sinh", "floor", "ceil", "round", "trunc",
        "fma", "fmod", "copysign", "nextafter", "isnan", "isinf",
        "normcdf", "normcdfinv", "lgamma", "rsqrt", "sqrt",
    ]
    lib_available = [n for n in lib_most_used if n in set(lib_names)]
    lib_str = ", ".join(f"libdevice.{n}" for n in lib_available)

    return (
        f"\nTRITON API REFERENCE (v{triton.__version__}):\n"
        f"\n## Core Functions (with signatures)"
        + "\n".join(core_lines) +
        f"\n\n## Additional tl.* Functions\n"
        f"  {remaining_str}\n"
        f"\n## libdevice (import: from triton.language.extra import libdevice)\n"
        f"  {lib_str}\n"
        f"  (and {max(0, len(lib_names) - len(lib_available))} more — use for transcendentals not on tl.*)\n"
        f"\n## Hardware Constraints\n"
        f"- num_warps: 1, 2, 4, or 8 (powers of 2, max 8)\n"
        f"- num_stages: 1-16 (pipeline depth)\n"
        f"- BLOCK sizes: must be powers of 2, typically 16-2048\n"
        f"- Max threads per block: 1024 (= num_warps * 32)\n"
        f"- tl.dot requires both input dimensions >= 16\n"
        f"- tl.dot requires both inputs to have the same dtype\n"
        f"- Accumulator for tl.dot should be float32 for numerical stability\n"
        f"\n## Common Mistakes to AVOID\n"
        f"- triton_helpers.* does NOT exist in standalone Triton — only inside torch._inductor internals\n"
        f"- Use tl.maximum(a, b) instead of triton_helpers.maximum(a, b)\n"
        f"- Use tl.math.* or libdevice.* for transcendentals not directly available as tl.*\n"
        f"- Import libdevice as: from triton.language.extra import libdevice\n"
        f"- Do NOT use reinterpret_tensor inside @triton.jit — it is a PyTorch function\n"
        f"- tl.full([1], 0, tl.int32) creates int32 — cast with .to(tl.float32) before mixing with float ops\n"
        f"- Do NOT import or use async_compile, empty_strided_cuda, or assert_size_stride — torch._inductor internals\n"
        f"- tl.range() is for loop iteration, tl.arange() is for index generation — do not confuse them\n"
        f"- Do NOT use tl.tensor — use tl.zeros/tl.full instead\n"
    )
