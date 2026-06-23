"""Patch a synthesized model source for kernel fusion.

Provides utilities to:
1. Remove an async_compile.triton() block for a specific kernel.
2. Rewrite the call() method to remove intermediate kernel calls, drop
   eliminated buffer allocations, and insert a fused kernel invocation.
3. Combine (1) and (2) into a single full-restitch operation.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 1. Remove async_compile block
# ---------------------------------------------------------------------------


def remove_async_compile_block(source: str, kernel_name: str) -> str:
    """Remove an async_compile.triton() block entirely.

    Finds the assignment of the form::

        kernel_name = async_compile.triton('kernel_name', '''...''', device_str='cuda')

    and deletes the entire multi-line block including any trailing blank lines.

    Parameters
    ----------
    source:
        Full model source text.
    kernel_name:
        Exact name of the kernel variable to remove.

    Returns
    -------
    str
        Source with the block removed.

    Raises
    ------
    ValueError
        If the kernel_name is not found in any async_compile block.
    """
    # Pattern: <kernel_name> = async_compile.triton(...)
    # The value can span multiple lines (triple-quoted strings).
    # Strategy: find the line that starts the assignment, then scan forward
    # until the closing parenthesis that balances the open one.

    lines = source.splitlines(keepends=True)

    # Find start line
    start_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{kernel_name} = async_compile.triton("):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(
            f"async_compile block for kernel '{kernel_name}' not found in source."
        )

    # Scan forward to find the closing ')' that ends the call.
    # We count triple-quote nesting to avoid being fooled by parens inside strings.
    depth = 0
    in_triple = False
    triple_char: str | None = None
    end_idx: int | None = None

    block_text = "".join(lines[start_idx:])
    char_pos = 0
    line_offset = start_idx

    # Walk character by character through all lines from start_idx onward.
    flat = "".join(lines[start_idx:])
    i = 0
    while i < len(flat):
        c = flat[i]

        if in_triple:
            # Look for closing triple quote
            if flat[i : i + 3] == triple_char * 3:
                in_triple = False
                triple_char = None
                i += 3
                continue
            i += 1
            continue

        # Detect triple quotes
        if c in ('"', "'") and flat[i : i + 3] in ('"""', "'''"):
            in_triple = True
            triple_char = c
            i += 3
            continue

        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                # This is the closing paren of async_compile.triton(...)
                # Find which original line this corresponds to
                chars_so_far = flat[: i + 1]
                newline_count = chars_so_far.count("\n")
                end_idx = start_idx + newline_count
                break

        i += 1

    if end_idx is None:
        # Fallback: remove just the start line
        end_idx = start_idx

    # Remove lines[start_idx .. end_idx] inclusive, plus any immediately
    # following blank lines.
    remove_end = end_idx + 1  # exclusive
    while remove_end < len(lines) and lines[remove_end].strip() == "":
        remove_end += 1

    result_lines = lines[:start_idx] + lines[remove_end:]
    return "".join(result_lines)


# ---------------------------------------------------------------------------
# 2. Patch call() method
# ---------------------------------------------------------------------------


def patch_call_for_fusion(
    call_source: str,
    kernels_to_remove: list[str],
    buffers_to_remove: list[str],
    fused_call_line: str,
    insert_after_line: str,
    extern_lines_to_remove: list[str] | None = None,
) -> str:
    """Rewrite the call() method body.

    - Removes lines where any of ``kernels_to_remove`` are run via ``.run(``.
    - Removes lines where any of ``buffers_to_remove`` are allocated via
      ``empty_strided_cuda`` or freed via ``del``.
    - Removes extern_kernels call lines matching ``extern_lines_to_remove``.
    - Removes buffer alias lines (``buf1 = buf0; del buf0  # reuse``)
      and reinterpret_tensor alias lines where both sides are in
      ``buffers_to_remove``.
    - Inserts ``fused_call_line`` after the line matching ``insert_after_line``.

    Parameters
    ----------
    call_source:
        Full text of the call() method (or entire model file).
    kernels_to_remove:
        List of kernel names whose ``.run(...)`` lines should be deleted.
    buffers_to_remove:
        List of buffer names whose allocation / ``del`` lines should be deleted.
    fused_call_line:
        The new call line to insert (e.g. ``"            fused_kernel_call(...)"``)
    insert_after_line:
        Exact content (stripped) of the line after which to insert.
        Insertion is placed immediately after the first matching line.
    extern_lines_to_remove:
        List of substrings matching extern_kernels call lines to remove.
        When a source line contains any of these substrings, it is removed.

    Returns
    -------
    str
        Patched source.
    """
    kernels_set = set(kernels_to_remove)
    buffers_set = set(buffers_to_remove)
    extern_subs = extern_lines_to_remove or []

    out_lines: list[str] = []
    inserted = False

    for line in call_source.splitlines(keepends=True):
        stripped = line.strip()

        # Drop kernel .run() lines
        skip = False
        for kname in kernels_set:
            if f"{kname}.run(" in line:
                skip = True
                break

        # Drop extern_kernels lines
        if not skip:
            for sub in extern_subs:
                if sub in line:
                    skip = True
                    break

        if not skip:
            # Drop buffer allocation lines for eliminated buffers
            for bname in buffers_set:
                if re.search(
                    rf"\b{re.escape(bname)}\s*=\s*empty_strided_cuda\(", line
                ):
                    skip = True
                    break

        if not skip:
            # Drop `del bufN` lines for eliminated buffers
            for bname in buffers_set:
                if re.match(rf"^\s*del\s+{re.escape(bname)}\s*$", line):
                    skip = True
                    break

        if not skip:
            # Drop buffer alias lines: buf1 = buf0; del buf0  # reuse
            m = re.match(r"(\w+)\s*=\s*(\w+)\s*;\s*del\s+\w+\s*#\s*reuse", stripped)
            if m:
                alias, original = m.group(1), m.group(2)
                if alias in buffers_set or original in buffers_set:
                    skip = True

        if not skip:
            # Drop reinterpret_tensor alias lines for eliminated buffers
            m = re.match(r"(\w+)\s*=\s*reinterpret_tensor\((\w+)", stripped)
            if m and "# reuse" in stripped:
                alias, original = m.group(1), m.group(2)
                if alias in buffers_set or original in buffers_set:
                    skip = True

        if not skip:
            out_lines.append(line)

        # Insert fused call after the target line (first match)
        if not inserted and stripped == insert_after_line.strip():
            # Preserve indentation of the matched line
            indent = len(line) - len(line.lstrip())
            out_lines.append(" " * indent + fused_call_line.strip() + "\n")
            inserted = True

    return "".join(out_lines)


# ---------------------------------------------------------------------------
# 3. Full restitch
# ---------------------------------------------------------------------------


def restitch_fused(
    model_source: str,
    kernel_names_to_remove: list[str],
    first_kernel_name: str,
    fused_block: str,
    call_patch_fn,
) -> str:
    """Full restitch: replace first kernel block with fused, remove others, patch call().

    Parameters
    ----------
    model_source:
        Full source of the synthesized model file.
    kernel_names_to_remove:
        All kernel names in the fusion group (including the first one).
    first_kernel_name:
        The kernel whose ``async_compile`` block will be REPLACED with
        ``fused_block``.
    fused_block:
        The replacement async_compile block (or raw kernel source block) to
        insert in place of the first kernel's block.
    call_patch_fn:
        A callable ``(source: str) -> str`` that patches the ``call()`` method.
        Typically a closure calling :func:`patch_call_for_fusion`.

    Returns
    -------
    str
        Fully patched model source with fused kernel inserted and originals removed.
    """
    result = model_source

    # Step 1: Replace first kernel's async_compile block with the fused block
    lines = result.splitlines(keepends=True)
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(f"{first_kernel_name} = async_compile.triton("):
            start_idx = i
            break

    if start_idx is not None:
        # Find the end of this block (same logic as remove_async_compile_block)
        flat = "".join(lines[start_idx:])
        depth = 0
        in_triple = False
        triple_char: str | None = None
        end_idx = start_idx
        idx = 0
        while idx < len(flat):
            c = flat[idx]
            if in_triple:
                if flat[idx : idx + 3] == triple_char * 3:
                    in_triple = False
                    triple_char = None
                    idx += 3
                    continue
                idx += 1
                continue
            if c in ('"', "'") and flat[idx : idx + 3] in ('"""', "'''"):
                in_triple = True
                triple_char = c
                idx += 3
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    chars_so_far = flat[: idx + 1]
                    newline_count = chars_so_far.count("\n")
                    end_idx = start_idx + newline_count
                    break
            idx += 1

        replace_end = end_idx + 1
        # Preserve indentation of original block
        indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        fused_lines = [" " * indent + l + "\n" for l in fused_block.splitlines()]
        result_lines = lines[:start_idx] + fused_lines + lines[replace_end:]
        result = "".join(result_lines)

    # Step 2: Remove async_compile blocks for the remaining kernels
    others_to_remove = [k for k in kernel_names_to_remove if k != first_kernel_name]
    for kname in others_to_remove:
        try:
            result = remove_async_compile_block(result, kname)
        except ValueError:
            pass  # Already removed or not found; continue

    # Step 3: Patch the call() method
    result = call_patch_fn(result)

    return result
