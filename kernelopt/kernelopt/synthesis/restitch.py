"""Re-stitch an optimized Triton kernel back into the full compiled model."""

from __future__ import annotations

import logging
import re

_log = logging.getLogger(__name__)


def restitch_kernel(
    synthesized_source: str,
    target_kernel_name: str,
    optimized_body: str,
) -> str:
    """Replace a Triton kernel body inside a synthesized_triton.py file.

    Finds the ``async_compile.triton('{target_kernel_name}', '''...''')``
    block and replaces the ``@triton.jit`` function body within it,
    preserving the Inductor ``@triton_heuristics.*`` decorator, imports,
    ``get_args()``, ``call()``, and benchmark harness.

    Any ``@triton.autotune`` in the optimized body is stripped — it
    conflicts with the ``@triton_heuristics`` wrapper that Inductor uses
    for its ``.run()`` calling convention.

    Returns the modified full source.

    Raises ValueError if *target_kernel_name* is not found.
    """
    block_start, block_end, inner_source = _find_async_compile_block(
        synthesized_source, target_kernel_name,
    )

    new_inner = _replace_jit_body(inner_source, optimized_body)

    rebuilt = _rebuild_async_block(target_kernel_name, new_inner)

    return (
        synthesized_source[:block_start]
        + rebuilt
        + synthesized_source[block_end:]
    )


def _find_async_compile_block(
    source: str, kernel_name: str,
) -> tuple[int, int, str]:
    """Locate the async_compile.triton() block for *kernel_name*.

    Returns (start, end, inner_source) where inner_source is the
    triple-quoted string content (without the quotes themselves).
    """
    pattern = re.compile(
        re.escape(kernel_name)
        + r"""\s*=\s*async_compile\.triton\(\s*'"""
        + re.escape(kernel_name)
        + r"""'\s*,\s*'''"""
    )
    m = pattern.search(source)
    if m is None:
        raise ValueError(
            f"Kernel '{kernel_name}' not found in async_compile.triton() block"
        )

    block_start = m.start()
    inner_start = m.end()

    end_marker = "''', device_str='cuda')"
    inner_end = source.find(end_marker, inner_start)
    if inner_end == -1:
        end_marker = "''', device_str=\"cuda\")"
        inner_end = source.find(end_marker, inner_start)
    if inner_end == -1:
        raise ValueError(
            f"Cannot find closing marker for kernel '{kernel_name}'"
        )

    block_end = inner_end + len(end_marker)
    inner_source = source[inner_start:inner_end]

    return block_start, block_end, inner_source


def _replace_jit_body(inner_source: str, optimized_body: str) -> str:
    """Replace the @triton.jit function body inside the inner string.

    The optimized body comes from a standalone file that may have
    different hardcoded numels and modulo constants (because the
    standalone was built from a per-kernel benchmark file with different
    tensor shapes). We preserve the ORIGINAL numels/modulo from the
    synthesized file and apply only the optimization changes (eviction
    policy, block sizes, new instructions).

    Strips @triton.autotune from the optimized body (conflicts with
    @triton_heuristics).
    """
    best_config = _extract_best_autotune_config(optimized_body)
    clean_body = _strip_autotune(optimized_body)
    clean_body = _strip_jit_decorator(clean_body)

    jit_match = re.search(r'^@triton\.jit\s*\n', inner_source, re.MULTILINE)
    if jit_match is None:
        raise ValueError("No @triton.jit found in async_compile block")

    after_jit = inner_source[jit_match.end():]
    first_def = re.search(r'^def\s', after_jit, re.MULTILINE)
    if first_def:
        after_first_def = after_jit[first_def.end():]
        next_def = re.search(r'^(?:def |class |if )', after_first_def, re.MULTILINE)
        if next_def:
            old_body_end = jit_match.end() + first_def.end() + next_def.start()
        else:
            old_body_end = len(inner_source)
    else:
        old_body_end = len(inner_source)

    preamble = inner_source[:jit_match.start()]
    suffix = inner_source[old_body_end:]
    original_body = inner_source[jit_match.end():old_body_end]

    merged = _merge_kernel_body(original_body, clean_body)

    if best_config:
        preamble = _inject_config_into_heuristics(preamble, best_config)

    return preamble + "@triton.jit\n" + merged.rstrip() + "\n\n" + suffix


def _inject_config_into_heuristics(preamble: str, config: dict) -> str:
    """Update @triton_heuristics size_hints to include the best autotune config.

    Bumps size_hints so the Inductor heuristics will try block sizes at
    least as large as the executor's best config.
    """
    xblock = config.get("XBLOCK", 0)
    if xblock <= 0:
        return preamble

    # When no_x_dim is True, the kernel has XBLOCK hardcoded to 1 and it's
    # not a parameter. Bumping size_hints would generate configs with XBLOCK>1
    # which triggers an assertion in Inductor's heuristics runtime.
    if "'no_x_dim': True" in preamble or "no_x_dim=True" in preamble:
        return preamble

    # Bump size_hints['x'] to at least xblock * num_multiprocessors so
    # the heuristics generates configs that include this block size.
    # The heuristics picks XBLOCK based on ceil(size_hint / num_SMs).
    try:
        import torch
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        num_sms = 132
        _log.warning("Could not query GPU SM count; falling back to %d", num_sms)
    min_hint = xblock * num_sms
    m = re.search(r"(size_hints\s*=\s*\{[^}]*'x'\s*:\s*)(\d+)", preamble)
    if m:
        current = int(m.group(2))
        if current < min_hint:
            preamble = preamble[:m.start(2)] + str(min_hint) + preamble[m.end(2):]

    return preamble


def _merge_kernel_body(original: str, optimized: str) -> str:
    """Merge optimization changes into the original kernel body.

    Preserves from the original:
    - Hardcoded numels (xnumel = N)
    - Modulo constants (xindex % N)
    - Variable assignments derived from tensor shapes

    Takes from the optimized:
    - eviction_policy changes
    - New tl.* calls (replacing triton_helpers.*)
    - Any new instructions the optimizer added
    - Block size changes in constexpr defaults

    If the function signatures differ (different param count), falls back
    to returning the optimized body as-is.
    """
    orig_lines = original.strip().split('\n')
    opt_lines = optimized.strip().split('\n')

    # Always preserve the original function signature — the heuristics
    # wrapper passes arguments matching the original def line.  If the
    # optimized body has a different signature (e.g. dropped XBLOCK),
    # replace only its def line with the original's.
    orig_sig_line = None
    opt_sig_line = None
    for line in original.strip().split('\n'):
        if re.match(r'^def\s+\w+\s*\(', line):
            orig_sig_line = line
            break
    for i, line in enumerate(optimized.strip().split('\n')):
        if re.match(r'^def\s+\w+\s*\(', line):
            opt_sig_line = line
            break
    if orig_sig_line and opt_sig_line and orig_sig_line != opt_sig_line:
        optimized = optimized.replace(opt_sig_line, orig_sig_line, 1)

    orig_lines = original.strip().split('\n')
    opt_lines = optimized.strip().split('\n')

    # Apply optimized body but restore original numels and shape constants.
    # The numels are hardcoded as "xnumel = <int>" or "ynumel = <int>" and
    # shape-derived as "x0 = (xindex % <int>)".
    _NUMEL_RE = re.compile(r'^(\s+\w*numel)\s*=\s*(\d+)\s*$')
    _MODULO_RE = re.compile(r'^(\s+\w+\s*=\s*\(?\w+\s*%\s*)\d+(\)?)\s*$')

    orig_numels = {}
    orig_modulos = {}
    for line in orig_lines:
        m = _NUMEL_RE.match(line)
        if m:
            orig_numels[m.group(1).strip()] = m.group(2)
        m = _MODULO_RE.match(line)
        if m:
            orig_modulos[m.group(1).strip()] = line.rstrip()

    result = []
    for line in opt_lines:
        m = _NUMEL_RE.match(line)
        if m and m.group(1).strip() in orig_numels:
            indent = line[:len(line) - len(line.lstrip())]
            var = m.group(1).strip()
            result.append(f"{indent}{var} = {orig_numels[var]}")
            continue
        m = _MODULO_RE.match(line)
        if m and m.group(1).strip() in orig_modulos:
            result.append(orig_modulos[m.group(1).strip()])
            continue
        result.append(line.rstrip())

    return '\n'.join(result) + '\n'


def _extract_signature(body: str) -> str | None:
    """Extract 'def name(params)' line from a kernel body."""
    m = re.search(r'^def\s+(\w+\s*\([^)]+\))', body, re.MULTILINE)
    return m.group(1) if m else None


def extract_kernel_body(standalone_source: str) -> str:
    """Extract the @triton.jit function (with decorators) from a standalone file.

    The standalone file has imports, the kernel, kernel_function(),
    get_inputs(), and __main__. This returns only the kernel portion:
    from the first decorator (@triton.autotune or @triton.jit) through
    the end of the function body.
    """
    autotune = re.search(r'^@triton\.autotune', standalone_source, re.MULTILINE)
    jit = re.search(r'^@triton\.jit', standalone_source, re.MULTILINE)

    if autotune:
        start = autotune.start()
    elif jit:
        start = jit.start()
    else:
        raise ValueError("No @triton.jit found in standalone source")

    func_match = re.search(r'^def\s+(\w+)\s*\(', standalone_source[start:], re.MULTILINE)
    if not func_match:
        raise ValueError("No def found after @triton.jit")

    after_start = standalone_source[start:]
    end_match = re.search(
        r'^(?:def |class |if __name__)',
        after_start[func_match.end():],
        re.MULTILINE,
    )
    if end_match:
        end = start + func_match.end() + end_match.start()
    else:
        end = len(standalone_source)

    return standalone_source[start:end].rstrip() + '\n'


def _extract_best_autotune_config(body: str) -> dict | None:
    """Extract the first (best) config from @triton.autotune if present.

    Returns dict with keys like XBLOCK, YBLOCK, num_warps, num_stages,
    or None if no autotune.
    """
    if "@triton.autotune" not in body:
        return None

    config = {}
    # Extract the first triton.Config({...}) block so we only look at one config
    config_m = re.search(r'triton\.Config\(\s*\{([^}]+)\}', body)
    if config_m:
        config_text = config_m.group(1)
        # Use finditer to get ALL BLOCK_* params, not just the first
        for block_m in re.finditer(r"'(\w*BLOCK\w*)'\s*:\s*(\d+)", config_text):
            config[block_m.group(1)] = int(block_m.group(2))
    warps_m = re.search(r'num_warps\s*=\s*(\d+)', body)
    if warps_m:
        config["num_warps"] = int(warps_m.group(1))
    stages_m = re.search(r'num_stages\s*=\s*(\d+)', body)
    if stages_m:
        config["num_stages"] = int(stages_m.group(1))

    return config if config else None


def _strip_autotune(body: str) -> str:
    """Remove @triton.autotune(...) decorator if present.

    Uses a balanced-parentheses scanner instead of regex to correctly
    handle nested parentheses inside the autotune config.
    """
    if "@triton.autotune" not in body:
        return body

    # Find the start of @triton.autotune
    start = body.find("@triton.autotune")
    if start == -1:
        return body

    # Find the opening paren
    open_paren = body.find("(", start)
    if open_paren == -1:
        return body

    # Walk forward counting depth to find the matching close paren
    depth = 0
    pos = open_paren
    while pos < len(body):
        ch = body[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                # pos is the matching close paren
                end = pos + 1
                # Strip trailing whitespace/newline after the decorator
                while end < len(body) and body[end] in (" ", "\t", "\r", "\n"):
                    end += 1
                return body[:start] + body[end:]
        pos += 1

    # Unbalanced parens — return body unchanged as a safe fallback
    return body


def _strip_jit_decorator(body: str) -> str:
    """Remove @triton.jit from the body if present (we add it ourselves)."""
    return re.sub(r'^@triton\.jit\s*\n', '', body, count=1, flags=re.MULTILINE)


def _rebuild_async_block(kernel_name: str, inner: str) -> str:
    """Wrap inner source back into async_compile.triton()."""
    return (
        f"{kernel_name} = async_compile.triton('{kernel_name}', '''\n"
        f"{inner}"
        f"''', device_str='cuda')"
    )
