"""Run torch.compile in a subprocess, extract generated Triton source.

Subprocess isolation is deliberate: torch.compile modifies global state
(Dynamo cache, Inductor cache) and we don't want that inside kernelopt's
main process.  The driver script uses ``run_and_get_code`` to get the
Inductor-generated source as a string -- no file-hunting required.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


class SynthesisError(Exception):
    """Raised when torch.compile extraction fails."""


def _detect_pytorch_pattern(source: str) -> str:
    """Classify which PyTorch input pattern the source matches.

    Returns one of: "kernelbench", "run_function", "forward_function", "none".
    """
    has_nn_module = bool(re.search(r'class\s+\w+\s*\(\s*(?:nn\.Module|torch\.nn\.Module)\s*\)', source))
    has_get_inputs = "def get_inputs(" in source
    has_run = bool(re.search(r'def\s+run\s*\(', source))

    if has_nn_module and has_get_inputs:
        return "kernelbench"
    if has_run:
        return "run_function"
    if has_nn_module:
        return "kernelbench"
    return "none"


def _build_driver_script(
    pytorch_file: str,
    output_json: str,
    pattern: str,
) -> str:
    """Build the Python driver script that runs torch.compile in a subprocess.

    The driver:
      1. Imports the user's PyTorch code
      2. Instantiates model / locates the callable
      3. Calls ``run_and_get_code(compiled_fn, *inputs)``
      4. Writes the generated source(s) + metadata to *output_json*
    """
    pytorch_dir = str(Path(pytorch_file).parent.resolve())
    pytorch_module = Path(pytorch_file).stem

    return textwrap.dedent(f"""\
        import json, sys, os, torch
        os.environ["TORCH_COMPILE_DEBUG"] = "0"

        sys.path.insert(0, {pytorch_dir!r})
        _mod = __import__({pytorch_module!r})

        from torch._inductor.utils import run_and_get_code

        torch._inductor.config.max_autotune = True
        torch._inductor.config.max_autotune_gemm = True

        pattern = {pattern!r}

        if pattern == "kernelbench":
            get_init = getattr(_mod, "get_init_inputs", lambda: [])
            model = _mod.Model(*get_init()).cuda()
            model.eval()
            raw_inputs = _mod.get_inputs()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in raw_inputs]
            compiled = torch.compile(model)
            with torch.no_grad():
                result, sources = run_and_get_code(compiled, *inputs)

        elif pattern == "run_function":
            run_fn = _mod.run
            get_inputs_fn = getattr(_mod, "get_inputs", None)
            if get_inputs_fn is not None:
                raw_inputs = get_inputs_fn()
                inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in raw_inputs]
            else:
                inputs = []
            compiled = torch.compile(run_fn)
            with torch.no_grad():
                result, sources = run_and_get_code(compiled, *inputs)
        else:
            print("ERROR: unknown pattern", file=sys.stderr)
            sys.exit(1)

        graph_break_count = 0
        graph_breaks = []
        try:
            callable_fn = compiled if pattern == "kernelbench" else compiled
            explanation = torch._dynamo.explain(callable_fn)(*inputs)
            graph_break_count = explanation.graph_break_count
            if graph_break_count > 0:
                for reason in explanation.break_reasons:
                    graph_breaks.append(str(reason))
        except Exception as _explain_err:
            graph_breaks = [f"explain() failed: {{_explain_err}}"]

        out = {{
            "num_sources": len(sources),
            "sources": list(sources),
            "graph_break_count": graph_break_count,
            "graph_breaks": graph_breaks,
        }}
        with open({output_json!r}, "w") as f:
            json.dump(out, f)
    """)


def extract_triton_from_pytorch(
    pytorch_file: str,
    *,
    timeout: int = 180,
) -> list[dict[str, Any]]:
    """Run torch.compile on a PyTorch file and extract generated Triton code.

    Args:
        pytorch_file: Path to the .py file containing PyTorch code
            (KernelBench Model+get_inputs, SOL-ExecBench run(), or generic).
        timeout: Subprocess timeout in seconds.

    Returns:
        List of dicts, each with:
          - ``inductor_source``: full Inductor output_code.py contents
          - ``triton_kernels``: list of extracted ``@triton.jit`` kernel source strings
          - ``call_source``: the ``call(args)`` function source
          - ``benchmark_source``: the ``benchmark_compiled_module()`` source (for input shapes)
          - ``pattern``: detected input pattern
    """
    source = Path(pytorch_file).read_text()
    pattern = _detect_pytorch_pattern(source)
    if pattern == "none":
        raise SynthesisError(
            f"Cannot detect PyTorch pattern in {pytorch_file}. "
            "Expected class Model(nn.Module) + get_inputs(), or def run()."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kpt_synth_driver_"
    ) as driver_f:
        driver_path = driver_f.name

    output_json = driver_path.replace(".py", "_out.json")

    driver_code = _build_driver_script(pytorch_file, output_json, pattern)
    Path(driver_path).write_text(driver_code)

    try:
        result = subprocess.run(
            [sys.executable, driver_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
            raise SynthesisError(
                f"torch.compile subprocess failed (exit {result.returncode}):\n{stderr_tail}"
            )

        if not Path(output_json).exists():
            raise SynthesisError(
                "torch.compile subprocess completed but produced no output. "
                f"stdout: {result.stdout[-500:]}"
            )

        with open(output_json) as f:
            data = json.load(f)

        results = []
        for src in data["sources"]:
            triton_kernel_blocks = _extract_triton_jit_blocks(src)
            triton_kernels = [b["source"] for b in triton_kernel_blocks]
            call_source = _extract_function(src, "call")
            benchmark_source = _extract_function(src, "benchmark_compiled_module")

            results.append({
                "inductor_source": src,
                "triton_kernels": triton_kernels,
                "triton_kernel_blocks": triton_kernel_blocks,
                "call_source": call_source,
                "benchmark_source": benchmark_source,
                "pattern": pattern,
                "graph_break_count": data.get("graph_break_count", 0),
                "graph_breaks": data.get("graph_breaks", []),
            })

        return results

    except subprocess.TimeoutExpired:
        raise SynthesisError(
            f"torch.compile timed out after {timeout}s on {pytorch_file}"
        )
    finally:
        for p in (driver_path, output_json):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def _extract_triton_jit_blocks(inductor_source: str) -> list[dict[str, Any]]:
    """Extract all @triton.jit kernel source blocks from Inductor output.

    Inductor wraps them in ``async_compile.triton('name', '''...''')`` or
    ``async_compile.triton_template('name', '''...''')`` calls.

    Returns list of dicts with keys: source, name, is_template.
    """
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    patterns = [
        (r"async_compile\.triton\(\s*'([^']+)'\s*,\s*'''(.*?)'''\s*,", False),
        (r'async_compile\.triton\(\s*\'([^\']+)\'\s*,\s*"""(.*?)"""\s*,', False),
        (r"async_compile\.triton_template\(\s*'([^']+)'\s*,\s*'''(.*?)'''\s*,", True),
        (r'async_compile\.triton_template\(\s*\'([^\']+)\'\s*,\s*"""(.*?)"""\s*,', True),
    ]

    for pat, is_template in patterns:
        for m in re.finditer(pat, inductor_source, re.DOTALL):
            name = m.group(1)
            if name not in seen_names:
                seen_names.add(name)
                results.append({
                    "source": m.group(2),
                    "name": name,
                    "is_template": is_template,
                })

    return results


def _extract_function(source: str, func_name: str) -> str:
    """Extract a top-level function definition from source by name.

    Handles multi-line function signatures like ``def call(\\n    arg1,\\n):``.
    """
    lines = source.split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^def\s+{func_name}\s*\(", line):
            start = i
            break
    if start is None:
        return ""

    func_lines = [lines[start]]

    # Phase 1: consume the full signature (may span multiple lines)
    sig_complete = bool(re.search(r'\)\s*(->\s*\S+\s*)?:\s*(#.*)?$', lines[start]))
    body_start = start + 1
    if not sig_complete:
        for i in range(start + 1, len(lines)):
            func_lines.append(lines[i])
            if re.search(r'\)\s*(->\s*\S+\s*)?:\s*(#.*)?$', lines[i]):
                body_start = i + 1
                break
        else:
            body_start = len(func_lines) + start

    # Phase 2: consume the indented body
    for i in range(body_start, len(lines)):
        line = lines[i]
        if line.strip() == "":
            func_lines.append(line)
            continue
        if line[0] not in (" ", "\t"):
            break
        func_lines.append(line)

    while func_lines and func_lines[-1].strip() == "":
        func_lines.pop()

    return "\n".join(func_lines)
