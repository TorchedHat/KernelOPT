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


def _find_python() -> str:
    """Find the best Python for running torch.compile.

    Prefers the Python that has the newest torch installed, since
    Inductor's codegen may use syntax features from newer Python versions.
    Falls back to sys.executable.
    """
    import shutil

    for candidate in ("python3.11", "python3.12", "python3.10", "python3"):
        path = shutil.which(candidate)
        if path:
            try:
                r = subprocess.run(
                    [path, "-c", "import torch; print(torch.__version__)"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    return path
            except (subprocess.TimeoutExpired, OSError):
                continue
    return sys.executable


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

        out = {{
            "num_sources": len(sources),
            "sources": list(sources),
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

    python = _find_python()

    try:
        run_kwargs = dict(
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if timeout:
            run_kwargs["timeout"] = timeout
        result = subprocess.run([python, driver_path], **run_kwargs)

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
            triton_kernels = _extract_triton_jit_blocks(src)
            call_source = _extract_function(src, "call")
            benchmark_source = _extract_function(src, "benchmark_compiled_module")

            results.append({
                "inductor_source": src,
                "triton_kernels": triton_kernels,
                "call_source": call_source,
                "benchmark_source": benchmark_source,
                "pattern": pattern,
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


def _extract_triton_jit_blocks(inductor_source: str) -> list[str]:
    """Extract all @triton.jit kernel source blocks from Inductor output.

    Inductor wraps them in ``async_compile.triton('name', '''...''')`` calls.
    We extract the triple-quoted content and strip inductor-specific imports/decorators.
    """
    pattern = re.compile(
        r"async_compile\.triton\(\s*'[^']+'\s*,\s*'''(.*?)'''\s*,",
        re.DOTALL,
    )
    blocks = pattern.findall(inductor_source)

    if not blocks:
        pattern2 = re.compile(
            r"async_compile\.triton\(\s*'[^']+'\s*,\s*\"\"\"(.*?)\"\"\"\s*,",
            re.DOTALL,
        )
        blocks = pattern2.findall(inductor_source)

    return blocks


def _extract_function(source: str, func_name: str) -> str:
    """Extract a top-level function definition from source by name."""
    lines = source.split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^def\s+{func_name}\s*\(", line):
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
