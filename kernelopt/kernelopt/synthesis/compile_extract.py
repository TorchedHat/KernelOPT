"""Run torch.compile in a subprocess, extract generated Triton source.

Subprocess isolation is deliberate: torch.compile modifies global state
(Dynamo cache, Inductor cache) and we don't want that inside kernelopt's
main process.  The driver script uses ``run_and_get_code`` to get the
Inductor-generated source as a string -- no file-hunting required.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SynthesisError(Exception):
    """Raised when torch.compile extraction fails."""


@dataclass
class CallableInfo:
    """Describes a compilable callable found in a PyTorch source file."""

    kind: str                          # "module_class" | "function"
    class_name: str | None = None      # e.g. "MyModel", "Attention"
    function_name: str | None = None   # e.g. "run", "forward_pass"
    has_get_inputs: bool = False
    has_get_init_inputs: bool = False


_HELPER_NAMES = frozenset({"get_inputs", "get_init_inputs", "get_init"})


def _detect_callable(
    source: str,
    *,
    entry_point: str | None = None,
) -> CallableInfo | None:
    """Find a compilable callable in *source* using AST analysis.

    Returns ``None`` when nothing compilable is found.

    Detection priority (most specific first):
      1. User-specified *entry_point* (class or function name)
      2. nn.Module subclass + get_inputs()
      3. Any non-helper function + get_inputs()
      4. nn.Module subclass without get_inputs
      5. Any function containing torch operations
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # Collect top-level classes and functions
    module_classes: list[str] = []
    functions: list[str] = []
    has_get_inputs = "def get_inputs(" in source
    has_get_init_inputs = "def get_init_inputs(" in source

    _nn_module_bases = {"nn.Module", "torch.nn.Module", "Module"}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if any(b in ast.unparse(base) for b in _nn_module_bases):
                    module_classes.append(node.name)
                    break
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name not in _HELPER_NAMES and not node.name.startswith("_"):
                functions.append(node.name)

    # 1. User-specified entry point
    if entry_point:
        # "ClassName.method" or "ClassName" or "function_name"
        if "." in entry_point:
            cls, _method = entry_point.rsplit(".", 1)
        else:
            cls = entry_point

        # Check if it's a class
        all_classes = [n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)]
        if cls in all_classes:
            return CallableInfo(
                kind="module_class",
                class_name=cls,
                has_get_inputs=has_get_inputs,
                has_get_init_inputs=has_get_init_inputs,
            )

        # Check if it's a function
        all_funcs = [n.name for n in ast.iter_child_nodes(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if entry_point in all_funcs:
            return CallableInfo(
                kind="function",
                function_name=entry_point,
                has_get_inputs=has_get_inputs,
                has_get_init_inputs=has_get_init_inputs,
            )
        return None

    # 2. nn.Module subclass + get_inputs
    if module_classes and has_get_inputs:
        cls = "Model" if "Model" in module_classes else module_classes[0]
        return CallableInfo(
            kind="module_class",
            class_name=cls,
            has_get_inputs=True,
            has_get_init_inputs=has_get_init_inputs,
        )

    # 3. Any function + get_inputs
    if functions and has_get_inputs:
        preferred = ["run", "forward", "main"]
        fn = next((f for f in preferred if f in functions), functions[0])
        return CallableInfo(
            kind="function",
            function_name=fn,
            has_get_inputs=True,
            has_get_init_inputs=has_get_init_inputs,
        )

    # 4. nn.Module subclass without get_inputs
    if module_classes:
        cls = "Model" if "Model" in module_classes else module_classes[0]
        return CallableInfo(
            kind="module_class",
            class_name=cls,
            has_get_inputs=False,
            has_get_init_inputs=has_get_init_inputs,
        )

    # 5. Any function with torch operations
    if functions:
        has_torch_import = "import torch" in source
        has_torch_decorator = "@torch." in source
        if has_torch_import or has_torch_decorator:
            preferred = ["run", "forward", "main"]
            fn = next((f for f in preferred if f in functions), functions[0])
            return CallableInfo(
                kind="function",
                function_name=fn,
                has_get_inputs=False,
                has_get_init_inputs=False,
            )

    return None


def _find_python() -> str:
    """Find the Python for running torch.compile.

    Uses sys.executable (the current interpreter) to ensure we run
    with the same torch installation as the calling process.
    """
    return sys.executable



def _build_driver_script(
    pytorch_file: str,
    output_json: str,
    info: CallableInfo,
    *,
    input_shapes: list[tuple[int, ...]] | None = None,
    force_triton: bool = False,
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

    # --- Build the input-generation snippet ---
    if info.has_get_inputs:
        input_block = textwrap.dedent("""\
            raw_inputs = _mod.get_inputs()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in raw_inputs]
        """)
    elif input_shapes:
        shape_lines = ", ".join(
            f"torch.randn({s}, device='cuda')" for s in input_shapes
        )
        input_block = f"inputs = [{shape_lines}]\n"
    else:
        input_block = textwrap.dedent("""\
            print("WARNING: no get_inputs() and no --input-shapes; using default (1, 64) tensor", file=sys.stderr)
            inputs = [torch.randn(1, 64, device='cuda')]
        """)

    # --- Build the callable-setup + compile snippet ---
    if info.kind == "module_class":
        cls = info.class_name or "Model"
        if info.has_get_init_inputs:
            init_line = f"init_args = getattr(_mod, 'get_init_inputs', lambda: [])()"
        else:
            init_line = "init_args = []"
        callable_block = textwrap.dedent(f"""\
            {init_line}
            model = _mod.{cls}(*init_args).cuda()
            model.eval()
            {{input_block}}
            compiled = torch.compile(model)
            with torch.no_grad():
                result, sources = run_and_get_code(compiled, *inputs)
        """).replace("{input_block}", input_block)
    else:
        fn = info.function_name or "run"
        callable_block = textwrap.dedent(f"""\
            fn = _mod.{fn}
            {{input_block}}
            compiled = torch.compile(fn)
            with torch.no_grad():
                result, sources = run_and_get_code(compiled, *inputs)
        """).replace("{input_block}", input_block)

    return textwrap.dedent(f"""\
        import json, sys, os, torch, glob
        os.environ["TORCH_COMPILE_DEBUG"] = "0"
        os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
        os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"

        sys.path.insert(0, {pytorch_dir!r})
        _mod = __import__({pytorch_module!r})

        from torch._inductor.utils import run_and_get_code

        torch._inductor.config.max_autotune = True
        torch._inductor.config.max_autotune_gemm = True
        {"torch._inductor.config.max_autotune_gemm_backends = 'TRITON'" if force_triton else "pass"}
        {"torch._inductor.config.max_autotune_conv_backends = 'TRITON,CUDNN,ATEN'" if force_triton else "pass"}

    """) + callable_block + textwrap.dedent(f"""\

        _bk_files = []
        _cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR",
            os.path.join("/tmp", "torchinductor_" + os.environ.get("USER", "root")))
        if os.path.isdir(_cache):
            for _f in glob.glob(os.path.join(_cache, "**", "*.py"), recursive=True):
                try:
                    _c = open(_f).read()
                except OSError:
                    continue
                if "@triton.jit" in _c and "def get_args" in _c and "def call" in _c:
                    _bk_files.append({{"name": os.path.basename(_f), "source": _c, "path": _f}})

        out = {{
            "num_sources": len(sources),
            "sources": list(sources),
            "benchmark_kernel_files": _bk_files,
        }}
        with open({output_json!r}, "w") as f:
            json.dump(out, f)
    """)


def extract_triton_from_pytorch(
    pytorch_file: str,
    *,
    timeout: int | None = None,
    entry_point: str | None = None,
    input_shapes: list[tuple[int, ...]] | None = None,
    force_triton: bool = True,
) -> list[dict[str, Any]]:
    """Run torch.compile on a PyTorch file and extract generated Triton code.

    Args:
        pytorch_file: Path to the .py file containing PyTorch code.
            Supports nn.Module subclasses, standalone functions, or any
            callable with ``import torch``.  Use *entry_point* to specify
            a particular class or function when auto-detection is ambiguous.
        timeout: Subprocess timeout in seconds.
        entry_point: Explicit callable name — ``"ClassName"``,
            ``"ClassName.method"``, or ``"function_name"``.
        input_shapes: Tensor shapes for synthesis when no ``get_inputs()``
            is present, e.g. ``[(8, 512), (8, 512, 64)]``.

    Returns:
        List of dicts, each with:
          - ``inductor_source``: full Inductor output_code.py contents
          - ``triton_kernels``: list of extracted ``@triton.jit`` kernel source strings
          - ``call_source``: the ``call(args)`` function source
          - ``benchmark_source``: the ``benchmark_compiled_module()`` source (for input shapes)
          - ``callable_info``: the :class:`CallableInfo` that was detected
    """
    source = Path(pytorch_file).read_text()
    info = _detect_callable(source, entry_point=entry_point)
    if info is None:
        raise SynthesisError(
            f"Cannot find a compilable callable in {pytorch_file}. "
            "Expected an nn.Module subclass or a function using torch ops. "
            "Use --entry-point to specify which class or function to compile."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kpt_synth_driver_"
    ) as driver_f:
        driver_path = driver_f.name

    output_json = driver_path.replace(".py", "_out.json")

    driver_code = _build_driver_script(
        pytorch_file, output_json, info, input_shapes=input_shapes,
        force_triton=force_triton,
    )
    Path(driver_path).write_text(driver_code)

    python = _find_python()

    try:
        run_kwargs = dict(
            capture_output=True,
            text=True,
            env={k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"} | {"PYTHONDONTWRITEBYTECODE": "1"},
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
                "callable_info": info,
                "benchmark_kernel_files": data.get("benchmark_kernel_files", []),
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
