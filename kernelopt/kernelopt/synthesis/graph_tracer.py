"""Trace a PyTorch model to extract ATen ops and Inductor fusion decisions.

Runs torch.export to get the computation graph, then torch.compile to get
Inductor's kernel fusion decisions. Returns structured data describing each
kernel group: which ATen ops, input/output shapes, kernel type.

All heavy work runs in a subprocess to isolate torch.compile state.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from kernelopt.constants import MAX_SOURCE_CHARS


def trace_model(
    pytorch_path: str,
    *,
    timeout: int = 300,
) -> list[dict]:
    """Trace a PyTorch model and return kernel group descriptions.

    Args:
        pytorch_path: Path to .py file with Model class + get_inputs().

    Returns:
        List of kernel group dicts, each with:
            - kernel_name: Inductor's kernel name (e.g., triton_poi_fused_relu_0)
            - kernel_type: "pointwise" | "reduction" | "template" | "persistent"
            - fused_aten_ops: list of ATen op names fused into this kernel
            - input_shapes: list of (shape, dtype) tuples
            - output_shapes: list of (shape, dtype) tuples
            - model_description: human-readable summary of what this kernel computes
            - inductor_source: the raw Inductor kernel source (for reference)
    """
    pytorch_path = str(Path(pytorch_path).resolve())
    script = _build_tracer_script(pytorch_path)

    import os as _os
    env = {k: v for k, v in _os.environ.items()}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    python = sys.executable
    pipeline_venv = Path(__file__).resolve().parents[3] / ".pipeline_venv" / "bin" / "python"
    if pipeline_venv.exists():
        python = str(pipeline_venv)

    proc = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd="/tmp",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Graph tracer failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )

    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)

    raise RuntimeError(
        f"No JSON output from graph tracer.\nstdout: {proc.stdout[-1000:]}\n"
        f"stderr: {proc.stderr[-1000:]}"
    )


def _build_tracer_script(pytorch_path: str) -> str:
    """Build the subprocess script that traces the model."""
    return textwrap.dedent(f"""\
        import json, sys, re, os, torch
        import importlib.util

        torch.cuda.set_device(0)

        # Load model
        spec = importlib.util.spec_from_file_location("_model", {pytorch_path!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_model"] = mod
        spec.loader.exec_module(mod)

        model_cls = getattr(mod, "Model", None)
        get_inputs = getattr(mod, "get_inputs", None)
        get_init = getattr(mod, "get_init_inputs", None)

        if model_cls is None or get_inputs is None:
            print(json.dumps([]))
            sys.exit(0)

        init_args = get_init() if get_init else []
        model = model_cls(*init_args).cuda().eval()
        inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_inputs()]

        # Step 1: Export to get ATen graph
        aten_ops = []
        input_info = []
        output_info = []
        try:
            from torch.export import export
            with torch.no_grad():
                exported = export(model, tuple(inputs), strict=True)
            for node in exported.graph_module.graph.nodes:
                if node.op == "call_function":
                    name = node.target.__name__ if hasattr(node.target, "__name__") else str(node.target)
                    meta = node.meta.get("val", None)
                    shape = list(meta.shape) if hasattr(meta, "shape") else []
                    dtype = str(meta.dtype) if hasattr(meta, "dtype") else "?"
                    aten_ops.append({{"name": name, "shape": shape, "dtype": dtype}})
                elif node.op == "placeholder":
                    meta = node.meta.get("val", None)
                    shape = list(meta.shape) if hasattr(meta, "shape") else []
                    dtype = str(meta.dtype) if hasattr(meta, "dtype") else "?"
                    input_info.append({{"name": node.name, "shape": shape, "dtype": dtype}})
        except Exception as e:
            aten_ops = [{{"name": "unknown", "shape": [], "dtype": "?", "error": str(e)}}]

        # Step 2: Compile to get Inductor fusion decisions
        kernel_groups = []
        code = ""  # populated by run_and_get_code; empty if compile fails
        try:
            from torch._inductor.utils import run_and_get_code
            torch._inductor.config.max_autotune = True
            torch._inductor.config.max_autotune_gemm = True
            _c = torch._inductor.config
            if hasattr(_c, "max_autotune_gemm_backends"):
                _c.max_autotune_gemm_backends = "TRITON"
            if hasattr(_c, "max_autotune_conv_backends"):
                _c.max_autotune_conv_backends = "TRITON"

            compiled = torch.compile(model)
            with torch.no_grad():
                _, _source_codes = run_and_get_code(compiled, *inputs)
                code = _source_codes[0] if _source_codes else ""

            # Extract each kernel block: name = async_compile.triton('name', '''...''')
            tq = chr(39)*3  # triple single-quote
            pattern = r"(triton_\\w+)\\s*=\\s*async_compile\\.triton\\(.*?" + tq + r"(.*?)" + tq
            blocks = re.findall(pattern, code, re.DOTALL)

            for kname, ksource in blocks:
                # Classify kernel type
                if "triton_poi_" in kname:
                    ktype = "pointwise"
                elif "triton_red_" in kname:
                    ktype = "reduction"
                elif "triton_tem_" in kname:
                    ktype = "template"
                elif "triton_per_" in kname:
                    ktype = "persistent"
                else:
                    ktype = "unknown"

                # Extract fused ATen ops from comments
                fused_ops = re.findall(r"Original ATen: \\[(.+?)\\]", ksource)
                source_ops = re.findall(r"Source node.*?:\\s*(.+)", ksource)

                # Extract function signature
                sig_match = re.search(r"def\\s+\\w+\\s*\\(([^)]+)\\)", ksource)
                params = sig_match.group(1) if sig_match else ""

                kernel_groups.append({{
                    "kernel_name": kname,
                    "kernel_type": ktype,
                    "fused_aten_ops": fused_ops,
                    "source_ops": source_ops,
                    "params": params,
                    "aten_graph": aten_ops,
                    "model_inputs": input_info,
                    "inductor_source": ksource[:{MAX_SOURCE_CHARS}],
                }})

            # Detect extern_kernels calls (cuBLAS/cuDNN) that need Triton replacement
            extern_calls = re.findall(r"extern_kernels\\.(\w+)\\(([^)]+)\\)", code)
            for ext_op, ext_args in extern_calls:
                kernel_groups.append({{
                    "kernel_name": f"extern_{{ext_op}}",
                    "kernel_type": "extern",
                    "fused_aten_ops": [ext_op],
                    "source_ops": [f"extern_kernels.{{ext_op}}"],
                    "params": ext_args.strip(),
                    "aten_graph": aten_ops,
                    "model_inputs": input_info,
                    "inductor_source": "",
                    "needs_triton_replacement": True,
                }})

            # If no kernel blocks found at all, create groups from ATen ops
            if not kernel_groups:
                for op in aten_ops:
                    kernel_groups.append({{
                        "kernel_name": f"aten_{{op['name'].replace('.', '_')}}",
                        "kernel_type": "from_aten",
                        "fused_aten_ops": [op["name"]],
                        "source_ops": [],
                        "params": "",
                        "aten_graph": aten_ops,
                        "model_inputs": input_info,
                        "inductor_source": "",
                    }})

        except Exception as e:
            kernel_groups = [{{
                "kernel_name": "unknown",
                "kernel_type": "unknown",
                "fused_aten_ops": [op["name"] for op in aten_ops],
                "source_ops": [],
                "params": "",
                "aten_graph": aten_ops,
                "model_inputs": input_info,
                "inductor_source": "",
                "compile_error": str(e),
            }}]

        # Step 3: Build human-readable description + attach full Inductor output
        for kg in kernel_groups:
            ops_str = ", ".join(kg["fused_aten_ops"]) or ", ".join(op["name"] for op in aten_ops)
            shapes_str = ", ".join(f"{{inp['name']}}={{inp['shape']}}" for inp in input_info)
            kg["model_description"] = (
                f"Kernel '{{kg['kernel_name']}}' (type={{kg['kernel_type']}}): "
                f"Fused ops: {{ops_str}}. "
                f"Model inputs: {{shapes_str}}."
            )
            kg["full_inductor_source"] = code

        print(json.dumps(kernel_groups))
    """)