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


def _extraction_hooks_code() -> str:
    """Python source installing compiler context extraction hooks.

    Inserted into the subprocess driver BEFORE torch.compile runs.
    All variables are prefixed with ``_cc_`` to avoid collisions.
    """
    return textwrap.dedent("""\
# --- Compiler context extraction hooks ---
_cc_warnings = []
_cc_pre_decomp_ops = []
_cc_ir_nodes = []
_cc_pre_nodes = []
_cc_post_nodes = []
_cc_why_no_fuse = []
_cc_fusion_ok = False
_cc_triton_compilations = []

try:
    import torch._inductor.config as _cc_cfg
    if hasattr(_cc_cfg, "pre_grad_custom_pass"):
        def _cc_pre_grad(graph):
            for node in graph.nodes:
                if node.op == "call_function":
                    target = node.target
                    op_name = getattr(target, "__name__", str(target))
                    if hasattr(target, "overloadpacket"):
                        op_name = str(target.overloadpacket)
                    in_shapes = []
                    for arg in node.args:
                        if hasattr(arg, "meta") and "val" in arg.meta:
                            val = arg.meta["val"]
                            if hasattr(val, "shape"):
                                in_shapes.append(tuple(int(s) for s in val.shape))
                    out_shape = ()
                    if "val" in node.meta:
                        val = node.meta["val"]
                        if hasattr(val, "shape"):
                            out_shape = tuple(int(s) for s in val.shape)
                    _cc_pre_decomp_ops.append({
                        "op_name": op_name,
                        "input_shapes": tuple(in_shapes),
                        "output_shape": out_shape,
                        "is_compound": any(c in op_name.lower() for c in (
                            "layer_norm", "group_norm", "batch_norm", "softmax",
                            "gelu", "silu", "linear", "conv",
                        )),
                    })
            return graph
        _cc_cfg.pre_grad_custom_pass = _cc_pre_grad
    if hasattr(_cc_cfg, "post_grad_custom_pre_pass"):
        def _cc_post_grad(graph):
            try:
                _nodes = graph.nodes if hasattr(graph, 'nodes') else (graph.graph.nodes if hasattr(graph, 'graph') else [])
                for node in _nodes:
                    if node.op != "call_function":
                        continue
                    val = node.meta.get("val")
                    if val is None or not hasattr(val, "shape"):
                        continue
                    shape = tuple(int(s) for s in val.shape)
                    dtype_str = str(val.dtype)
                    bpe = val.element_size() if hasattr(val, "element_size") else 4
                    total_elements = 1
                    for s in shape:
                        total_elements *= s
                    is_reduction = any(k in str(node.target) for k in ("reduce", "sum", "mean", "var", "amax"))
                    reduction = None
                    if is_reduction:
                        reduce_dim = node.kwargs.get("dim", node.kwargs.get("dims"))
                        if reduce_dim is not None:
                            if isinstance(reduce_dim, int):
                                reduce_dim = [reduce_dim]
                            input_val = None
                            if node.args:
                                arg0 = node.args[0]
                                if hasattr(arg0, "meta"):
                                    input_val = arg0.meta.get("val")
                            if input_val is not None and hasattr(input_val, "shape"):
                                in_shape = tuple(int(s) for s in input_val.shape)
                                red_ranges = tuple(in_shape[d] for d in reduce_dim if d < len(in_shape))
                                reduction = {
                                    "reduction_type": str(node.target).split(".")[-1],
                                    "reduction_ranges": red_ranges,
                                }
                    _cc_ir_nodes.append({
                        "buffer_name": node.name,
                        "node_type": "Reduction" if is_reduction else "Pointwise",
                        "iteration_ranges": shape,
                        "total_elements": total_elements,
                        "dtype": dtype_str,
                        "bytes_per_element": bpe,
                        "total_memory_bytes": total_elements * bpe,
                        "num_reads": len(node.args),
                        "num_writes": 1,
                        "read_buffer_names": tuple(a.name for a in node.args if hasattr(a, "name")),
                        "write_buffer_names": (node.name,),
                        "reduction": reduction,
                    })
            except Exception:
                pass
            return graph
        _cc_cfg.post_grad_custom_pre_pass = _cc_post_grad
    if hasattr(_cc_cfg, "_pre_fusion_custom_pass"):
        from torch._inductor.scheduler import WhyNoFuse, FusedSchedulerNode
        def _cc_pre_fusion(nodes):
            _cc_pre_nodes.clear()
            for n in nodes:
                try:
                    _cc_pre_nodes.append({"name": n.get_name(), "names": list(n.get_operation_names()), "is_reduction": n.is_reduction()})
                except Exception:
                    pass
            return nodes
        def _cc_post_fusion(nodes):
            _cc_post_nodes.clear()
            for n in nodes:
                try:
                    entry = {"name": n.get_name(), "names": list(n.get_operation_names()), "is_fused": isinstance(n, FusedSchedulerNode)}
                    if isinstance(n, FusedSchedulerNode):
                        entry["snodes"] = [sn.get_name() for sn in n.snodes]
                    _cc_post_nodes.append(entry)
                except Exception:
                    pass
            return nodes
        _cc_cfg._pre_fusion_custom_pass = _cc_pre_fusion
        _cc_cfg._post_fusion_custom_pass = _cc_post_fusion
        _cc_fusion_ok = True
        _cc_orig_wnf = WhyNoFuse.__call__
        def _cc_patched_wnf(self, reason, *args):
            try:
                formatted = reason % args if args else reason
                _cc_why_no_fuse.append({"node1": self.name1, "node2": self.name2, "reason": formatted})
            except Exception:
                pass
            return _cc_orig_wnf(self, reason, *args)
        WhyNoFuse.__call__ = _cc_patched_wnf
except Exception as _cc_e:
    _cc_warnings.append(f"Inductor hooks: {_cc_e}")

try:
    import triton
    def _cc_triton_listener(*, src, metadata, metadata_group, times, cache_hit):
        entry = {
            "kernel_name": metadata.get("name", ""),
            "cache_hit": cache_hit,
            "num_warps": metadata.get("num_warps", 4),
            "num_stages": metadata.get("num_stages", 1),
            "shared_bytes": metadata.get("shared", 0),
            "ir_files": dict(metadata_group) if metadata_group else {},
            "compilation_time_us": 0.0,
        }
        if times:
            total = getattr(times, "ir_initialization", 0)
            for _, dur in getattr(times, "lowering_stages", []):
                total += dur
            total += getattr(times, "store_results", 0)
            entry["compilation_time_us"] = total
        _cc_triton_compilations.append(entry)
    triton.knobs.compilation.listener = _cc_triton_listener
except Exception as _cc_e:
    _cc_warnings.append(f"Triton listener: {_cc_e}")

""")


def _extraction_collect_code() -> str:
    """Python source collecting compiler context data after compilation.

    Inserted into the subprocess driver AFTER torch.compile has run.
    Expects ``sources`` (from ``run_and_get_code``) and the ``_cc_*``
    variables populated by the hooks from ``_extraction_hooks_code()``.
    Defines ``_compiler_context`` dict in the script namespace.
    """
    return textwrap.dedent("""\
# --- Collect compiler context data ---
import re as _cc_re

_cc_structural_kw = {"device mismatch", "extern but not a triton", "different devices",
                     "not a GPU node", "will create cycle", "data dependency"}
_cc_fusion_data = {"fusion_groups": [], "fusion_rejections": [], "intermediate_buffers": []}
if _cc_fusion_ok:
    _cc_fusion_data["fusion_groups"] = [n for n in _cc_post_nodes if n.get("is_fused")]
    for _rej in _cc_why_no_fuse[:50]:
        _reason = _rej.get("reason", "")
        _is_structural = any(kw in _reason.lower() for kw in _cc_structural_kw)
        _is_heuristic = any(kw in _reason.lower() for kw in ("score", "threshold", "too many"))
        _cc_fusion_data["fusion_rejections"].append({
            "node1_name": _rej["node1"],
            "node2_name": _rej["node2"],
            "reason": _reason,
            "near_threshold": _is_heuristic and not _is_structural,
            "fixable_by_manual_fusion": not _is_structural,
        })

_cc_codegen = []
_cc_generated = sources[0] if sources else ""
_cc_cat_map = {"per": "persistent_reduction", "red": "streaming_reduction",
               "poi": "pointwise", "tem": "template"}
_cc_tq = chr(39) * 3
_cc_kpat = r"(triton_\\w+)\\s*=\\s*async_compile\\.triton\\(.*?" + _cc_tq + r"(.*?)" + _cc_tq
for _kname, _ksrc in _cc_re.findall(_cc_kpat, _cc_generated, _cc_re.DOTALL):
    _prefix = _kname.split("_")[1][:3] if "_" in _kname else "unk"
    _cat = _cc_cat_map.get(_prefix, "unknown")
    _persistent = _cat == "persistent_reduction"
    _cooperative = ("RSPLIT" in _ksrc) and _prefix == "red"
    if _cooperative:
        _cat = "cooperative_reduction"
    _blk = {}
    for _bm in _cc_re.finditer(r'"(XBLOCK|RBLOCK|YBLOCK|BLOCK_M|BLOCK_N|BLOCK_K)"\\s*:\\s*(\\d+)', _ksrc):
        _blk[_bm.group(1)] = int(_bm.group(2))
    for _bm in _cc_re.finditer(r'(XBLOCK|RBLOCK|YBLOCK)\\s*:\\s*tl\\.constexpr\\s*=\\s*(\\d+)', _ksrc):
        _blk[_bm.group(1)] = int(_bm.group(2))
    _nw = _cc_re.search(r"num_warps\\s*=\\s*(\\d+)", _ksrc)
    _ns = _cc_re.search(r"num_stages\\s*=\\s*(\\d+)", _ksrc)
    _hint = _cc_re.search(r"reduction_hint\\s*=\\s*ReductionHint\\.(\\w+)", _ksrc)
    _cc_codegen.append({
        "kernel_name": _kname,
        "kernel_category": _cat,
        "persistent_reduction": _persistent,
        "cooperative_reduction": _cooperative,
        "no_x_dim": "XBLOCK" not in _ksrc,
        "uses_block_pointers": "block_ptr" in _ksrc,
        "num_reduction_dims": len(_cc_re.findall(r"r\\d+_numel|rnumel", _ksrc[:500])) or (1 if _persistent or _cooperative or _prefix == "red" else 0),
        "reduction_hint": _hint.group(1) if _hint else "N/A",
        "inductor_block_sizes": _blk,
        "inductor_num_warps": int(_nw.group(1)) if _nw else 4,
        "inductor_num_stages": int(_ns.group(1)) if _ns else 1,
    })

_cc_triton_info = []
for _tc in _cc_triton_compilations:
    _ttir, _ttgir = "", ""
    for _fk, _fp in _tc.get("ir_files", {}).items():
        try:
            if _fk.endswith(".ttir"):
                _ttir = open(_fp).read()[:5000]
            elif _fk.endswith(".ttgir"):
                _ttgir = open(_fp).read()[:5000]
        except OSError:
            pass
    _dot_count = len(_cc_re.findall(r"tt\\.dot\\b", _ttir)) if _ttir else 0
    _load_count = len(_cc_re.findall(r"tt\\.load\\b", _ttir)) if _ttir else 0
    _store_count = len(_cc_re.findall(r"tt\\.store\\b", _ttir)) if _ttir else 0
    _cc_triton_info.append({
        "kernel_name": _tc["kernel_name"],
        "num_warps": _tc["num_warps"],
        "num_stages": _tc["num_stages"],
        "shared_bytes": _tc["shared_bytes"],
        "n_regs": 0, "n_spills": 0, "n_max_threads": 0,
        "ttir_text": _ttir,
        "ttgir_text": _ttgir,
        "compilation_time_us": _tc["compilation_time_us"],
        "dot_count": _dot_count,
        "load_count": _load_count,
        "store_count": _store_count,
        "has_async_copy": "async_copy" in _ttir or "async_copy" in _ttgir,
        "has_warp_specialize": "warp_specialize" in _ttir or "warp_specialize" in _ttgir,
        "mma_type": "wgmma" if "wgmma" in (_ttir + _ttgir).lower() else ("mma" if "nvidia_mma" in (_ttir + _ttgir) else ("dot" if _dot_count > 0 else "")),
    })

_cc_aot_decomps = []
_cc_known_decomps = {
    "layer_norm": ("var_mean", "rsqrt", "sub", "mul", "add"),
    "native_layer_norm": ("var_mean", "rsqrt", "sub", "mul", "add"),
    "gelu": ("mul", "erf", "add", "mul"),
    "group_norm": ("reshape", "var_mean", "rsqrt", "sub", "mul", "add", "reshape"),
    "native_group_norm": ("reshape", "var_mean", "rsqrt", "sub", "mul", "add", "reshape"),
    "batch_norm": ("mean", "var", "sub", "rsqrt", "mul", "add"),
    "softmax": ("max", "sub", "exp", "sum", "div"),
    "linear": ("addmm",),
}
for _op in _cc_pre_decomp_ops:
    if not _op.get("is_compound"):
        continue
    _name = _op["op_name"].lower()
    for _dk, _dv in _cc_known_decomps.items():
        if _dk in _name:
            _cc_aot_decomps.append({
                "original_op": _op["op_name"],
                "decomposed_into": _dv,
                "num_intermediates": max(0, len(_dv) - 1),
                "decomposition_source": "inductor_decompositions",
            })
            break

_compiler_context = {
    "pytorch_version": torch.__version__,
    "extraction_success": True,
    "extraction_warnings": _cc_warnings,
    "dynamo_info": {
        "original_ops": _cc_pre_decomp_ops,
        "graph_breaks": [],
        "num_subgraphs": 1,
        "unsupported_ops": [],
    },
    "aot_decomps": _cc_aot_decomps,
    "ir_nodes": _cc_ir_nodes,
    "fusion_data": _cc_fusion_data,
    "codegen_strategies": _cc_codegen,
    "triton_info": _cc_triton_info,
}

""")


def _build_driver_script(
    pytorch_file: str,
    output_json: str,
    info: CallableInfo,
    *,
    input_shapes: list[tuple[int, ...]] | None = None,
    force_triton: bool = False,
    extract_compiler_context: bool = False,
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

    header = textwrap.dedent(f"""\
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

    """)

    if extract_compiler_context:
        header += _extraction_hooks_code()

    footer = textwrap.dedent(f"""\

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
    """)

    if extract_compiler_context:
        footer += _extraction_collect_code()
        footer += 'out["compiler_context"] = _compiler_context\n'

    footer += textwrap.dedent(f"""\
        with open({output_json!r}, "w") as f:
            json.dump(out, f)
    """)

    return header + callable_block + footer


def extract_triton_from_pytorch(
    pytorch_file: str,
    *,
    timeout: int | None = None,
    entry_point: str | None = None,
    input_shapes: list[tuple[int, ...]] | None = None,
    force_triton: bool = True,
    extract_compiler_context: bool = False,
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
        extract_compiler_context=extract_compiler_context,
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

        compiler_context_data = data.get("compiler_context")

        results = []
        for src in data["sources"]:
            triton_kernels = _extract_triton_jit_blocks(src)
            call_source = _extract_function(src, "call")
            benchmark_source = _extract_function(src, "benchmark_compiled_module")

            result = {
                "inductor_source": src,
                "triton_kernels": triton_kernels,
                "call_source": call_source,
                "benchmark_source": benchmark_source,
                "callable_info": info,
                "benchmark_kernel_files": data.get("benchmark_kernel_files", []),
            }
            if compiler_context_data is not None:
                result["compiler_context"] = compiler_context_data
            results.append(result)

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
