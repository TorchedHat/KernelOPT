"""Build the subprocess script for compiler context extraction.

Generates a self-contained Python script that:
1. Installs Dynamo pre-grad capture hook
2. Installs Inductor fusion capture hooks
3. Installs Triton compilation listener
4. Runs torch.compile + run_and_get_code
5. Walks IR nodes after compilation
6. Reads autotune cache
7. Parses codegen strategies from generated code
8. Collects Triton compilation metadata
9. Serializes everything as JSON between sentinel markers

The script runs in an isolated subprocess to avoid polluting the
main KernelOPT process with torch.compile global state.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


def build_extraction_script(
    pytorch_path: str,
    *,
    entry_point: str | None = None,
    timeout_seconds: int = 300,
) -> str:
    """Build the complete extraction subprocess script."""
    pytorch_dir = str(Path(pytorch_path).parent.resolve())
    pytorch_module = Path(pytorch_path).stem
    entry_class = entry_point or "Model"

    return textwrap.dedent(f"""\
import json, sys, os, signal, re, threading
import torch

signal.alarm({timeout_seconds})
torch.cuda.set_device(0)

pytorch_version = torch.__version__
extraction_warnings = []

# ================================================================
# Step 1: Install Dynamo pre-grad capture hook
# ================================================================
_pre_decomp_ops = []
try:
    import torch._inductor.config as _ind_config
    if hasattr(_ind_config, "pre_grad_custom_pass"):
        def _capture_pre_decomp(graph):
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
                    _pre_decomp_ops.append({{
                        "op_name": op_name,
                        "input_shapes": tuple(in_shapes),
                        "output_shape": out_shape,
                        "is_compound": any(c in op_name.lower() for c in (
                            "layer_norm", "group_norm", "batch_norm", "softmax",
                            "gelu", "silu", "linear", "conv",
                        )),
                    }})
            return graph
        _ind_config.pre_grad_custom_pass = _capture_pre_decomp
    else:
        extraction_warnings.append("pre_grad_custom_pass not available")
except Exception as e:
    extraction_warnings.append(f"Dynamo capture setup failed: {{e}}")

# ================================================================
# Step 2a: Install post_grad_custom_pre_pass for IR node capture
# ================================================================
# post_grad_custom_pre_pass receives an FX GraphModule (post-decomposition).
# We extract per-node metadata from node.meta["val"] (FakeTensor with shape/dtype)
# and node.meta["tensor_meta"] when available.
_ir_nodes = []
try:
    import torch._inductor.config as _ind_config
    if hasattr(_ind_config, "post_grad_custom_pre_pass"):
        def _capture_ir_nodes(graph):
            try:
                _nodes = graph.nodes if hasattr(graph, 'nodes') else (graph.graph.nodes if hasattr(graph, 'graph') else [])
                for node in _nodes:
                    if node.op != "call_function":
                        continue
                    target = node.target
                    op_name = getattr(target, "__name__", str(target))

                    val = node.meta.get("val")
                    if val is None:
                        continue

                    if hasattr(val, "shape"):
                        shape = tuple(int(s) for s in val.shape)
                        dtype_str = str(val.dtype)
                        bpe = val.element_size() if hasattr(val, "element_size") else 4
                        total_elements = 1
                        for s in shape:
                            total_elements *= s

                        is_reduction = "reduce" in op_name or "sum" in op_name or "mean" in op_name or "var" in op_name or "amax" in op_name
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
                                    rnumel = 1
                                    for r in red_ranges:
                                        rnumel *= r
                                    padded = 1
                                    while padded < rnumel:
                                        padded *= 2
                                    reduction = {{
                                        "reduction_type": op_name.split(".")[-1] if "." in op_name else op_name,
                                        "reduction_ranges": red_ranges,
                                        "reduction_hint": "DEFAULT",
                                        "src_dtype": str(input_val.dtype),
                                        "dst_dtype": dtype_str,
                                        "is_persistent": False,
                                        "is_cooperative": False,
                                        "rnumel_padded": padded,
                                        "wasted_sram_fraction": round((padded - rnumel) / padded, 4) if padded > 0 else 0.0,
                                    }}

                        _ir_nodes.append({{
                            "buffer_name": node.name,
                            "node_type": "Reduction" if is_reduction else "Pointwise",
                            "iteration_ranges": shape,
                            "total_elements": total_elements,
                            "dtype": dtype_str,
                            "bytes_per_element": bpe,
                            "total_memory_bytes": total_elements * bpe,
                            "num_reads": len(node.args),
                            "num_writes": 1,
                            "read_buffer_names": tuple(
                                a.name for a in node.args if hasattr(a, "name")
                            ),
                            "write_buffer_names": (node.name,),
                            "reduction": reduction,
                        }})
            except Exception as _e:
                extraction_warnings.append(f"IR capture in pass failed: {{_e}}")
            return graph
        _ind_config.post_grad_custom_pre_pass = _capture_ir_nodes
    else:
        extraction_warnings.append("post_grad_custom_pre_pass not available")
except Exception as e:
    extraction_warnings.append(f"IR capture hook setup failed: {{e}}")

# ================================================================
# Step 2b: Install Inductor fusion capture hooks
# ================================================================
_pre_nodes = []
_post_nodes = []
_why_no_fuse = []
_fusion_hooks_ok = False
try:
    import torch._inductor.config as _ind_config
    from torch._inductor.scheduler import WhyNoFuse

    def _pre_pass(nodes):
        _pre_nodes.clear()
        _pre_errors = 0
        for n in nodes:
            try:
                _pre_nodes.append({{
                    "name": n.get_name(),
                    "names": list(n.get_operation_names()),
                    "is_reduction": n.is_reduction(),
                }})
            except Exception:
                _pre_errors += 1
        if _pre_errors:
            extraction_warnings.append(f"Skipped {{_pre_errors}} nodes in pre-fusion capture")
        return nodes

    def _post_pass(nodes):
        from torch._inductor.scheduler import FusedSchedulerNode
        _post_nodes.clear()
        _post_errors = 0
        for n in nodes:
            try:
                entry = {{
                    "name": n.get_name(),
                    "names": list(n.get_operation_names()),
                    "is_fused": isinstance(n, FusedSchedulerNode),
                }}
                if isinstance(n, FusedSchedulerNode):
                    entry["snodes"] = [sn.get_name() for sn in n.snodes]
                _post_nodes.append(entry)
            except Exception:
                _post_errors += 1
        if _post_errors:
            extraction_warnings.append(f"Skipped {{_post_errors}} nodes in post-fusion capture")
        return nodes

    if hasattr(_ind_config, "_pre_fusion_custom_pass"):
        _ind_config._pre_fusion_custom_pass = _pre_pass
        _ind_config._post_fusion_custom_pass = _post_pass
        _fusion_hooks_ok = True

    _orig_wnf = WhyNoFuse.__call__
    def _patched_wnf(self, reason, *args):
        try:
            formatted = reason % args if args else reason
        except (TypeError, ValueError):
            formatted = reason
        _why_no_fuse.append({{
            "node1": self.name1,
            "node2": self.name2,
            "reason": formatted,
        }})
        return _orig_wnf(self, reason, *args)
    WhyNoFuse.__call__ = _patched_wnf
except Exception as e:
    extraction_warnings.append(f"Fusion hooks failed: {{e}}")

# ================================================================
# Step 3: Install Triton compilation listener
# ================================================================
_triton_compilations = []
try:
    import triton
    def _on_triton_compile(*, src, metadata, metadata_group, times, cache_hit):
        entry = {{
            "kernel_name": metadata.get("name", ""),
            "cache_hit": cache_hit,
            "num_warps": metadata.get("num_warps", 4),
            "num_stages": metadata.get("num_stages", 1),
            "shared_bytes": metadata.get("shared", 0),
            "ir_files": dict(metadata_group) if metadata_group else {{}},
            "compilation_time_us": 0.0,
        }}
        if times:
            total = getattr(times, "ir_initialization", 0)
            for _, dur in getattr(times, "lowering_stages", []):
                total += dur
            total += getattr(times, "store_results", 0)
            entry["compilation_time_us"] = total
        _triton_compilations.append(entry)
    triton.knobs.compilation.listener = _on_triton_compile
except Exception as e:
    extraction_warnings.append(f"Triton listener failed: {{e}}")

# ================================================================
# Step 4: Load model and compile
# ================================================================
sys.path.insert(0, {pytorch_dir!r})
import importlib.util
_spec = importlib.util.spec_from_file_location("_model", {pytorch_path!r})
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_model"] = _mod
_spec.loader.exec_module(_mod)

_model_cls = getattr(_mod, {entry_class!r}, None) or getattr(_mod, "Model", None)
_get_inputs = getattr(_mod, "get_inputs", None)
_get_init = getattr(_mod, "get_init_inputs", None)

if _model_cls is None or _get_inputs is None:
    print("__COMPILER_CONTEXT_START__")
    print(json.dumps({{"extraction_success": False, "extraction_warnings": ["No Model/get_inputs"]}}))
    print("__COMPILER_CONTEXT_END__")
    sys.exit(0)

_init_args = _get_init() if _get_init else []
_model = _model_cls(*_init_args).cuda().eval()
_inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in _get_inputs()]

_ind_config.max_autotune = True
_ind_config.max_autotune_gemm = True
_ind_config.triton.unique_kernel_names = True

# Reset Dynamo + Inductor caches to ensure fresh compilation
# (otherwise Triton listener sees cache_hit=True and we miss IR files)
torch._dynamo.reset()
try:
    import triton.runtime.cache as _tri_cache
    if hasattr(_tri_cache, "default_cache_manager"):
        pass  # cache manager reset not needed — listener fires even on cache hit
except Exception as _e:
    extraction_warnings.append(f"Triton cache check failed: {{_e}}")

from torch._inductor.utils import run_and_get_code
_compiled = torch.compile(_model)
with torch.no_grad():
    _, _source_codes = run_and_get_code(_compiled, *_inputs)
_generated_code = _source_codes[0] if _source_codes else ""

# ================================================================
# Step 5: IR nodes already captured by post_grad_custom_pre_pass (Step 2a)
# No post-compilation walk needed — V.graph is cleaned up by this point.
# ================================================================
# _ir_nodes is populated by the _capture_ir_nodes pass during compilation.

# _ir_nodes is populated by the _capture_ir_nodes pass installed in Step 2a.
# The pass runs DURING compilation when the graph is still alive.
# No post-compilation fallback needed — V.graph is cleaned up by this point.
if not _ir_nodes:
    extraction_warnings.append("IR nodes: post_grad_custom_pre_pass did not fire or graph had no ComputedBuffers")

# ================================================================
# Step 6: Collect fusion data
# ================================================================
_structural_kw = {{"device mismatch", "extern but not a triton", "different devices",
                   "not a GPU node", "will create cycle", "data dependency"}}
_fusion_data = {{"fusion_groups": [], "fusion_rejections": [], "intermediate_buffers": []}}
if _fusion_hooks_ok:
    _fusion_data["fusion_groups"] = [n for n in _post_nodes if n.get("is_fused")]
    for rej in _why_no_fuse[:50]:
        reason = rej.get("reason", "")
        is_structural = any(kw in reason.lower() for kw in _structural_kw)
        is_heuristic = any(kw in reason.lower() for kw in ("score", "threshold", "too many"))
        _fusion_data["fusion_rejections"].append({{
            "node1_name": rej["node1"],
            "node2_name": rej["node2"],
            "reason": reason,
            "near_threshold": is_heuristic and not is_structural,
            "fixable_by_manual_fusion": not is_structural,
        }})

# ================================================================
# Step 7: Read autotune cache
# ================================================================
_autotune_state = []
try:
    from torch._inductor.select_algorithm import get_algorithm_selector_cache
    _cache = get_algorithm_selector_cache()
    _local = getattr(_cache, "local_cache", getattr(_cache, "_local_cache", None))
    if _local:
        for key, val in _local.items():
            if not isinstance(val, dict):
                continue
            timings = []
            for ck, tv in val.items():
                if isinstance(tv, (int, float)) and tv < float("inf"):
                    timings.append({{"config": str(ck)[:200], "timing_ms": float(tv)}})
            if timings:
                timings.sort(key=lambda x: x["timing_ms"])
                _autotune_state.append({{
                    "operation": str(key)[:300],
                    "num_configs_tried": len(timings),
                    "winner_type": "",
                    "winner_timing_ms": timings[0]["timing_ms"],
                    "runner_up_type": "",
                    "runner_up_timing_ms": timings[1]["timing_ms"] if len(timings) > 1 else 0.0,
                    "all_timings": tuple(timings[:20]),
                }})
except Exception as e:
    extraction_warnings.append(f"Autotune capture failed: {{e}}")

# ================================================================
# Step 8: Parse codegen strategies from generated code
# ================================================================
_codegen_strategies = []
_cat_map = {{"per": "persistent_reduction", "red": "streaming_reduction",
            "poi": "pointwise", "tem": "template", "for": "foreach",
            "spl": "split_scan", "unk": "unknown"}}
_tq = chr(39) * 3
_kpat = r"(triton_\\w+)\\s*=\\s*async_compile\\.triton\\(.*?" + _tq + r"(.*?)" + _tq
for _kname, _ksrc in re.findall(_kpat, _generated_code, re.DOTALL):
    _prefix = _kname.split("_")[1][:3] if "_" in _kname else "unk"
    _cat = _cat_map.get(_prefix, "unknown")
    _persistent = _cat == "persistent_reduction"
    _cooperative = ("RSPLIT" in _ksrc or "rsplit" in _ksrc) and _prefix == "red"
    if _cooperative:
        _cat = "cooperative_reduction"
    _no_x = "XBLOCK" not in _ksrc
    if not _no_x:
        _no_x = bool(re.search(r"XBLOCK\\s*:\\s*tl\\.constexpr\\s*=\\s*1\\b", _ksrc))
    _blk = {{}}
    for _bm in re.finditer(r'"(XBLOCK|RBLOCK|YBLOCK|BLOCK_M|BLOCK_N|BLOCK_K)"\\s*:\\s*(\\d+)', _ksrc):
        _blk[_bm.group(1)] = int(_bm.group(2))
    for _bm in re.finditer(r'(XBLOCK|RBLOCK|YBLOCK)\\s*:\\s*tl\\.constexpr\\s*=\\s*(\\d+)', _ksrc):
        _blk[_bm.group(1)] = int(_bm.group(2))
    _nw = re.search(r"num_warps\\s*=\\s*(\\d+)", _ksrc)
    _ns = re.search(r"num_stages\\s*=\\s*(\\d+)", _ksrc)
    _hint = re.search(r"reduction_hint\\s*=\\s*ReductionHint\\.(\\w+)", _ksrc)
    _codegen_strategies.append({{
        "kernel_name": _kname,
        "kernel_category": _cat,
        "persistent_reduction": _persistent,
        "cooperative_reduction": _cooperative,
        "no_x_dim": _no_x,
        "uses_block_pointers": "block_ptr" in _ksrc,
        "num_reduction_dims": len(re.findall(r"r\\d+_numel|rnumel", _ksrc[:500])) or (1 if _persistent or _cooperative or _prefix == "red" else 0),
        "reduction_hint": _hint.group(1) if _hint else "N/A",
        "inductor_block_sizes": _blk,
        "inductor_num_warps": int(_nw.group(1)) if _nw else 4,
        "inductor_num_stages": int(_ns.group(1)) if _ns else 1,
    }})

# ================================================================
# Step 9: Collect Triton compilation metadata + read IR text
# ================================================================
_triton_info = []
for _tc in _triton_compilations:
    _ttir = ""
    _ttgir = ""
    for _fk, _fp in _tc.get("ir_files", {{}}).items():
        try:
            if _fk.endswith(".ttir"):
                _ttir = open(_fp).read()[:5000]
            elif _fk.endswith(".ttgir"):
                _ttgir = open(_fp).read()[:5000]
        except (OSError, IOError):
            pass
    _dot_count = len(re.findall(r"tt\\.dot\\b", _ttir)) if _ttir else 0
    _load_count = len(re.findall(r"tt\\.load\\b", _ttir)) if _ttir else 0
    _store_count = len(re.findall(r"tt\\.store\\b", _ttir)) if _ttir else 0
    _triton_info.append({{
        "kernel_name": _tc["kernel_name"],
        "num_warps": _tc["num_warps"],
        "num_stages": _tc["num_stages"],
        "shared_bytes": _tc["shared_bytes"],
        "n_regs": 0,
        "n_spills": 0,
        "n_max_threads": 0,
        "ttir_text": _ttir,
        "ttgir_text": _ttgir,
        "compilation_time_us": _tc["compilation_time_us"],
        "dot_count": _dot_count,
        "load_count": _load_count,
        "store_count": _store_count,
        "has_async_copy": "async_copy" in _ttir or "async_copy" in _ttgir,
        "has_warp_specialize": "warp_specialize" in _ttir or "warp_specialize" in _ttgir,
        "mma_type": "wgmma" if "wgmma" in (_ttir + _ttgir).lower() else ("mma" if "nvidia_mma" in (_ttir + _ttgir) else ("dot" if _dot_count > 0 else "")),
    }})

# ================================================================
# Step 10: AOT decomposition diff
# ================================================================
_aot_decomps = []
_known_decomps = {{
    "layer_norm": ("var_mean", "rsqrt", "sub", "mul", "add"),
    "native_layer_norm": ("var_mean", "rsqrt", "sub", "mul", "add"),
    "gelu": ("mul", "erf", "add", "mul"),
    "group_norm": ("reshape", "var_mean", "rsqrt", "sub", "mul", "add", "reshape"),
    "native_group_norm": ("reshape", "var_mean", "rsqrt", "sub", "mul", "add", "reshape"),
    "batch_norm": ("mean", "var", "sub", "rsqrt", "mul", "add"),
    "softmax": ("max", "sub", "exp", "sum", "div"),
    "linear": ("addmm",),
}}
for _op in _pre_decomp_ops:
    if not _op.get("is_compound"):
        continue
    _name = _op["op_name"].lower()
    for _dk, _dv in _known_decomps.items():
        if _dk in _name:
            _aot_decomps.append({{
                "original_op": _op["op_name"],
                "decomposed_into": _dv,
                "num_intermediates": max(0, len(_dv) - 1),
                "decomposition_source": "inductor_decompositions",
            }})
            break

# ================================================================
# Output
# ================================================================
_output = {{
    "pytorch_version": pytorch_version,
    "extraction_success": True,
    "extraction_warnings": extraction_warnings,
    "dynamo_info": {{
        "original_ops": _pre_decomp_ops,
        "graph_breaks": [],
        "num_subgraphs": 1,
        "unsupported_ops": [],
    }},
    "aot_decomps": _aot_decomps,
    "ir_nodes": _ir_nodes,
    "fusion_data": _fusion_data,
    "autotune_entries": _autotune_state,
    "codegen_strategies": _codegen_strategies,
    "triton_info": _triton_info,
}}
print("__COMPILER_CONTEXT_START__")
print(json.dumps(_output))
print("__COMPILER_CONTEXT_END__")
""")
