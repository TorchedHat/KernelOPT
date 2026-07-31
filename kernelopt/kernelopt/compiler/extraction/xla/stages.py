"""Extract compilation metadata from JAX/XLA stages API.

JAX exposes a clean three-stage pipeline:
  jaxpr → StableHLO (lowered) → Optimized HLO (compiled)

Each stage provides structured metadata that tells the planner
what the computation looks like before and after XLA's optimizer.

References:
    - Lowered/Compiled classes: jax/_src/stages.py
    - cost_analysis: returns FLOPs, bytes accessed, utilization
    - memory_analysis: generated code size, argument/output/temp sizes
"""

from __future__ import annotations

from typing import Any, Callable


def extract_xla_stages(
    fn: Callable,
    *args: Any,
) -> dict:
    """Extract StableHLO, optimized HLO, cost, and memory from a JAX function.

    Args:
        fn: A JAX-traceable function (not yet jitted).
        *args: Example inputs for tracing.

    Returns:
        Dict with stablehlo_text, optimized_hlo_text, cost_analysis,
        memory_analysis. Empty dict on failure.
    """
    try:
        import jax
    except ImportError:
        return {"error": "JAX not installed"}

    try:
        jitted = jax.jit(fn)
        lowered = jitted.lower(*args)

        stablehlo_text = ""
        try:
            stablehlo_text = lowered.as_text()
        except Exception:
            pass

        compiled = lowered.compile()

        optimized_hlo_text = ""
        try:
            optimized_hlo_text = compiled.as_text()
        except Exception:
            pass

        cost = {}
        try:
            cost_raw = compiled.cost_analysis()
            if isinstance(cost_raw, list) and cost_raw:
                cost = cost_raw[0]
            elif isinstance(cost_raw, dict):
                cost = cost_raw
        except Exception:
            pass

        memory = {}
        try:
            mem_stats = compiled.memory_analysis()
            if mem_stats:
                memory = {
                    "generated_code_size_bytes": getattr(mem_stats, "generated_code_size_in_bytes", 0),
                    "argument_size_bytes": getattr(mem_stats, "argument_size_in_bytes", 0),
                    "output_size_bytes": getattr(mem_stats, "output_size_in_bytes", 0),
                    "temp_size_bytes": getattr(mem_stats, "temp_size_in_bytes", 0),
                }
        except Exception:
            pass

        hlo_proto = b""
        try:
            exe = compiled.runtime_executable()
            modules = exe.hlo_modules()
            if modules:
                hlo_proto = modules[0].as_serialized_hlo_module_proto()
        except Exception:
            pass

        return {
            "stablehlo_text": stablehlo_text[:10000],
            "optimized_hlo_text": optimized_hlo_text[:10000],
            "cost_analysis": cost,
            "memory_analysis": memory,
            "hlo_proto_size": len(hlo_proto),
        }

    except Exception as e:
        return {"error": str(e)}


def extract_jaxpr(fn: Callable, *args: Any) -> dict:
    """Extract jaxpr (JAX's functional SSA IR) for a function.

    Returns per-equation metadata: primitive name, input/output shapes.
    """
    try:
        import jax
    except ImportError:
        return {"error": "JAX not installed"}

    try:
        closed_jaxpr = jax.make_jaxpr(fn)(*args)
        jaxpr = closed_jaxpr.jaxpr

        equations = []
        for eqn in jaxpr.eqns:
            in_shapes = []
            for v in eqn.invars:
                if hasattr(v, "aval") and hasattr(v.aval, "shape"):
                    in_shapes.append(tuple(int(s) for s in v.aval.shape))

            out_shapes = []
            for v in eqn.outvars:
                if hasattr(v, "aval") and hasattr(v.aval, "shape"):
                    out_shapes.append(tuple(int(s) for s in v.aval.shape))

            equations.append({
                "primitive": eqn.primitive.name,
                "input_shapes": in_shapes,
                "output_shapes": out_shapes,
                "params": {
                    k: str(v)[:100] for k, v in eqn.params.items()
                    if k in ("dimension_numbers", "axes", "shape", "broadcast_dimensions")
                },
            })

        input_shapes = [
            tuple(int(s) for s in v.aval.shape)
            for v in jaxpr.invars
            if hasattr(v, "aval") and hasattr(v.aval, "shape")
        ]

        return {
            "num_equations": len(equations),
            "input_shapes": input_shapes,
            "equations": equations[:50],
        }

    except Exception as e:
        return {"error": str(e)}
