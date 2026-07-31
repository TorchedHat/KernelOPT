"""JAX/XLA compilation metadata extraction.

Provides extraction of StableHLO, optimized HLO, fusion decisions,
autotuning results, and Pallas kernel injection bridge.

All imports are lazy — JAX is an optional dependency.
"""

from kernelopt.compiler.extraction.xla.stages import (
    extract_xla_stages,
    extract_jaxpr,
)
from kernelopt.compiler.extraction.xla.fusion_dump import (
    parse_fusion_dump_text,
    setup_xla_dump_flags,
)
from kernelopt.compiler.extraction.xla.autotune import (
    parse_autotune_results,
    parse_per_fusion_cache,
    setup_autotune_dump_flags,
)
from kernelopt.compiler.extraction.xla.pallas_bridge import (
    create_pallas_kernel_wrapper,
    check_pallas_available,
)

__all__ = [
    "extract_xla_stages",
    "extract_jaxpr",
    "parse_fusion_dump_text",
    "setup_xla_dump_flags",
    "parse_autotune_results",
    "parse_per_fusion_cache",
    "setup_autotune_dump_flags",
    "create_pallas_kernel_wrapper",
    "check_pallas_available",
]
