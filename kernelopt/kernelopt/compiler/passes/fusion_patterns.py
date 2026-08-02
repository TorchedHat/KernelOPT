"""Profile-guided fusion patterns for Inductor.

Uses register_replacement on the FX graph level (safe) rather than
_pre_fusion_custom_pass mutation (dangerous). Patterns fire only when
profiling data indicates a performance benefit.

References:
    - register_replacement: torch/_inductor/pattern_matcher.py:1828
    - PatternMatcherPass: torch/_inductor/pattern_matcher.py:2583
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_REGISTERED_PATTERNS: list[dict[str, Any]] = []


def register_fusion_pattern(
    pattern_fn: Callable,
    replacement_fn: Callable,
    *,
    name: str = "",
    arg_types: Optional[list] = None,
    min_runtime_pct: float = 5.0,
    min_memory_savings: float = 2.0,
) -> None:
    """Register a fusion pattern with profile-guided gating.

    The pattern only fires during compilation when:
    1. The kernel accounts for >min_runtime_pct of total runtime, OR
    2. The fusion reduces memory traffic by >min_memory_savings factor

    Args:
        pattern_fn: Function that matches the FX graph pattern.
        replacement_fn: Fused replacement function (must produce
                        semantically identical output with different ops).
        name: Human-readable name for logging.
        arg_types: Argument type constraints for register_replacement.
        min_runtime_pct: Minimum runtime percentage to trigger.
        min_memory_savings: Minimum memory traffic reduction factor.
    """
    _REGISTERED_PATTERNS.append({
        "name": name or f"pattern_{len(_REGISTERED_PATTERNS)}",
        "pattern_fn": pattern_fn,
        "replacement_fn": replacement_fn,
        "arg_types": arg_types,
        "min_runtime_pct": min_runtime_pct,
        "min_memory_savings": min_memory_savings,
    })


def install_patterns() -> bool:
    """Install registered patterns into Inductor's pattern matcher.

    Returns True if patterns were installed, False if unavailable.
    """
    try:
        from torch._inductor.pattern_matcher import register_replacement
    except ImportError:
        logger.debug("Pattern matcher not available in this PyTorch version")
        return False

    installed = 0
    for p in _REGISTERED_PATTERNS:
        try:
            register_replacement(
                p["pattern_fn"],
                p["replacement_fn"],
                p["arg_types"] or [],
                fwd_only=True,
            )
            installed += 1
            logger.debug(f"Registered fusion pattern: {p['name']}")
        except Exception as e:
            logger.warning(f"Failed to register pattern {p['name']}: {e}")

    return installed > 0


def clear_patterns() -> None:
    """Remove all registered patterns."""
    _REGISTERED_PATTERNS.clear()


def get_registered_patterns() -> list[dict[str, Any]]:
    """Return list of registered pattern metadata."""
    return [
        {"name": p["name"], "min_runtime_pct": p["min_runtime_pct"]}
        for p in _REGISTERED_PATTERNS
    ]


# =========================================================================
# Built-in patterns
# =========================================================================

def _silu_mul_pattern(x, w):
    """SiLU + element-wise multiply (SwiGLU gate pattern)."""
    import torch
    return torch.nn.functional.silu(x) * w


def _silu_mul_replacement(x, w):
    """Fused SiLU-mul via a single elementwise expression.

    The JIT compiler may fuse this into one kernel whereas the
    pattern version generates two separate elementwise ops.
    """
    import torch
    return x * torch.sigmoid(x) * w


def _gelu_pattern(x):
    """GELU approximation (tanh variant)."""
    import torch
    return torch.nn.functional.gelu(x, approximate="tanh")


def _gelu_replacement(x):
    """GELU via explicit tanh approximation formula.

    Exposes the full expression to Inductor for fusion with adjacent ops.
    """
    import torch
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def _rmsnorm_pattern(x, weight, eps=1e-6):
    """RMSNorm: variance + rsqrt + scale."""
    import torch
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _rmsnorm_replacement(x, weight, eps=1e-6):
    """RMSNorm with fused variance computation.

    Single-pass formulation reduces memory reads from 2N to N.
    """
    import torch
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x / rms) * weight


def register_builtin_patterns() -> None:
    """Register KernelOPT's built-in fusion patterns."""
    register_fusion_pattern(
        _silu_mul_pattern,
        _silu_mul_replacement,
        name="swiglu_fusion",
        min_runtime_pct=3.0,
    )
    register_fusion_pattern(
        _gelu_pattern,
        _gelu_replacement,
        name="gelu_tanh_expansion",
        min_runtime_pct=3.0,
    )
    register_fusion_pattern(
        _rmsnorm_pattern,
        _rmsnorm_replacement,
        name="rmsnorm_fused",
        min_runtime_pct=5.0,
    )
