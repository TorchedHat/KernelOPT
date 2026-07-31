"""Profile-guided overrides for Inductor's heuristic decisions.

Subclasses InductorChoices to inject profiling-derived tile sizes,
reduction strategies, and warp counts instead of Inductor's defaults.

Set via: config.inductor_choices_class = KernelOPTChoices

References:
    - InductorChoices: torch/_inductor/codegen/choices.py:442
    - config.inductor_choices_class: torch/_inductor/config.py:957
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProfileGuidedChoices:
    """KernelOPT overrides for Inductor's heuristic decisions.

    Wraps the default InductorChoices and overrides specific decisions
    when profiling data is available. Falls back to default behavior
    when no profiling guidance exists.
    """

    def __init__(
        self,
        profiling_data: Optional[dict] = None,
        base_choices: Any = None,
    ) -> None:
        self._profiling_data = profiling_data or {}
        self._base = base_choices
        self._overrides: dict[str, Any] = {}

    def set_override(self, key: str, value: Any) -> None:
        """Set a specific override for a heuristic decision."""
        self._overrides[key] = value

    def should_use_persistent_reduction(
        self,
        numel: int,
        reduction_numel: int,
    ) -> bool:
        """Decide between persistent and streaming reduction.

        Uses profiling data when available: persistent is better when
        reduction_numel fits in shared memory and the kernel is
        compute-bound (not memory-bound).
        """
        if "persistent_reduction" in self._overrides:
            return self._overrides["persistent_reduction"]

        profiling = self._profiling_data.get("reduction_analysis", {})
        if profiling.get("force_persistent"):
            return True
        if profiling.get("force_streaming"):
            return False

        if self._base and hasattr(self._base, "should_use_persistent_reduction"):
            return self._base.should_use_persistent_reduction(numel, reduction_numel)

        return reduction_numel <= 8192

    def get_tile_sizes(
        self,
        kernel_name: str,
        input_shapes: list[tuple[int, ...]],
    ) -> Optional[dict[str, int]]:
        """Get profile-guided tile sizes for a kernel.

        Returns None to fall back to Inductor defaults.
        """
        tile_overrides = self._profiling_data.get("tile_sizes", {})
        if kernel_name in tile_overrides:
            return tile_overrides[kernel_name]
        return None

    def get_num_warps(
        self,
        kernel_name: str,
        numel: int,
    ) -> Optional[int]:
        """Get profile-guided num_warps for a kernel.

        Returns None to fall back to Inductor defaults.
        """
        warp_overrides = self._profiling_data.get("num_warps", {})
        if kernel_name in warp_overrides:
            return warp_overrides[kernel_name]

        if numel >= 65536:
            return 8
        elif numel >= 4096:
            return 4
        return None


def install_choices(profiling_data: Optional[dict] = None) -> bool:
    """Install KernelOPT choices into Inductor config.

    Returns True if successful.
    """
    try:
        import torch._inductor.config as cfg
        if hasattr(cfg, "inductor_choices_class"):
            choices = ProfileGuidedChoices(profiling_data=profiling_data)
            cfg.inductor_choices_class = type(
                "KernelOPTChoices",
                (choices.__class__,),
                {
                    "should_use_persistent_reduction": choices.should_use_persistent_reduction,
                    "get_tile_sizes": choices.get_tile_sizes,
                    "get_num_warps": choices.get_num_warps,
                },
            )
            return True
    except ImportError:
        pass

    return False
