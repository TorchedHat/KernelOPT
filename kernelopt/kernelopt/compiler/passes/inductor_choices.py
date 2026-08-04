"""Profile-guided overrides for Inductor's heuristic decisions.

Subclasses the actual InductorChoices class to inject profiling-derived
tile sizes, reduction strategies, and warp counts.

Set via: config.inductor_choices_class = KernelOPTChoices

References:
    - InductorChoices: torch/_inductor/codegen/choices.py
    - config.inductor_choices_class: torch/_inductor/config.py:957
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_base_class():
    """Import InductorChoices at runtime — torch is optional."""
    try:
        from torch._inductor.choices import InductorChoices
        return InductorChoices
    except ImportError:
        try:
            from torch._inductor.codegen.choices import InductorChoices
            return InductorChoices
        except ImportError:
            return object


class ProfileGuidedChoices(_get_base_class()):
    """KernelOPT overrides for Inductor's heuristic decisions.

    Inherits from InductorChoices and overrides specific methods
    when profiling data is available. Falls back to parent behavior
    when no profiling guidance exists.
    """

    def __init__(
        self,
        profiling_data: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._profiling_data = profiling_data or {}
        self._overrides: dict[str, Any] = {}

    def uuid(self) -> str:
        """Cache key for Inductor's config serialization."""
        data = json.dumps(self._profiling_data, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

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

        return super().should_use_persistent_reduction(numel, reduction_numel)


def install_choices(profiling_data: Optional[dict] = None) -> bool:
    """Install KernelOPT choices class into Inductor config.

    Returns True if successful.
    """
    try:
        import torch._inductor.config as cfg
        try:
            from torch._inductor.choices import InductorChoices
        except ImportError:
            from torch._inductor.codegen.choices import InductorChoices
    except ImportError:
        logger.debug("InductorChoices not available in this PyTorch version")
        return False

    if not hasattr(cfg, "inductor_choices_class"):
        logger.debug("inductor_choices_class config not available")
        return False

    class KernelOPTChoices(ProfileGuidedChoices):
        def __init__(self):
            super().__init__(profiling_data=profiling_data)

    cfg.inductor_choices_class = KernelOPTChoices
    logger.debug("Installed KernelOPTChoices as Inductor choices class")
    return True
