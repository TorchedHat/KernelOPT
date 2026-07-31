"""KernelOPT autotuning plugin for Inductor's CachingAutotuner.

Injects profile-guided configs before Inductor's autotuning loop,
prunes known-bad configs, and explores neighbors that Inductor's
coordinate descent misses.

References:
    - CachingAutotunerPlugin: torch/_inductor/runtime/triton_heuristics.py:452
    - IncrementalAutotunePlugin: FB-internal (PR #182010)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class KernelOPTAutotunePlugin:
    """Autotuning plugin that injects profile-guided configs.

    Three hooks:
      pre_compile: inject known-good configs before compilation
      pre_dispatch: filter configs before GPU dispatch
      pre_autotune: modify config list before autotuning loop
    """

    def __init__(
        self,
        profiling_data: Optional[dict] = None,
        avoid_configs: Optional[list[dict]] = None,
    ) -> None:
        self._profiling_data = profiling_data or {}
        self._avoid_configs = avoid_configs or []
        self._injected_count = 0
        self._pruned_count = 0

    def pre_compile(
        self,
        kernel_name: str,
        configs: list[dict],
    ) -> list[dict]:
        """Inject profile-guided configs before compilation.

        Adds configs derived from NCU metrics (occupancy-optimal
        num_warps, register-safe block sizes).
        """
        suggested = self._profiling_data.get("suggested_configs", {})
        if kernel_name in suggested:
            for cfg in suggested[kernel_name]:
                if cfg not in configs:
                    configs.append(cfg)
                    self._injected_count += 1
        return configs

    def pre_dispatch(
        self,
        kernel_name: str,
        configs: list[dict],
    ) -> list[dict]:
        """Filter configs before dispatch to GPU.

        Removes configs that previously caused register spills
        or had occupancy below 25%.
        """
        if not self._avoid_configs:
            return configs

        filtered = []
        for cfg in configs:
            if self._should_avoid(cfg):
                self._pruned_count += 1
                continue
            filtered.append(cfg)

        return filtered if filtered else configs

    def pre_autotune(
        self,
        kernel_name: str,
        configs: list[dict],
    ) -> list[dict]:
        """Add diagonal neighbor configs that coordinate descent misses.

        Inductor's coordinate descent changes one parameter at a time.
        This adds configs that simultaneously adjust num_warps + BLOCK_M.
        """
        neighbors = self._profiling_data.get("neighbor_configs", {})
        if kernel_name in neighbors:
            for cfg in neighbors[kernel_name]:
                if cfg not in configs:
                    configs.append(cfg)
        return configs

    def _should_avoid(self, config: dict) -> bool:
        """Check if a config matches any known-bad configuration."""
        for bad in self._avoid_configs:
            if all(config.get(k) == v for k, v in bad.items()):
                return True
        return False

    @property
    def stats(self) -> dict:
        """Return plugin statistics."""
        return {
            "injected": self._injected_count,
            "pruned": self._pruned_count,
        }


def install_autotune_plugin(
    profiling_data: Optional[dict] = None,
    avoid_configs: Optional[list[dict]] = None,
) -> Optional[KernelOPTAutotunePlugin]:
    """Install the autotuning plugin into Inductor's config.

    Returns the plugin instance, or None if unavailable.
    """
    plugin = KernelOPTAutotunePlugin(
        profiling_data=profiling_data,
        avoid_configs=avoid_configs,
    )

    try:
        import torch._inductor.config as cfg
        if hasattr(cfg, "triton") and hasattr(cfg.triton, "autotune_caching_plugin"):
            cfg.triton.autotune_caching_plugin = plugin
            return plugin
    except ImportError:
        pass

    logger.debug("CachingAutotunerPlugin not available in this PyTorch version")
    return None
