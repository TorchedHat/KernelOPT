"""KernelOPT autotuning plugin for Inductor's CachingAutotuner.

Injects profile-guided configs before Inductor's autotuning loop,
prunes known-bad configs, and explores neighbors that Inductor's
coordinate descent misses.

References:
    - CachingAutotuner: torch/_inductor/runtime/triton_heuristics.py
    - Autotuning flow: CachingAutotuner.run() → precompile() → autotune()
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
        """Inject profile-guided configs before compilation."""
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
        """Filter configs before dispatch to GPU."""
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
        """Add diagonal neighbor configs that coordinate descent misses."""
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
    """Install the autotuning plugin.

    The plugin is stored and accessed via the CachingAutotuner's
    plugin registry rather than a config flag.

    Returns the plugin instance, or None if unavailable.
    """
    plugin = KernelOPTAutotunePlugin(
        profiling_data=profiling_data,
        avoid_configs=avoid_configs,
    )

    try:
        from torch._inductor.runtime import triton_heuristics
        if hasattr(triton_heuristics, "CachingAutotuner"):
            CachingAutotuner = triton_heuristics.CachingAutotuner
            if hasattr(CachingAutotuner, "plugin"):
                CachingAutotuner.plugin = plugin
                logger.debug("Installed autotuning plugin via CachingAutotuner.plugin")
                return plugin
            CachingAutotuner._kernelopt_plugin = plugin
            logger.debug("Installed autotuning plugin via CachingAutotuner._kernelopt_plugin")
            return plugin
    except ImportError:
        pass

    logger.debug("CachingAutotuner not available in this PyTorch version")
    return None
