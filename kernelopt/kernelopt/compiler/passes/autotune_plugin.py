"""KernelOPT autotuning plugin for Inductor's CachingAutotuner.

Subclasses CachingAutotunerPlugin and injects via monkey-patching
get_caching_autotuner_plugins() to add our plugin to the chain.

References:
    - CachingAutotunerPlugin: torch/_inductor/runtime/triton_heuristics.py:454
    - get_caching_autotuner_plugins: torch/_inductor/runtime/triton_heuristics.py:514
    - DEFER sentinel: returned to fall through to default behavior
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PLUGIN_INSTALLED = False
_SAVED_GET_PLUGINS = None


def _get_base_plugin_class():
    """Import CachingAutotunerPlugin at runtime."""
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotunerPlugin
        return CachingAutotunerPlugin
    except ImportError:
        return object


class KernelOPTAutotunePlugin(_get_base_plugin_class()):
    """Autotuning plugin that injects profile-guided configs.

    Subclasses the actual CachingAutotunerPlugin so hooks are called
    by Inductor's autotuning pipeline. Returns DEFER to fall through
    to default behavior after mutation.
    """

    def __init__(
        self,
        profiling_data: Optional[dict] = None,
        avoid_configs: Optional[list[dict]] = None,
    ) -> None:
        super().__init__()
        self._profiling_data = profiling_data or {}
        self._avoid_configs = avoid_configs or []
        self._injected_count = 0
        self._pruned_count = 0

    def pre_compile(self, autotuner: Any) -> object:
        """Inject profile-guided configs before compilation."""
        try:
            from torch._inductor.runtime.triton_heuristics import DEFER
        except ImportError:
            return None

        suggested = self._profiling_data.get("suggested_configs", {})
        kernel_name = getattr(autotuner, "kernel_name", "")
        if kernel_name in suggested:
            for cfg in suggested[kernel_name]:
                if cfg not in (autotuner.configs if hasattr(autotuner, "configs") else []):
                    if hasattr(autotuner, "configs"):
                        autotuner.configs.append(cfg)
                        self._injected_count += 1

        return DEFER

    def pre_dispatch(
        self, autotuner: Any, *args: object, stream: object = None, **kwargs: object
    ) -> object:
        """Filter configs before dispatch to GPU."""
        try:
            from torch._inductor.runtime.triton_heuristics import DEFER
        except ImportError:
            return None

        if self._avoid_configs and hasattr(autotuner, "launchers"):
            original_count = len(autotuner.launchers)
            autotuner.launchers = [
                l for l in autotuner.launchers
                if not self._should_avoid_launcher(l)
            ]
            if not autotuner.launchers:
                autotuner.launchers = [l for l in autotuner.launchers] or autotuner.launchers
            self._pruned_count += original_count - len(autotuner.launchers)

        return DEFER

    def pre_autotune(
        self, autotuner: Any, *args: object, stream: object = None, **kwargs: object
    ) -> object:
        """Mutate launcher list before autotuning, then fall through."""
        try:
            from torch._inductor.runtime.triton_heuristics import DEFER
        except ImportError:
            return None

        return DEFER

    def _should_avoid_launcher(self, launcher: Any) -> bool:
        """Check if a launcher's config matches any known-bad configuration."""
        if not self._avoid_configs:
            return False
        config = getattr(launcher, "config", None)
        if config is None:
            return False
        config_dict = config.kwargs if hasattr(config, "kwargs") else {}
        for bad in self._avoid_configs:
            if all(config_dict.get(k) == v for k, v in bad.items()):
                return True
        return False

    @property
    def stats(self) -> dict:
        return {"injected": self._injected_count, "pruned": self._pruned_count}


def install_autotune_plugin(
    profiling_data: Optional[dict] = None,
    avoid_configs: Optional[list[dict]] = None,
) -> Optional[KernelOPTAutotunePlugin]:
    """Install the autotuning plugin into Inductor's plugin chain.

    Monkey-patches get_caching_autotuner_plugins() to append our plugin.
    """
    global _PLUGIN_INSTALLED, _SAVED_GET_PLUGINS

    if _PLUGIN_INSTALLED:
        logger.warning("Autotune plugin already installed")
        return None

    plugin = KernelOPTAutotunePlugin(
        profiling_data=profiling_data,
        avoid_configs=avoid_configs,
    )

    try:
        import torch._inductor.runtime.triton_heuristics as th
        original_get_plugins = th.get_caching_autotuner_plugins
        _SAVED_GET_PLUGINS = original_get_plugins

        def patched_get_plugins(autotuner):
            plugins = original_get_plugins(autotuner)
            plugins.append(plugin)
            return plugins

        th.get_caching_autotuner_plugins = patched_get_plugins
        _PLUGIN_INSTALLED = True
        logger.debug("Installed KernelOPT autotuning plugin via get_caching_autotuner_plugins patch")
        return plugin

    except (ImportError, AttributeError) as e:
        logger.debug(f"CachingAutotunerPlugin not available: {e}")
        return None


def uninstall_autotune_plugin() -> None:
    """Remove the autotuning plugin and restore original function."""
    global _PLUGIN_INSTALLED, _SAVED_GET_PLUGINS

    if not _PLUGIN_INSTALLED:
        return

    try:
        import torch._inductor.runtime.triton_heuristics as th
        if _SAVED_GET_PLUGINS is not None:
            th.get_caching_autotuner_plugins = _SAVED_GET_PLUGINS
    except ImportError:
        pass

    _PLUGIN_INSTALLED = False
    _SAVED_GET_PLUGINS = None
