"""Compiler passes and hook installation for KernelOPT.

Provides:
- Hook installer API (install/uninstall extraction hooks into Inductor)
- Context manager for compilation capture
- Profile-guided fusion patterns
- InductorChoices overrides
- CachingAutotuner plugin
- ExternalTritonTemplateKernel registration
- Custom decomposition table
"""

from kernelopt.compiler.passes.hook_installer import (
    install_hooks,
    uninstall_hooks,
    is_installed,
    get_captured_data,
    clear_captured_data,
)
from kernelopt.compiler.passes.capture import (
    compiler_capture,
    CaptureResult,
)

__all__ = [
    "install_hooks",
    "uninstall_hooks",
    "is_installed",
    "get_captured_data",
    "clear_captured_data",
    "compiler_capture",
    "CaptureResult",
]
