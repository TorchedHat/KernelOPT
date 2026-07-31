"""Context manager for capturing compiler metadata during torch.compile.

Usage:
    with compiler_capture() as ctx:
        compiled = torch.compile(model)
        compiled(input)
    # ctx.data contains captured compilation metadata
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from kernelopt.compiler.passes.hook_installer import (
    clear_captured_data,
    get_captured_data,
    install_hooks,
    uninstall_hooks,
)


class CaptureResult:
    """Container for captured compiler metadata."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    @property
    def pre_grad_ops(self) -> list[dict]:
        return self.data.get("pre_grad", [])

    @property
    def post_grad_nodes(self) -> list[dict]:
        return self.data.get("post_grad", [])

    @property
    def fusion_pre(self) -> list[dict]:
        return self.data.get("fusion_pre", [])

    @property
    def fusion_post(self) -> list[dict]:
        return self.data.get("fusion_post", [])

    @property
    def fusion_rejections(self) -> list[dict]:
        return self.data.get("fusion_rejections", [])

    @property
    def triton_kernels(self) -> list[dict]:
        return self.data.get("triton", [])


@contextmanager
def compiler_capture(
    *,
    extract: bool = True,
    triton: bool = True,
) -> Generator[CaptureResult, None, None]:
    """Context manager that installs hooks, yields, then collects results.

    Args:
        extract: Capture Inductor IR and fusion data.
        triton: Capture Triton compilation metadata.

    Yields:
        CaptureResult that is populated after the context exits.
    """
    result = CaptureResult()
    clear_captured_data()

    try:
        install_hooks(extract=extract, triton=triton)
        yield result
    finally:
        result.data = get_captured_data()
        uninstall_hooks()
        clear_captured_data()
