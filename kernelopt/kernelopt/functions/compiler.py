"""CUDA kernel compiler: wraps nvcc, formats errors."""

from __future__ import annotations

import re
import subprocess


def compile_cuda(cu_path: str) -> tuple[str, str | None]:
    """Compile a .cu file with nvcc.

    Returns:
        (binary_path, None)           on success
        (binary_path, error_message)  on failure

    binary_path is always cu_path with .cu stripped — predictable without
    inspecting the return value.
    """
    binary = cu_path.replace(".cu", "")
    result = subprocess.run(
        ["nvcc", "-O2", "-o", binary, cu_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return binary, _format_nvcc_errors(result.stderr)
    return binary, None


def _format_nvcc_errors(stderr: str) -> str:
    """Strip temp-file path prefixes from nvcc stderr.

    Keeps lines matching: filename.cu(line): error/warning/note: message
    Strips any leading /tmp/... prefix so errors are readable.
    """
    if not stderr.strip():
        return ""
    lines = stderr.strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        # Strip leading /path/to/tmpXXX/ prefix before the filename
        line = re.sub(r"^[^\s]*?/([^/]+\.cu)", r"\1", line)
        cleaned.append(line)
    return "\n".join(cleaned)
