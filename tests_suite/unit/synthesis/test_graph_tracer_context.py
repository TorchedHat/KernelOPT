"""Tests that graph_tracer propagates full Inductor source through kernel groups."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "kernelopt"))

from kernelopt.synthesis.graph_tracer import _build_tracer_script


def test_full_inductor_source_in_kernel_groups():
    """Each kernel group must contain full_inductor_source = complete output_code.py."""
    script = _build_tracer_script("/fake/model.py")
    assert "full_inductor_source" in script, (
        "Tracer script must assign full_inductor_source to each kernel group"
    )


def test_full_inductor_source_uses_code_variable():
    """full_inductor_source must be set from the `code` variable, not a sub-kernel."""
    script = _build_tracer_script("/fake/model.py")
    assert '"full_inductor_source": code' in script or "'full_inductor_source': code" in script or 'full_inductor_source"] = code' in script, (
        "full_inductor_source must be assigned from the full `code` variable"
    )


def test_code_initialized_before_try():
    """code must be initialized to empty string before the try block (safe fallback)."""
    script = _build_tracer_script("/fake/model.py")
    # code = "" must appear before the try: block that calls run_and_get_code
    code_init_pos = script.find('code = ""')
    try_pos = script.find("run_and_get_code")
    assert code_init_pos != -1, "code = '' initializer must exist in the script"
    assert code_init_pos < try_pos, "code = '' must be initialized before run_and_get_code is called"
