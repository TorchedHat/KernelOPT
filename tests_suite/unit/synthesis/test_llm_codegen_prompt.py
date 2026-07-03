"""Tests that _build_codegen_prompt uses full Inductor source with correct framing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "kernelopt"))

from kernelopt.synthesis.llm_codegen import _build_codegen_prompt


FULL_INDUCTOR = """\
# AOT ID: ['0_inference']
triton_red_fused__softmax_0 = async_compile.triton(...)
triton_red_fused__softmax_exp = async_compile.triton(...)
def call(args):
    triton_red_fused__softmax_0[grid](...)
    triton_red_fused__softmax_exp[grid](...)
    return (buf1,)
"""

SUB_KERNEL = "@triton.jit\ndef triton_red_fused__softmax_exp_prepare_so(...):\n    pass\n" * 5


def _make_group(**overrides):
    base = {
        "kernel_name": "triton_red_fused__softmax_exp_prepare_so",
        "kernel_type": "reduction",
        "fused_aten_ops": ["_softmax"],
        "aten_graph": [{"name": "_softmax", "shape": [4096, 393216], "dtype": "float32"}],
        "model_inputs": [{"name": "x", "shape": [4096, 393216], "dtype": "float32"}],
    }
    base.update(overrides)
    return base


def test_uses_full_inductor_source_when_available():
    """Prompt must include full_inductor_source when present."""
    group = _make_group(full_inductor_source=FULL_INDUCTOR)
    prompt = _build_codegen_prompt(group, gpu_specs=None)
    assert "def call(args):" in prompt, "full_inductor_source (including call()) must appear in prompt"


def test_correct_framing_not_reference_only():
    """Prompt must NOT say 'reference only' or 'write clean code'."""
    group = _make_group(full_inductor_source=FULL_INDUCTOR)
    prompt = _build_codegen_prompt(group, gpu_specs=None)
    assert "reference only" not in prompt.lower(), "Wrong framing: tells LLM to ignore Inductor"
    assert "write clean code" not in prompt.lower(), "Wrong framing: tells LLM to ignore Inductor"


def test_framing_tells_llm_to_understand_decomposition():
    """Prompt must instruct LLM to understand the full decomposition."""
    group = _make_group(full_inductor_source=FULL_INDUCTOR)
    prompt = _build_codegen_prompt(group, gpu_specs=None)
    assert "decomposition" in prompt.lower() or "full" in prompt.lower(), (
        "Prompt framing must guide LLM to understand the full Inductor decomposition"
    )


def test_falls_back_to_sub_kernel_when_no_full_source():
    """When full_inductor_source is absent, fall back to inductor_source sub-kernel."""
    group = _make_group(inductor_source=SUB_KERNEL)
    prompt = _build_codegen_prompt(group, gpu_specs=None)
    assert "triton_red_fused__softmax_exp_prepare_so" in prompt, (
        "Sub-kernel fallback must appear in prompt when full source is absent"
    )


def test_full_source_preferred_over_sub_kernel():
    """When both are present, full_inductor_source takes precedence."""
    group = _make_group(full_inductor_source=FULL_INDUCTOR, inductor_source=SUB_KERNEL)
    prompt = _build_codegen_prompt(group, gpu_specs=None)
    assert "def call(args):" in prompt, "full_inductor_source must take precedence"


def test_no_inductor_source_at_all():
    """When neither source is present, prompt must still be valid (no crash)."""
    group = _make_group()
    prompt = _build_codegen_prompt(group, gpu_specs=None)
    assert "Operations to implement" in prompt  # basic prompt structure intact
