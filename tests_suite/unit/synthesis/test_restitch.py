"""Unit tests for kernelopt.synthesis.restitch."""

from __future__ import annotations

import pytest

from kernelopt.synthesis.restitch import restitch_kernel, extract_kernel_body


SINGLE_KERNEL_SOURCE = """\
import torch
from torch._inductor.async_compile import AsyncCompile
async_compile = AsyncCompile()

triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics

@triton_heuristics.pointwise(
    size_hints={'x': 1024},
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}},
)
@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
    tmp1 = tl.maximum(0, tmp0)
    tl.store(in_out_ptr0 + xindex, tmp1, xmask)


def get_args():
    return torch.randn(1024, device='cuda'), 1024,


def call(args):
    triton_poi_fused_relu_0.run(*args)
''', device_str='cuda')

async_compile.wait(globals())

class Runner:
    def call(self, args):
        triton_poi_fused_relu_0.run(args[0], 1024, stream=None)
        return (args[0],)
"""

MULTI_KERNEL_SOURCE = """\
import torch
from torch._inductor.async_compile import AsyncCompile
async_compile = AsyncCompile()

triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
    tmp0 = tl.load(in_out_ptr0 + xindex, xindex < xnumel)
    tl.store(in_out_ptr0 + xindex, tl.maximum(0, tmp0), xindex < xnumel)
''', device_str='cuda')

triton_poi_fused_sigmoid_1 = async_compile.triton('triton_poi_fused_sigmoid_1', '''
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_sigmoid_1(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
    tmp0 = tl.load(in_out_ptr0 + xindex, xindex < xnumel)
    tl.store(in_out_ptr0 + xindex, tl.sigmoid(tmp0), xindex < xnumel)
''', device_str='cuda')

async_compile.wait(globals())
"""


class TestRestitchSingleKernel:

    def test_replaces_body_inside_async_compile(self):
        optimized = """\
@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
    tmp1 = tl.maximum(0, tmp0)
    tl.store(in_out_ptr0 + xindex, tmp1, xmask)
"""
        result = restitch_kernel(SINGLE_KERNEL_SOURCE, "triton_poi_fused_relu_0", optimized)

        assert "async_compile.triton('triton_poi_fused_relu_0'" in result
        assert "@triton.jit" in result
        assert "def triton_poi_fused_relu_0(" in result
        assert "class Runner:" in result
        assert "triton_poi_fused_relu_0.run" in result
        assert "@triton_heuristics.pointwise" in result

    def test_runner_call_untouched(self):
        optimized = "@triton.jit\ndef triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass\n"
        result = restitch_kernel(SINGLE_KERNEL_SOURCE, "triton_poi_fused_relu_0", optimized)

        assert "class Runner:" in result
        assert "triton_poi_fused_relu_0.run(args[0], 1024, stream=None)" in result


class TestRestitchMultiKernel:

    def test_only_target_replaced(self):
        optimized = "@triton.jit\ndef triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    # optimized\n    pass\n"
        result = restitch_kernel(MULTI_KERNEL_SOURCE, "triton_poi_fused_relu_0", optimized)

        assert "# optimized" in result
        assert "tl.sigmoid(tmp0)" in result  # sigmoid kernel untouched

    def test_both_kernels_in_async_compile(self):
        optimized = "@triton.jit\ndef triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass\n"
        result = restitch_kernel(MULTI_KERNEL_SOURCE, "triton_poi_fused_relu_0", optimized)

        assert "async_compile.triton('triton_poi_fused_relu_0'" in result
        assert "async_compile.triton('triton_poi_fused_sigmoid_1'" in result


class TestRestitchAutotune:

    def test_autotune_stripped_heuristics_kept(self):
        optimized = """\
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
    tmp1 = tl.maximum(0, tmp0)
    tl.store(in_out_ptr0 + xindex, tmp1, xmask)
"""
        result = restitch_kernel(SINGLE_KERNEL_SOURCE, "triton_poi_fused_relu_0", optimized)

        assert "@triton.autotune" not in result
        assert "@triton.jit" in result
        assert "@triton_heuristics.pointwise" in result
        assert "async_compile.triton('triton_poi_fused_relu_0'" in result


class TestRestitchErrors:

    def test_kernel_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            restitch_kernel(SINGLE_KERNEL_SOURCE, "nonexistent_kernel", "@triton.jit\ndef k(): pass")


class TestExtractKernelBody:

    def test_extract_simple(self):
        source = """\
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, n, XBLOCK: tl.constexpr):
    idx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    tl.store(x_ptr + idx, tl.load(x_ptr + idx) * 2)

def kernel_function(*inputs):
    pass
"""
        body = extract_kernel_body(source)
        assert body.startswith("@triton.jit")
        assert "def my_kernel(" in body
        assert "kernel_function" not in body

    def test_extract_with_autotune(self):
        source = """\
import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'XBLOCK': 512})], key=['n'])
@triton.jit
def my_kernel(x_ptr, n, XBLOCK: tl.constexpr):
    idx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    tl.store(x_ptr + idx, tl.load(x_ptr + idx) * 2)

def kernel_function(*inputs):
    pass
"""
        body = extract_kernel_body(source)
        assert body.startswith("@triton.autotune")
        assert "@triton.jit" in body
        assert "kernel_function" not in body


class TestRestitchRealData:

    def test_l3_003_real_file(self):
        from pathlib import Path
        synth = Path("pipeline_output/test_l3_003/synthesis/synthesized_triton.py")
        if not synth.exists():
            pytest.skip("No pipeline output available")

        source = synth.read_text()

        optimized = """\
@triton.jit
def triton_poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = (xindex % 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_first')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
"""
        result = restitch_kernel(source, "triton_poi_fused_addmm_relu_0", optimized)

        assert "evict_first" in result
        assert "class Runner:" in result
        assert "extern_kernels.mm" in result
        assert "async_compile.triton('triton_poi_fused_addmm_relu_0'" in result
        assert "triton_poi_fused_addmm_relu_0.run(" in result
        assert "@triton.jit" in result
