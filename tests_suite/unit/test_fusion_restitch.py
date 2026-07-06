"""Unit tests for fusion restitch utilities."""
from __future__ import annotations

import pytest

from kernelopt.fusion.restitch import (
    remove_async_compile_block,
    patch_call_for_fusion,
)


# ---------------------------------------------------------------------------
# Fixtures — a minimal synthesized model source with three kernels
# ---------------------------------------------------------------------------

MODEL_SOURCE = """\
import torch
from torch._inductor.runtime.runtime_utils import async_compile

triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_convolution_0(arg1_1, buf0, xnumel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    buf0[pid] = arg1_1[pid]
''', device_str='cuda')

triton_tem_fused_convolution_1 = async_compile.triton('triton_tem_fused_convolution_1', '''
import triton
import triton.language as tl

@triton.jit
def triton_tem_fused_convolution_1(buf0, arg0_1, buf1, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    buf1[pid] = buf0[pid] + arg0_1[pid]
''', device_str='cuda')

triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl

@triton.jit
def triton_poi_fused_convolution_2(buf1, buf2, xnumel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    buf2[pid] = buf1[pid]
''', device_str='cuda')


class Runner:
    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16, 64), (64, 1), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 1024, stream=raw_stream0)
            del arg1_1
            buf1 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_1.run(buf0, arg0_1, buf1, 16, 128, 64, stream=raw_stream0)
            del arg0_1
            del buf0
            buf2 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_2.run(buf1, buf2, 2048, stream=raw_stream0)
            del buf1
        return (buf2, )
"""


# ---------------------------------------------------------------------------
# 1. remove_async_compile_block removes K1's block, keeps K0 and K2
# ---------------------------------------------------------------------------


def test_remove_block():
    result = remove_async_compile_block(MODEL_SOURCE, "triton_tem_fused_convolution_1")

    # K1 block must be gone
    assert "triton_tem_fused_convolution_1 = async_compile.triton(" not in result
    assert "triton_tem_fused_convolution_1" not in result.split("def call")[0] or \
           "async_compile" not in result.split("triton_tem_fused_convolution_1")[0].split("\n")[-1]

    # K0 and K2 blocks must still exist
    assert "triton_poi_fused_convolution_0 = async_compile.triton(" in result
    assert "triton_poi_fused_convolution_2 = async_compile.triton(" in result


# ---------------------------------------------------------------------------
# 2. remove_async_compile_block raises ValueError for nonexistent kernel
# ---------------------------------------------------------------------------


def test_remove_nonexistent_raises():
    with pytest.raises(ValueError, match="not found"):
        remove_async_compile_block(MODEL_SOURCE, "triton_nonexistent_kernel_999")


# ---------------------------------------------------------------------------
# 3. patch_call_for_fusion removes intermediate kernel lines, buffer allocs,
#    inserts fused call, preserves buf2 and return
# ---------------------------------------------------------------------------


def test_patch_call_removes_intermediate_kernel_lines():
    kernels_to_remove = [
        "triton_poi_fused_convolution_0",
        "triton_tem_fused_convolution_1",
        "triton_poi_fused_convolution_2",
    ]
    buffers_to_remove = ["buf0", "buf1"]
    fused_call_line = "            fused_kernel_call(arg1_1, arg0_1, buf2, stream=raw_stream0)"
    insert_after_line = "raw_stream0 = get_raw_stream(0)"

    result = patch_call_for_fusion(
        MODEL_SOURCE,
        kernels_to_remove=kernels_to_remove,
        buffers_to_remove=buffers_to_remove,
        fused_call_line=fused_call_line,
        insert_after_line=insert_after_line,
    )

    # All three original .run() calls must be gone
    assert "triton_poi_fused_convolution_0.run(" not in result
    assert "triton_tem_fused_convolution_1.run(" not in result
    assert "triton_poi_fused_convolution_2.run(" not in result

    # Eliminated buffer allocations must be gone
    assert "buf0 = empty_strided_cuda(" not in result
    assert "buf1 = empty_strided_cuda(" not in result

    # del buf0, del buf1 lines should be gone
    # (but del arg1_1, del arg0_1 may remain — they're not in buffers_to_remove)
    lines = result.splitlines()
    for line in lines:
        stripped = line.strip()
        assert stripped != "del buf0", f"Expected 'del buf0' removed, still present: {stripped!r}"
        assert stripped != "del buf1", f"Expected 'del buf1' removed, still present: {stripped!r}"

    # buf2 allocation must be preserved
    assert "buf2 = empty_strided_cuda(" in result

    # return statement must be preserved
    assert "return (buf2, )" in result

    # fused call must be present
    assert "fused_kernel_call(" in result
