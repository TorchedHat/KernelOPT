"""Unit tests for FusionGroup detection from buffer dependency graphs."""
from __future__ import annotations

import pytest

from kernelopt.fusion.graph import parse_call_graph
from kernelopt.fusion.detect import detect_fusion_groups, FusionGroup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 3-kernel chain: poi -> tem -> poi
# buf0 is fusible (K0 writes, K1 reads, K0 is its sole producer, K1 its sole consumer)
# buf1 is fusible (K1 writes, K2 reads, same logic)
# buf2 is a model output → not fusible
CALL_SOURCE_3KERNEL = """\
class Runner:
    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16, 64, 1024, 1024), (67108864, 1, 65536, 64), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 1024, 1048576, stream=raw_stream0)
            del arg1_1
            buf1 = empty_strided_cuda((16777216, 128), (128, 1), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_1.run(buf0, arg0_1, buf1, 131072, 1, 1, stream=raw_stream0)
            del arg0_1
            del buf0
            buf2 = empty_strided_cuda((16, 128, 1024, 1024), (134217728, 1048576, 1024, 1), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_2.run(buf1, buf2, 2048, 1048576, stream=raw_stream0)
            del buf1
        return (buf2, )
"""

# 2 independent kernels that share NO buffers — nothing fusible
CALL_SOURCE_INDEPENDENT = """\
class Runner:
    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1024,), (1,), torch.float32)
            raw_stream0 = get_raw_stream(0)
            triton_poi_kernel_a.run(arg0_1, buf0, 1024, stream=raw_stream0)
            buf1 = empty_strided_cuda((1024,), (1,), torch.float32)
            triton_poi_kernel_b.run(arg1_1, buf1, 1024, stream=raw_stream0)
        return (buf0, buf1)
"""

KERNEL_SOURCES_3K: dict[str, str] = {
    "triton_poi_fused_convolution_0": "@triton.jit\ndef triton_poi_fused_convolution_0(arg1_1, buf0, xnumel, BLOCK: tl.constexpr):\n    pass\n",
    "triton_tem_fused_convolution_1": "@triton.jit\ndef triton_tem_fused_convolution_1(buf0, arg0_1, buf1, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):\n    pass\n",
    "triton_poi_fused_convolution_2": "@triton.jit\ndef triton_poi_fused_convolution_2(buf1, buf2, xnumel, BLOCK: tl.constexpr):\n    pass\n",
}


@pytest.fixture(scope="module")
def graph_3k():
    return parse_call_graph(CALL_SOURCE_3KERNEL)


@pytest.fixture(scope="module")
def groups_3k(graph_3k):
    return detect_fusion_groups(graph_3k, KERNEL_SOURCES_3K)


@pytest.fixture(scope="module")
def graph_ind():
    return parse_call_graph(CALL_SOURCE_INDEPENDENT)


# ---------------------------------------------------------------------------
# 1. Detects full_gemm_fusion for poi -> tem -> poi
# ---------------------------------------------------------------------------


def test_detects_full_gemm_fusion(groups_3k):
    assert len(groups_3k) == 1, f"Expected 1 fusion group, got {len(groups_3k)}"
    group = groups_3k[0]
    assert group.fusion_type == "full_gemm_fusion"
    kernel_names = [k.name for k in group.kernels]
    assert kernel_names == [
        "triton_poi_fused_convolution_0",
        "triton_tem_fused_convolution_1",
        "triton_poi_fused_convolution_2",
    ], f"Unexpected kernel order: {kernel_names}"


# ---------------------------------------------------------------------------
# 2. Identifies eliminated buffers correctly
# ---------------------------------------------------------------------------


def test_identifies_eliminated_buffers(groups_3k):
    group = groups_3k[0]
    elim_names = {buf.name for buf in group.eliminated_buffers}
    assert "buf0" in elim_names, f"buf0 should be eliminated, got: {elim_names}"
    assert "buf1" in elim_names, f"buf1 should be eliminated, got: {elim_names}"
    assert "buf2" not in elim_names, f"buf2 is a model output, must NOT be eliminated"


# ---------------------------------------------------------------------------
# 3. Identifies external inputs and outputs
# ---------------------------------------------------------------------------


def test_identifies_external_io(groups_3k):
    group = groups_3k[0]
    inputs_set = set(group.input_buffers)
    outputs_set = set(group.output_buffers)

    # Both model-input args feed the chain
    assert "arg0_1" in inputs_set, f"arg0_1 missing from inputs: {inputs_set}"
    assert "arg1_1" in inputs_set, f"arg1_1 missing from inputs: {inputs_set}"

    # buf0 and buf1 are eliminated — must NOT appear as external inputs or outputs
    assert "buf0" not in inputs_set, "buf0 is eliminated, should not be in input_buffers"
    assert "buf1" not in inputs_set, "buf1 is eliminated, should not be in input_buffers"

    # buf2 is the external output
    assert "buf2" in outputs_set, f"buf2 missing from outputs: {outputs_set}"


# ---------------------------------------------------------------------------
# 4. No fusion groups for independent kernels
# ---------------------------------------------------------------------------


def test_no_fusion_for_independent_kernels(graph_ind):
    groups = detect_fusion_groups(graph_ind, {})
    assert len(groups) == 0, (
        f"Expected 0 fusion groups for independent kernels, got {len(groups)}: "
        f"{[g.group_id for g in groups]}"
    )
