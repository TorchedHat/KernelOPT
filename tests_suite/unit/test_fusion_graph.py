"""Unit tests for the buffer dependency graph parser (fusion phase)."""

import pytest

from kernelopt.fusion.graph import parse_call_graph


CALL_SOURCE = """\
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


@pytest.fixture(scope="module")
def graph():
    return parse_call_graph(CALL_SOURCE)


# ---------------------------------------------------------------------------
# 1. Three kernels with correct names
# ---------------------------------------------------------------------------

def test_parse_finds_three_kernels(graph):
    assert len(graph.kernels) == 3
    names = [k.name for k in graph.kernels]
    assert "triton_poi_fused_convolution_0" in names
    assert "triton_tem_fused_convolution_1" in names
    assert "triton_poi_fused_convolution_2" in names


# ---------------------------------------------------------------------------
# 2. Kernel type classification: poi, tem, poi
# ---------------------------------------------------------------------------

def test_parse_classifies_kernel_types(graph):
    types = [k.kernel_type for k in graph.kernels]
    assert types == ["poi", "tem", "poi"]


# ---------------------------------------------------------------------------
# 3. Intermediate buffers buf0, buf1, buf2 with correct shapes
# ---------------------------------------------------------------------------

def test_parse_finds_intermediate_buffers(graph):
    assert set(graph.buffers.keys()) == {"buf0", "buf1", "buf2"}

    buf0 = graph.buffers["buf0"]
    assert buf0.shape == (16, 64, 1024, 1024)
    assert buf0.strides == (67108864, 1, 65536, 64)
    assert buf0.dtype == "torch.float32"

    buf1 = graph.buffers["buf1"]
    assert buf1.shape == (16777216, 128)
    assert buf1.strides == (128, 1)

    buf2 = graph.buffers["buf2"]
    assert buf2.shape == (16, 128, 1024, 1024)
    assert buf2.strides == (134217728, 1048576, 1024, 1)


# ---------------------------------------------------------------------------
# 4. Edges: (K0, buf0, K1) and (K1, buf1, K2)
# ---------------------------------------------------------------------------

def test_parse_builds_edges(graph):
    # Normalise to a set of tuples for order-independent comparison.
    edges = set(graph.edges)
    assert ("triton_poi_fused_convolution_0", "buf0", "triton_tem_fused_convolution_1") in edges
    assert ("triton_tem_fused_convolution_1", "buf1", "triton_poi_fused_convolution_2") in edges
    # Exactly two edges expected.
    assert len(edges) == 2


# ---------------------------------------------------------------------------
# 5. Model inputs and outputs
# ---------------------------------------------------------------------------

def test_parse_identifies_model_inputs_outputs(graph):
    assert graph.model_inputs == {"arg0_1", "arg1_1"}
    assert graph.model_outputs == ["buf2"]


# ---------------------------------------------------------------------------
# 6. Kernel reads and writes
# ---------------------------------------------------------------------------

def test_parse_tracks_kernel_reads_writes(graph):
    by_name = {k.name: k for k in graph.kernels}

    k0 = by_name["triton_poi_fused_convolution_0"]
    assert "arg1_1" in k0.reads
    assert "buf0" in k0.writes
    assert "arg1_1" not in k0.writes

    k1 = by_name["triton_tem_fused_convolution_1"]
    assert "buf0" in k1.reads
    assert "arg0_1" in k1.reads
    assert "buf1" in k1.writes

    k2 = by_name["triton_poi_fused_convolution_2"]
    assert "buf1" in k2.reads
    assert "buf2" in k2.writes
