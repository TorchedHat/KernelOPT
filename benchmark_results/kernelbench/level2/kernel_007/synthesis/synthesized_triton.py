# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
try:
    empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
except (AttributeError, ImportError):
    pass
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
try:
    empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
except (AttributeError, ImportError):
    pass
try:
    empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
except (AttributeError, ImportError):
    pass
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
try:
    empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
except (AttributeError, ImportError):
    pass


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ur/cureutt3guhddv6h2psfoo7fnqeapqf3e63wxsmdgllguhd4vgwq.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.convolution, aten.relu, aten.leaky_relu, aten.gelu, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => relu
#   x_2 => gt, mul, where
#   x_3 => add, erf, mul_1, mul_2, mul_3
#   x_4 => sigmoid
#   x_5 => add_1
# Graph fragment:
#   %buf0 : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %gt : Tensor "b8[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%relu, 0), kwargs = {})
#   %mul : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, 0.01), kwargs = {})
#   %where : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %relu, %mul), kwargs = {})
#   %mul_1 : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, 0.5), kwargs = {})
#   %mul_2 : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2,), kwargs = {})
#   %add : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_3 : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %add), kwargs = {})
#   %sigmoid : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_3,), kwargs = {})
#   %add_1 : Tensor "f32[64, 32, 30, 62, 62][3690240, 115320, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, %arg3_1), kwargs = {})
#   return %add_1
triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0 = async_compile.triton('triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2834104320}, 'kernel_num_gb': 1.889403136, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 236175360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 115320) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tmp4 > tmp5
    tmp7 = tl.full([1], 0.01, tl.float32)
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tl.full([1], 0.5, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp13 = tmp9 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = tl.full([1], 1.0, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = tl.sigmoid(tmp17)
    tmp20 = tmp18 + tmp19
    tl.store(in_out_ptr0 + (x3), tmp20, None)


def get_args():
    arg_0 = rand_strided((64, 32, 30, 62, 62), (3690240, 115320, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 236175360,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.889403136
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (32, 8, 3, 3, 3), (216, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (32, ), (1, ))
        assert_size_stride(arg2_1, (64, 8, 32, 64, 64), (1048576, 131072, 4096, 64, 1))
        assert_size_stride(arg3_1, (32, 1, 1, 1), (1, 1, 1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (64, 32, 30, 62, 62), (3690240, 115320, 3844, 62, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.convolution, aten.relu, aten.leaky_relu, aten.gelu, aten.sigmoid, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0.run(buf1, arg1_1, arg3_1, 236175360, stream=stream0)
            del arg1_1
            del arg3_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 8, 3, 3, 3), (216, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 8, 32, 64, 64), (1048576, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 236175360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 115320) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tmp4 > tmp5
    tmp7 = tl.full([1], 0.01, tl.float32)
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tl.full([1], 0.5, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp13 = tmp9 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = tl.full([1], 1.0, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = tl.sigmoid(tmp17)
    tmp20 = tmp18 + tmp19
    tl.store(in_out_ptr0 + (x3), tmp20, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((32, 8, 3, 3, 3), (216, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 8, 32, 64, 64), (1048576, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    _args = parser.parse_args()
    _inputs = get_inputs()
    if _args.profile_only:
        kernel_function(*_inputs)
        torch.cuda.synchronize()
    else:
        kernel_function(*_inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")
