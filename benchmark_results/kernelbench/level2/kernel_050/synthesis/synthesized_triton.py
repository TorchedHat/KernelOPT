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


# kernel path: /workspace/tmp/torchinductor_user/cm/ccmdxyy6vh62naligbx24iyd5n3c7pqvtmjpvlzrqgmhzat4d24d.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.mul]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => mul
# Graph fragment:
#   %buf0 : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %mul : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %arg3_1), kwargs = {})
#   return %mul
triton_poi_fused_convolution_mul_0 = async_compile.triton('triton_poi_fused_convolution_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3023806464}, 'kernel_num_gb': 2.015871044, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 251983872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = ((xindex // 123039) % 16)
    x0 = (xindex % 3969)
    x5 = xindex // 3969
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x0 + 4000*x5), tmp5, xmask)


def get_args():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 31, 63, 63), (1984000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 251983872,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_mul_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 2.015871044
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ug/cug3ft4pyqttgzpxfa6uhj2yxwje4wjljzhnplhnznircn7c3373.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten.mul, aten.avg_pool3d, aten.add]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => mul
#   x_2 => avg_pool3d
#   x_3 => add
#   x_4 => mul_1
# Graph fragment:
#   %mul : Tensor "f32[128, 16, 31, 63, 63][1984000, 124000, 4000, 63, 1]cuda:0" = PlaceHolder[target=mul]
#   %arg4_1 : Tensor "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add : Tensor "f32[128, 16, 15, 31, 31][230912, 14432, 961, 31, 1]cuda:0" = PlaceHolder[target=add]
#   %arg5_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %mul : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %arg3_1), kwargs = {})
#   %avg_pool3d : Tensor "f32[128, 16, 15, 31, 31][230640, 14415, 961, 31, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%mul, [2, 2, 2], [2, 2, 2]), kwargs = {})
#   %add : Tensor "f32[128, 16, 15, 31, 31][230640, 14415, 961, 31, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%avg_pool3d, %arg4_1), kwargs = {})
#   %mul_1 : Tensor "f32[128, 16, 15, 31, 31][230640, 14415, 961, 31, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %arg5_1), kwargs = {})
#   return %add,%mul_1
triton_poi_fused_add_avg_pool3d_convolution_mul_1 = async_compile.triton('triton_poi_fused_add_avg_pool3d_convolution_mul_1', '''
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
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool3d_convolution_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 236175360}, 'kernel_num_gb': 1.062789188, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29521920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 31)
    x1 = ((xindex // 31) % 31)
    x2 = ((xindex // 961) % 15)
    x6 = xindex // 14415
    x3 = ((xindex // 14415) % 16)
    x5 = (xindex % 14415)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (63 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (64 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4000 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4001 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4063 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4064 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp18 = tmp16 + tmp17
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr1 + (x7), tmp21, xmask)


def get_args():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1984000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 15, 31, 31), (230640, 14415, 961, 31, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 29521920,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool3d_convolution_mul_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_avg_pool3d_convolution_mul_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.062789188
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        assert_size_stride(arg0_1, (3, 16, 3, 3, 3), (432, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (16, ), (1, ))
        assert_size_stride(arg2_1, (128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1))
        assert_size_stride(arg3_1, (), ())
        assert_size_stride(arg4_1, (16, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(arg5_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((128, 16, 31, 63, 63), (1984000, 124000, 4000, 63, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_mul_0.run(buf0, arg1_1, arg3_1, buf1, 251983872, stream=stream0)
            del arg1_1
            del arg3_1
            del buf0
            buf3 = empty_strided_cuda((128, 16, 15, 31, 31), (230640, 14415, 961, 31, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten.mul, aten.avg_pool3d, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_avg_pool3d_convolution_mul_1.run(buf1, arg4_1, arg5_1, buf3, 29521920, stream=stream0)
            del arg4_1
            del arg5_1
            del buf1
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((3, 16, 3, 3, 3), (432, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale1
        x = self.avg_pool(x)
        x = x + self.bias
        x = x * self.scale2
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 251983872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = ((xindex // 123039) % 16)
    x0 = (xindex % 3969)
    x5 = xindex // 3969
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x0 + 4000*x5), tmp5, xmask)

''',
    '''@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29521920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 31)
    x1 = ((xindex // 31) % 31)
    x2 = ((xindex // 961) % 15)
    x6 = xindex // 14415
    x3 = ((xindex // 14415) % 16)
    x5 = (xindex % 14415)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (63 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (64 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4000 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4001 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4063 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4064 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp18 = tmp16 + tmp17
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr1 + (x7), tmp21, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((3, 16, 3, 3, 3), (432, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]

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
