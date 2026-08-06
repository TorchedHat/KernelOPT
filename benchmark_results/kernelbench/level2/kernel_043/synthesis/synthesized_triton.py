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


# kernel path: /workspace/tmp/torchinductor_user/u4/cu4zao56w776f4qvyfx6scuhzog2dou6zxgouj3tiybi6j4pgyjz.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %buf0 : Tensor "f32[4, 64, 32, 128, 128][33554432, 524288, 16384, 128, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[4, 64, 32, 128, 128][33554432, 524288, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1610612736}, 'kernel_num_gb': 1.07374208, 'kernel_flop': 231928233984},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 524288) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)


def get_args():
    arg_0 = rand_strided((4, 64, 32, 128, 128), (33554432, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 134217728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.07374208
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/e2/ce24yop37xth6ak6zvod7w6kyweli4abello34ohxjxho6cdjrxf.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.max_pool3d_with_indices, aten.logsumexp, aten.relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => _low_memory_max_pool_with_offsets
#   x_2 => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
#   x_3 => relu
# Graph fragment:
#   %convolution : Tensor "f32[4, 64, 32, 128, 128][33554432, 524288, 16384, 128, 1]cuda:0" = PlaceHolder[target=convolution]
#   %getitem : Tensor "f32[4, 64, 16, 64, 64][4194304, 65536, 4096, 64, 1]cuda:0" = PlaceHolder[target=getitem]
#   %amax : Tensor "f32[4, 1, 16, 64, 64][65536, 262144, 4096, 64, 1]cuda:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[4, 1, 16, 64, 64][65536, 262144, 4096, 64, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %convolution : Tensor "f32[4, 64, 32, 128, 128][33554432, 524288, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%convolution, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %amax : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%getitem, [1], True), kwargs = {})
#   %abs_1 : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : Tensor "b8[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : Tensor "f32[4, 64, 16, 64, 64][4194304, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %where), kwargs = {})
#   %exp : Tensor "f32[4, 64, 16, 64, 64][4194304, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %where), kwargs = {})
#   %relu : Tensor "f32[4, 1, 16, 64, 64][65536, 65536, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   return %getitem,%amax,%sum_1,%relu
triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1 = async_compile.triton('triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2097152, 'r0_': 0}, 'kernel_num_gb': 0.538968064, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_4 = r0_index
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 16)
    x3 = xindex // 65536
    x5 = (xindex % 65536)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16384 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16385 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (16512 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (16513 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = triton_helpers.max2(tmp15, 1)[:, None].to(tl.float32)
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tl.full([1, 1], float("inf"), tl.float32)
    tmp20 = tmp18 == tmp19
    tmp21 = tl.full([1, 1], 0.0, tl.float32)
    tmp22 = tl.where(tmp20, tmp21, tmp17)
    tmp23 = tmp14 - tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tl_math.log(tmp27)
    tmp29 = tmp28 + tmp22
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x6), tmp31, None)


def get_args():
    arg_0 = rand_strided((4, 1, 16, 64, 64), (65536, 65536, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4, 64, 32, 128, 128), (33554432, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 262144, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.538968064
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (4, 32, 32, 128, 128), (16777216, 524288, 16384, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (4, 64, 32, 128, 128), (33554432, 524288, 16384, 128, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(buf1, arg1_1, 134217728, stream=stream0)
            del arg1_1
            buf4 = empty_strided_cuda((4, 1, 16, 64, 64), (65536, 262144, 4096, 64, 1), torch.float32)
            buf5 = reinterpret_tensor(buf4, (4, 1, 16, 64, 64), (65536, 65536, 4096, 64, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.max_pool3d_with_indices, aten.logsumexp, aten.relu]
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1.run(buf5, buf1, 262144, 64, stream=stream0)
            del buf1
        return (buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 32, 32, 128, 128), (16777216, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 524288) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)

''',
    '''@triton.jit
def triton_per_fused_convolution_logsumexp_max_pool3d_with_indices_relu_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_4 = r0_index
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 16)
    x3 = xindex // 65536
    x5 = (xindex % 65536)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16384 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16385 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (16512 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (16513 + 2*x0 + 256*x1 + 32768*x2 + 524288*r0_4 + 33554432*x3), None, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = triton_helpers.max2(tmp15, 1)[:, None].to(tl.float32)
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tl.full([1, 1], float("inf"), tl.float32)
    tmp20 = tmp18 == tmp19
    tmp21 = tl.full([1, 1], 0.0, tl.float32)
    tmp22 = tl.where(tmp20, tmp21, tmp17)
    tmp23 = tmp14 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tl.log(tmp27)
    tmp29 = tmp28 + tmp22
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x6), tmp31, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((64, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 32, 32, 128, 128), (16777216, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]

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
