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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/vl/cvl2vrk5w7m7nukwedpl2al5qretayioimovasno2bltng2eodex.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %buf0 : Tensor "f32[128, 16, 32, 64, 64][2097152, 131072, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[128, 16, 32, 64, 64][2097152, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [1, 1, 1], 1), kwargs = {})
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
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3221225472}, 'kernel_num_gb': 2.147483712, 'kernel_flop': 5435817984},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 131072) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)


def get_args():
    arg_0 = rand_strided((128, 16, 32, 64, 64), (2097152, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 268435456,


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
    num_gb = 2.147483712
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/7m/c7mjhcdlzfbxt52zk46rpc5g5otqas3jm6onjkxcxybykr54u36w.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, view, x_3, sigmoid, x_4, max_1], Original ATen: [aten.convolution, aten.max_pool3d_with_indices, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.view, aten.sigmoid, aten.mul, aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   sigmoid => sigmoid
#   view => view
#   x => convolution
#   x_1 => _low_memory_max_pool_with_offsets
#   x_2 => div
#   x_3 => sub_1
#   x_4 => mul
# Graph fragment:
#   %convolution : Tensor "f32[128, 16, 32, 64, 64][2097152, 131072, 4096, 64, 1]cuda:0" = PlaceHolder[target=convolution]
#   %getitem : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_4 : Tensor "f32[128, 1, 16, 32, 32][16384, 2097152, 1024, 32, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %getitem_5 : Tensor "f32[128, 1, 16, 32, 32][16384, 2097152, 1024, 32, 1]cuda:0" = PlaceHolder[target=getitem_5]
#   %arg3_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[128, 16, 32, 64, 64][2097152, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [1, 1, 1], 1), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%convolution, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%getitem, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %getitem_4), kwargs = {})
#   %exp_default : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_5), kwargs = {})
#   %view : Tensor "f32[1, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [1, -1, 1, 1, 1]), kwargs = {})
#   %sub_1 : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %view), kwargs = {})
#   %sigmoid : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sub_1,), kwargs = {})
#   %mul : Tensor "f32[128, 16, 16, 32, 32][262144, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_1), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%mul, 1), kwargs = {})
#   return %getitem,%getitem_4,%getitem_5,%getitem_2
triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1 = async_compile.triton('triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1', '''
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
    size_hints={'x': 2097152, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 5, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16777216, 'r0_': 64}, 'kernel_num_gb': 1.090519104, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = ((xindex // 1024) % 16)
    x3 = xindex // 16384
    x5 = (xindex % 16384)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (r0_4), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp19 = triton_helpers.max2(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = tmp15 - tmp19
    tmp21 = libdevice.exp(tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp25 = tmp14 - tmp19
    tmp26 = libdevice.exp(tmp25)
    tmp27 = (tmp26 / tmp24)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp30 * tmp29
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = triton_helpers.max2(tmp32, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (x6), tmp34, None)


def get_args():
    arg_0 = rand_strided((128, 16, 32, 32), (16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 32, 64, 64), (2097152, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 2097152, 16,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.090519104
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
        assert_size_stride(arg0_1, (3, 16, 3, 3, 3), (432, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (16, ), (1, ))
        assert_size_stride(arg2_1, (128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1))
        assert_size_stride(arg3_1, (16, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
            assert_size_stride(buf0, (128, 16, 32, 64, 64), (2097152, 131072, 4096, 64, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(buf1, arg1_1, 268435456, stream=stream0)
            del arg1_1
            buf3 = empty_strided_cuda((128, 1, 16, 32, 32), (16384, 2097152, 1024, 32, 1), torch.float32)
            buf5 = reinterpret_tensor(buf3, (128, 16, 32, 32), (16384, 1024, 32, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2, view, x_3, sigmoid, x_4, max_1], Original ATen: [aten.convolution, aten.max_pool3d_with_indices, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.view, aten.sigmoid, aten.mul, aten.max]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1.run(buf5, buf1, arg3_1, 2097152, 16, stream=stream0)
            del arg3_1
            del buf1
        return (buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((3, 16, 3, 3, 3), (432, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels)) # Assuming subtraction is element-wise across channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = torch.softmax(x, dim=1) # Apply softmax across channels (dim=1)
        x = x - self.subtract.view(1, -1, 1, 1, 1) # Subtract across channels
        x = torch.sigmoid(x) * x # Swish activation
        x = torch.max(x, dim=1)[0] # Max pooling across channels
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 131072) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)

''',
    '''@triton.jit
def triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = ((xindex // 1024) % 16)
    x3 = xindex // 16384
    x5 = (xindex % 16384)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2 + 131072*r0_4 + 2097152*x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (r0_4), None, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp19 = triton_helpers.max2(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = tmp15 - tmp19
    tmp21 = tl.exp(tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp25 = tmp14 - tmp19
    tmp26 = tl.exp(tmp25)
    tmp27 = (tmp26 / tmp24)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp30 * tmp29
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = triton_helpers.max2(tmp32, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (x6), tmp34, None)

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
    arg3_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
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
