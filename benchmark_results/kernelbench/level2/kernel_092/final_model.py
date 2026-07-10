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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ia/cia457agdcvazmolrxt7oet3uwnbk6d7t3kircblvh3xyvdcz6ji.py
# Topologically Sorted Source Nodes: [x_conv], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_conv => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[128, 8, 128, 128][131072, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
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
    size_hints={'y': 1024, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 134217728, 'x': 67108864}, 'kernel_num_gb': 0.134217728, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 16384*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 131072*y1), tmp0, None)


def get_args():
    arg_0 = rand_strided((128, 8, 128, 128), (131072, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 128, 128), (131072, 1, 1024, 8), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1024, 16384,


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
    num_gb = 0.134217728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/yg/cyggg4ke2deuarwqt76km52xh63wigodq5qcjvs7frv4x4cwuitz.py
# Topologically Sorted Source Nodes: [x_conv], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_conv => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 8, 3, 3][72, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 36864, 'x': 18432}, 'kernel_num_gb': 3.6864e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 72*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 512, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 3.6864e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/gb/cgbpm7yw6gi3e43bubcxqx6wmrdzkesjomhky6uwkybeacmtjflg.py
# Topologically Sorted Source Nodes: [x_conv, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_conv => convolution
#   x_norm => var_mean, view
# Graph fragment:
#   %buf2 : Tensor "f32[128, 64, 126, 126][1016064, 1, 8064, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[128, 16, 4, 15876][1016064, 63504, 15876, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [128, 16, 4, 15876]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf4
triton_red_fused_convolution_native_group_norm_2 = async_compile.triton('triton_red_fused_convolution_native_group_norm_2', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 520225024}, 'kernel_num_gb': 0.520241408, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 63504
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        flat_offset = r0_2 + 4*x0 + 64*r0_3
        flat_offset = tl.max_contiguous(tl.multiple_of(flat_offset, [1, 4]), [1, 4])
        bias_offset = r0_2 + 4*x0
        bias_offset = tl.max_contiguous(tl.multiple_of(bias_offset, [1, 4]), [1, 4])
        tmp0 = tl.load(in_ptr0 + (flat_offset + 1016064*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + bias_offset, mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp4, mask=xmask, eviction_policy='evict_first')
    tl.store(out_ptr1 + (x4), tmp8, mask=xmask, eviction_policy='evict_first')

def get_args():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 2048, 63504,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_convolution_native_group_norm_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.520241408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/qi/cqi5c3yzrfnmvg3j4pizijyv6u4gwi5dmz4ituqo5l7xqe4nonaz.py
# Topologically Sorted Source Nodes: [x_conv, x_norm, x_tanh, x_hard_swish, x_res, x_logsumexp], Original ATen: [aten.convolution, aten.native_group_norm, aten.tanh, aten.hardswish, aten.add, aten.logsumexp]
# Source node to ATen node mapping:
#   x_conv => convolution
#   x_hard_swish => add_2, clamp_max, clamp_min, div, mul_2
#   x_logsumexp => abs_1, add_4, amax, eq, exp, full_default, log, sub_1, sum_1, where
#   x_norm => add, add_1, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, var_mean, view, view_1
#   x_res => add_3
#   x_tanh => tanh
# Graph fragment:
#   %buf2 : Tensor "f32[128, 64, 126, 126][1016064, 1, 8064, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_1 : Tensor "f32[128, 16, 1, 1][16, 1, 2048, 2048]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf4 : Tensor "f32[128, 16, 1, 1][16, 1, 2048, 2048]cuda:0" = PlaceHolder[target=buf4]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %tanh : Tensor "f32[128, 64, 126, 126][1016064, 1, 8064, 64]cuda:0" = PlaceHolder[target=tanh]
#   %amax : Tensor "f32[128, 1, 126, 126][15904, 2035712, 126, 1]cuda:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[128, 1, 126, 126][15904, 2035712, 126, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %convolution : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[128, 16, 4, 15876][1016064, 63504, 15876, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [128, 16, 4, 15876]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[128, 16, 4, 15876][1016064, 63504, 15876, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[128, 16, 1, 1][16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[128, 16, 1, 1][16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 16, 4, 15876][1016064, 63504, 15876, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [128, 64, 126, 126]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %mul_1 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_3, 2), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 3), kwargs = {})
#   %add_1 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %tanh : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_1,), kwargs = {})
#   %add_2 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 3), kwargs = {})
#   %clamp_min : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %clamp_max : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_2 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %clamp_max), kwargs = {})
#   %div : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, 6), kwargs = {})
#   %add_3 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %div), kwargs = {})
#   %amax : Tensor "f32[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%add_3, [1], True), kwargs = {})
#   %abs_1 : Tensor "f32[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : Tensor "b8[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub_1 : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %where), kwargs = {})
#   %exp : Tensor "f32[128, 64, 126, 126][1016064, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : Tensor "f32[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add_4 : Tensor "f32[128, 1, 126, 126][15876, 15876, 126, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %where), kwargs = {})
#   return %tanh,%amax,%sum_1,%add_4
triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3 = async_compile.triton('triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3', '''
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
    size_hints={'x': 2097152, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16257024, 'r0_': 520241920}, 'kernel_num_gb': 0.528370432, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2032128
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 15876
    x0 = (xindex % 15876)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16*x1 + (r0_2 // 4)), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (16*x1 + (r0_2 // 4)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 63504.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = tl.full([1, 1], 3.0, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1, 1], 0.0, tl.float32)
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tl.full([1, 1], 6.0, tl.float32)
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp16 * tmp22
    tmp24 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(xmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tl.full([1, 1], float("inf"), tl.float32)
    tmp33 = tmp31 == tmp32
    tmp34 = tl.where(tmp33, tmp19, tmp30)
    tmp35 = tmp26 - tmp34
    tmp36 = libdevice.exp(tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp41 = tl_math.log(tmp40)
    tmp42 = tmp41 + tmp34
    tl.store(out_ptr3 + (x3), tmp42, xmask)


def get_args():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 1, 126, 126), (15876, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 2032128, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.528370432
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 8, 3, 3), (72, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (128, 8, 128, 128), (131072, 16384, 128, 1))
        assert_size_stride(arg3_1, (64, ), (1, ))
        assert_size_stride(arg4_1, (64, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 8, 128, 128), (131072, 1, 1024, 8), torch.float32)
            # Topologically Sorted Source Nodes: [x_conv], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 1024, 16384, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 8, 3, 3), (72, 1, 24, 8), torch.float32)
            # Topologically Sorted Source Nodes: [x_conv], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 512, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [x_conv], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (128, 64, 126, 126), (1016064, 1, 8064, 64), 'unknown_op')
            del buf0
            del buf1
            buf3 = empty_strided_cuda((128, 16, 1, 1), (16, 1, 2048, 2048), torch.float32)
            buf4 = empty_strided_cuda((128, 16, 1, 1), (16, 1, 2048, 2048), torch.float32)
            # Topologically Sorted Source Nodes: [x_conv, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_2.run(buf2, arg1_1, buf3, buf4, 2048, 63504, stream=stream0)
            buf9 = empty_strided_cuda((128, 1, 126, 126), (15876, 15876, 126, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_conv, x_norm, x_tanh, x_hard_swish, x_res, x_logsumexp], Original ATen: [aten.convolution, aten.native_group_norm, aten.tanh, aten.hardswish, aten.add, aten.logsumexp]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3.run(buf2, arg1_1, buf3, buf4, arg3_1, arg4_1, buf9, 2032128, 64, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del buf2
            del buf3
            del buf4
        return (buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 8, 128, 128), (131072, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Tanh
        x_tanh = self.tanh(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish(x_tanh)
        # Residual Addition
        x_res = x_conv + x_hard_swish
        # LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 16384*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 131072*y1), tmp0, None)

''',
    '''@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 72*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_red_fused_convolution_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 63504
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 64*r0_3 + 1016064*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp8, xmask)

''',
    '''@triton.jit
def triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2032128
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 15876
    x0 = (xindex % 15876)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16*x1 + (r0_2 // 4)), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (16*x1 + (r0_2 // 4)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 63504.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = tl.full([1, 1], 3.0, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1, 1], 0.0, tl.float32)
    tmp20 = tl.maximum(tmp18, tmp19)
    tmp21 = tl.full([1, 1], 6.0, tl.float32)
    tmp22 = tl.minimum(tmp20, tmp21)
    tmp23 = tmp16 * tmp22
    tmp24 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(xmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tl.full([1, 1], float("inf"), tl.float32)
    tmp33 = tmp31 == tmp32
    tmp34 = tl.where(tmp33, tmp19, tmp30)
    tmp35 = tmp26 - tmp34
    tmp36 = tl.exp(tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp41 = tl.log(tmp40)
    tmp42 = tmp41 + tmp34
    tl.store(out_ptr3 + (x3), tmp42, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 8, 128, 128), (131072, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]

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
