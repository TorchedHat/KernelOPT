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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/wn/cwnrxgphceloi3yc4j67ni32t2ialcet4wx33iy6xcd6trn6mzaq.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.div]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
# Graph fragment:
#   %buf0 : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, 2.0), kwargs = {})
#   return %div
triton_poi_fused_convolution_div_0 = async_compile.triton('triton_poi_fused_convolution_div_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1322582016}, 'kernel_num_gb': 0.881721408, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110215168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    x2 = ((xindex // 53816) % 16)
    x0 = (xindex % 3844)
    x5 = xindex // 3844
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x0 + 3872*x5), tmp4, None)


def get_args():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (861056, 53816, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 14, 62, 62), (867328, 54208, 3872, 62, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 110215168,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_div_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.881721408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/tr/ctrxyjdlhmnrzkrettshdvf7rfbjtyclki2jixm2aelh5vazo7gn.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.div, aten.max_pool3d_with_indices, aten.mean]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
#   x_2 => _low_memory_max_pool_with_offsets
#   x_3 => mean
# Graph fragment:
#   %div : Tensor "f32[128, 16, 14, 62, 62][867328, 54208, 3872, 62, 1]cuda:0" = PlaceHolder[target=div]
#   %convolution : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, 2.0), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %mean : Tensor "f32[128, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%getitem, [-1, -2, -3], True), kwargs = {})
#   return %buf2
triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1 = async_compile.triton('triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1', '''
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
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 868352, 'r0_': 0}, 'kernel_num_gb': 0.441556992, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 108544
    r0_numel = 127
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 53)
    x1 = xindex // 53
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 127*x0
        tmp1 = tl.full([1, 1], 6727, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr0 + (1 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = tl.load(in_ptr0 + (62 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = triton_helpers.maximum(tmp5, tmp6)
        tmp8 = tl.load(in_ptr0 + (63 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = triton_helpers.maximum(tmp7, tmp8)
        tmp10 = tl.load(in_ptr0 + (3872 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = triton_helpers.maximum(tmp9, tmp10)
        tmp12 = tl.load(in_ptr0 + (3873 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = triton_helpers.maximum(tmp11, tmp12)
        tmp14 = tl.load(in_ptr0 + (3934 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = triton_helpers.maximum(tmp13, tmp14)
        tmp16 = tl.load(in_ptr0 + (3935 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)


def get_args():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (867328, 54208, 3872, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 1, 1, 1, 53), (848, 53, 108544, 108544, 108544, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 108544, 127,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.441556992
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ke/cketd4v3bb5y4hw6vntrpngqhbthsazd4pjckrocvueg7lzsuou2.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.div, aten.max_pool3d_with_indices, aten.mean]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
#   x_2 => _low_memory_max_pool_with_offsets
#   x_3 => mean
# Graph fragment:
#   %buf2 : Tensor "f32[128, 16, 1, 1, 1, 53][848, 53, 108544, 108544, 108544, 1]cuda:0" = PlaceHolder[target=buf2]
#   %convolution : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, 2.0), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %mean : Tensor "f32[128, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%getitem, [-1, -2, -3], True), kwargs = {})
#   return %buf3
triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2 = async_compile.triton('triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2', '''

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
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16384, 'r0_': 434176}, 'kernel_num_gb': 0.000442368, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 53
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 53*x0), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, mask=xmask)

def get_args():
    arg_0 = rand_strided((128, 16, 1, 1, 1, 53), (848, 53, 108544, 108544, 108544, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 1, 1, 1), (16, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 2048, 53,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000442368
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/jk/cjkbuhcsm6igwfqbyadekmfant6f7kqsf6xahsyofor3ppxob2y3.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.convolution, aten.div, aten.max_pool3d_with_indices, aten.mean, aten.add, aten.sum]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
#   x_2 => _low_memory_max_pool_with_offsets
#   x_3 => mean
#   x_4 => add
#   x_5 => sum_1
# Graph fragment:
#   %buf3 : Tensor "f32[128, 16, 1, 1, 1][16, 1, 2048, 2048, 2048]cuda:0" = PlaceHolder[target=buf3]
#   %arg3_1 : Tensor "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 62, 62][861056, 53816, 3844, 62, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, 2.0), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %mean : Tensor "f32[128, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%getitem, [-1, -2, -3], True), kwargs = {})
#   %add : Tensor "f32[128, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %arg3_1), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add, [1]), kwargs = {})
#   return %sum_1
triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3 = async_compile.triton('triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3', '''

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
    size_hints={'x': 132, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 8256}, 'kernel_num_gb': 8.768e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 6727.0, tl.float32)
    tmp2 = (tmp0 / tmp1)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)

def get_args():
    arg_0 = rand_strided((128, 16, 1, 1, 1), (16, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 128, 16,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 8.768e-06
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
        assert_size_stride(arg0_1, (16, 8, 3, 3, 3), (216, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (16, ), (1, ))
        assert_size_stride(arg2_1, (128, 8, 16, 64, 64), (524288, 65536, 4096, 64, 1))
        assert_size_stride(arg3_1, (16, 1, 1, 1), (1, 1, 1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (128, 16, 14, 62, 62), (861056, 53816, 3844, 62, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((128, 16, 14, 62, 62), (867328, 54208, 3872, 62, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.div]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_div_0.run(buf0, arg1_1, buf1, 110215168, stream=stream0)
            del arg1_1
            del buf0
            buf2 = empty_strided_cuda((128, 16, 1, 1, 1, 53), (848, 53, 108544, 108544, 108544, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.div, aten.max_pool3d_with_indices, aten.mean]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1.run(buf1, buf2, 108544, 127, stream=stream0)
            del buf1
            buf3 = empty_strided_cuda((128, 16, 1, 1, 1), (16, 1, 2048, 2048, 2048), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.div, aten.max_pool3d_with_indices, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2.run(buf2, buf3, 2048, 53, stream=stream0)
            del buf2
            buf4 = empty_strided_cuda((128, 1, 1, 1), (1, 1, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.convolution, aten.div, aten.max_pool3d_with_indices, aten.mean, aten.add, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3.run(buf3, arg3_1, buf4, 128, 16, stream=stream0)
            del arg3_1
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16, 8, 3, 3, 3), (216, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 8, 16, 64, 64), (524288, 65536, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_div_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110215168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    x2 = ((xindex // 53816) % 16)
    x0 = (xindex % 3844)
    x5 = xindex // 3844
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x0 + 3872*x5), tmp4, None)

''',
    '''@triton.jit
def triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 108544
    r0_numel = 127
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 53)
    x1 = xindex // 53
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 127*x0
        tmp1 = tl.full([1, 1], 6727, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr0 + (1 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.maximum(tmp3, tmp4)
        tmp6 = tl.load(in_ptr0 + (62 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.maximum(tmp5, tmp6)
        tmp8 = tl.load(in_ptr0 + (63 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.maximum(tmp7, tmp8)
        tmp10 = tl.load(in_ptr0 + (3872 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.maximum(tmp9, tmp10)
        tmp12 = tl.load(in_ptr0 + (3873 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.maximum(tmp11, tmp12)
        tmp14 = tl.load(in_ptr0 + (3934 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.maximum(tmp13, tmp14)
        tmp16 = tl.load(in_ptr0 + (3935 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.maximum(tmp15, tmp16)
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)

''',
    '''@triton.jit
def triton_per_fused_convolution_div_max_pool3d_with_indices_mean_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 53
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 53*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)

''',
    '''@triton.jit
def triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 6727.0, tl.float32)
    tmp2 = (tmp0 / tmp1)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((16, 8, 3, 3, 3), (216, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 8, 16, 64, 64), (524288, 65536, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
