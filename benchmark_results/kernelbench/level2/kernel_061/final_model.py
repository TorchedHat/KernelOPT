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


# kernel path: /workspace/tmp/torchinductor_user/6p/c6p6kwmig6px3dsaeljqdjt25elvgh2lcjdzyxwbhw6l2hbz2voh.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_1 => relu
#   x_2 => var_mean, view
# Graph fragment:
#   %convolution : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0" = PlaceHolder[target=convolution]
#   %relu : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %view : Tensor "f32[16, 8, 16, 39304][5030912, 628864, 39304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%relu, [16, 8, 16, 39304]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf1,%buf2,%buf3
triton_red_fused_native_group_norm_relu_0 = async_compile.triton('triton_red_fused_native_group_norm_relu_0', '''

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
    size_hints={'x': 1024, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_relu_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15360, 'r0_': 321978368}, 'kernel_num_gb': 0.321986048, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_native_group_norm_relu_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 125773
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        # Boundary check removed: r0_2 + 125773*x0 is always < 628864
        # Proof: max = 125772 + 125773*4 = 628863 < 628864, so modulo is a no-op
        tmp3 = tl.load(in_ptr0 + (628864*x1 + r0_2 + 125773*x0), mask=xmask & r0_mask, other=0.0, eviction_policy='evict_first')
        tmp5 = tl.maximum(tl.zeros([1, 1], tl.float32), tmp3)  # ReLU preserved
        tmp14 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        # tmp15 (mean2) is always 0.0 — eliminated
        # tmp16 (weight) is always 1.0 — use literal 1.0 in Welford update
        _valid = r0_mask & xmask
        _new_count    = tmp17_weight + 1.0
        _delta        = tmp14 - tmp17_mean
        _weight_ratio = 1.0 / _new_count
        _mean_next    = tl.math.fma(_delta, _weight_ratio, tmp17_mean)
        _delta2       = tmp14 - _mean_next
        _m2_next      = tl.math.fma(_delta, _delta2, tmp17_m2)
        tmp17_mean    = tl.where(_valid, _mean_next,   tmp17_mean)
        tmp17_m2      = tl.where(_valid, _m2_next,     tmp17_m2)
        tmp17_weight  = tl.where(_valid, _new_count,   tmp17_weight)
    tmp18, tmp19, tmp20 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tmp22 = tmp20[:, None]
    tl.store(out_ptr0 + (x3), tmp17, mask=xmask)
    tl.store(out_ptr1 + (x3), tmp21, mask=xmask)
    tl.store(out_ptr2 + (x3), tmp22, mask=xmask)

def get_args():
    arg_0 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 640, 125773,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_relu_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_native_group_norm_relu_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.321986048
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/3h/c3h6yjru6q5xp6jioh3zf377p7z7x33nm6ineibplltmvhith46d.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_1 => relu
#   x_2 => var_mean, view
# Graph fragment:
#   %buf1 : Tensor "f32[16, 8, 1, 1, 5][40, 5, 640, 640, 1]cuda:0" = PlaceHolder[target=buf1]
#   %buf2 : Tensor "f32[16, 8, 1, 1, 5][40, 5, 640, 640, 1]cuda:0" = PlaceHolder[target=buf2]
#   %buf3 : Tensor "f32[16, 8, 1, 1, 5][40, 5, 640, 640, 1]cuda:0" = PlaceHolder[target=buf3]
#   %relu : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %view : Tensor "f32[16, 8, 16, 39304][5030912, 628864, 39304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%relu, [16, 8, 16, 39304]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf5
triton_per_fused_native_group_norm_relu_1 = async_compile.triton('triton_per_fused_native_group_norm_relu_1', '''

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
    size_hints={'x': 132, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2048, 'r0_': 4800}, 'kernel_num_gb': 8.704e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_native_group_norm_relu_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 5*x0), mask=r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, mask=xmask)
    tl.store(out_ptr1 + (x0), tmp14, mask=xmask)

def get_args():
    arg_0 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 128, 5,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_relu_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_native_group_norm_relu_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 8.704e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/6c/c6chdljdp4gbo7bgc2xxx4v26nmxdxtiadb5mpd3ptfnpvhfk5aa.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_1 => relu
#   x_2 => add, add_1, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, var_mean, view, view_1
# Graph fragment:
#   %convolution : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_1 : Tensor "f32[16, 8, 1, 1][8, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf5 : Tensor "f32[16, 8, 1, 1][8, 1, 128, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg2_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %relu : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %view : Tensor "f32[16, 8, 16, 39304][5030912, 628864, 39304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%relu, [16, 8, 16, 39304]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[16, 8, 16, 39304][5030912, 628864, 39304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[16, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[16, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[16, 8, 16, 39304][5030912, 628864, 39304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [16, 128, 34, 34, 34]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 128, 1, 1, 1][128, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 4), kwargs = {})
#   %mul_1 : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 128, 1, 1, 1][128, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 4), kwargs = {})
#   %add_1 : Tensor "f32[16, 128, 34, 34, 34][5030912, 39304, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_7), kwargs = {})
#   return %add_1
triton_poi_fused_native_group_norm_relu_2 = async_compile.triton('triton_poi_fused_native_group_norm_relu_2', '''

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 965935104}, 'kernel_num_gb': 0.643958784, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80494592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_chan  = xindex // 39304
    x_group = x_chan // 16
    x2      = x_chan % 128
    tmp0     = tl.load(in_ptr0 + xindex, xmask)
    tmp_relu = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp0)
    tmp_mean     = tl.load(in_ptr1 + x_group, xmask, eviction_policy='evict_last')
    tmp_centered = tmp_relu - tmp_mean
    tmp_var     = tl.load(in_ptr2 + x_group, xmask, eviction_policy='evict_last')
    tmp_inv_std = libdevice.rsqrt(tmp_var * 1.5908420085906982e-06 + 1e-05)
    tmp_weight = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp_bias   = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp_out    = tmp_centered * tmp_inv_std * tmp_weight + tmp_bias
    tl.store(out_ptr0 + xindex, tmp_out, xmask)

def get_args():
    arg_0 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1), (8, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 80494592,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_native_group_norm_relu_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.643958784
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
        assert_size_stride(arg0_1, (64, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (16, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        assert_size_stride(arg2_1, (128, ), (1, ))
        assert_size_stride(arg3_1, (128, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg1_1
            buf1 = empty_strided_cuda((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), torch.float32)
            buf2 = empty_strided_cuda((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), torch.float32)
            buf3 = empty_strided_cuda((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_group_norm_relu_0.run(buf0, buf1, buf2, buf3, 640, 125773, stream=stream0)
            buf4 = empty_strided_cuda((16, 8, 1, 1), (8, 1, 128, 128), torch.float32)
            buf5 = empty_strided_cuda((16, 8, 1, 1), (8, 1, 128, 128), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_relu_1.run(buf1, buf2, buf3, buf4, buf5, 128, 5, stream=stream0)
            del buf1
            del buf2
            del buf3
            buf7 = empty_strided_cuda((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.relu, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_relu_2.run(buf0, buf4, buf5, arg2_1, arg3_1, buf7, 80494592, stream=stream0)
            del arg2_1
            del arg3_1
            del buf0
            del buf4
            del buf5
        return (buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_red_fused_native_group_norm_relu_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 125773
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 125773*x0
        tmp1 = tl.full([1, 1], 628864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (628864*x1 + (((r0_2 + 125773*x0) % 628864))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tl.maximum(tmp4, tmp3)
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.full([1, 1], 0.0, tl.float32)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(r0_mask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(r0_mask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(r0_mask & xmask, tmp17_weight_next, tmp17_weight)
    tmp18, tmp19, tmp20 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tmp22 = tmp20[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp21, xmask)
    tl.store(out_ptr2 + (x3), tmp22, xmask)

''',
    '''@triton.jit
def triton_per_fused_native_group_norm_relu_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 5*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)

''',
    '''@triton.jit
def triton_poi_fused_native_group_norm_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80494592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    x6 = xindex // 39304
    x2 = ((xindex // 39304) % 128)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + (x6 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x6 // 16), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0)
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 628864.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x4), tmp15, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((64, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
