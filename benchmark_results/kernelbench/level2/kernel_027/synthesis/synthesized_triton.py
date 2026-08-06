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


# kernel path: /workspace/tmp/torchinductor_user/qx/cqxtm3doh65qw27qlhqw2cv4tsaisvcsumwumytxmzuoulwecz6q.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.hardswish, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, clamp_max, clamp_min, div, mul
#   x_2 => var_mean, view
# Graph fragment:
#   %buf0 : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, 3), kwargs = {})
#   %clamp_min : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0), kwargs = {})
#   %clamp_max : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %clamp_max), kwargs = {})
#   %div : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 6), kwargs = {})
#   %view : Tensor "f32[1024, 4, 4, 10933][174928, 43732, 10933, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [1024, 4, 4, 10933]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf2
triton_red_fused_convolution_hardswish_native_group_norm_0 = async_compile.triton('triton_red_fused_convolution_hardswish_native_group_norm_0', '''
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
    size_hints={'x': 4096, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_hardswish_native_group_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 65536, 'r0_': 716505088}, 'kernel_num_gb': 0.71653792, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_convolution_hardswish_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 43732
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 4)
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_3 = r0_index // 10933
        tmp0 = tl.load(in_ptr0 + (r0_5 + 43732*x4), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 3.0, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0.0, tl.float32)
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = tl.full([1, 1], 6.0, tl.float32)
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask, tmp13_weight_next, tmp13_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tl.store(out_ptr0 + (x4), tmp13, None)
    tl.store(out_ptr1 + (x4), tmp17, None)


def get_args():
    arg_0 = rand_strided((1024, 16, 13, 29, 29), (174928, 10933, 841, 29, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 4096, 43732,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_hardswish_native_group_norm_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_convolution_hardswish_native_group_norm_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.71653792
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/d2/cd2ylknbvrlejp4jllii4wof3sndkbpfoayikal2e666q6zssku4.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.hardswish, aten.native_group_norm, aten.mean]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, clamp_max, clamp_min, div, mul
#   x_2 => add_1, add_2, mul_1, mul_2, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, var_mean, view, view_1
#   x_3 => mean
# Graph fragment:
#   %buf0 : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_1 : Tensor "f32[1024, 4, 1, 1][4, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[1024, 4, 1, 1][4, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf2]
#   %arg3_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf4 : Tensor "f32[1024, 16][16, 1]cuda:0" = PlaceHolder[target=buf4]
#   %convolution : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, 3), kwargs = {})
#   %clamp_min : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0), kwargs = {})
#   %clamp_max : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %clamp_max), kwargs = {})
#   %div : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 6), kwargs = {})
#   %view : Tensor "f32[1024, 4, 4, 10933][174928, 43732, 10933, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [1024, 4, 4, 10933]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1024, 4, 4, 10933][174928, 43732, 10933, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[1024, 4, 1, 1][4, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1024, 4, 1, 1][4, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_1 : Tensor "f32[1024, 4, 4, 10933][174928, 43732, 10933, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_1, [1024, 16, 13, 29, 29]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 4), kwargs = {})
#   %mul_2 : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 16, 1, 1, 1][16, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 4), kwargs = {})
#   %add_2 : Tensor "f32[1024, 16, 13, 29, 29][174928, 10933, 841, 29, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %mean : Tensor "f32[1024, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_2, [2, 3, 4]), kwargs = {})
#   return %buf4,%mean
triton_red_fused_convolution_hardswish_mean_native_group_norm_1 = async_compile.triton('triton_red_fused_convolution_hardswish_mean_native_group_norm_1', '''
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
    size_hints={'x': 16384, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_hardswish_mean_native_group_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 164032, 'r0_': 716505088}, 'kernel_num_gb': 0.71666912, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_convolution_hardswish_mean_native_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 10933
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 16)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x4 // 4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_5 + 10933*x4), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 3.0, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0.0, tl.float32)
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = tl.full([1, 1], 6.0, tl.float32)
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 - tmp12
        tmp15 = tl.full([1, 1], 43732.0, tl.float32)
        tmp16 = (tmp14 / tmp15)
        tmp17 = tl.full([1, 1], 1e-05, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp13 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.full([1, 1], 10933.0, tl.float32)
    tmp29 = (tmp26 / tmp28)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp29, None)


def get_args():
    arg_0 = rand_strided((1024, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 16, 13, 29, 29), (174928, 10933, 841, 29, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 10933,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_hardswish_mean_native_group_norm_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_convolution_hardswish_mean_native_group_norm_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.71666912
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
        assert_size_stride(arg0_1, (16, 3, 4, 4, 4), (192, 64, 16, 4, 1))
        assert_size_stride(arg1_1, (16, ), (1, ))
        assert_size_stride(arg2_1, (1024, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1))
        assert_size_stride(arg3_1, (16, ), (1, ))
        assert_size_stride(arg4_1, (16, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (1024, 16, 13, 29, 29), (174928, 10933, 841, 29, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((1024, 4, 1, 1), (4, 1, 4096, 4096), torch.float32)
            buf2 = empty_strided_cuda((1024, 4, 1, 1), (4, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.hardswish, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_hardswish_native_group_norm_0.run(buf0, arg1_1, buf1, buf2, 4096, 43732, stream=stream0)
            buf4 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.hardswish, aten.native_group_norm, aten.mean]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_hardswish_mean_native_group_norm_1.run(buf5, buf0, arg1_1, buf1, buf2, arg3_1, arg4_1, 16384, 10933, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del buf0
            del buf1
            del buf2
        return (buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16, 3, 4, 4, 4), (192, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = F.hardswish(x)                           # Nonlinear activation
        x = self.group_norm(x)                       # Normalization over channels
        x = torch.mean(x, dim=[2, 3, 4])             # Mean over spatial dims → (B, C)
        return x

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_red_fused_convolution_hardswish_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 43732
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 4)
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_3 = r0_index // 10933
        tmp0 = tl.load(in_ptr0 + (r0_5 + 43732*x4), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 3.0, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0.0, tl.float32)
        tmp6 = tl.maximum(tmp4, tmp5)
        tmp7 = tl.full([1, 1], 6.0, tl.float32)
        tmp8 = tl.minimum(tmp6, tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask, tmp13_weight_next, tmp13_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tl.store(out_ptr0 + (x4), tmp13, None)
    tl.store(out_ptr1 + (x4), tmp17, None)

''',
    '''@triton.jit
def triton_red_fused_convolution_hardswish_mean_native_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 10933
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 16)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x4 // 4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_5 + 10933*x4), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 3.0, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0.0, tl.float32)
        tmp6 = tl.maximum(tmp4, tmp5)
        tmp7 = tl.full([1, 1], 6.0, tl.float32)
        tmp8 = tl.minimum(tmp6, tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 - tmp12
        tmp15 = tl.full([1, 1], 43732.0, tl.float32)
        tmp16 = (tmp14 / tmp15)
        tmp17 = tl.full([1, 1], 1e-05, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp13 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.full([1, 1], 10933.0, tl.float32)
    tmp29 = (tmp26 / tmp28)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp29, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((16, 3, 4, 4, 4), (192, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
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
