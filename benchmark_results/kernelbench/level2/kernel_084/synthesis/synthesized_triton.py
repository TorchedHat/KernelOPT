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


# kernel path: /workspace/tmp/torchinductor_user/m2/cm2l7fxwjhl4qsvg3xj4a5sb5667jksjbre76suk4zm3lukeklfh.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.t, aten.addmm]
# Source node to ATen node mapping:
#   x => mm_default, permute
# Graph fragment:
#   %arg2_1 : Tensor "f32[1024, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %permute : Tensor "f32[8192, 8192][1, 8192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0]), kwargs = {})
#   %mm_default : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg2_1, %permute), kwargs = {})
#   return %mm_default
triton_tem_fused_addmm_t_0 = async_compile.triton('triton_tem_fused_addmm_t_0', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_t_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.33554432, 'kernel_flop': 137438953472, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_t_0(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1024
    N = 8192
    K = 8192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 8192
    stride_ak = 1
    stride_bk = 1
    stride_bn = 8192

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 8192*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 8192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 8192*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 512, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_t_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.33554432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/i3/ci3inu4rjxdosbewknqdr6q6bxsagocnh7fj757h2fa4pqqyi7g4.py
# Topologically Sorted Source Nodes: [x, x_1, x_3], Original ATen: [aten.addmm, aten._native_batch_norm_legit_no_training, aten.ge, aten.scalar_tensor, aten.neg, aten.where, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   x => add_tensor
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub
#   x_3 => div, exp, sum_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_default : Tensor "f32[1024, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg3_1 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %mul_tensor : Tensor "f32[1024, 8192][8192, 1]cuda:0" = PlaceHolder[target=mul_tensor]
#   %amax_default : Tensor "f32[1024, 1][1, 1024]cuda:0" = PlaceHolder[target=amax_default]
#   %sum_1 : Tensor "f32[1024, 1][1, 1024]cuda:0" = PlaceHolder[target=sum_1]
#   %add_tensor : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mm_default), kwargs = {})
#   %sub : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %arg3_1), kwargs = {})
#   %add : Tensor "f32[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %mul_1 : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %mul), kwargs = {})
#   %mul_2 : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %add_1 : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
#   %ge_scalar : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg7_1, 0), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (1,), kwargs = {dtype: torch.float32, device: cuda:0, pin_memory: False})
#   %neg_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%scalar_tensor_default,), kwargs = {})
#   %where_self : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_scalar, %scalar_tensor_default, %neg_default), kwargs = {})
#   %mul_tensor : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %where_self), kwargs = {})
#   %amax_default : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %arg7_1), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, %mul_tensor_1), kwargs = {})
#   %exp : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_2,), kwargs = {})
#   %sum_1 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : Tensor "f32[1024, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %mul_tensor,%amax_default,%sum_1,%div
triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1', '''
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
    size_hints={'x': 1024, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 100827136}, 'kernel_num_gb': 0.067272708, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp18 = tl.load(in_ptr5 + (0))
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp20 = tl.full([1, 1], 0.0, tl.float32)
        tmp21 = tmp19 >= tmp20
        tmp22 = tl.full([1, 1], -1.0, tl.float32)
        tmp23 = tl.where(tmp21, tmp11, tmp22)
        tmp24 = tmp17 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = triton_helpers.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp24, r0_mask & xmask)
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
    tmp30 = tl.load(in_ptr5 + (0))
    tmp31 = tl.broadcast_to(tmp30, [1, 1])
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp28 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tmp28 - tmp26
        tmp32 = tl.full([1, 1], 0.0, tl.float32)
        tmp33 = tmp31 >= tmp32
        tmp34 = tl.full([1, 1], 1.0, tl.float32)
        tmp35 = tl.full([1, 1], -1.0, tl.float32)
        tmp36 = tl.where(tmp33, tmp34, tmp35)
        tmp37 = tmp36 * tmp31
        tmp38 = tmp29 * tmp37
        tmp39 = libdevice.exp(tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp45 = tl.load(in_ptr5 + (0))
    tmp46 = tl.broadcast_to(tmp45, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp43 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tmp43 - tmp26
        tmp47 = tl.full([1, 1], 0.0, tl.float32)
        tmp48 = tmp46 >= tmp47
        tmp49 = tl.full([1, 1], 1.0, tl.float32)
        tmp50 = tl.full([1, 1], -1.0, tl.float32)
        tmp51 = tl.where(tmp48, tmp49, tmp50)
        tmp52 = tmp51 * tmp46
        tmp53 = tmp44 * tmp52
        tmp54 = libdevice.exp(tmp53)
        tmp55 = (tmp54 / tmp41)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp55, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1024, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.067272708
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
        assert_size_stride(arg1_1, (8192, ), (1, ))
        assert_size_stride(arg2_1, (1024, 8192), (8192, 1))
        assert_size_stride(arg3_1, (8192, ), (1, ))
        assert_size_stride(arg4_1, (8192, ), (1, ))
        assert_size_stride(arg5_1, (8192, ), (1, ))
        assert_size_stride(arg6_1, (8192, ), (1, ))
        assert_size_stride(arg7_1, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_0.run(arg2_1, arg0_1, buf0, 512, 1, 1, stream=stream0)
            del arg0_1
            del arg2_1
            buf1 = buf0; del buf0  # reuse
            buf4 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_3], Original ATen: [aten.addmm, aten._native_batch_norm_legit_no_training, aten.ge, aten.scalar_tensor, aten.neg, aten.where, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1.run(buf4, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, 1024, 8192, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.bn(x)
        x = self.scale * x
        x = self.softmax(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_tem_fused_addmm_t_0(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1024
    N = 8192
    K = 8192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 8192
    stride_ak = 1
    stride_bk = 1
    stride_bn = 8192

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 8192*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 8192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 8192*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp18 = tl.load(in_ptr5 + (0))
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp20 = tl.full([1, 1], 0.0, tl.float32)
        tmp21 = tmp19 >= tmp20
        tmp22 = tl.full([1, 1], -1.0, tl.float32)
        tmp23 = tl.where(tmp21, tmp11, tmp22)
        tmp24 = tmp17 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = tl.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp24, r0_mask & xmask)
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
    tmp30 = tl.load(in_ptr5 + (0))
    tmp31 = tl.broadcast_to(tmp30, [1, 1])
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp28 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tmp28 - tmp26
        tmp32 = tl.full([1, 1], 0.0, tl.float32)
        tmp33 = tmp31 >= tmp32
        tmp34 = tl.full([1, 1], 1.0, tl.float32)
        tmp35 = tl.full([1, 1], -1.0, tl.float32)
        tmp36 = tl.where(tmp33, tmp34, tmp35)
        tmp37 = tmp36 * tmp31
        tmp38 = tmp29 * tmp37
        tmp39 = tl.exp(tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp45 = tl.load(in_ptr5 + (0))
    tmp46 = tl.broadcast_to(tmp45, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp43 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tmp43 - tmp26
        tmp47 = tl.full([1, 1], 0.0, tl.float32)
        tmp48 = tmp46 >= tmp47
        tmp49 = tl.full([1, 1], 1.0, tl.float32)
        tmp50 = tl.full([1, 1], -1.0, tl.float32)
        tmp51 = tl.where(tmp48, tmp49, tmp50)
        tmp52 = tmp51 * tmp46
        tmp53 = tmp44 * tmp52
        tmp54 = tl.exp(tmp53)
        tmp55 = (tmp54 / tmp41)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp55, r0_mask & xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((8192, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1]

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
