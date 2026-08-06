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


# kernel path: /workspace/tmp/torchinductor_user/ob/cobkayukt3722cbmdseaevfk7rgb6f7sumirdyfgr3wgsoxhofge.py
# Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward], Original ATen: [aten.view, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
#   view => view
#   x => permute
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view : Tensor "f32[2, 128, 16384][2097152, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2, 128, 16384]), kwargs = {})
#   %permute : Tensor "f32[16384, 2, 128][1, 2097152, 16384]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view, [2, 0, 1]), kwargs = {})
#   %clone : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_permute_view_0 = async_compile.triton('triton_poi_fused_clone_permute_view_0', '''
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
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 33554432}, 'kernel_num_gb': 0.033554432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16384*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 16384, 256,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_view_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_permute_view_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033554432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ac/cacyl4vhfje6nmddf6lwwyrycoxrv2uw4jb4hkehvjmcvulv4ep6.py
# Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view, aten.t, aten.mm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone, mm, permute_1, view_1
#   view => view
#   x => permute
# Graph fragment:
#   %clone : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0" = PlaceHolder[target=clone]
#   %arg2_1 : Tensor "f32[384, 128][128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "f32[2, 128, 16384][2097152, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2, 128, 16384]), kwargs = {})
#   %permute : Tensor "f32[16384, 2, 128][1, 2097152, 16384]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view, [2, 0, 1]), kwargs = {})
#   %clone : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1 : Tensor "f32[32768, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [32768, 128]), kwargs = {})
#   %permute_1 : Tensor "f32[128, 384][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg2_1, [1, 0]), kwargs = {})
#   %mm : Tensor "f32[32768, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_1, %permute_1), kwargs = {})
#   return %mm
triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1 = async_compile.triton('triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.067305472, 'kernel_flop': 3221225472, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1(arg_A, arg_B, out_ptr0):
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

    M = 32768
    N = 384
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
        xindex = idx_n + 128*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((16384, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 768, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.067305472
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/5w/c5wrrfdqshn73cgp4jvtv6ojjppu2aqktyi4jc5xfyc4zhrjkfeq.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add, clone_1, mul, permute_2, permute_3, select, squeeze, unsqueeze, view_2, view_3, view_4
# Graph fragment:
#   %mm : Tensor "f32[32768, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg1_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %view_2 : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16384, 2, 384]), kwargs = {})
#   %add : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %arg1_1), kwargs = {})
#   %view_3 : Tensor "f32[16384, 2, 3, 128][768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [16384, 2, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 16384, 2, 3, 128][12582912, 768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 16384, 2, 1, 128][128, 768, 384, 12582912, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 16384, 2, 128][128, 768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 16384, 2, 128][4194304, 256, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %view_4 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [16384, 8, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [1, 0, 2]), kwargs = {})
#   %mul : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_3, 0.1767766952966369), kwargs = {})
#   return %mul
triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2', '''
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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50332672}, 'kernel_num_gb': 0.033555968, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 16384)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (384*((x0 + 32*x2) // 128) + 768*x1 + 768*((x0 + 32*x2) // 256) + (((x0 + 32*x2) % 128))), None)
    tmp1 = tl.load(in_ptr1 + (((x0 + 32*x2) % 128)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.1767766952966369, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)


def get_args():
    arg_0 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033555968
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ub/cubgogr2zy3ylciq4bgrhvmy5euouz2lndai5yzn2hhqcd7njl4k.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add, clone_1, permute_2, squeeze, unsqueeze, view_2, view_3
# Graph fragment:
#   %mm : Tensor "f32[32768, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg1_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %view_2 : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16384, 2, 384]), kwargs = {})
#   %add : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %arg1_1), kwargs = {})
#   %view_3 : Tensor "f32[16384, 2, 3, 128][768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [16384, 2, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 16384, 2, 3, 128][12582912, 768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 16384, 2, 1, 128][128, 768, 384, 12582912, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 16384, 2, 128][128, 768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 16384, 2, 128][4194304, 256, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3', '''
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
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 150996480}, 'kernel_num_gb': 0.100664832, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 32768)
    x2 = xindex // 4194304
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)


def get_args():
    arg_0 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((3, 16384, 2, 128), (4194304, 256, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 12582912,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.100664832
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/2p/c2ppk5p75k23p5a7dvzqxamyzhi5tuuy5rzpd66edmdvehiiptly.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add, bmm, clone_1, mul, permute_2, permute_3, permute_4, permute_6, select, select_1, squeeze, unsqueeze, view_2, view_3, view_4, view_5
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 16384, 2, 128][4194304, 256, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_2 : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16384, 2, 384]), kwargs = {})
#   %add : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %arg1_1), kwargs = {})
#   %view_3 : Tensor "f32[16384, 2, 3, 128][768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [16384, 2, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 16384, 2, 3, 128][12582912, 768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 16384, 2, 1, 128][128, 768, 384, 12582912, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 16384, 2, 128][128, 768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 16384, 2, 128][4194304, 256, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %view_4 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [16384, 8, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [1, 0, 2]), kwargs = {})
#   %mul : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_3, 0.1767766952966369), kwargs = {})
#   %select_1 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 1), kwargs = {})
#   %view_5 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [16384, 8, 32]), kwargs = {})
#   %permute_4 : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [1, 0, 2]), kwargs = {})
#   %permute_6 : Tensor "f32[8, 32, 16384][32, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [0, 2, 1]), kwargs = {})
#   %bmm : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bmm.default](args = (%mul, %permute_6), kwargs = {})
#   return %buf4
triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4', '''
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
    size_hints={'y': 256, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 33554432}, 'kernel_num_gb': 0.033554432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (4194304 + y0 + 256*x1), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp0, ymask)


def get_args():
    arg_0 = rand_strided((3, 16384, 2, 128), (4194304, 256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 32, 16384), (524288, 16384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 256, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033554432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/lb/clbbsnrwghdlcvbaph7bvzjiyksq4lt4zrerensqgm6kmlamekvj.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add, bmm, clone_1, mul, permute_2, permute_3, permute_4, permute_6, select, select_1, squeeze, unsqueeze, view_2, view_3, view_4, view_5
# Graph fragment:
#   %mul : Tensor "f32[8, 16384, 32][524288, 32, 1]cuda:0" = PlaceHolder[target=mul]
#   %buf4 : Tensor "f32[8, 32, 16384][524288, 16384, 1]cuda:0" = PlaceHolder[target=buf4]
#   %view_2 : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16384, 2, 384]), kwargs = {})
#   %add : Tensor "f32[16384, 2, 384][768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %arg1_1), kwargs = {})
#   %view_3 : Tensor "f32[16384, 2, 3, 128][768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [16384, 2, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 16384, 2, 3, 128][12582912, 768, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 16384, 2, 1, 128][128, 768, 384, 12582912, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 16384, 2, 128][128, 768, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 16384, 2, 128][4194304, 256, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %view_4 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [16384, 8, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [1, 0, 2]), kwargs = {})
#   %mul : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_3, 0.1767766952966369), kwargs = {})
#   %select_1 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 1), kwargs = {})
#   %view_5 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [16384, 8, 32]), kwargs = {})
#   %permute_4 : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [1, 0, 2]), kwargs = {})
#   %permute_6 : Tensor "f32[8, 32, 16384][32, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [0, 2, 1]), kwargs = {})
#   %bmm : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bmm.default](args = (%mul, %permute_6), kwargs = {})
#   return %bmm
triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5 = async_compile.triton('triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 8.623489024, 'kernel_flop': 137438953472, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B

    M = 16384
    N = 16384
    K = 32

    stride_aq = 524288
    stride_am = 32
    stride_ak = 1

    stride_bq = 524288
    stride_bk = 16384
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16384*idx_m + 268435456*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 32, 16384), (524288, 16384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 16384), (268435456, 16384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 65536, 8, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 8.623489024
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/jo/cjo3iqupad5x2h4fi6ynnpl5mvicu7x5w6odi3nibjfmivdb3p5q.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#   multi_head_attention_forward => div
# Graph fragment:
#   %bmm : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0" = PlaceHolder[target=bmm]
#   %getitem_2 : Tensor "f32[8, 16384, 1][16384, 1, 131072]cuda:0" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f32[8, 16384, 1][16384, 1, 131072]cuda:0" = PlaceHolder[target=getitem_3]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%bmm, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%bmm, %getitem_2), kwargs = {})
#   %exp_default : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_3), kwargs = {})
#   return %getitem_2,%getitem_3,%div
triton_red_fused__softmax_exp_prepare_softmax_online_sub_6 = async_compile.triton('triton_red_fused__softmax_exp_prepare_softmax_online_sub_6', '''
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
    size_hints={'x': 131072, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_exp_prepare_softmax_online_sub_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 25769803776}, 'kernel_num_gb': 17.179869184, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__softmax_exp_prepare_softmax_online_sub_6(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp2_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp2_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])

        _tmp2_max_next, _tmp2_sum_next = triton_helpers.online_softmax_combine(
            _tmp2_max, _tmp2_sum, tmp1, False
        )

        _tmp2_max = tl.where(r0_mask, _tmp2_max_next, _tmp2_max)
        _tmp2_sum = tl.where(r0_mask, _tmp2_sum_next, _tmp2_sum)

    tmp2, tmp3 = triton_helpers.online_softmax_reduce(
        _tmp2_max, _tmp2_sum, 1, False)
    tmp2 = tmp2[:, None]
    tmp3 = tmp3[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp4 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = libdevice.exp(tmp5)
        tmp7 = (tmp6 / tmp3)
        tl.store(in_out_ptr0 + (r0_1 + 16384*x0), tmp7, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 16384, 16384), (268435456, 16384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, 131072, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_exp_prepare_softmax_online_sub_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__softmax_exp_prepare_softmax_online_sub_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 17.179869184
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/c6/cc6ihn5gjubwu5tl7xe5jkexzfz3gmbybuun7iv5hnxctgg4pv6a.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.select, aten.view, aten.transpose, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => bmm_1, div, permute_5, select_2, view_6
# Graph fragment:
#   %div : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0" = PlaceHolder[target=div]
#   %clone_1 : Tensor "f32[3, 16384, 2, 128][4194304, 256, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %sub_tensor : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%bmm, %getitem_2), kwargs = {})
#   %exp_default : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[8, 16384, 16384][268435456, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_3), kwargs = {})
#   %select_2 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 2), kwargs = {})
#   %view_6 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [16384, 8, 32]), kwargs = {})
#   %permute_5 : Tensor "f32[8, 16384, 32][32, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %bmm_1 : Tensor "f32[8, 16384, 32][524288, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%div, %permute_5), kwargs = {})
#   return %bmm_1
triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7 = async_compile.triton('triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 8.623489024, 'kernel_flop': 137438953472, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B + 8388608

    M = 16384
    N = 32
    K = 16384

    stride_aq = 268435456
    stride_am = 16384
    stride_ak = 1

    stride_bq = 32
    stride_bk = 256
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 32*idx_m + 524288*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((8, 16384, 16384), (268435456, 16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3, 16384, 2, 128), (4194304, 256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 128, 8, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 8.623489024
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/bw/cbwp6njrydfucn3sl6o7tyqirgbbs7xxkqqpyboyzsil5tvcooar.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2, permute_7
# Graph fragment:
#   %bmm_1 : Tensor "f32[8, 16384, 32][524288, 32, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %permute_7 : Tensor "f32[16384, 8, 32][32, 524288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_transpose_8 = async_compile.triton('triton_poi_fused_clone_transpose_8', '''
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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50331648}, 'kernel_num_gb': 0.033554432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 8)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 524288*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)


def get_args():
    arg_0 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 8, 32), (256, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_transpose_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033554432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/nl/cnlcxoabrxcvjelctb3okpx4odqfrfgy6ur4xme7kkhg2vl27gng.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2, mm_default, permute_7, permute_8, view_7
# Graph fragment:
#   %clone_2 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0" = PlaceHolder[target=clone_2]
#   %arg3_1 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %permute_7 : Tensor "f32[16384, 8, 32][32, 524288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[16384, 8, 32][256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_7 : Tensor "f32[32768, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [32768, 128]), kwargs = {})
#   %permute_8 : Tensor "f32[128, 128][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0]), kwargs = {})
#   %mm_default : Tensor "f32[32768, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_7, %permute_8), kwargs = {})
#   return %mm_default
triton_tem_fused_addmm_clone_t_transpose_view_9 = async_compile.triton('triton_tem_fused_addmm_clone_t_transpose_view_9', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_clone_t_transpose_view_9', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.033619968, 'kernel_flop': 1073741824, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_clone_t_transpose_view_9(arg_A, arg_B, out_ptr0):
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

    M = 32768
    N = 128
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
        xindex = idx_n + 128*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((16384, 8, 32), (256, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 256, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_clone_t_transpose_view_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_clone_t_transpose_view_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033619968
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/lx/clxotdvsqbuw2sa3uuqmyqe77ezmv5hgaf5rcuykk5tcix43npn6.py
# Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward, add, x_1], Original ATen: [aten.view, aten.permute, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_1
#   multi_head_attention_forward => add_tensor, view_8
#   view => view
#   x => permute
#   x_1 => var_mean
# Graph fragment:
#   %arg4_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mm_default : Tensor "f32[32768, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg0_1 : Tensor "f32[2, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view : Tensor "f32[2, 128, 16384][2097152, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2, 128, 16384]), kwargs = {})
#   %permute : Tensor "f32[16384, 2, 128][1, 2097152, 16384]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view, [2, 0, 1]), kwargs = {})
#   %add_tensor : Tensor "f32[32768, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %mm_default), kwargs = {})
#   %view_8 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [16384, 2, 128]), kwargs = {})
#   %add_1 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %permute), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf13
triton_red_fused_add_addmm_native_layer_norm_permute_view_10 = async_compile.triton('triton_red_fused_add_addmm_native_layer_norm_permute_view_10', '''
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
    size_hints={'y': 16384, 'x': 2, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_native_layer_norm_permute_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 131072, 'r0_': 16777728}, 'kernel_num_gb': 0.033817088, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_add_addmm_native_layer_norm_permute_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 2
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    tmp6_mean = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 256*y0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (y0 + 16384*r0_2 + 2097152*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 2)
    tmp6 = tmp7[:, :, None]
    tmp10 = tmp8[:, :, None]
    tmp11 = tmp9[:, :, None]
    tl.store(out_ptr0 + (x1 + 2*y0), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + 2*y0), tmp10, xmask)


def get_args():
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 16384, 2, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_native_layer_norm_permute_view_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_add_addmm_native_layer_norm_permute_view_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033817088
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/fm/cfm4srbitozq5qrlh2jbz72beve2w4huhq3eug3d2axjgtmzzc6y.py
# Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward, add, x_1], Original ATen: [aten.view, aten.permute, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_1
#   multi_head_attention_forward => add_tensor, view_8
#   view => view
#   x => permute
#   x_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# Graph fragment:
#   %arg4_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mm_default : Tensor "f32[32768, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg0_1 : Tensor "f32[2, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_1 : Tensor "f32[16384, 2, 1][2, 1, 32768]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf13 : Tensor "f32[16384, 2, 1][2, 1, 32768]cuda:0" = PlaceHolder[target=buf13]
#   %arg5_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %view : Tensor "f32[2, 128, 16384][2097152, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2, 128, 16384]), kwargs = {})
#   %permute : Tensor "f32[16384, 2, 128][1, 2097152, 16384]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view, [2, 0, 1]), kwargs = {})
#   %add_tensor : Tensor "f32[32768, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %mm_default), kwargs = {})
#   %view_8 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [16384, 2, 128]), kwargs = {})
#   %add_1 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %permute), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[16384, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[16384, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %add_3 : Tensor "f32[16384, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
#   return %add_3
triton_poi_fused_add_addmm_native_layer_norm_permute_view_11 = async_compile.triton('triton_poi_fused_add_addmm_native_layer_norm_permute_view_11', '''
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
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_native_layer_norm_permute_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 50333184}, 'kernel_num_gb': 0.050595328, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_native_layer_norm_permute_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 128)
    x3 = xindex
    y0 = yindex
    x2 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + 256*y0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 16384*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 2*y0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 2*y0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tl.full([1, 1], 128.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 256*y0), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((16384, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 256,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_native_layer_norm_permute_view_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_addmm_native_layer_norm_permute_view_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.050595328
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2, 128, 128, 128), (2097152, 16384, 128, 1))
        assert_size_stride(arg1_1, (384, ), (1, ))
        assert_size_stride(arg2_1, (384, 128), (128, 1))
        assert_size_stride(arg3_1, (128, 128), (128, 1))
        assert_size_stride(arg4_1, (128, ), (1, ))
        assert_size_stride(arg5_1, (128, ), (1, ))
        assert_size_stride(arg6_1, (128, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16384, 2, 128), (256, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward], Original ATen: [aten.view, aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_view_0.run(arg0_1, buf0, 16384, 256, stream=stream0)
            buf1 = empty_strided_cuda((32768, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1.run(buf0, arg2_1, buf1, 768, 1, 1, stream=stream0)
            del arg2_1
            buf2 = reinterpret_tensor(buf0, (8, 16384, 32), (524288, 32, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf1, arg1_1, buf2, 4194304, stream=stream0)
            buf3 = empty_strided_cuda((3, 16384, 2, 128), (4194304, 256, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3.run(buf1, arg1_1, buf3, 12582912, stream=stream0)
            del arg1_1
            del buf1
            buf4 = empty_strided_cuda((8, 32, 16384), (524288, 16384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4.run(buf3, buf4, 256, 16384, stream=stream0)
            buf5 = empty_strided_cuda((8, 16384, 16384), (268435456, 16384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5.run(buf2, buf4, buf5, 65536, 8, 1, stream=stream0)
            del buf2
            del buf4
            buf8 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__softmax_exp_prepare_softmax_online_sub_6.run(buf8, 131072, 16384, stream=stream0)
            buf9 = empty_strided_cuda((8, 16384, 32), (524288, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.select, aten.view, aten.transpose, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7.run(buf8, buf3, buf9, 128, 8, 1, stream=stream0)
            del buf3
            del buf8
            buf10 = empty_strided_cuda((16384, 8, 32), (256, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_8.run(buf9, buf10, 4194304, stream=stream0)
            buf11 = reinterpret_tensor(buf9, (32768, 128), (128, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_t_transpose_view_9.run(buf10, arg3_1, buf11, 256, 1, 1, stream=stream0)
            del arg3_1
            del buf10
            buf12 = empty_strided_cuda((16384, 2, 1), (2, 1, 32768), torch.float32)
            buf13 = empty_strided_cuda((16384, 2, 1), (2, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward, add, x_1], Original ATen: [aten.view, aten.permute, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_layer_norm_permute_view_10.run(arg4_1, buf11, arg0_1, buf12, buf13, 16384, 2, 128, stream=stream0)
            buf15 = reinterpret_tensor(buf11, (16384, 2, 128), (256, 128, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [view, x, multi_head_attention_forward, add, x_1], Original ATen: [aten.view, aten.permute, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_native_layer_norm_permute_view_11.run(buf15, arg4_1, arg0_1, buf12, buf13, arg5_1, arg6_1, 16384, 256, stream=stream0)
            del arg0_1
            del arg4_1
            del arg5_1
            del arg6_1
            del buf12
            del buf13
        return (reinterpret_tensor(buf15, (2, 128, 128, 128), (128, 1, 32768, 256), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_clone_permute_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16384*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_clone_mm_permute_t_view_1(arg_A, arg_B, out_ptr0):
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

    M = 32768
    N = 384
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
        xindex = idx_n + 128*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 16384)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (384*((x0 + 32*x2) // 128) + 768*x1 + 768*((x0 + 32*x2) // 256) + (((x0 + 32*x2) % 128))), None)
    tmp1 = tl.load(in_ptr1 + (((x0 + 32*x2) % 128)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.1767766952966369, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 32768)
    x2 = xindex // 4194304
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (4194304 + y0 + 256*x1), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 16384*y0), tmp0, ymask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_add_bmm_clone_mul_select_squeeze_transpose_unsqueeze_view_5(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B

    M = 16384
    N = 16384
    K = 32

    stride_aq = 524288
    stride_am = 32
    stride_ak = 1

    stride_bq = 524288
    stride_bk = 16384
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16384*idx_m + 268435456*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_red_fused__softmax_exp_prepare_softmax_online_sub_6(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp2_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp2_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])

        _tmp2_max_next, _tmp2_sum_next = triton_helpers.online_softmax_combine(
            _tmp2_max, _tmp2_sum, tmp1, False
        )

        _tmp2_max = tl.where(r0_mask, _tmp2_max_next, _tmp2_max)
        _tmp2_sum = tl.where(r0_mask, _tmp2_sum_next, _tmp2_sum)

    tmp2, tmp3 = triton_helpers.online_softmax_reduce(
        _tmp2_max, _tmp2_sum, 1, False)
    tmp2 = tmp2[:, None]
    tmp3 = tmp3[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp4 = tl.load(in_out_ptr0 + (r0_1 + 16384*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = (tmp6 / tmp3)
        tl.store(in_out_ptr0 + (r0_1 + 16384*x0), tmp7, r0_mask)

''',
    '''@triton.jit
def triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B + 8388608

    M = 16384
    N = 32
    K = 16384

    stride_aq = 268435456
    stride_am = 16384
    stride_ak = 1

    stride_bq = 32
    stride_bk = 256
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 32*idx_m + 524288*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 8)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 524288*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_clone_t_transpose_view_9(arg_A, arg_B, out_ptr0):
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

    M = 32768
    N = 128
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
        xindex = idx_n + 128*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_red_fused_add_addmm_native_layer_norm_permute_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 2
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    tmp6_mean = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 256*y0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (y0 + 16384*r0_2 + 2097152*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 2)
    tmp6 = tmp7[:, :, None]
    tmp10 = tmp8[:, :, None]
    tmp11 = tmp9[:, :, None]
    tl.store(out_ptr0 + (x1 + 2*y0), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + 2*y0), tmp10, xmask)

''',
    '''@triton.jit
def triton_poi_fused_add_addmm_native_layer_norm_permute_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 128)
    x3 = xindex
    y0 = yindex
    x2 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + 256*y0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 16384*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 2*y0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 2*y0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tl.full([1, 1], 128.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 256*y0), tmp17, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]

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
