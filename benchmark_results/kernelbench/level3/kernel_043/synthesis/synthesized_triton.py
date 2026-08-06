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


# kernel path: /workspace/tmp/torchinductor_user/hl/chletlaewpfr57wosfkomzzj4pl52zxxjkg6ue3qg6kqid5jtchl.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear => mm_default, permute, view
# Graph fragment:
#   %arg0_1 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[2304, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %view : Tensor "f32[65536, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [65536, 768]), kwargs = {})
#   %permute : Tensor "f32[768, 2304][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [1, 0]), kwargs = {})
#   %mm_default : Tensor "f32[65536, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %permute), kwargs = {})
#   return %mm_default
triton_tem_fused_addmm_t_view_0 = async_compile.triton('triton_tem_fused_addmm_t_view_0', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_t_view_0', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.812384256, 'kernel_flop': 231928233984, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_t_view_0(arg_A, arg_B, out_ptr0):
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

    M = 65536
    N = 2304
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2304*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2304*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 9216, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_t_view_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.812384256
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/4v/c4vbcurszjwtfgmrgeo3eazowapnurqnoxu3kvrhgthoaknr7y47.py
# Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   linear => add_tensor, view_1
#   matmul => clone
#   q_1 => permute_2
#   split => split
#   view_1 => view_3
# Graph fragment:
#   %arg2_1 : Tensor "f32[2304][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mm_default : Tensor "f32[65536, 2304][2304, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[65536, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mm_default), kwargs = {})
#   %view_1 : Tensor "f32[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_3 : Tensor "f32[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [128, 512, 8, 96]), kwargs = {})
#   %permute_2 : Tensor "f32[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_addmm_clone_split_transpose_view_1 = async_compile.triton('triton_poi_fused_addmm_clone_split_transpose_view_1', '''
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
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_split_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603982848}, 'kernel_num_gb': 0.4026624, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x2 = ((xindex // 49152) % 8)
    x1 = ((xindex // 96) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)


def get_args():
    arg_0 = rand_strided((2304,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 50331648,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_split_transpose_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_clone_split_transpose_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.4026624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/gv/cgvakcymtmp6tsnif4nffczfi5xjetu7aa7rzwl3vjcjovoun7z4.py
# Topologically Sorted Source Nodes: [linear, split, view, k_1, transpose_3, matmul], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   k_1 => permute_1
#   linear => add_tensor, view_1
#   matmul => clone_1
#   split => split
#   transpose_3 => permute_4
#   view => view_2
# Graph fragment:
#   %arg2_1 : Tensor "f32[2304][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mm_default : Tensor "f32[65536, 2304][2304, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[65536, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mm_default), kwargs = {})
#   %view_1 : Tensor "f32[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_2 : Tensor "f32[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [128, 512, 8, 96]), kwargs = {})
#   %permute_1 : Tensor "f32[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %permute_4 : Tensor "f32[128, 8, 96, 512][1179648, 96, 1, 2304]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[128, 8, 96, 512][393216, 49152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_addmm_clone_split_transpose_view_2 = async_compile.triton('triton_poi_fused_addmm_clone_split_transpose_view_2', '''
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
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_split_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 201329664, 'x': 402653184}, 'kernel_num_gb': 0.4026624, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 512
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 768)
    x2 = xindex
    y1 = yindex // 768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, xmask & ymask)


def get_args():
    arg_0 = rand_strided((2304,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 96, 512), (393216, 49152, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 98304, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_split_transpose_view_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_clone_split_transpose_view_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.4026624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/pd/cpdwcbu6g6jer3wzfvuo46xuh7ndumjaenwx6kj4n2xx3gp5myds.py
# Topologically Sorted Source Nodes: [linear, split, view_2, v_1, y], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   linear => add_tensor, view_1
#   split => split
#   v_1 => permute_3
#   view_2 => view_4
#   y => clone_3
# Graph fragment:
#   %arg2_1 : Tensor "f32[2304][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mm_default : Tensor "f32[65536, 2304][2304, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[65536, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mm_default), kwargs = {})
#   %view_1 : Tensor "f32[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_4 : Tensor "f32[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [128, 512, 8, 96]), kwargs = {})
#   %permute_3 : Tensor "f32[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_addmm_clone_split_transpose_view_3 = async_compile.triton('triton_poi_fused_addmm_clone_split_transpose_view_3', '''
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
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_split_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603982848}, 'kernel_num_gb': 0.4026624, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x2 = ((xindex // 49152) % 8)
    x1 = ((xindex // 96) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)


def get_args():
    arg_0 = rand_strided((2304,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 50331648,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_split_transpose_view_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_clone_split_transpose_view_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.4026624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/rj/crjb2ygxn7sdafofdhy45u5tnncie73tcyjivjfe73tomvlslqv4.py
# Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul, view, k_1, transpose_3], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
# Source node to ATen node mapping:
#   k_1 => permute_1
#   linear => add_tensor, view_1
#   matmul => bmm, clone, clone_1, view_5, view_6
#   q_1 => permute_2
#   split => split
#   transpose_3 => permute_4
#   view => view_2
#   view_1 => view_3
# Graph fragment:
#   %clone : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0" = PlaceHolder[target=clone]
#   %clone_1 : Tensor "f32[128, 8, 96, 512][393216, 49152, 512, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %add_tensor : Tensor "f32[65536, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mm_default), kwargs = {})
#   %view_1 : Tensor "f32[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_3 : Tensor "f32[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [128, 512, 8, 96]), kwargs = {})
#   %permute_2 : Tensor "f32[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_5 : Tensor "f32[1024, 512, 96][49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1024, 512, 96]), kwargs = {})
#   %view_2 : Tensor "f32[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [128, 512, 8, 96]), kwargs = {})
#   %permute_1 : Tensor "f32[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %permute_4 : Tensor "f32[128, 8, 96, 512][1179648, 96, 1, 2304]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[128, 8, 96, 512][393216, 49152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f32[1024, 96, 512][49152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [1024, 96, 512]), kwargs = {})
#   %bmm : Tensor "f32[1024, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_5, %view_6), kwargs = {})
#   return %bmm
triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4 = async_compile.triton('triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 1.476395008, 'kernel_flop': 51539607552, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 512
    K = 96

    stride_aq = 49152
    stride_am = 96
    stride_ak = 1

    stride_bq = 49152
    stride_bk = 512
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
    xindex = idx_n + 512*idx_m + 262144*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 96, 512), (393216, 49152, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 64, 1024, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.476395008
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/tk/ctk375g2qhvauuogy47vdv5bvdwfyc5727mfge3q4xq3r4xgjqx2.py
# Topologically Sorted Source Nodes: [getitem_3, eq, att_1, matmul, att, att_2], Original ATen: [aten.slice, aten.eq, aten.masked_fill, aten.view, aten.mul, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#   att => mul
#   att_1 => full_default, where
#   att_2 => div
#   eq => eq
#   getitem_3 => slice_1, slice_2
#   matmul => view_7
# Graph fragment:
#   %arg3_1 : Tensor "f32[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %bmm : Tensor "f32[1024, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %getitem_3 : Tensor "f32[128, 8, 512, 1][4096, 512, 1, 524288]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "f32[128, 8, 512, 1][4096, 512, 1, 524288]cuda:0" = PlaceHolder[target=getitem_4]
#   %slice_1 : Tensor "f32[1, 1, 512, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 2, 0, 512), kwargs = {})
#   %slice_2 : Tensor "f32[1, 1, 512, 512][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 3, 0, 512), kwargs = {})
#   %eq : Tensor "b8[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%slice_2, 0), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_7 : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [128, 8, 512, 512]), kwargs = {})
#   %mul : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 0.10206207261596577), kwargs = {})
#   %where : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%where, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %getitem_3), kwargs = {})
#   %exp_default : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_4), kwargs = {})
#   return %getitem_3,%getitem_4,%expand_2
triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5 = async_compile.triton('triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5', '''
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
    size_hints={'x': 524288, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3222274048}, 'kernel_num_gb': 2.151677952, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x3 = xindex
    _tmp9_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp9_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp7 = tl.where(tmp2, tmp6, tmp5)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])

        _tmp9_max_next, _tmp9_sum_next = triton_helpers.online_softmax_combine(
            _tmp9_max, _tmp9_sum, tmp8, False
        )

        _tmp9_max = tl.where(r0_mask, _tmp9_max_next, _tmp9_max)
        _tmp9_sum = tl.where(r0_mask, _tmp9_sum_next, _tmp9_sum)

    tmp9, tmp10 = triton_helpers.online_softmax_reduce(
        _tmp9_max, _tmp9_sum, 1, False)
    tmp9 = tmp9[:, None]
    tmp10 = tmp10[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.full([1, 1], 0.0, tl.float32)
        tmp13 = tmp11 == tmp12
        tmp15 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp18 = tl.where(tmp13, tmp17, tmp16)
        tmp19 = tmp18 - tmp9
        tmp20 = libdevice.exp(tmp19)
        tmp21 = (tmp20 / tmp10)
        tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp21, r0_mask)


def get_args():
    arg_0 = rand_strided((128, 8, 512, 512), (2097152, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 524288, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 2.151677952
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/rq/crq5azxapzgmz3kpsrovykr4mobhced2uwwyslfkscuthdutkudm.py
# Topologically Sorted Source Nodes: [linear, split, getitem_3, eq, att_1, matmul, att, att_2, y, view_2, v_1], Original ATen: [aten.addmm, aten.view, aten.split, aten.slice, aten.eq, aten.masked_fill, aten.mul, aten.sub, aten.exp, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
# Source node to ATen node mapping:
#   att => mul
#   att_1 => full_default, where
#   att_2 => div
#   eq => eq
#   getitem_3 => slice_1, slice_2
#   linear => add_tensor, view_1
#   matmul => view_7
#   split => split
#   v_1 => permute_3
#   view_2 => view_4
#   y => bmm_1, clone_3, view_8, view_9
# Graph fragment:
#   %expand_2 : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0" = PlaceHolder[target=expand_2]
#   %clone_3 : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0" = PlaceHolder[target=clone_3]
#   %add_tensor : Tensor "f32[65536, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mm_default), kwargs = {})
#   %view_1 : Tensor "f32[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %slice_1 : Tensor "f32[1, 1, 512, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 2, 0, 512), kwargs = {})
#   %slice_2 : Tensor "f32[1, 1, 512, 512][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 3, 0, 512), kwargs = {})
#   %eq : Tensor "b8[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%slice_2, 0), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_7 : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [128, 8, 512, 512]), kwargs = {})
#   %mul : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 0.10206207261596577), kwargs = {})
#   %where : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %getitem_3), kwargs = {})
#   %exp_default : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_4), kwargs = {})
#   %view_8 : Tensor "f32[1024, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_2, [1024, 512, 512]), kwargs = {})
#   %view_4 : Tensor "f32[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [128, 512, 8, 96]), kwargs = {})
#   %permute_3 : Tensor "f32[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_9 : Tensor "f32[1024, 512, 96][49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1024, 512, 96]), kwargs = {})
#   %bmm_1 : Tensor "f32[1024, 512, 96][49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_8, %view_9), kwargs = {})
#   return %bmm_1
triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6 = async_compile.triton('triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 1.476395008, 'kernel_flop': 51539607552, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 96
    K = 512

    stride_aq = 262144
    stride_am = 512
    stride_ak = 1

    stride_bq = 49152
    stride_bk = 96
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
    xindex = idx_n + 96*idx_m + 49152*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((128, 8, 512, 512), (2097152, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 16, 1024, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.476395008
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/vc/cvcsaz3c4vgeva5nedp5udwobl75mqlvuy7uwpghxjbk45w4x4es.py
# Topologically Sorted Source Nodes: [y, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_4
#   transpose_4 => permute_5
#   y => view_10
# Graph fragment:
#   %bmm_1 : Tensor "f32[1024, 512, 96][49152, 96, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_10 : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [128, 8, 512, 96]), kwargs = {})
#   %permute_5 : Tensor "f32[128, 512, 8, 96][393216, 96, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[128, 512, 8, 96][393216, 768, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_transpose_view_7 = async_compile.triton('triton_poi_fused_clone_transpose_view_7', '''
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
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603979776}, 'kernel_num_gb': 0.402653184, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)


def get_args():
    arg_0 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 50331648,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_transpose_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.402653184
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/5l/c5lsnur4swrkfb2soi3kfslk6jhs6licwccczx5cpv26dhs22mug.py
# Topologically Sorted Source Nodes: [y, transpose_4, contiguous, y_1, linear_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   contiguous => clone_4
#   linear_1 => addmm_1, permute_6, view_12
#   transpose_4 => permute_5
#   y => view_10
#   y_1 => view_11
# Graph fragment:
#   %arg5_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %clone_4 : Tensor "f32[128, 512, 8, 96][393216, 768, 96, 1]cuda:0" = PlaceHolder[target=clone_4]
#   %arg4_1 : Tensor "f32[768, 768][768, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %view_10 : Tensor "f32[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [128, 8, 512, 96]), kwargs = {})
#   %permute_5 : Tensor "f32[128, 512, 8, 96][393216, 96, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[128, 512, 8, 96][393216, 768, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_11 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [128, 512, 768]), kwargs = {})
#   %view_12 : Tensor "f32[65536, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [65536, 768]), kwargs = {})
#   %permute_6 : Tensor "f32[768, 768][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg4_1, [1, 0]), kwargs = {})
#   %addmm_1 : Tensor "f32[65536, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg5_1, %view_12, %permute_6), kwargs = {})
#   return %addmm_1
triton_tem_fused_addmm_clone_t_transpose_view_8 = async_compile.triton('triton_tem_fused_addmm_clone_t_transpose_view_8', '''
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
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_clone_t_transpose_view_8', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.606339072, 'kernel_flop': 77309411328, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_clone_t_transpose_view_8(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 65536
    N = 768
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def get_args():
    arg_0 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 3072, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_clone_t_transpose_view_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_clone_t_transpose_view_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.606339072
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
        assert_size_stride(arg0_1, (128, 512, 768), (393216, 768, 1))
        assert_size_stride(arg1_1, (2304, 768), (768, 1))
        assert_size_stride(arg2_1, (2304, ), (1, ))
        assert_size_stride(arg3_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
        assert_size_stride(arg4_1, (768, 768), (768, 1))
        assert_size_stride(arg5_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((65536, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_0.run(arg0_1, arg1_1, buf0, 9216, 1, 1, stream=stream0)
            del arg0_1
            del arg1_1
            buf1 = empty_strided_cuda((128, 8, 512, 96), (393216, 49152, 96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_clone_split_transpose_view_1.run(arg2_1, buf0, buf1, 50331648, stream=stream0)
            buf2 = empty_strided_cuda((128, 8, 96, 512), (393216, 49152, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, split, view, k_1, transpose_3, matmul], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_clone_split_transpose_view_2.run(arg2_1, buf0, buf2, 98304, 512, stream=stream0)
            buf7 = empty_strided_cuda((128, 8, 512, 96), (393216, 49152, 96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, split, view_2, v_1, y], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_clone_split_transpose_view_3.run(arg2_1, buf0, buf7, 50331648, stream=stream0)
            del arg2_1
            del buf0
            buf3 = empty_strided_cuda((1024, 512, 512), (262144, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul, view, k_1, transpose_3], Original ATen: [aten.addmm, aten.view, aten.split, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4.run(buf1, buf2, buf3, 64, 1024, 1, stream=stream0)
            del buf1
            del buf2
            buf6 = reinterpret_tensor(buf3, (128, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [getitem_3, eq, att_1, matmul, att, att_2], Original ATen: [aten.slice, aten.eq, aten.masked_fill, aten.view, aten.mul, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5.run(buf6, arg3_1, 524288, 512, stream=stream0)
            del arg3_1
            buf8 = empty_strided_cuda((1024, 512, 96), (49152, 96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, split, getitem_3, eq, att_1, matmul, att, att_2, y, view_2, v_1], Original ATen: [aten.addmm, aten.view, aten.split, aten.slice, aten.eq, aten.masked_fill, aten.mul, aten.sub, aten.exp, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6.run(buf6, buf7, buf8, 16, 1024, 1, stream=stream0)
            del buf6
            buf9 = reinterpret_tensor(buf7, (128, 512, 8, 96), (393216, 768, 96, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [y, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf8, buf9, 50331648, stream=stream0)
            buf10 = reinterpret_tensor(buf8, (65536, 768), (768, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [y, transpose_4, contiguous, y_1, linear_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_t_transpose_view_8.run(arg5_1, buf9, arg4_1, buf10, 3072, 1, 1, stream=stream0)
            del arg4_1
            del arg5_1
            del buf9
        return (reinterpret_tensor(buf10, (128, 512, 768), (393216, 768, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class Model(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_tem_fused_addmm_t_view_0(arg_A, arg_B, out_ptr0):
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

    M = 65536
    N = 2304
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2304*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2304*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x2 = ((xindex // 49152) % 8)
    x1 = ((xindex // 96) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)

''',
    '''@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 512
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 768)
    x2 = xindex
    y1 = yindex // 768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x2 = ((xindex // 49152) % 8)
    x1 = ((xindex // 96) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 512
    K = 96

    stride_aq = 49152
    stride_am = 96
    stride_ak = 1

    stride_bq = 49152
    stride_bk = 512
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
    xindex = idx_n + 512*idx_m + 262144*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_5(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x3 = xindex
    _tmp9_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp9_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp7 = tl.where(tmp2, tmp6, tmp5)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])

        _tmp9_max_next, _tmp9_sum_next = triton_helpers.online_softmax_combine(
            _tmp9_max, _tmp9_sum, tmp8, False
        )

        _tmp9_max = tl.where(r0_mask, _tmp9_max_next, _tmp9_max)
        _tmp9_sum = tl.where(r0_mask, _tmp9_sum_next, _tmp9_sum)

    tmp9, tmp10 = triton_helpers.online_softmax_reduce(
        _tmp9_max, _tmp9_sum, 1, False)
    tmp9 = tmp9[:, None]
    tmp10 = tmp10[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_2 + 1024*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.full([1, 1], 0.0, tl.float32)
        tmp13 = tmp11 == tmp12
        tmp15 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp18 = tl.where(tmp13, tmp17, tmp16)
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = (tmp20 / tmp10)
        tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp21, r0_mask)

''',
    '''@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 96
    K = 512

    stride_aq = 262144
    stride_am = 512
    stride_ak = 1

    stride_bq = 49152
    stride_bk = 96
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
    xindex = idx_n + 96*idx_m + 49152*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_clone_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_clone_t_transpose_view_8(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 65536
    N = 768
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2304,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
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
