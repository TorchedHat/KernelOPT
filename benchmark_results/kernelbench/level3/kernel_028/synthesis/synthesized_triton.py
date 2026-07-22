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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4q/c4qsfh7owg2zqcoimq7p56g766flocli36u5oijlx4ztqatr5oep.py
# Topologically Sorted Source Nodes: [unfold, unfold_1, x], Original ATen: [aten.unfold, aten.clone]
# Source node to ATen node mapping:
#   unfold => unfold
#   unfold_1 => unfold_1
#   x => clone
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %unfold : Tensor "f32[2, 3, 14, 224, 16][150528, 50176, 3584, 1, 224]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unfold.default](args = (%arg0_1, 2, 16, 16), kwargs = {})
#   %unfold_1 : Tensor "f32[2, 3, 14, 14, 16, 16][150528, 50176, 3584, 16, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unfold.default](args = (%unfold, 3, 16, 16), kwargs = {})
#   %clone : Tensor "f32[2, 3, 14, 14, 16, 16][150528, 50176, 3584, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%unfold_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_unfold_0 = async_compile.triton('triton_poi_fused_clone_unfold_0', '''
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
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_unfold_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3612672}, 'kernel_num_gb': 0.002408448, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_unfold_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x2 = ((xindex // 256) % 14)
    x3 = xindex // 3584
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 224*x1 + 3584*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((2, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 3, 14, 14, 16, 16), (150528, 50176, 3584, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 301056,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_unfold_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_unfold_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002408448
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/mq/cmqcwvzt6pfhpxxej4ov76qprnm3ahg4mpb6nwbz55urkd336n7q.py
# Topologically Sorted Source Nodes: [unfold, unfold_1, x, x_1], Original ATen: [aten.unfold, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   unfold => unfold
#   unfold_1 => unfold_1
#   x => clone, view
#   x_1 => addmm, permute, view_1
# Graph fragment:
#   %clone : Tensor "f32[2, 3, 14, 14, 16, 16][150528, 50176, 3584, 256, 16, 1]cuda:0" = PlaceHolder[target=clone]
#   %arg2_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %buf1 : Tensor "f32[392, 768][768, 1]cuda:0" = PlaceHolder[target=buf1]
#   %arg1_1 : Tensor "f32[512, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %unfold : Tensor "f32[2, 3, 14, 224, 16][150528, 50176, 3584, 1, 224]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unfold.default](args = (%arg0_1, 2, 16, 16), kwargs = {})
#   %unfold_1 : Tensor "f32[2, 3, 14, 14, 16, 16][150528, 50176, 3584, 16, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unfold.default](args = (%unfold, 3, 16, 16), kwargs = {})
#   %clone : Tensor "f32[2, 3, 14, 14, 16, 16][150528, 50176, 3584, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%unfold_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[2, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [2, 196, 768]), kwargs = {})
#   %view_1 : Tensor "f32[392, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [392, 768]), kwargs = {})
#   %permute : Tensor "f32[768, 512][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "f32[392, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg2_1, %view_1, %permute), kwargs = {})
#   return %buf1,%addmm
triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1 = async_compile.triton('triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_B': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00438272, 'kernel_flop': 308281344, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1(in_ptr0, arg_B, in_ptr2, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    B = arg_B

    M = 392
    N = 512
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
        tmp2 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 768*idx_m + 150528*((idx_n + 768*((idx_m % 196))) // 150528), [BLOCK_M, BLOCK_K])), None, eviction_policy='evict_last')
        a = tmp2.broadcast_to(xindex.shape)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def get_args():
    arg_0 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 3, 14, 14, 16, 16), (150528, 50176, 3584, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 112, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00438272
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/5d/c5dp6yekt36fo5old4gznp4yhxk6eccpj2jqecud6uihuhgxmd4s.py
# Topologically Sorted Source Nodes: [cls_tokens, x_1, x_2, x_3], Original ATen: [aten.expand, aten.view, aten.cat, aten.add]
# Source node to ATen node mapping:
#   cls_tokens => expand
#   x_1 => view_2
#   x_2 => cat
#   x_3 => add
# Graph fragment:
#   %arg3_1 : Tensor "f32[1, 1, 512][512, 512, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm : Tensor "f32[392, 512][512, 1]cuda:0" = PlaceHolder[target=addmm]
#   %arg4_1 : Tensor "f32[1, 197, 512][100864, 512, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %expand : Tensor "f32[2, 1, 512][0, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%arg3_1, [2, -1, -1]), kwargs = {})
#   %view_2 : Tensor "f32[2, 196, 512][100352, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [2, 196, 512]), kwargs = {})
#   %cat : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %view_2], 1), kwargs = {})
#   %add : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %arg4_1), kwargs = {})
#   return %add
triton_poi_fused_add_cat_expand_view_2 = async_compile.triton('triton_poi_fused_add_cat_expand_view_2', '''
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_expand_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2822144}, 'kernel_num_gb': 0.002015232, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_expand_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 512) % 197)
    x0 = (xindex % 512)
    x2 = xindex // 100864
    x3 = (xindex % 100864)
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 512*((-1) + x1) + 100352*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, xmask)


def get_args():
    arg_0 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 201728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_expand_view_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_cat_expand_view_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002015232
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/2x/c2xvl7b7zquemywojcwwuxyrjns36orl7a452g24fionbxf5nji3.py
# Topologically Sorted Source Nodes: [cls_tokens, x_1, x_2, x_3, multi_head_attention_forward], Original ATen: [aten.expand, aten.view, aten.cat, aten.add, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   cls_tokens => expand
#   multi_head_attention_forward => mm_default_24, permute_1, view_3
#   x_1 => view_2
#   x_2 => cat
#   x_3 => add
# Graph fragment:
#   %add : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0" = PlaceHolder[target=add]
#   %arg6_1 : Tensor "f32[1536, 512][512, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %expand : Tensor "f32[2, 1, 512][0, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%arg3_1, [2, -1, -1]), kwargs = {})
#   %view_2 : Tensor "f32[2, 196, 512][100352, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [2, 196, 512]), kwargs = {})
#   %cat : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %view_2], 1), kwargs = {})
#   %add : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %arg4_1), kwargs = {})
#   %view_3 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [394, 512]), kwargs = {})
#   %permute_1 : Tensor "f32[512, 1536][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg6_1, [1, 0]), kwargs = {})
#   %mm_default_24 : Tensor "f32[394, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_3, %permute_1), kwargs = {})
#   return %mm_default_24
triton_tem_fused_add_addmm_cat_expand_t_view_3 = async_compile.triton('triton_tem_fused_add_addmm_cat_expand_t_view_3', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_cat_expand_t_view_3', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.006373376, 'kernel_flop': 619708416, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_add_addmm_cat_expand_t_view_3(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 1536
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1536*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1536*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((2, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((394, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 84, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_cat_expand_t_view_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_add_addmm_cat_expand_t_view_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006373376
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ud/cudmc64rqomw4pfz5rrl53faxa5xma6fyw3n2nbh6pxjwucpylhn.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_efficient_attention, add_tensor_24, clone_2, permute_2, permute_3, permute_4, permute_5, select, select_1, select_2, squeeze, unsqueeze, view_10, view_11, view_4, view_5, view_6, view_7, view_8, view_9
# Graph fragment:
#   %arg5_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %mm_default_24 : Tensor "f32[394, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_24]
#   %add_tensor_24 : Tensor "f32[394, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, %mm_default_24), kwargs = {})
#   %view_4 : Tensor "f32[2, 197, 1536][302592, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [2, 197, 1536]), kwargs = {})
#   %view_5 : Tensor "f32[2, 197, 3, 512][302592, 1536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [2, 197, 3, 512]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 2, 197, 3, 512][605184, 302592, 1536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 2, 197, 1, 512][512, 302592, 1536, 605184, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 2, 197, 512][512, 302592, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_2 : Tensor "f32[3, 2, 197, 512][201728, 100864, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 0), kwargs = {})
#   %view_6 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [2, 1576, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_9 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [197, 8, 2, 64]), kwargs = {})
#   %select_1 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 1), kwargs = {})
#   %view_7 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [2, 1576, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_10 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [197, 8, 2, 64]), kwargs = {})
#   %select_2 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 2), kwargs = {})
#   %view_8 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [2, 1576, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_11 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [197, 8, 2, 64]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%view_9, %view_10, %view_11, None, False), kwargs = {})
#   return %buf5
triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4', '''
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2422784}, 'kernel_num_gb': 0.001619968, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((394, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((197, 8, 2, 64), (512, 64, 100864, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 201728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001619968
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pr/cpr4hcwqdpapvz3rxyuffk54q4hvzksiimpgwj4c4kamd5ro6har.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_efficient_attention, add_tensor_24, clone_2, permute_2, permute_3, permute_4, permute_5, select, select_1, select_2, squeeze, unsqueeze, view_10, view_11, view_4, view_5, view_6, view_7, view_8, view_9
# Graph fragment:
#   %arg5_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %mm_default_24 : Tensor "f32[394, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_24]
#   %add_tensor_24 : Tensor "f32[394, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, %mm_default_24), kwargs = {})
#   %view_4 : Tensor "f32[2, 197, 1536][302592, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [2, 197, 1536]), kwargs = {})
#   %view_5 : Tensor "f32[2, 197, 3, 512][302592, 1536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [2, 197, 3, 512]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 2, 197, 3, 512][605184, 302592, 1536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 2, 197, 1, 512][512, 302592, 1536, 605184, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 2, 197, 512][512, 302592, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_2 : Tensor "f32[3, 2, 197, 512][201728, 100864, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 0), kwargs = {})
#   %view_6 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [2, 1576, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_9 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [197, 8, 2, 64]), kwargs = {})
#   %select_1 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 1), kwargs = {})
#   %view_7 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [2, 1576, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_10 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [197, 8, 2, 64]), kwargs = {})
#   %select_2 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 2), kwargs = {})
#   %view_8 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [2, 1576, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_11 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [197, 8, 2, 64]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%view_9, %view_10, %view_11, None, False), kwargs = {})
#   return %buf6
triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5', '''
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2422784}, 'kernel_num_gb': 0.001619968, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((394, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((197, 8, 2, 64), (512, 64, 100864, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 201728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001619968
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/az/cazuyuw7xhppaclyinv6hhz6gj2n5743hmckvopyjd7bbyljbjde.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_efficient_attention, add_tensor_24, clone_2, permute_2, permute_3, permute_4, permute_5, select, select_1, select_2, squeeze, unsqueeze, view_10, view_11, view_4, view_5, view_6, view_7, view_8, view_9
# Graph fragment:
#   %arg5_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %mm_default_24 : Tensor "f32[394, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_24]
#   %add_tensor_24 : Tensor "f32[394, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, %mm_default_24), kwargs = {})
#   %view_4 : Tensor "f32[2, 197, 1536][302592, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [2, 197, 1536]), kwargs = {})
#   %view_5 : Tensor "f32[2, 197, 3, 512][302592, 1536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [2, 197, 3, 512]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 2, 197, 3, 512][605184, 302592, 1536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 2, 197, 1, 512][512, 302592, 1536, 605184, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 2, 197, 512][512, 302592, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_2 : Tensor "f32[3, 2, 197, 512][201728, 100864, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 0), kwargs = {})
#   %view_6 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [2, 1576, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_9 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [197, 8, 2, 64]), kwargs = {})
#   %select_1 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 1), kwargs = {})
#   %view_7 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [2, 1576, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_10 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [197, 8, 2, 64]), kwargs = {})
#   %select_2 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 2), kwargs = {})
#   %view_8 : Tensor "f32[2, 1576, 64][100864, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [2, 1576, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[1576, 2, 64][64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_11 : Tensor "f32[197, 8, 2, 64][512, 64, 100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [197, 8, 2, 64]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%view_9, %view_10, %view_11, None, False), kwargs = {})
#   return %buf7
triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6', '''
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2422784}, 'kernel_num_gb': 0.001619968, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((394, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((197, 8, 2, 64), (512, 64, 100864, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 201728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001619968
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/p4/cp4u3sqbc4cbmq3rul32hdi65k6wti3dryxsudzan2nst47brmur.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_3, permute_6
# Graph fragment:
#   %getitem : Tensor "f32[197, 8, 2, 64][1024, 64, 512, 1]cuda:0" = PlaceHolder[target=getitem]
#   %permute_6 : Tensor "f32[2, 197, 8, 64][512, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem, [2, 0, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[2, 197, 8, 64][100864, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_permute_7 = async_compile.triton('triton_poi_fused_clone_permute_7', '''
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2420736}, 'kernel_num_gb': 0.001613824, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((197, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 197, 8, 64), (100864, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 201728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_permute_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001613824
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/np/cnp4lgt6nygkwc72ieeplephy6oop3rajrixd5rmwz5d53jupd4b.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_3, mm_default_23, permute_6, permute_7, view_12
# Graph fragment:
#   %clone_3 : Tensor "f32[2, 197, 8, 64][100864, 512, 64, 1]cuda:0" = PlaceHolder[target=clone_3]
#   %arg7_1 : Tensor "f32[512, 512][512, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %permute_6 : Tensor "f32[2, 197, 8, 64][512, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem, [2, 0, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[2, 197, 8, 64][100864, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_12 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [394, 512]), kwargs = {})
#   %permute_7 : Tensor "f32[512, 512][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg7_1, [1, 0]), kwargs = {})
#   %mm_default_23 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_12, %permute_7), kwargs = {})
#   return %mm_default_23
triton_tem_fused_addmm_clone_permute_t_view_8 = async_compile.triton('triton_tem_fused_addmm_clone_permute_t_view_8', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_clone_permute_t_view_8', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0026624, 'kernel_flop': 206569472, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_clone_permute_t_view_8(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 512
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((2, 197, 8, 64), (100864, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((394, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 112, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_clone_permute_t_view_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_clone_permute_t_view_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0026624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4u/c4ucqpiz3tekjrqx3rmhagkqpcdar2a6hm5pq2cujrrb3tncbap2.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, add, x_5], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_1
#   multi_head_attention_forward => add_tensor_23, view_13
#   x_5 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %add : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0" = PlaceHolder[target=add]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %mm_default_23 : Tensor "f32[394, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %getitem_5 : Tensor "f32[2, 197, 1][197, 1, 394]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf16 : Tensor "f32[2, 197, 1][197, 1, 394]cuda:0" = PlaceHolder[target=buf16]
#   %arg9_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %add_tensor_23 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, %mm_default_23), kwargs = {})
#   %view_13 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [2, 197, 512]), kwargs = {})
#   %add_1 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_13), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_5), kwargs = {})
#   %add_2 : Tensor "f32[2, 197, 1][197, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[2, 197, 1][197, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg9_1), kwargs = {})
#   %add_3 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg10_1), kwargs = {})
#   return %getitem_5,%buf16,%add_3
triton_per_fused_add_addmm_native_layer_norm_view_9 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_9', '''
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
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3233792}, 'kernel_num_gb': 0.00242688, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 394
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 512.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp31, xmask)


def get_args():
    arg_0 = rand_strided((2, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((394, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 394, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_addmm_native_layer_norm_view_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00242688
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/oo/cooyi3fmx4po4rchrn2iwcdwgj4y3ox25jbsvbdodfktx3wuocn2.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_1 => mm_default_22, permute_8, view_14
# Graph fragment:
#   %add_3 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0" = PlaceHolder[target=add_3]
#   %arg11_1 : Tensor "f32[2048, 512][512, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %view_14 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [394, 512]), kwargs = {})
#   %permute_8 : Tensor "f32[512, 2048][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg11_1, [1, 0]), kwargs = {})
#   %mm_default_22 : Tensor "f32[394, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_14, %permute_8), kwargs = {})
#   return %mm_default_22
triton_tem_fused_addmm_t_view_10 = async_compile.triton('triton_tem_fused_addmm_t_view_10', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_t_view_10', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.008228864, 'kernel_flop': 826277888, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_t_view_10(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 2048
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2048*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2048*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((2, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((394, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 112, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_t_view_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008228864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bp/cbp6wure3jwuy4fu42keqjd7p5m25zv32urcspe426byrw5b2xmq.py
# Topologically Sorted Source Nodes: [linear_1, relu], Original ATen: [aten.addmm, aten.view, aten.relu]
# Source node to ATen node mapping:
#   linear_1 => add_tensor_22, view_15
#   relu => relu
# Graph fragment:
#   %arg12_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %mm_default_22 : Tensor "f32[394, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_default_22]
#   %add_tensor_22 : Tensor "f32[394, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default_22), kwargs = {})
#   %view_15 : Tensor "f32[2, 197, 2048][403456, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [2, 197, 2048]), kwargs = {})
#   %relu : Tensor "f32[2, 197, 2048][403456, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_15,), kwargs = {})
#   return %relu
triton_poi_fused_addmm_relu_view_11 = async_compile.triton('triton_poi_fused_addmm_relu_view_11', '''
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9691136}, 'kernel_num_gb': 0.006463488, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_view_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 806912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


def get_args():
    arg_0 = rand_strided((2, 197, 2048), (403456, 2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 806912,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_view_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_relu_view_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006463488
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pv/cpvrdbiabxsaxxczsphqukdyx4am7qiim44jjgq4gz55z3stuha7.py
# Topologically Sorted Source Nodes: [linear_1, relu, x_6], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
# Source node to ATen node mapping:
#   linear_1 => add_tensor_22, view_15
#   relu => relu
#   x_6 => mm_default_21, permute_9, view_16
# Graph fragment:
#   %relu : Tensor "f32[2, 197, 2048][403456, 2048, 1]cuda:0" = PlaceHolder[target=relu]
#   %arg13_1 : Tensor "f32[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %add_tensor_22 : Tensor "f32[394, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default_22), kwargs = {})
#   %view_15 : Tensor "f32[2, 197, 2048][403456, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [2, 197, 2048]), kwargs = {})
#   %relu : Tensor "f32[2, 197, 2048][403456, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_15,), kwargs = {})
#   %view_16 : Tensor "f32[394, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%relu, [394, 2048]), kwargs = {})
#   %permute_9 : Tensor "f32[2048, 512][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg13_1, [1, 0]), kwargs = {})
#   %mm_default_21 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_16, %permute_9), kwargs = {})
#   return %mm_default_21
triton_tem_fused_addmm_relu_t_view_12 = async_compile.triton('triton_tem_fused_addmm_relu_t_view_12', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_relu_t_view_12', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.008228864, 'kernel_flop': 826277888, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_relu_t_view_12(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 512
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

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
        xindex = idx_n + 2048*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((2, 197, 2048), (403456, 2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((394, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 112, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_relu_t_view_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_relu_t_view_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008228864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ft/cftexwagnmfop4x3h3ho6543iqfcpniiva3wiibgx6kcvgvzlvyj.py
# Topologically Sorted Source Nodes: [x_21, add_11, x_22, x_23, input_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.select, aten.t]
# Source node to ATen node mapping:
#   add_11 => add_34
#   input_1 => mm_default, permute_55
#   x_21 => add_tensor_1, view_92
#   x_22 => add_35, add_36, mul_22, mul_23, rsqrt_11, sub_11, var_mean_11
#   x_23 => select_18
# Graph fragment:
#   %add_36 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0" = PlaceHolder[target=add_36]
#   %arg77_1 : Tensor "f32[2048, 512][512, 1]cuda:0" = PlaceHolder[target=arg77_1]
#   %add_tensor_1 : Tensor "f32[394, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg74_1, %mm_default_1), kwargs = {})
#   %view_92 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [2, 197, 512]), kwargs = {})
#   %add_34 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_92), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %getitem_47), kwargs = {})
#   %add_35 : Tensor "f32[2, 197, 1][197, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_11 : Tensor "f32[2, 197, 1][197, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_22 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_23 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %arg75_1), kwargs = {})
#   %add_36 : Tensor "f32[2, 197, 512][100864, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %arg76_1), kwargs = {})
#   %select_18 : Tensor "f32[2, 512][100864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%add_36, 1, 0), kwargs = {})
#   %permute_55 : Tensor "f32[512, 2048][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg77_1, [1, 0]), kwargs = {})
#   %mm_default : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_18, %permute_55), kwargs = {})
#   return %mm_default
triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13 = async_compile.triton('triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=2,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.004214784, 'kernel_flop': 4194304, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 2048
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 100864
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + ((tl.broadcast_to(idx_n + 100864*idx_m, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2048*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2048*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((2, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 64, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004214784
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/tm/ctmutdzkm2aog4ucakseovsaodst56gw6c5uywvfsydjdsa2f6t2.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.gelu]
# Source node to ATen node mapping:
#   input_1 => add_tensor
#   input_2 => add_37, erf, mul_24, mul_25, mul_26
# Graph fragment:
#   %arg78_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg78_1]
#   %mm_default : Tensor "f32[2, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg78_1, %mm_default), kwargs = {})
#   %mul_24 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.5), kwargs = {})
#   %mul_25 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_25,), kwargs = {})
#   %add_37 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_26 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %add_37), kwargs = {})
#   return %mul_26
triton_poi_fused_addmm_gelu_14 = async_compile.triton('triton_poi_fused_addmm_gelu_14', '''
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 57344}, 'kernel_num_gb': 4.096e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)


def get_args():
    arg_0 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 4096,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_gelu_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_gelu_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 4.096e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4f/c4fp3x2scbr4jwtqwd6fbuwft57e2ri2bt2b6toqzf7moe5vui5h.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_4], Original ATen: [aten.addmm, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   input_1 => add_tensor
#   input_2 => add_37, erf, mul_24, mul_25, mul_26
#   input_4 => addmm_26, permute_56
# Graph fragment:
#   %arg80_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg80_1]
#   %mul_26 : Tensor "f32[2, 2048][2048, 1]cuda:0" = PlaceHolder[target=mul_26]
#   %arg79_1 : Tensor "f32[10, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg79_1]
#   %add_tensor : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg78_1, %mm_default), kwargs = {})
#   %mul_24 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.5), kwargs = {})
#   %mul_25 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_25,), kwargs = {})
#   %add_37 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_26 : Tensor "f32[2, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %add_37), kwargs = {})
#   %permute_56 : Tensor "f32[2048, 10][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg79_1, [1, 0]), kwargs = {})
#   %addmm_26 : Tensor "f32[2, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg80_1, %mul_26, %permute_56), kwargs = {})
#   return %addmm_26
triton_tem_fused_addmm_gelu_t_15 = async_compile.triton('triton_tem_fused_addmm_gelu_t_15', '''
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=1,
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_t_15', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 9.8464e-05, 'kernel_flop': 81920, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_gelu_t_15(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 10
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

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
        xindex = idx_n + 2048*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 10*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 10*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def get_args():
    arg_0 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_gelu_t_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_gelu_t_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 9.8464e-05
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg1_1, (512, 768), (768, 1))
        assert_size_stride(arg2_1, (512, ), (1, ))
        assert_size_stride(arg3_1, (1, 1, 512), (512, 512, 1))
        assert_size_stride(arg4_1, (1, 197, 512), (100864, 512, 1))
        assert_size_stride(arg5_1, (1536, ), (1, ))
        assert_size_stride(arg6_1, (1536, 512), (512, 1))
        assert_size_stride(arg7_1, (512, 512), (512, 1))
        assert_size_stride(arg8_1, (512, ), (1, ))
        assert_size_stride(arg9_1, (512, ), (1, ))
        assert_size_stride(arg10_1, (512, ), (1, ))
        assert_size_stride(arg11_1, (2048, 512), (512, 1))
        assert_size_stride(arg12_1, (2048, ), (1, ))
        assert_size_stride(arg13_1, (512, 2048), (2048, 1))
        assert_size_stride(arg14_1, (512, ), (1, ))
        assert_size_stride(arg15_1, (512, ), (1, ))
        assert_size_stride(arg16_1, (512, ), (1, ))
        assert_size_stride(arg17_1, (1536, ), (1, ))
        assert_size_stride(arg18_1, (1536, 512), (512, 1))
        assert_size_stride(arg19_1, (512, 512), (512, 1))
        assert_size_stride(arg20_1, (512, ), (1, ))
        assert_size_stride(arg21_1, (512, ), (1, ))
        assert_size_stride(arg22_1, (512, ), (1, ))
        assert_size_stride(arg23_1, (2048, 512), (512, 1))
        assert_size_stride(arg24_1, (2048, ), (1, ))
        assert_size_stride(arg25_1, (512, 2048), (2048, 1))
        assert_size_stride(arg26_1, (512, ), (1, ))
        assert_size_stride(arg27_1, (512, ), (1, ))
        assert_size_stride(arg28_1, (512, ), (1, ))
        assert_size_stride(arg29_1, (1536, ), (1, ))
        assert_size_stride(arg30_1, (1536, 512), (512, 1))
        assert_size_stride(arg31_1, (512, 512), (512, 1))
        assert_size_stride(arg32_1, (512, ), (1, ))
        assert_size_stride(arg33_1, (512, ), (1, ))
        assert_size_stride(arg34_1, (512, ), (1, ))
        assert_size_stride(arg35_1, (2048, 512), (512, 1))
        assert_size_stride(arg36_1, (2048, ), (1, ))
        assert_size_stride(arg37_1, (512, 2048), (2048, 1))
        assert_size_stride(arg38_1, (512, ), (1, ))
        assert_size_stride(arg39_1, (512, ), (1, ))
        assert_size_stride(arg40_1, (512, ), (1, ))
        assert_size_stride(arg41_1, (1536, ), (1, ))
        assert_size_stride(arg42_1, (1536, 512), (512, 1))
        assert_size_stride(arg43_1, (512, 512), (512, 1))
        assert_size_stride(arg44_1, (512, ), (1, ))
        assert_size_stride(arg45_1, (512, ), (1, ))
        assert_size_stride(arg46_1, (512, ), (1, ))
        assert_size_stride(arg47_1, (2048, 512), (512, 1))
        assert_size_stride(arg48_1, (2048, ), (1, ))
        assert_size_stride(arg49_1, (512, 2048), (2048, 1))
        assert_size_stride(arg50_1, (512, ), (1, ))
        assert_size_stride(arg51_1, (512, ), (1, ))
        assert_size_stride(arg52_1, (512, ), (1, ))
        assert_size_stride(arg53_1, (1536, ), (1, ))
        assert_size_stride(arg54_1, (1536, 512), (512, 1))
        assert_size_stride(arg55_1, (512, 512), (512, 1))
        assert_size_stride(arg56_1, (512, ), (1, ))
        assert_size_stride(arg57_1, (512, ), (1, ))
        assert_size_stride(arg58_1, (512, ), (1, ))
        assert_size_stride(arg59_1, (2048, 512), (512, 1))
        assert_size_stride(arg60_1, (2048, ), (1, ))
        assert_size_stride(arg61_1, (512, 2048), (2048, 1))
        assert_size_stride(arg62_1, (512, ), (1, ))
        assert_size_stride(arg63_1, (512, ), (1, ))
        assert_size_stride(arg64_1, (512, ), (1, ))
        assert_size_stride(arg65_1, (1536, ), (1, ))
        assert_size_stride(arg66_1, (1536, 512), (512, 1))
        assert_size_stride(arg67_1, (512, 512), (512, 1))
        assert_size_stride(arg68_1, (512, ), (1, ))
        assert_size_stride(arg69_1, (512, ), (1, ))
        assert_size_stride(arg70_1, (512, ), (1, ))
        assert_size_stride(arg71_1, (2048, 512), (512, 1))
        assert_size_stride(arg72_1, (2048, ), (1, ))
        assert_size_stride(arg73_1, (512, 2048), (2048, 1))
        assert_size_stride(arg74_1, (512, ), (1, ))
        assert_size_stride(arg75_1, (512, ), (1, ))
        assert_size_stride(arg76_1, (512, ), (1, ))
        assert_size_stride(arg77_1, (2048, 512), (512, 1))
        assert_size_stride(arg78_1, (2048, ), (1, ))
        assert_size_stride(arg79_1, (10, 2048), (2048, 1))
        assert_size_stride(arg80_1, (10, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((2, 3, 14, 14, 16, 16), (150528, 50176, 3584, 256, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unfold, unfold_1, x], Original ATen: [aten.unfold, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_unfold_0.run(arg0_1, buf0, 301056, stream=stream0)
            del arg0_1
            buf2 = empty_strided_cuda((392, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unfold, unfold_1, x, x_1], Original ATen: [aten.unfold, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1.run(arg2_1, arg1_1, buf0, buf2, 112, 1, 1, stream=stream0)
            del arg1_1
            del arg2_1
            del buf0
            buf3 = empty_strided_cuda((2, 197, 512), (100864, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cls_tokens, x_1, x_2, x_3], Original ATen: [aten.expand, aten.view, aten.cat, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_expand_view_2.run(arg3_1, buf2, arg4_1, buf3, 201728, stream=stream0)
            del arg3_1
            del arg4_1
            del buf2
            buf4 = empty_strided_cuda((394, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cls_tokens, x_1, x_2, x_3, multi_head_attention_forward], Original ATen: [aten.expand, aten.view, aten.cat, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_cat_expand_t_view_3.run(buf3, arg6_1, buf4, 84, 1, 1, stream=stream0)
            del arg6_1
            buf5 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(arg5_1, buf4, buf5, 201728, stream=stream0)
            buf6 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(arg5_1, buf4, buf6, 201728, stream=stream0)
            buf7 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(arg5_1, buf4, buf7, 201728, stream=stream0)
            del arg5_1
            del buf4
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            buf8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf5, buf6, buf7, None, False)
            del buf5
            del buf6
            buf9 = buf8[0]
            assert_size_stride(buf9, (197, 8, 2, 64), (1024, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf9, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf8
            buf13 = reinterpret_tensor(buf7, (2, 197, 8, 64), (100864, 512, 64, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf9, buf13, 201728, stream=stream0)
            buf14 = reinterpret_tensor(buf9, (394, 512), (512, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_permute_t_view_8.run(buf13, arg7_1, buf14, 112, 1, 1, stream=stream0)
            del arg7_1
            del buf13
            buf18 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, x_5], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf18, arg8_1, buf14, arg9_1, arg10_1, 394, 512, stream=stream0)
            del arg10_1
            del arg8_1
            del arg9_1
            del buf14
            buf19 = empty_strided_cuda((394, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_10.run(buf18, arg11_1, buf19, 112, 1, 1, stream=stream0)
            del arg11_1
            buf20 = reinterpret_tensor(buf19, (2, 197, 2048), (403456, 2048, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_1, relu], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_11.run(buf20, arg12_1, 806912, stream=stream0)
            del arg12_1
            buf21 = empty_strided_cuda((394, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, relu, x_6], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_view_12.run(buf20, arg13_1, buf21, 112, 1, 1, stream=stream0)
            del arg13_1
            del buf20
            buf25 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [x_6, add_1, x_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf25, arg14_1, buf21, arg15_1, arg16_1, 394, 512, stream=stream0)
            del arg14_1
            del arg15_1
            del arg16_1
            buf26 = empty_strided_cuda((394, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_cat_expand_t_view_3.run(buf25, arg18_1, buf26, 84, 1, 1, stream=stream0)
            del arg18_1
            buf27 = reinterpret_tensor(buf21, (197, 8, 2, 64), (512, 64, 100864, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(arg17_1, buf26, buf27, 201728, stream=stream0)
            buf28 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(arg17_1, buf26, buf28, 201728, stream=stream0)
            buf29 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(arg17_1, buf26, buf29, 201728, stream=stream0)
            del arg17_1
            del buf26
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            buf30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf27, buf28, buf29, None, False)
            del buf27
            del buf28
            buf31 = buf30[0]
            assert_size_stride(buf31, (197, 8, 2, 64), (1024, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf31, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf30
            buf35 = reinterpret_tensor(buf29, (2, 197, 8, 64), (100864, 512, 64, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf31, buf35, 201728, stream=stream0)
            buf36 = reinterpret_tensor(buf31, (394, 512), (512, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_permute_t_view_8.run(buf35, arg19_1, buf36, 112, 1, 1, stream=stream0)
            del arg19_1
            del buf35
            buf40 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, add_2, x_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf40, arg20_1, buf36, arg21_1, arg22_1, 394, 512, stream=stream0)
            del arg20_1
            del arg21_1
            del arg22_1
            del buf36
            buf41 = empty_strided_cuda((394, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_10.run(buf40, arg23_1, buf41, 112, 1, 1, stream=stream0)
            del arg23_1
            buf42 = reinterpret_tensor(buf41, (2, 197, 2048), (403456, 2048, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [linear_3, relu_1], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_11.run(buf42, arg24_1, 806912, stream=stream0)
            del arg24_1
            buf43 = empty_strided_cuda((394, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, relu_1, x_9], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_view_12.run(buf42, arg25_1, buf43, 112, 1, 1, stream=stream0)
            del arg25_1
            del buf42
            buf47 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [x_9, add_3, x_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf47, arg26_1, buf43, arg27_1, arg28_1, 394, 512, stream=stream0)
            del arg26_1
            del arg27_1
            del arg28_1
            buf48 = empty_strided_cuda((394, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_cat_expand_t_view_3.run(buf47, arg30_1, buf48, 84, 1, 1, stream=stream0)
            del arg30_1
            buf49 = reinterpret_tensor(buf43, (197, 8, 2, 64), (512, 64, 100864, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(arg29_1, buf48, buf49, 201728, stream=stream0)
            buf50 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(arg29_1, buf48, buf50, 201728, stream=stream0)
            buf51 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(arg29_1, buf48, buf51, 201728, stream=stream0)
            del arg29_1
            del buf48
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            buf52 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf49, buf50, buf51, None, False)
            del buf49
            del buf50
            buf53 = buf52[0]
            assert_size_stride(buf53, (197, 8, 2, 64), (1024, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf53, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf52
            buf57 = reinterpret_tensor(buf51, (2, 197, 8, 64), (100864, 512, 64, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf53, buf57, 201728, stream=stream0)
            buf58 = reinterpret_tensor(buf53, (394, 512), (512, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_permute_t_view_8.run(buf57, arg31_1, buf58, 112, 1, 1, stream=stream0)
            del arg31_1
            del buf57
            buf62 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2, add_4, x_11], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf62, arg32_1, buf58, arg33_1, arg34_1, 394, 512, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del buf58
            buf63 = empty_strided_cuda((394, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_10.run(buf62, arg35_1, buf63, 112, 1, 1, stream=stream0)
            del arg35_1
            buf64 = reinterpret_tensor(buf63, (2, 197, 2048), (403456, 2048, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [linear_5, relu_2], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_11.run(buf64, arg36_1, 806912, stream=stream0)
            del arg36_1
            buf65 = empty_strided_cuda((394, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, relu_2, x_12], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_view_12.run(buf64, arg37_1, buf65, 112, 1, 1, stream=stream0)
            del arg37_1
            del buf64
            buf69 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [x_12, add_5, x_13], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf69, arg38_1, buf65, arg39_1, arg40_1, 394, 512, stream=stream0)
            del arg38_1
            del arg39_1
            del arg40_1
            buf70 = empty_strided_cuda((394, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_cat_expand_t_view_3.run(buf69, arg42_1, buf70, 84, 1, 1, stream=stream0)
            del arg42_1
            buf71 = reinterpret_tensor(buf65, (197, 8, 2, 64), (512, 64, 100864, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(arg41_1, buf70, buf71, 201728, stream=stream0)
            buf72 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(arg41_1, buf70, buf72, 201728, stream=stream0)
            buf73 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(arg41_1, buf70, buf73, 201728, stream=stream0)
            del arg41_1
            del buf70
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            buf74 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf71, buf72, buf73, None, False)
            del buf71
            del buf72
            buf75 = buf74[0]
            assert_size_stride(buf75, (197, 8, 2, 64), (1024, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf75, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf74
            buf79 = reinterpret_tensor(buf73, (2, 197, 8, 64), (100864, 512, 64, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf75, buf79, 201728, stream=stream0)
            buf80 = reinterpret_tensor(buf75, (394, 512), (512, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_permute_t_view_8.run(buf79, arg43_1, buf80, 112, 1, 1, stream=stream0)
            del arg43_1
            del buf79
            buf84 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3, add_6, x_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf84, arg44_1, buf80, arg45_1, arg46_1, 394, 512, stream=stream0)
            del arg44_1
            del arg45_1
            del arg46_1
            del buf80
            buf85 = empty_strided_cuda((394, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_10.run(buf84, arg47_1, buf85, 112, 1, 1, stream=stream0)
            del arg47_1
            buf86 = reinterpret_tensor(buf85, (2, 197, 2048), (403456, 2048, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [linear_7, relu_3], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_11.run(buf86, arg48_1, 806912, stream=stream0)
            del arg48_1
            buf87 = empty_strided_cuda((394, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, relu_3, x_15], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_view_12.run(buf86, arg49_1, buf87, 112, 1, 1, stream=stream0)
            del arg49_1
            del buf86
            buf91 = buf84; del buf84  # reuse
            # Topologically Sorted Source Nodes: [x_15, add_7, x_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf91, arg50_1, buf87, arg51_1, arg52_1, 394, 512, stream=stream0)
            del arg50_1
            del arg51_1
            del arg52_1
            buf92 = empty_strided_cuda((394, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_cat_expand_t_view_3.run(buf91, arg54_1, buf92, 84, 1, 1, stream=stream0)
            del arg54_1
            buf93 = reinterpret_tensor(buf87, (197, 8, 2, 64), (512, 64, 100864, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(arg53_1, buf92, buf93, 201728, stream=stream0)
            buf94 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(arg53_1, buf92, buf94, 201728, stream=stream0)
            buf95 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(arg53_1, buf92, buf95, 201728, stream=stream0)
            del arg53_1
            del buf92
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            buf96 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf93, buf94, buf95, None, False)
            del buf93
            del buf94
            buf97 = buf96[0]
            assert_size_stride(buf97, (197, 8, 2, 64), (1024, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf97, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf96
            buf101 = reinterpret_tensor(buf95, (2, 197, 8, 64), (100864, 512, 64, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf97, buf101, 201728, stream=stream0)
            buf102 = reinterpret_tensor(buf97, (394, 512), (512, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_permute_t_view_8.run(buf101, arg55_1, buf102, 112, 1, 1, stream=stream0)
            del arg55_1
            del buf101
            buf106 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4, add_8, x_17], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf106, arg56_1, buf102, arg57_1, arg58_1, 394, 512, stream=stream0)
            del arg56_1
            del arg57_1
            del arg58_1
            del buf102
            buf107 = empty_strided_cuda((394, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_10.run(buf106, arg59_1, buf107, 112, 1, 1, stream=stream0)
            del arg59_1
            buf108 = reinterpret_tensor(buf107, (2, 197, 2048), (403456, 2048, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [linear_9, relu_4], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_11.run(buf108, arg60_1, 806912, stream=stream0)
            del arg60_1
            buf109 = empty_strided_cuda((394, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, relu_4, x_18], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_view_12.run(buf108, arg61_1, buf109, 112, 1, 1, stream=stream0)
            del arg61_1
            del buf108
            buf113 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [x_18, add_9, x_19], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf113, arg62_1, buf109, arg63_1, arg64_1, 394, 512, stream=stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            buf114 = empty_strided_cuda((394, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_cat_expand_t_view_3.run(buf113, arg66_1, buf114, 84, 1, 1, stream=stream0)
            del arg66_1
            buf115 = reinterpret_tensor(buf109, (197, 8, 2, 64), (512, 64, 100864, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(arg65_1, buf114, buf115, 201728, stream=stream0)
            buf116 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5.run(arg65_1, buf114, buf116, 201728, stream=stream0)
            buf117 = empty_strided_cuda((197, 8, 2, 64), (512, 64, 100864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(arg65_1, buf114, buf117, 201728, stream=stream0)
            del arg65_1
            del buf114
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_efficient_attention]
            buf118 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf115, buf116, buf117, None, False)
            del buf115
            del buf116
            buf119 = buf118[0]
            assert_size_stride(buf119, (197, 8, 2, 64), (1024, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf119, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf118
            buf123 = reinterpret_tensor(buf117, (2, 197, 8, 64), (100864, 512, 64, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf119, buf123, 201728, stream=stream0)
            buf124 = reinterpret_tensor(buf119, (394, 512), (512, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_clone_permute_t_view_8.run(buf123, arg67_1, buf124, 112, 1, 1, stream=stream0)
            del arg67_1
            del buf123
            buf128 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5, add_10, x_20], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf128, arg68_1, buf124, arg69_1, arg70_1, 394, 512, stream=stream0)
            del arg68_1
            del arg69_1
            del arg70_1
            del buf124
            buf129 = empty_strided_cuda((394, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_t_view_10.run(buf128, arg71_1, buf129, 112, 1, 1, stream=stream0)
            del arg71_1
            buf130 = reinterpret_tensor(buf129, (2, 197, 2048), (403456, 2048, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [linear_11, relu_5], Original ATen: [aten.addmm, aten.view, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_view_11.run(buf130, arg72_1, 806912, stream=stream0)
            del arg72_1
            buf131 = empty_strided_cuda((394, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_11, relu_5, x_21], Original ATen: [aten.addmm, aten.view, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_view_12.run(buf130, arg73_1, buf131, 112, 1, 1, stream=stream0)
            del arg73_1
            del buf130
            buf135 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [x_21, add_11, x_22], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_9.run(buf135, arg74_1, buf131, arg75_1, arg76_1, 394, 512, stream=stream0)
            del arg74_1
            del arg75_1
            del arg76_1
            del buf131
            buf136 = empty_strided_cuda((2, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_21, add_11, x_22, x_23, input_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.select, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13.run(buf135, arg77_1, buf136, 64, 1, 1, stream=stream0)
            del arg77_1
            del buf135
            buf137 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_14.run(buf137, arg78_1, 4096, stream=stream0)
            del arg78_1
            buf138 = empty_strided_cuda((2, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_4], Original ATen: [aten.addmm, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_15.run(arg80_1, buf137, arg79_1, buf138, 1, 1, 1, stream=stream0)
            del arg79_1
            del arg80_1
            del buf137
        return (buf138, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((10, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
        super(Model, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, img):
        """
        Forward pass of the Vision Transformer.

        :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: The output tensor, shape (batch_size, num_classes).
        """
        p = self.patch_size
        
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.rand(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_clone_unfold_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x2 = ((xindex // 256) % 14)
    x3 = xindex // 3584
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 224*x1 + 3584*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1(in_ptr0, arg_B, in_ptr2, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    B = arg_B

    M = 392
    N = 512
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
        tmp2 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 768*idx_m + 150528*((idx_n + 768*((idx_m % 196))) // 150528), [BLOCK_M, BLOCK_K])), None, eviction_policy='evict_last')
        a = tmp2.broadcast_to(xindex.shape)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

''',
    '''@triton.jit
def triton_poi_fused_add_cat_expand_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 512) % 197)
    x0 = (xindex % 512)
    x2 = xindex // 100864
    x3 = (xindex % 100864)
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 512*((-1) + x1) + 100352*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, xmask)

''',
    '''@triton.jit
def triton_tem_fused_add_addmm_cat_expand_t_view_3(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 1536
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1536*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1536*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)

''',
    '''@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)

''',
    '''@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + 1536*x1 + 302592*x2 + 302592*((x0 + 512*x1) // 100864)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)

''',
    '''@triton.jit
def triton_poi_fused_clone_permute_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 197)
    x2 = xindex // 100864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_clone_permute_t_view_8(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 512
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 394
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 512.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp31, xmask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_t_view_10(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 2048
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2048*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2048*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_relu_view_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 806912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_relu_t_view_12(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 512
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

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
        xindex = idx_n + 2048*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_tem_fused_add_addmm_native_layer_norm_select_t_view_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 2048
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 100864
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
        xindex = idx_n + 512*idx_m
        a = tl.load(A + ((tl.broadcast_to(idx_n + 100864*idx_m, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2048*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2048*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_gelu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_gelu_t_15(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 10
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

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
        xindex = idx_n + 2048*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 10*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 10*idx_m
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
    arg0_1 = rand_strided((2, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((10, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1]

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
