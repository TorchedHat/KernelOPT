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


# kernel path: /workspace/tmp/torchinductor_user/6e/c6eovtv22omoxs2vmslgjik3gz6b2lp6h3234zlhyidtrzataywq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[10, 3, 32, 32][3072, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[10, 128, 8, 8][8192, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 32, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 245760, 'x': 122880}, 'kernel_num_gb': 0.00024576, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 3072*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 32, 32), (3072, 1, 96, 3), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 30, 1024,


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
    num_gb = 0.00024576
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/65/c65eteorarl4u5mmmy4i6cbbexclhzckz3esdxe62bovk2cmmor6.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[128, 3, 4, 4][48, 16, 4, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[10, 128, 8, 8][8192, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 49152, 'x': 24576}, 'kernel_num_gb': 4.9152e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 48*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 384, 16,


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
    num_gb = 4.9152e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/7r/c7r4jmult72hjtykyxy7gxlwegie5bdfwvsfe45mworzktt45ex2.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %buf2 : Tensor "f32[10, 128, 8, 8][8192, 1, 1024, 128]cuda:0" = PlaceHolder[target=buf2]
#   %arg2_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[10, 128, 8, 8][8192, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''

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
    size_hints={'y': 2048, 'x': 2112}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 328192, 'x': 655360}, 'kernel_num_gb': 0.000655872, 'kernel_flop': 7864320},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 8192*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 64*y3), tmp2, xmask & ymask)

def get_args():
    arg_0 = rand_strided((10, 128, 8, 8), (8192, 1, 1024, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 128, 8, 8), (8192, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1280, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000655872
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/kh/ckhnaa7auot5wluze6d45c3kqfssgj3bwx274nrlwswcpweko66u.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => view
#   x_2 => addmm, permute
# Graph fragment:
#   %arg4_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f32[10, 128, 8, 8][8192, 64, 8, 1]cuda:0" = PlaceHolder[target=convolution]
#   %arg3_1 : Tensor "f32[128, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[10, 128, 8, 8][8192, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[10, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [10, 8192]), kwargs = {})
#   %permute : Tensor "f32[8192, 128][1, 8192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "f32[10, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg4_1, %view, %permute), kwargs = {})
#   return %addmm
triton_tem_fused_addmm_convolution_t_view_3 = async_compile.triton('triton_tem_fused_addmm_convolution_t_view_3', '''

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
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_convolution_t_view_3', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.004532224, 'kernel_flop': 20971520, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_convolution_t_view_3(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 10
    N = 128
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
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def get_args():
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 128, 8, 8), (8192, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 4, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_convolution_t_view_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_convolution_t_view_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004532224
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ec/ceckp5gugporhedqwmsrea65k4ylkbbs7zyfmqy2cevsfqf3hdj6.py
# Topologically Sorted Source Nodes: [cls_tokens, unsqueeze, x_3], Original ATen: [aten.expand, aten.unsqueeze, aten.cat]
# Source node to ATen node mapping:
#   cls_tokens => expand
#   unsqueeze => unsqueeze
#   x_3 => cat
# Graph fragment:
#   %arg5_1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %expand : Tensor "f32[10, 1, 128][0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%arg5_1, [10, -1, -1]), kwargs = {})
#   %unsqueeze : Tensor "f32[10, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%addmm, 1), kwargs = {})
#   %cat : Tensor "f32[10, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %unsqueeze], 1), kwargs = {})
#   return %buf5
triton_poi_fused_cat_expand_unsqueeze_4 = async_compile.triton('triton_poi_fused_cat_expand_unsqueeze_4', '''
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
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_expand_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 10752}, 'kernel_num_gb': 5.632e-06, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + 256*x1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1280,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_expand_unsqueeze_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_cat_expand_unsqueeze_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 5.632e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/33/c33y6zs5ryr44a436262zvnagh77ccna2hucxnrbk5sisup5ztil.py
# Topologically Sorted Source Nodes: [cls_tokens, unsqueeze, x_3], Original ATen: [aten.expand, aten.unsqueeze, aten.cat]
# Source node to ATen node mapping:
#   cls_tokens => expand
#   unsqueeze => unsqueeze
#   x_3 => cat
# Graph fragment:
#   %addmm : Tensor "f32[10, 128][128, 1]cuda:0" = PlaceHolder[target=addmm]
#   %expand : Tensor "f32[10, 1, 128][0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%arg5_1, [10, -1, -1]), kwargs = {})
#   %unsqueeze : Tensor "f32[10, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%addmm, 1), kwargs = {})
#   %cat : Tensor "f32[10, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %unsqueeze], 1), kwargs = {})
#   return %buf6
triton_poi_fused_cat_expand_unsqueeze_5 = async_compile.triton('triton_poi_fused_cat_expand_unsqueeze_5', '''
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
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_expand_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15360}, 'kernel_num_gb': 1.024e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 256*x1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1280,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_expand_unsqueeze_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_cat_expand_unsqueeze_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.024e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/r4/cr47sns24ycbyevjqyd5nq3ozofwjiirauk7ywqahe6ldklfc5o2.py
# Topologically Sorted Source Nodes: [getitem, linear_1], Original ATen: [aten.select, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   getitem => select
#   linear_1 => addmm_1, permute_1
# Graph fragment:
#   %arg79_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %_transformer_encoder_layer_fwd_5 : Tensor "f32[10, 2, 128][256, 128, 1]cuda:0" = PlaceHolder[target=_transformer_encoder_layer_fwd_5]
#   %arg78_1 : Tensor "f32[1000, 128][128, 1]cuda:0" = PlaceHolder[target=arg78_1]
#   %select : Tensor "f32[10, 128][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%_transformer_encoder_layer_fwd_5, 1, 0), kwargs = {})
#   %permute_1 : Tensor "f32[128, 1000][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg78_1, [1, 0]), kwargs = {})
#   %addmm_1 : Tensor "f32[10, 1000][1000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg79_1, %select, %permute_1), kwargs = {})
#   return %addmm_1
triton_tem_fused_addmm_select_t_6 = async_compile.triton('triton_tem_fused_addmm_select_t_6', '''
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
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_select_t_6', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00059712, 'kernel_flop': 2560000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_select_t_6(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 10
    N = 1000
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
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
        a = tl.load(A + ((tl.broadcast_to(idx_n + 256*idx_m, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1000*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def get_args():
    arg_0 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 32, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_select_t_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_select_t_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00059712
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1 = args
        args.clear()
        assert_size_stride(arg0_1, (10, 3, 32, 32), (3072, 1024, 32, 1))
        assert_size_stride(arg1_1, (128, 3, 4, 4), (48, 16, 4, 1))
        assert_size_stride(arg2_1, (128, ), (1, ))
        assert_size_stride(arg3_1, (128, 8192), (8192, 1))
        assert_size_stride(arg4_1, (128, ), (1, ))
        assert_size_stride(arg5_1, (1, 1, 128), (128, 128, 1))
        assert_size_stride(arg6_1, (384, ), (1, ))
        assert_size_stride(arg7_1, (384, 128), (128, 1))
        assert_size_stride(arg8_1, (128, 128), (128, 1))
        assert_size_stride(arg9_1, (128, ), (1, ))
        assert_size_stride(arg10_1, (128, ), (1, ))
        assert_size_stride(arg11_1, (128, ), (1, ))
        assert_size_stride(arg12_1, (128, ), (1, ))
        assert_size_stride(arg13_1, (128, ), (1, ))
        assert_size_stride(arg14_1, (512, 128), (128, 1))
        assert_size_stride(arg15_1, (512, ), (1, ))
        assert_size_stride(arg16_1, (128, 512), (512, 1))
        assert_size_stride(arg17_1, (128, ), (1, ))
        assert_size_stride(arg18_1, (384, ), (1, ))
        assert_size_stride(arg19_1, (384, 128), (128, 1))
        assert_size_stride(arg20_1, (128, 128), (128, 1))
        assert_size_stride(arg21_1, (128, ), (1, ))
        assert_size_stride(arg22_1, (128, ), (1, ))
        assert_size_stride(arg23_1, (128, ), (1, ))
        assert_size_stride(arg24_1, (128, ), (1, ))
        assert_size_stride(arg25_1, (128, ), (1, ))
        assert_size_stride(arg26_1, (512, 128), (128, 1))
        assert_size_stride(arg27_1, (512, ), (1, ))
        assert_size_stride(arg28_1, (128, 512), (512, 1))
        assert_size_stride(arg29_1, (128, ), (1, ))
        assert_size_stride(arg30_1, (384, ), (1, ))
        assert_size_stride(arg31_1, (384, 128), (128, 1))
        assert_size_stride(arg32_1, (128, 128), (128, 1))
        assert_size_stride(arg33_1, (128, ), (1, ))
        assert_size_stride(arg34_1, (128, ), (1, ))
        assert_size_stride(arg35_1, (128, ), (1, ))
        assert_size_stride(arg36_1, (128, ), (1, ))
        assert_size_stride(arg37_1, (128, ), (1, ))
        assert_size_stride(arg38_1, (512, 128), (128, 1))
        assert_size_stride(arg39_1, (512, ), (1, ))
        assert_size_stride(arg40_1, (128, 512), (512, 1))
        assert_size_stride(arg41_1, (128, ), (1, ))
        assert_size_stride(arg42_1, (384, ), (1, ))
        assert_size_stride(arg43_1, (384, 128), (128, 1))
        assert_size_stride(arg44_1, (128, 128), (128, 1))
        assert_size_stride(arg45_1, (128, ), (1, ))
        assert_size_stride(arg46_1, (128, ), (1, ))
        assert_size_stride(arg47_1, (128, ), (1, ))
        assert_size_stride(arg48_1, (128, ), (1, ))
        assert_size_stride(arg49_1, (128, ), (1, ))
        assert_size_stride(arg50_1, (512, 128), (128, 1))
        assert_size_stride(arg51_1, (512, ), (1, ))
        assert_size_stride(arg52_1, (128, 512), (512, 1))
        assert_size_stride(arg53_1, (128, ), (1, ))
        assert_size_stride(arg54_1, (384, ), (1, ))
        assert_size_stride(arg55_1, (384, 128), (128, 1))
        assert_size_stride(arg56_1, (128, 128), (128, 1))
        assert_size_stride(arg57_1, (128, ), (1, ))
        assert_size_stride(arg58_1, (128, ), (1, ))
        assert_size_stride(arg59_1, (128, ), (1, ))
        assert_size_stride(arg60_1, (128, ), (1, ))
        assert_size_stride(arg61_1, (128, ), (1, ))
        assert_size_stride(arg62_1, (512, 128), (128, 1))
        assert_size_stride(arg63_1, (512, ), (1, ))
        assert_size_stride(arg64_1, (128, 512), (512, 1))
        assert_size_stride(arg65_1, (128, ), (1, ))
        assert_size_stride(arg66_1, (384, ), (1, ))
        assert_size_stride(arg67_1, (384, 128), (128, 1))
        assert_size_stride(arg68_1, (128, 128), (128, 1))
        assert_size_stride(arg69_1, (128, ), (1, ))
        assert_size_stride(arg70_1, (128, ), (1, ))
        assert_size_stride(arg71_1, (128, ), (1, ))
        assert_size_stride(arg72_1, (128, ), (1, ))
        assert_size_stride(arg73_1, (128, ), (1, ))
        assert_size_stride(arg74_1, (512, 128), (128, 1))
        assert_size_stride(arg75_1, (512, ), (1, ))
        assert_size_stride(arg76_1, (128, 512), (512, 1))
        assert_size_stride(arg77_1, (128, ), (1, ))
        assert_size_stride(arg78_1, (1000, 128), (128, 1))
        assert_size_stride(arg79_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((10, 3, 32, 32), (3072, 1, 96, 3), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg0_1, buf0, 30, 1024, stream=stream0)
            del arg0_1
            buf1 = empty_strided_cuda((128, 3, 4, 4), (48, 1, 12, 3), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg1_1, buf1, 384, 16, stream=stream0)
            del arg1_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (10, 128, 8, 8), (8192, 1, 1024, 128), 'unknown_op')
            del buf0
            del buf1
            buf3 = empty_strided_cuda((10, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_2.run(buf2, arg2_1, buf3, 1280, 64, stream=stream0)
            del arg2_1
            del buf2
            buf4 = empty_strided_cuda((10, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_convolution_t_view_3.run(arg4_1, buf3, arg3_1, buf4, 4, 1, 1, stream=stream0)
            del arg3_1
            del arg4_1
            del buf3
            buf7 = empty_strided_cuda((10, 2, 128), (256, 128, 1), torch.float32)
            buf5 = reinterpret_tensor(buf7, (10, 1, 128), (256, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [cls_tokens, unsqueeze, x_3], Original ATen: [aten.expand, aten.unsqueeze, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_expand_unsqueeze_4.run(arg5_1, buf5, 1280, stream=stream0)
            del arg5_1
            buf6 = reinterpret_tensor(buf7, (10, 1, 128), (256, 128, 1), 128)  # alias
            # Topologically Sorted Source Nodes: [cls_tokens, unsqueeze, x_3], Original ATen: [aten.expand, aten.unsqueeze, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_expand_unsqueeze_5.run(buf4, buf6, 1280, stream=stream0)
            del buf4
            del buf5
            del buf6
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._transformer_encoder_layer_fwd]
            buf8 = torch.ops.aten._transformer_encoder_layer_fwd.default(buf7, 128, 4, arg7_1, arg6_1, arg8_1, arg9_1, False, False, 1e-05, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1)
            del arg10_1
            del arg11_1
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            del arg16_1
            del arg17_1
            del arg6_1
            del arg7_1
            del arg8_1
            del arg9_1
            del buf7
            buf9 = buf8
            assert_size_stride(buf9, (10, 2, 128), (256, 128, 1), 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            assert_alignment(buf9, 16, 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            del buf8
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._transformer_encoder_layer_fwd]
            buf10 = torch.ops.aten._transformer_encoder_layer_fwd.default(buf9, 128, 4, arg19_1, arg18_1, arg20_1, arg21_1, False, False, 1e-05, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1)
            del arg18_1
            del arg19_1
            del arg20_1
            del arg21_1
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del arg26_1
            del arg27_1
            del arg28_1
            del arg29_1
            del buf9
            buf11 = buf10
            assert_size_stride(buf11, (10, 2, 128), (256, 128, 1), 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            assert_alignment(buf11, 16, 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            del buf10
            # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._transformer_encoder_layer_fwd]
            buf12 = torch.ops.aten._transformer_encoder_layer_fwd.default(buf11, 128, 4, arg31_1, arg30_1, arg32_1, arg33_1, False, False, 1e-05, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1)
            del arg30_1
            del arg31_1
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            del arg36_1
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            del arg41_1
            del buf11
            buf13 = buf12
            assert_size_stride(buf13, (10, 2, 128), (256, 128, 1), 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            assert_alignment(buf13, 16, 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            del buf12
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._transformer_encoder_layer_fwd]
            buf14 = torch.ops.aten._transformer_encoder_layer_fwd.default(buf13, 128, 4, arg43_1, arg42_1, arg44_1, arg45_1, False, False, 1e-05, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            del arg46_1
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            del arg51_1
            del arg52_1
            del arg53_1
            del buf13
            buf15 = buf14
            assert_size_stride(buf15, (10, 2, 128), (256, 128, 1), 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            del buf14
            # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._transformer_encoder_layer_fwd]
            buf16 = torch.ops.aten._transformer_encoder_layer_fwd.default(buf15, 128, 4, arg55_1, arg54_1, arg56_1, arg57_1, False, False, 1e-05, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1)
            del arg54_1
            del arg55_1
            del arg56_1
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            del arg61_1
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            del buf15
            buf17 = buf16
            assert_size_stride(buf17, (10, 2, 128), (256, 128, 1), 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            assert_alignment(buf17, 16, 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            del buf16
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._transformer_encoder_layer_fwd]
            buf18 = torch.ops.aten._transformer_encoder_layer_fwd.default(buf17, 128, 4, arg67_1, arg66_1, arg68_1, arg69_1, False, False, 1e-05, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1)
            del arg66_1
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            del arg71_1
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            del arg76_1
            del arg77_1
            del buf17
            buf19 = buf18
            assert_size_stride(buf19, (10, 2, 128), (256, 128, 1), 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            assert_alignment(buf19, 16, 'torch.ops.aten._transformer_encoder_layer_fwd.default')
            del buf18
            buf20 = empty_strided_cuda((10, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem, linear_1], Original ATen: [aten.select, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_select_t_6.run(arg79_1, buf19, arg78_1, buf20, 32, 1, 1, stream=stream0)
            del arg78_1
            del arg79_1
            del buf19
        return (buf20, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((10, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Convolutional Vision Transformer (CViT) implementation.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param image_size: Height/width of the square input image.
        """
        super(Model, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the CViT model.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        B = x.size(0)
        x = self.conv1(x)                  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(start_dim=1)         # (B, embed_dim * num_patches)
        x = self.linear_proj(x)            # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])        # Use [CLS] token for classification

# === Test config ===
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]
'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 3072*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 48*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 8192*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 64*y3), tmp2, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_convolution_t_view_3(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 10
    N = 128
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
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

''',
    '''@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + 256*x1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 256*x1), tmp0, xmask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_select_t_6(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 10
    N = 1000
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
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
        a = tl.load(A + ((tl.broadcast_to(idx_n + 256*idx_m, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1000*idx_m
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
    arg0_1 = rand_strided((10, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1]

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
