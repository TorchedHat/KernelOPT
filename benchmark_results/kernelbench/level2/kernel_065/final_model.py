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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/kt/cktsjo55xpiacvb6f2uvudv4sp2rmdz4lsd5ztk4xlquezzypnju.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[128, 8, 384, 384][1179648, 147456, 384, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[128, 64, 382, 382][9339136, 145924, 382, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 1024, 'x': 262144}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1207959552, 'x': 603979776}, 'kernel_num_gb': 1.207959552, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 147456
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
    tmp0 = tl.load(in_ptr0 + (x2 + 147456*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 1179648*y1), tmp0, None)


def get_args():
    arg_0 = rand_strided((128, 8, 384, 384), (1179648, 147456, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 384, 384), (1179648, 1, 3072, 8), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1024, 147456,


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
    num_gb = 1.207959552
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/yg/cyggg4ke2deuarwqt76km52xh63wigodq5qcjvs7frv4x4cwuitz.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 8, 3, 3][72, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[128, 64, 382, 382][9339136, 145924, 382, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/zb/czb27h636e7izaucq2mgthhqjnzuuafpkdsjd3kipmbm5u3dg3sa.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %buf2 : Tensor "f32[128, 64, 382, 382][9339136, 1, 24448, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[128, 64, 382, 382][9339136, 145924, 382, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 14344913152}, 'kernel_num_gb': 9.56327552, 'kernel_flop': 172138954752},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1195409408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)


def get_args():
    arg_0 = rand_strided((128, 64, 382, 382), (9339136, 1, 24448, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1195409408,


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
    num_gb = 9.56327552
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ap/caplkpp42nzbsfqxe73cylpt3g2mafheiyrjla73q677p3nw766p.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.avg_pool2d]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => avg_pool2d
# Graph fragment:
#   %convolution : Tensor "f32[128, 64, 382, 382][9339136, 1, 24448, 64]cuda:0" = PlaceHolder[target=convolution]
#   %convolution : Tensor "f32[128, 64, 382, 382][9339136, 145924, 382, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d : Tensor "f32[128, 64, 95, 95][577600, 9025, 95, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution, [4, 4], [4, 4]), kwargs = {})
#   return %avg_pool2d
triton_poi_fused_avg_pool2d_convolution_3 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_3', '''

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
    size_hints={'y': 2097152, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 16, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 591462400, 'x': 4731699200}, 'kernel_num_gb': 5.0274304, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1155200
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 95)
    y1 = ((yindex // 95) % 95)
    y2 = yindex // 9025
    y4 = (yindex % 9025)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (24448 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (24512 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (24576 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (24640 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (48896 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (48960 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (49024 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (49088 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (73344 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (73408 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (73472 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (73536 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1, 1], 16.0, tl.float32)
    tmp32 = (tmp30 / tmp31)
    tl.store(out_ptr0 + (y4 + 9025*x3 + 577600*y2), tmp32, xmask & ymask)

def get_args():
    arg_0 = rand_strided((128, 64, 382, 382), (9339136, 1, 24448, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 95, 95), (577600, 9025, 95, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1155200, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_convolution_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_avg_pool2d_convolution_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 5.0274304
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/na/cnaerrbocxvkcya3xqtdbt3oyi5ky3st4fur7lalydr4airaihgn.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sigmoid, aten.sum]
# Source node to ATen node mapping:
#   x_2 => sigmoid
#   x_3 => sum_1
# Graph fragment:
#   %avg_pool2d : Tensor "f32[128, 64, 95, 95][577600, 9025, 95, 1]cuda:0" = PlaceHolder[target=avg_pool2d]
#   %sigmoid : Tensor "f32[128, 64, 95, 95][577600, 9025, 95, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%avg_pool2d,), kwargs = {})
#   %sum_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid, [1, 2, 3]), kwargs = {})
#   return %buf5
triton_red_fused_sigmoid_sum_4 = async_compile.triton('triton_red_fused_sigmoid_sum_4', '''

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sigmoid_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5120, 'r0_': 295731200}, 'kernel_num_gb': 0.29573376, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_sigmoid_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 115520
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, mask=xmask)

def get_args():
    arg_0 = rand_strided((128, 64, 95, 95), (577600, 9025, 95, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 640, 115520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_sigmoid_sum_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_sigmoid_sum_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.29573376
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ix/cixcydjhyabgf3m6spmr7klthcsd4eeg3rxfxt7ubbujqg63nwjt.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sigmoid, aten.sum]
# Source node to ATen node mapping:
#   x_2 => sigmoid
#   x_3 => sum_1
# Graph fragment:
#   %buf5 : Tensor "f32[128, 5][5, 1]cuda:0" = PlaceHolder[target=buf5]
#   %sigmoid : Tensor "f32[128, 64, 95, 95][577600, 9025, 95, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%avg_pool2d,), kwargs = {})
#   %sum_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid, [1, 2, 3]), kwargs = {})
#   return %sum_1
triton_per_fused_sigmoid_sum_5 = async_compile.triton('triton_per_fused_sigmoid_sum_5', '''

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
    size_hints={'x': 128, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sigmoid_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 1600}, 'kernel_num_gb': 3.072e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_sigmoid_sum_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, mask=xmask)

def get_args():
    arg_0 = rand_strided((128, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 128, 5,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_sigmoid_sum_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_sigmoid_sum_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 3.072e-06
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 8, 3, 3), (72, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (128, 8, 384, 384), (1179648, 147456, 384, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 8, 384, 384), (1179648, 1, 3072, 8), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 1024, 147456, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 8, 3, 3), (72, 1, 24, 8), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 512, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (128, 64, 382, 382), (9339136, 1, 24448, 64), 'unknown_op')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_2.run(buf3, arg1_1, 1195409408, stream=stream0)
            del arg1_1
            buf4 = empty_strided_cuda((128, 64, 95, 95), (577600, 9025, 95, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.avg_pool2d]
            stream0 = get_raw_stream(0)
            triton_poi_fused_avg_pool2d_convolution_3.run(buf3, buf4, 1155200, 64, stream=stream0)
            del buf3
            buf5 = empty_strided_cuda((128, 5), (5, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sigmoid, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sigmoid_sum_4.run(buf4, buf5, 640, 115520, stream=stream0)
            del buf4
            buf6 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sigmoid, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_sigmoid_sum_5.run(buf5, buf6, 128, 5, stream=stream0)
            del buf5
        return (buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 8, 384, 384), (1179648, 147456, 384, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=[1,2,3]) # Sum over all spatial dimensions
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 147456
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
    tmp0 = tl.load(in_ptr0 + (x2 + 147456*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 1179648*y1), tmp0, None)

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
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1195409408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)

''',
    '''@triton.jit
def triton_poi_fused_avg_pool2d_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1155200
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 95)
    y1 = ((yindex // 95) % 95)
    y2 = yindex // 9025
    y4 = (yindex % 9025)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (24448 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (24512 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (24576 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (24640 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (48896 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (48960 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (49024 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (49088 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (73344 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (73408 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (73472 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (73536 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1, 1], 16.0, tl.float32)
    tmp32 = (tmp30 / tmp31)
    tl.store(out_ptr0 + (y4 + 9025*x3 + 577600*y2), tmp32, xmask & ymask)

''',
    '''@triton.jit
def triton_red_fused_sigmoid_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 115520
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)

''',
    '''@triton.jit
def triton_per_fused_sigmoid_sum_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 8, 384, 384), (1179648, 147456, 384, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]

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
