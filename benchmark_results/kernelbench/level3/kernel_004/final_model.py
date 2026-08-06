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


# kernel path: /workspace/tmp/torchinductor_user/7e/c7e2ewnqpfokc7vqywzfgwmzyvsriw4jbt2qjypasl5qwfn27nq5.py
# Topologically Sorted Source Nodes: [conv2d, x], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv2d => convolution
#   x => relu
# Graph fragment:
#   %buf0 : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[6][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_0 = async_compile.triton('triton_poi_fused_convolution_relu_0', '''

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
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 231211008}, 'kernel_num_gb': 0.154140696, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.max_contiguous(
                 tl.multiple_of(xoffset + tl.arange(0, XBLOCK)[:], XBLOCK),
                 XBLOCK)
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 784) % 6)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)

def get_args():
    arg_0 = rand_strided((4096, 6, 28, 28), (4704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 19267584,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_relu_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.154140696
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/k2/ck24gza76etwjz6yarq4ufudezltfvjbyez6kxcg5bc4e7rasbnh.py
# Topologically Sorted Source Nodes: [conv2d, x, x_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   conv2d => convolution
#   x => relu
#   x_1 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0" = PlaceHolder[target=relu]
#   %convolution : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', '''

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
    size_hints={'y': 32768, 'x': 4224}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 38535168, 'x': 0}, 'kernel_num_gb': 0.09633792, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 14)
    x3 = xindex // 14
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 6*x5 + 1176*y1), tmp6, mask=ymask & xmask)

def get_args():
    arg_0 = rand_strided((4096, 6, 28, 28), (4704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 6, 14, 14), (1176, 1, 84, 6), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 24576, 196,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_max_pool2d_with_indices_relu_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.09633792
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/4a/c4arvkfu67wwbhcbls6cqjep5yu7d3rsivozqbbcle4zv4evjn4k.py
# Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   conv2d => convolution
#   conv2d_1 => convolution_1
#   x => relu
#   x_1 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %arg3_1 : Tensor "f32[16, 6, 5, 5][150, 25, 5, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf3
triton_poi_fused_convolution_max_pool2d_with_indices_relu_2 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_2', '''

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
    size_hints={'y': 128, 'x': 4224}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 19200, 'x': 9600}, 'kernel_num_gb': 1.92e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 6*x2 + 150*y1), tmp0, xmask & ymask)

def get_args():
    arg_0 = rand_strided((16, 6, 5, 5), (150, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 6, 5, 5), (150, 1, 30, 6), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 96, 25,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_max_pool2d_with_indices_relu_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.92e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ey/cey4hdrhpvxdhay7gsxdr6w74y3zngbwvkebkxuprsfh5bkfyw42.py
# Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   conv2d => convolution
#   conv2d_1 => convolution_1
#   x => relu
#   x_1 => _low_memory_max_pool_with_offsets
#   x_2 => relu_1
# Graph fragment:
#   %buf4 : Tensor "f32[4096, 16, 10, 10][1600, 1, 160, 16]cuda:0" = PlaceHolder[target=buf4]
#   %arg4_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', '''
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
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 78643264}, 'kernel_num_gb': 0.052428864, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6553600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


def get_args():
    arg_0 = rand_strided((4096, 16, 10, 10), (1600, 1, 160, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 6553600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.052428864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/7w/c7wf6c5ttfjzmd2jipmuaavzbv4f2gc67gbft2dzihzqudn4vgbj.py
# Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1, x_2, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   conv2d => convolution
#   conv2d_1 => convolution_1
#   x => relu
#   x_1 => _low_memory_max_pool_with_offsets
#   x_2 => relu_1
#   x_3 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %relu_1 : Tensor "f32[4096, 16, 10, 10][1600, 1, 160, 16]cuda:0" = PlaceHolder[target=relu_1]
#   %convolution : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', '''

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
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 13107200, 'x': 26214400}, 'kernel_num_gb': 0.032768, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Output: (N=4096, C=16, OH=5, OW=5), strides=(400, 25, 5, 1)
    # Input:  (N=4096, C=16, IH=10, IW=10), strides=(1600, 1, 160, 16)
    # Max-pool 2x2 stride=2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # Decode yindex -> (n, oh, ow): yindex = n*25 + oh*5 + ow
    n  = yindex // 25
    rem = yindex % 25
    oh = rem // 5
    ow = rem % 5
    c = xindex  # channel index
    # Input base offset: n*1600 + c*1 + oh*2*160 + ow*2*16
    base = n * 1600 + c + oh * 320 + ow * 32
    mask = xmask & ymask
    # Load 2x2 pool window: offsets 0, +16, +160, +176
    v00 = tl.load(in_ptr0 + base,       mask=mask, other=-3.4e38)
    v01 = tl.load(in_ptr0 + base + 16,  mask=mask, other=-3.4e38)
    v10 = tl.load(in_ptr0 + base + 160, mask=mask, other=-3.4e38)
    v11 = tl.load(in_ptr0 + base + 176, mask=mask, other=-3.4e38)
    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    # Output offset: n*400 + c*25 + oh*5 + ow
    out_offset = n * 400 + c * 25 + oh * 5 + ow
    tl.store(out_ptr0 + out_offset, result, mask=mask)

def get_args():
    arg_0 = rand_strided((4096, 16, 10, 10), (1600, 1, 160, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 102400, 16,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.032768
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/4r/c4rf4eu3g3pi63qqds27af75kvzi6hzfxyuw3x44ms4oy6gtp5gu.py
# Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1, x_2, x_3, x_4, linear, x_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   conv2d => convolution
#   conv2d_1 => convolution_1
#   linear => add_tensor_1, mm_default_1, permute
#   x => relu
#   x_1 => _low_memory_max_pool_with_offsets
#   x_2 => relu_1
#   x_3 => _low_memory_max_pool_with_offsets_1
#   x_4 => view
#   x_5 => relu_2
# Graph fragment:
#   %getitem_2 : Tensor "f32[4096, 16, 5, 5][400, 25, 5, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %arg5_1 : Tensor "f32[120, 400][400, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %mm_default_1 : Tensor "f32[4096, 120][120, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %convolution : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[4096, 6, 28, 28][4704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[4096, 16, 10, 10][1600, 100, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %view : Tensor "f32[4096, 400][400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [-1, 400]), kwargs = {})
#   %permute : Tensor "f32[400, 120][1, 400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0]), kwargs = {})
#   %mm_default_1 : Tensor "f32[4096, 120][120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %permute), kwargs = {})
#   %add_tensor_1 : Tensor "f32[4096, 120][120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg6_1, %mm_default_1), kwargs = {})
#   %relu_2 : Tensor "f32[4096, 120][120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   return %mm_default_1,%relu_2
triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5 = async_compile.triton('triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5', '''

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
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00871168, 'kernel_flop': 393216000, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5(arg_A, arg_B, in_ptr2, out_ptr1):
    EVEN_K : tl.constexpr = False
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

    M = 4096
    N = 120
    K = 400
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 400
    stride_ak = 1
    stride_bk = 1
    stride_bn = 400

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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Define block pointers for TMA-style loads before the k loop
    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # Load B^T directly as (BLOCK_K, BLOCK_N) tile to avoid tl.trans() smem bank conflicts.
    # B is stored as (120, 400) with strides (400, 1).
    # B^T[k, n] = B_flat[n*400 + k*1], so shape=(K,N), strides=(stride_bk=1, stride_bn=400).
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
        # b is already (BLOCK_K, BLOCK_N) = B^T tile, no tl.trans() needed
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))  # advance along K dim (axis=0)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 120*idx_m
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = tmp0 + acc
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tl.maximum(tmp2, tmp1)
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 120*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)

def get_args():
    arg_0 = rand_strided((4096, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((120, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((4096, 120), (120, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 128, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00871168
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/rn/crnhrjugvxkdghjww2aftoaffelqppygqwxen3665mz3idsr43mb.py
# Topologically Sorted Source Nodes: [linear, x_5, linear_1], Original ATen: [aten.addmm, aten.relu, aten.t]
# Source node to ATen node mapping:
#   linear => add_tensor_1
#   linear_1 => mm_default, permute_1
#   x_5 => relu_2
# Graph fragment:
#   %relu_2 : Tensor "f32[4096, 120][120, 1]cuda:0" = PlaceHolder[target=relu_2]
#   %arg7_1 : Tensor "f32[84, 120][120, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %add_tensor_1 : Tensor "f32[4096, 120][120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg6_1, %mm_default_1), kwargs = {})
#   %relu_2 : Tensor "f32[4096, 120][120, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   %permute_1 : Tensor "f32[120, 84][1, 120]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg7_1, [1, 0]), kwargs = {})
#   %mm_default : Tensor "f32[4096, 84][84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%relu_2, %permute_1), kwargs = {})
#   return %mm_default
triton_tem_fused_addmm_relu_t_6 = async_compile.triton('triton_tem_fused_addmm_relu_t_6', '''

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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_relu_t_6', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.003382656, 'kernel_flop': 82575360, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_relu_t_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
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

    M = 4096
    N = 84
    K = 120
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 120
    stride_ak = 1
    stride_bk = 1
    stride_bn = 120

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

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 120*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0, eviction_policy='evict_first')

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 84*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 120*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0, eviction_policy='evict_first')


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 84*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')

def get_args():
    arg_0 = rand_strided((4096, 120), (120, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((84, 120), (120, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4096, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 128, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_relu_t_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_relu_t_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.003382656
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/j7/cj7bgcvroufhou3uzu6bzpbzfxc7iavuap5gy4ertgj23vnhpl7j.py
# Topologically Sorted Source Nodes: [linear_1, x_6], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_1 => add_tensor
#   x_6 => relu_3
# Graph fragment:
#   %arg8_1 : Tensor "f32[84][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %mm_default : Tensor "f32[4096, 84][84, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[4096, 84][84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, %mm_default), kwargs = {})
#   %relu_3 : Tensor "f32[4096, 84][84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   return %relu_3
triton_poi_fused_addmm_relu_7 = async_compile.triton('triton_poi_fused_addmm_relu_7', '''

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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4129104}, 'kernel_num_gb': 0.002752848, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 84)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.where(tmp2 > 0.0, tmp2, 0.0)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)

def get_args():
    arg_0 = rand_strided((4096, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((84,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 344064,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_relu_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002752848
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/37/c37ifkph57ntwv2ferlwzbgclasjexstjszhur5uskxxwxopdmys.py
# Topologically Sorted Source Nodes: [linear_1, x_6, x_7], Original ATen: [aten.addmm, aten.relu, aten.t]
# Source node to ATen node mapping:
#   linear_1 => add_tensor
#   x_6 => relu_3
#   x_7 => addmm_2, permute_2
# Graph fragment:
#   %arg10_1 : Tensor "f32[20][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %relu_3 : Tensor "f32[4096, 84][84, 1]cuda:0" = PlaceHolder[target=relu_3]
#   %arg9_1 : Tensor "f32[20, 84][84, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %add_tensor : Tensor "f32[4096, 84][84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, %mm_default), kwargs = {})
#   %relu_3 : Tensor "f32[4096, 84][84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   %permute_2 : Tensor "f32[84, 20][1, 84]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg9_1, [1, 0]), kwargs = {})
#   %addmm_2 : Tensor "f32[4096, 20][20, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg10_1, %relu_3, %permute_2), kwargs = {})
#   return %addmm_2
triton_tem_fused_addmm_relu_t_8 = async_compile.triton('triton_tem_fused_addmm_relu_t_8', '''

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
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_relu_t_8', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.002038336, 'kernel_flop': 13762560, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_relu_t_8(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 20
    K = 84
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 84
    stride_ak = 1
    stride_bk = 1
    stride_bn = 84

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

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        idx_k = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 84*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        # B layout: (20, 84) with strides (84, 1) — B[n,k] = B_ptr + n*84 + k
        # Load as [BLOCK_K, BLOCK_N]: k contiguous (stride=1), n strided (stride=84)
        xindex_b = idx_k + offs_b_n[None, :] * 84
        b = tl.load(B + xindex_b, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 20*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def get_args():
    arg_0 = rand_strided((20,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((20, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((4096, 20), (20, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 128, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_relu_t_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_relu_t_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002038336
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
        args.clear()
        assert_size_stride(arg0_1, (6, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg1_1, (6, ), (1, ))
        assert_size_stride(arg2_1, (4096, 1, 32, 32), (1024, 1024, 32, 1))
        assert_size_stride(arg3_1, (16, 6, 5, 5), (150, 25, 5, 1))
        assert_size_stride(arg4_1, (16, ), (1, ))
        assert_size_stride(arg5_1, (120, 400), (400, 1))
        assert_size_stride(arg6_1, (120, ), (1, ))
        assert_size_stride(arg7_1, (84, 120), (120, 1))
        assert_size_stride(arg8_1, (84, ), (1, ))
        assert_size_stride(arg9_1, (20, 84), (84, 1))
        assert_size_stride(arg10_1, (20, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (4096, 6, 28, 28), (4704, 784, 28, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [conv2d, x], Original ATen: [aten.convolution, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_relu_0.run(buf1, arg1_1, 19267584, stream=stream0)
            del arg1_1
            buf2 = empty_strided_cuda((4096, 6, 14, 14), (1176, 1, 84, 6), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d, x, x_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_1.run(buf1, buf2, 24576, 196, stream=stream0)
            del buf1
            buf3 = empty_strided_cuda((16, 6, 5, 5), (150, 1, 30, 6), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_2.run(arg3_1, buf3, 96, 25, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf4, (4096, 16, 10, 10), (1600, 1, 160, 16), 'unknown_op')
            del buf2
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(buf5, arg4_1, 6553600, stream=stream0)
            del arg4_1
            buf6 = empty_strided_cuda((4096, 16, 5, 5), (400, 25, 5, 1), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1, x_2, x_3], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.run(buf5, buf6, 102400, 16, stream=stream0)
            del buf5
            buf8 = empty_strided_cuda((4096, 120), (120, 1), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d, x, x_1, conv2d_1, x_2, x_3, x_4, linear, x_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5.run(buf6, arg5_1, arg6_1, buf8, 128, 1, 1, stream=stream0)
            del arg5_1
            del arg6_1
            del buf6
            buf9 = empty_strided_cuda((4096, 84), (84, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, x_5, linear_1], Original ATen: [aten.addmm, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_6.run(buf8, arg7_1, buf9, 128, 1, 1, stream=stream0)
            del arg7_1
            del buf8
            buf10 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_1, x_6], Original ATen: [aten.addmm, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_7.run(buf10, arg8_1, 344064, stream=stream0)
            del arg8_1
            buf11 = empty_strided_cuda((4096, 20), (20, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, x_6, x_7], Original ATen: [aten.addmm, aten.relu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_t_8.run(arg10_1, buf10, arg9_1, buf11, 128, 1, 1, stream=stream0)
            del arg10_1
            del arg9_1
            del buf10
        return (buf11, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((6, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4096, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 6, 5, 5), (150, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((120, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((84, 120), (120, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((20, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    return [torch.rand(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 784) % 6)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)

''',
    '''@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 14)
    x3 = xindex // 14
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + 2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + 2*x2 + 56*x3 + 784*y4), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 6*x5 + 1176*y1), tmp6, xmask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 6*x2 + 150*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6553600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)

''',
    '''@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 5)
    y4 = yindex // 5
    y2 = yindex // 25
    y5 = (yindex % 25)
    tmp0 = tl.load(in_ptr0 + (x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (16 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (160 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (176 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y5 + 25*x3 + 400*y2), tmp6, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_convolution_max_pool2d_with_indices_relu_t_view_5(arg_A, arg_B, in_ptr2, out_ptr1):
    EVEN_K : tl.constexpr = False
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

    M = 4096
    N = 120
    K = 400
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 400
    stride_ak = 1
    stride_bk = 1
    stride_bn = 400

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

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 400*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 120*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 400*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 120*idx_m
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = tmp0 + acc
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tl.maximum(tmp2, tmp1)
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 120*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_relu_t_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
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

    M = 4096
    N = 84
    K = 120
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 120
    stride_ak = 1
    stride_bk = 1
    stride_bn = 120

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

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 120*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 84*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 120*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 84*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 344064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 84)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_relu_t_8(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 20
    K = 84
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 84
    stride_ak = 1
    stride_bk = 1
    stride_bn = 84

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

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 84*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 20*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 84*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 20*idx_m
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
    arg0_1 = rand_strided((6, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4096, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 6, 5, 5), (150, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((120, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((84, 120), (120, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((84,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((20, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((20,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]

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
