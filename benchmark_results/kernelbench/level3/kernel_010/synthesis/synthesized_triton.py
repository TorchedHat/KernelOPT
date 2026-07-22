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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/du/cducrdlblbsf4gq6eunicd6n4g434tsltmrvdpkdnowcimnjc5dq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[10, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 32, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 12042240, 'x': 6021120}, 'kernel_num_gb': 0.01204224, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 30, 50176,


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
    num_gb = 0.01204224
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ya/cyaymclf32mw43t4kjgkvf7e4iizhs4xa3id3nix6frcgzwz2log.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 75264, 'x': 37632}, 'kernel_num_gb': 7.5264e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 192, 49,


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
    num_gb = 7.5264e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/3o/c3osxgfc22b35y5v4gsulwp2xd7hdcps3z6zy6emilznvhac3eu2.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => relu
# Graph fragment:
#   %convolution : Tensor "f32[10, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 96338944}, 'kernel_num_gb': 0.064226304, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((10, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 8028160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.064226304
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/vk/cvkusxnuydqlz62ewvd3twbk3zcbug2nacyi6b23rx4qsth3abom.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => relu
#   x_3 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[10, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = PlaceHolder[target=relu]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[10, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', '''
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 88309760}, 'kernel_num_gb': 0.0802816, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 3584) % 56)
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x5 = xindex // 3584
    x6 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x5), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x5), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x5), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x5), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x5), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x5), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x5), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x5), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x5), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, None)


def get_args():
    arg_0 = rand_strided((10, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0802816
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/3a/c3aar3x4ob4n4bfb34pwfwqglgcixshhjhq7tp3q36cpwv6pn7ty.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out => convolution_1
# Graph fragment:
#   %getitem : Tensor "f32[10, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=getitem]
#   %arg6_1 : Tensor "f32[64, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf5
triton_tem_fused_convolution_4 = async_compile.triton('triton_tem_fused_convolution_4', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_4', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.016072704, 'kernel_flop': 256901120, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 64
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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
        xindex = idx_n + 64*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 64*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016072704
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/wh/cwhxct7ujw2vhnejtg6qx23hp6xw3sl3g5fcchzktlj6a3b4dntk.py
# Topologically Sorted Source Nodes: [out, out_1, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out => convolution_1
#   out_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   out_2 => relu_1
# Graph fragment:
#   %buf5 : Tensor "f32[31360, 64][64, 1]cuda:0" = PlaceHolder[target=buf5]
#   %arg7_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convolution_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   return %relu_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 24085504}, 'kernel_num_gb': 0.016057344, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016057344
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/mj/cmjc34xd67o5sxyfrvvn6rhcrlllahwjgjg22gp32wa53pvkdqgo.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out => convolution_1
#   out_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   out_2 => relu_1
#   out_3 => convolution_2
# Graph fragment:
#   %arg11_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convolution_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %convolution_2 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', '''
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 294912, 'x': 147456}, 'kernel_num_gb': 0.000294912, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 4096, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000294912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/qi/cqi6bir5fk6huszokezf3377bjm2ebw4hykhhr6t4x6uiqpzlkas.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_4 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   out_5 => relu_2
#   out_6 => convolution_3
# Graph fragment:
#   %buf9 : Tensor "f32[10, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf9]
#   %arg16_1 : Tensor "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %unsqueeze_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %convolution_3 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf10
triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.040206336, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7(arg_A, arg_B, out_ptr0):
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

    M = 31360
    N = 256
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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
        xindex = idx_n + 64*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 490, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.040206336
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ph/cphik2kcdakp2opzawmvdytegvka4lgvhu6wgvqddw7cn4boxom5.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6, out_7, input_1, input_2, out_8, out_9, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_1 => convolution_4
#   input_2 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   out_10 => convolution_5
#   out_4 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   out_5 => relu_2
#   out_6 => convolution_3
#   out_7 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   out_8 => add_10
#   out_9 => relu_3
# Graph fragment:
#   %buf10 : Tensor "f32[31360, 256][256, 1]cuda:0" = PlaceHolder[target=buf10]
#   %arg17_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %buf11 : Tensor "f32[31360, 256][256, 1]cuda:0" = PlaceHolder[target=buf11]
#   %arg22_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %add_10 : Tensor "f32[10, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=add_10]
#   %unsqueeze_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
#   %convolution_3 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %convolution_4 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_9), kwargs = {})
#   %relu_3 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
#   %convolution_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_10,%buf13
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 192684032}, 'kernel_num_gb': 0.128458752, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


def get_args():
    arg_0 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((31360, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 8028160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.128458752
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/n2/cn27shj6ma6czagbltxg3rgyyaguu654f6ujgqwu3vdtuk5i7cbg.py
# Topologically Sorted Source Nodes: [out_9, out_10], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_10 => convolution_5
#   out_9 => relu_3
# Graph fragment:
#   %buf13 : Tensor "f32[10, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=buf13]
#   %arg26_1 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=arg26_1]
#   %relu_3 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
#   %convolution_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf14
triton_tem_fused_convolution_relu_9 = async_compile.triton('triton_tem_fused_convolution_relu_9', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_relu_9', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.040206336, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_relu_9(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 64
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 64*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_relu_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_relu_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.040206336
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/wl/cwlu4cq45ybnglltzrphrqezz75w6zlgst2vyo55wjsnfuw4uihc.py
# Topologically Sorted Source Nodes: [out_9, out_14, out_15, out_16, out_17, out_18, out_19], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_14 => add_13, add_14, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   out_15 => relu_5
#   out_16 => convolution_7
#   out_17 => add_15, add_16, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   out_18 => add_17
#   out_19 => relu_6
#   out_9 => relu_3
# Graph fragment:
#   %buf19 : Tensor "f32[31360, 256][256, 1]cuda:0" = PlaceHolder[target=buf19]
#   %arg37_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %add_10 : Tensor "f32[10, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=add_10]
#   %relu_3 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
#   %unsqueeze_48 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_13 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg33_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_13,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg34_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_14 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_5 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
#   %convolution_7 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_15 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_15,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_16 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_17 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %relu_3), kwargs = {})
#   %relu_6 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   return %relu_6
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 128454656}, 'kernel_num_gb': 0.096342016, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


def get_args():
    arg_0 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8028160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.096342016
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/lr/clrcwiig5baktcno73e2txe2vaz2xjxn7nlijfxkmxgns64n7ahb.py
# Topologically Sorted Source Nodes: [out_24, out_25, out_26, out_27, out_28, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_24 => add_20, add_21, mul_27, mul_28, mul_29, reciprocal_9, sqrt_9, sub_9, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, unsqueeze_77, unsqueeze_78, unsqueeze_79
#   out_25 => relu_8
#   out_26 => convolution_10
#   out_27 => add_22, add_23, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   out_28 => add_24
#   out_29 => relu_9
# Graph fragment:
#   %buf26 : Tensor "f32[31360, 256][256, 1]cuda:0" = PlaceHolder[target=buf26]
#   %arg52_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %arg53_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %arg54_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %arg55_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg55_1]
#   %relu_6 : Tensor "f32[10, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=relu_6]
#   %unsqueeze_72 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg47_1, -1), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_72, -1), kwargs = {})
#   %sub_9 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %add_20 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg48_1, 1e-05), kwargs = {})
#   %sqrt_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_20,), kwargs = {})
#   %reciprocal_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_9,), kwargs = {})
#   %mul_27 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 1), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_27, -1), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, -1), kwargs = {})
#   %mul_28 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg49_1, -1), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, -1), kwargs = {})
#   %mul_29 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg50_1, -1), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_78, -1), kwargs = {})
#   %add_21 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_8 : Tensor "f32[10, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
#   %convolution_10 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_80 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_22 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_22,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_23 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %add_24 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %relu_6), kwargs = {})
#   %relu_9 : Tensor "f32[10, 256, 56, 56][802816, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_24,), kwargs = {})
#   return %relu_9
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 128454656}, 'kernel_num_gb': 0.096342016, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 8028160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.096342016
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/zs/czsd4x4iy5e77v4jw7f7gdt6eammhxtic3qj7hg42wwa7eq4k4f4.py
# Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_30 => convolution_11
# Graph fragment:
#   %relu_9 : Tensor "f32[10, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = PlaceHolder[target=relu_9]
#   %arg56_1 : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=arg56_1]
#   %convolution_11 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf28
triton_tem_fused_convolution_12 = async_compile.triton('triton_tem_fused_convolution_12', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_12', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.048300032, 'kernel_flop': 2055208960, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_12(arg_A, arg_B, out_ptr0):
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

    M = 31360
    N = 128
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.048300032
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/dy/cdy2wwjq4eka7vgl5vqrjmh73fd5pmrq3d2t2yaml2dcawehqb3g.py
# Topologically Sorted Source Nodes: [out_30, out_31, out_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_30 => convolution_11
#   out_31 => add_25, add_26, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   out_32 => relu_10
# Graph fragment:
#   %buf28 : Tensor "f32[31360, 128][128, 1]cuda:0" = PlaceHolder[target=buf28]
#   %arg57_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %arg59_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %convolution_11 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_88 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_25 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_25,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_26 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_10 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   return %relu_10
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 48171008}, 'kernel_num_gb': 0.032114688, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((10, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4014080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.032114688
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/6d/c6d77wdfywhy7z5evw5zuh57kg7joq32zmzf63ec7yqg3iep2qt3.py
# Topologically Sorted Source Nodes: [out_30, out_31, out_32, out_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_30 => convolution_11
#   out_31 => add_25, add_26, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   out_32 => relu_10
#   out_33 => convolution_12
# Graph fragment:
#   %arg61_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg61_1]
#   %convolution_11 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_88 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_25 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_25,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_26 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_10 : Tensor "f32[10, 128, 56, 56][401408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   %convolution_12 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg61_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf30
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', '''
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 589824}, 'kernel_num_gb': 0.001179648, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 16384, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001179648
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/fl/cflh52hyefcbjq2dhhxv3xiy6ps7afcmvq2rn6w2pkynwicoyyhu.py
# Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_34 => add_27, add_28, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   out_35 => relu_11
#   out_36 => convolution_13
# Graph fragment:
#   %convolution_12 : Tensor "f32[10, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=convolution_12]
#   %arg62_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %arg63_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %unsqueeze_96 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_11 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
#   %convolution_13 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf32
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12044288}, 'kernel_num_gb': 0.008030208, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((10, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1003520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008030208
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/hb/chbofbwt2vyodwynm2on6nizby3idkikriq45swusettqo2pbp55.py
# Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_34 => add_27, add_28, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   out_35 => relu_11
#   out_36 => convolution_13
# Graph fragment:
#   %buf32 : Tensor "f32[10, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf32]
#   %arg66_1 : Tensor "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=arg66_1]
#   %unsqueeze_96 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_11 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
#   %convolution_13 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf33
triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.020332544, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 512
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
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 248, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.020332544
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/op/copjjcvabfoittoserz2yrml2c3ctakj6fw45wobaoja5syj63y2.py
# Topologically Sorted Source Nodes: [out_34, out_35, out_36, out_37, input_4, out_38, out_39, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_4 => add_31, add_32, mul_42, mul_43, mul_44, reciprocal_14, sqrt_14, sub_14, unsqueeze_112, unsqueeze_113, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119
#   out_34 => add_27, add_28, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   out_35 => relu_11
#   out_36 => convolution_13
#   out_37 => add_29, add_30, mul_39, mul_40, mul_41, reciprocal_13, sqrt_13, sub_13, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111
#   out_38 => add_33
#   out_39 => relu_12
#   out_40 => convolution_15
# Graph fragment:
#   %buf33 : Tensor "f32[7840, 512][512, 1]cuda:0" = PlaceHolder[target=buf33]
#   %arg67_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg67_1]
#   %arg68_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg68_1]
#   %arg69_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %arg70_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %convolution_14 : Tensor "f32[10, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=convolution_14]
#   %arg72_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %arg73_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %arg74_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %arg75_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %add_33 : Tensor "f32[10, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=add_33]
#   %unsqueeze_96 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_11 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
#   %convolution_13 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_104 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg67_1, -1), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_104, -1), kwargs = {})
#   %sub_13 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %add_29 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg68_1, 1e-05), kwargs = {})
#   %sqrt_13 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_29,), kwargs = {})
#   %reciprocal_13 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_13,), kwargs = {})
#   %mul_39 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 1), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_39, -1), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, -1), kwargs = {})
#   %mul_40 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg69_1, -1), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_108, -1), kwargs = {})
#   %mul_41 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg70_1, -1), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_110, -1), kwargs = {})
#   %add_30 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg72_1, -1), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, -1), kwargs = {})
#   %sub_14 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %add_31 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg73_1, 1e-05), kwargs = {})
#   %sqrt_14 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_31,), kwargs = {})
#   %reciprocal_14 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_14,), kwargs = {})
#   %mul_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, 1), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_42, -1), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, -1), kwargs = {})
#   %mul_43 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg74_1, -1), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, -1), kwargs = {})
#   %mul_44 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg75_1, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, -1), kwargs = {})
#   %add_32 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_33 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_32), kwargs = {})
#   %relu_12 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   %convolution_15 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_33,%buf36
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 96354304}, 'kernel_num_gb': 0.064241664, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


def get_args():
    arg_0 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 4014080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.064241664
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/er/cer532crdzx37tkkun7yf4l3hjwoiggyijsbr4nvocyvjjxbt7rc.py
# Topologically Sorted Source Nodes: [out_39, out_40], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_39 => relu_12
#   out_40 => convolution_15
# Graph fragment:
#   %buf36 : Tensor "f32[10, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=buf36]
#   %arg76_1 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=arg76_1]
#   %relu_12 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   %convolution_15 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf37
triton_tem_fused_convolution_relu_18 = async_compile.triton('triton_tem_fused_convolution_relu_18', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_relu_18', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.020332544, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_relu_18(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 128
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
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 123, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_relu_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_relu_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.020332544
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/mx/cmxk43arho7gexhcenehve2o6wuzunsywpcgqqtnkt6fa7ah7vv5.py
# Topologically Sorted Source Nodes: [out_39, out_44, out_45, out_46, out_47, out_48, out_49], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_39 => relu_12
#   out_44 => add_36, add_37, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135
#   out_45 => relu_14
#   out_46 => convolution_17
#   out_47 => add_38, add_39, mul_51, mul_52, mul_53, reciprocal_17, sqrt_17, sub_17, unsqueeze_136, unsqueeze_137, unsqueeze_138, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143
#   out_48 => add_40
#   out_49 => relu_15
# Graph fragment:
#   %buf42 : Tensor "f32[7840, 512][512, 1]cuda:0" = PlaceHolder[target=buf42]
#   %arg87_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg88_1]
#   %arg89_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg89_1]
#   %arg90_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %add_33 : Tensor "f32[10, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=add_33]
#   %relu_12 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   %unsqueeze_128 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, -1), kwargs = {})
#   %sub_16 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %add_36 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_36,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, -1), kwargs = {})
#   %mul_49 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_132, -1), kwargs = {})
#   %mul_50 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, -1), kwargs = {})
#   %add_37 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_14 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
#   %convolution_17 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_136 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg87_1, -1), kwargs = {})
#   %unsqueeze_137 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_136, -1), kwargs = {})
#   %sub_17 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %add_38 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg88_1, 1e-05), kwargs = {})
#   %sqrt_17 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %reciprocal_17 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_17,), kwargs = {})
#   %mul_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 1), kwargs = {})
#   %unsqueeze_138 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_51, -1), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_138, -1), kwargs = {})
#   %mul_52 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg89_1, -1), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, -1), kwargs = {})
#   %mul_53 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg90_1, -1), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, -1), kwargs = {})
#   %add_39 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_40 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %relu_12), kwargs = {})
#   %relu_15 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_40,), kwargs = {})
#   return %relu_15
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 64233472}, 'kernel_num_gb': 0.048177152, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


def get_args():
    arg_0 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4014080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.048177152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/dy/cdy4wx6fj6l2koej4dzpjwtygu5avw4lzjz6sa7j7vo3vzdfa6f3.py
# Topologically Sorted Source Nodes: [out_54, out_55, out_56, out_57, out_58, out_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_54 => add_43, add_44, mul_57, mul_58, mul_59, reciprocal_19, sqrt_19, sub_19, unsqueeze_152, unsqueeze_153, unsqueeze_154, unsqueeze_155, unsqueeze_156, unsqueeze_157, unsqueeze_158, unsqueeze_159
#   out_55 => relu_17
#   out_56 => convolution_20
#   out_57 => add_45, add_46, mul_60, mul_61, mul_62, reciprocal_20, sqrt_20, sub_20, unsqueeze_160, unsqueeze_161, unsqueeze_162, unsqueeze_163, unsqueeze_164, unsqueeze_165, unsqueeze_166, unsqueeze_167
#   out_58 => add_47
#   out_59 => relu_18
# Graph fragment:
#   %buf49 : Tensor "f32[7840, 512][512, 1]cuda:0" = PlaceHolder[target=buf49]
#   %arg102_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg102_1]
#   %arg103_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg103_1]
#   %arg104_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg104_1]
#   %arg105_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg105_1]
#   %relu_15 : Tensor "f32[10, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=relu_15]
#   %unsqueeze_152 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg97_1, -1), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, -1), kwargs = {})
#   %sub_19 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %add_43 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg98_1, 1e-05), kwargs = {})
#   %sqrt_19 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_43,), kwargs = {})
#   %reciprocal_19 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_19,), kwargs = {})
#   %mul_57 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_19, 1), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_57, -1), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, -1), kwargs = {})
#   %mul_58 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg99_1, -1), kwargs = {})
#   %unsqueeze_157 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_156, -1), kwargs = {})
#   %mul_59 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %unsqueeze_158 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg100_1, -1), kwargs = {})
#   %unsqueeze_159 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_158, -1), kwargs = {})
#   %add_44 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %relu_17 : Tensor "f32[10, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_44,), kwargs = {})
#   %convolution_20 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_160 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg102_1, -1), kwargs = {})
#   %unsqueeze_161 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_160, -1), kwargs = {})
#   %sub_20 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %add_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg103_1, 1e-05), kwargs = {})
#   %sqrt_20 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_45,), kwargs = {})
#   %reciprocal_20 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_20,), kwargs = {})
#   %mul_60 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_20, 1), kwargs = {})
#   %unsqueeze_162 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_60, -1), kwargs = {})
#   %unsqueeze_163 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_162, -1), kwargs = {})
#   %mul_61 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %unsqueeze_164 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg104_1, -1), kwargs = {})
#   %unsqueeze_165 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_164, -1), kwargs = {})
#   %mul_62 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %unsqueeze_166 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg105_1, -1), kwargs = {})
#   %unsqueeze_167 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_166, -1), kwargs = {})
#   %add_46 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_47 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %relu_15), kwargs = {})
#   %relu_18 : Tensor "f32[10, 512, 28, 28][401408, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   return %relu_18
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 64233472}, 'kernel_num_gb': 0.048177152, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4014080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.048177152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/5y/c5y73hm46x4egct3vjd3w7yzdf5oboc3ow3lftkaiy7ww33h6uwv.py
# Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_70 => convolution_24
# Graph fragment:
#   %relu_21 : Tensor "f32[10, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=relu_21]
#   %arg121_1 : Tensor "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=arg121_1]
#   %convolution_24 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf58
triton_tem_fused_convolution_21 = async_compile.triton('triton_tem_fused_convolution_21', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_21', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.024608768, 'kernel_flop': 2055208960, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_21(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 256
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
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 246, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_21.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_21.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024608768
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/3a/c3ancic7bxtzbpjfjxulqc44p6kfqdgbbkwoe3o5ppnul57gfu5f.py
# Topologically Sorted Source Nodes: [out_70, out_71, out_72], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_70 => convolution_24
#   out_71 => add_55, add_56, mul_72, mul_73, mul_74, reciprocal_24, sqrt_24, sub_24, unsqueeze_192, unsqueeze_193, unsqueeze_194, unsqueeze_195, unsqueeze_196, unsqueeze_197, unsqueeze_198, unsqueeze_199
#   out_72 => relu_22
# Graph fragment:
#   %buf58 : Tensor "f32[7840, 256][256, 1]cuda:0" = PlaceHolder[target=buf58]
#   %arg122_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg122_1]
#   %arg123_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %arg124_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg124_1]
#   %arg125_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %convolution_24 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_192 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg122_1, -1), kwargs = {})
#   %unsqueeze_193 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_192, -1), kwargs = {})
#   %sub_24 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %add_55 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, 1e-05), kwargs = {})
#   %sqrt_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_55,), kwargs = {})
#   %reciprocal_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_24,), kwargs = {})
#   %mul_72 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_24, 1), kwargs = {})
#   %unsqueeze_194 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_72, -1), kwargs = {})
#   %unsqueeze_195 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_194, -1), kwargs = {})
#   %mul_73 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %unsqueeze_196 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg124_1, -1), kwargs = {})
#   %unsqueeze_197 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_196, -1), kwargs = {})
#   %mul_74 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %unsqueeze_198 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg125_1, -1), kwargs = {})
#   %unsqueeze_199 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_198, -1), kwargs = {})
#   %add_56 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_22 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
#   return %relu_22
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22', '''
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 24088576}, 'kernel_num_gb': 0.016060416, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((10, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016060416
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/s7/cs7b6gohzih7t3qqcay2c34w67st7qjgi6swgcl6bc2iqp6m24d6.py
# Topologically Sorted Source Nodes: [out_70, out_71, out_72, out_73], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_70 => convolution_24
#   out_71 => add_55, add_56, mul_72, mul_73, mul_74, reciprocal_24, sqrt_24, sub_24, unsqueeze_192, unsqueeze_193, unsqueeze_194, unsqueeze_195, unsqueeze_196, unsqueeze_197, unsqueeze_198, unsqueeze_199
#   out_72 => relu_22
#   out_73 => convolution_25
# Graph fragment:
#   %arg126_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg126_1]
#   %convolution_24 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_192 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg122_1, -1), kwargs = {})
#   %unsqueeze_193 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_192, -1), kwargs = {})
#   %sub_24 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %add_55 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, 1e-05), kwargs = {})
#   %sqrt_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_55,), kwargs = {})
#   %reciprocal_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_24,), kwargs = {})
#   %mul_72 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_24, 1), kwargs = {})
#   %unsqueeze_194 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_72, -1), kwargs = {})
#   %unsqueeze_195 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_194, -1), kwargs = {})
#   %mul_73 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %unsqueeze_196 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg124_1, -1), kwargs = {})
#   %unsqueeze_197 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_196, -1), kwargs = {})
#   %mul_74 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %unsqueeze_198 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg125_1, -1), kwargs = {})
#   %unsqueeze_199 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_198, -1), kwargs = {})
#   %add_56 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_22 : Tensor "f32[10, 256, 28, 28][200704, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
#   %convolution_25 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg126_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf60
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23', '''
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}, 'kernel_num_gb': 0.004718592, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 65536, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004718592
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/l2/cl2dcy3xfr2lluwk6ts7lnyuuankvdjwlenust3vrdzwgw6zp4ku.py
# Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_74 => add_57, add_58, mul_75, mul_76, mul_77, reciprocal_25, sqrt_25, sub_25, unsqueeze_200, unsqueeze_201, unsqueeze_202, unsqueeze_203, unsqueeze_204, unsqueeze_205, unsqueeze_206, unsqueeze_207
#   out_75 => relu_23
#   out_76 => convolution_26
# Graph fragment:
#   %convolution_25 : Tensor "f32[10, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=convolution_25]
#   %arg127_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg127_1]
#   %arg128_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg128_1]
#   %arg129_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %arg130_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg130_1]
#   %unsqueeze_200 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg127_1, -1), kwargs = {})
#   %unsqueeze_201 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_200, -1), kwargs = {})
#   %sub_25 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %add_57 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_57,), kwargs = {})
#   %reciprocal_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_25,), kwargs = {})
#   %mul_75 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 1), kwargs = {})
#   %unsqueeze_202 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_75, -1), kwargs = {})
#   %unsqueeze_203 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_202, -1), kwargs = {})
#   %mul_76 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %unsqueeze_204 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, -1), kwargs = {})
#   %unsqueeze_205 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_204, -1), kwargs = {})
#   %mul_77 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %unsqueeze_206 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, -1), kwargs = {})
#   %unsqueeze_207 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_206, -1), kwargs = {})
#   %add_58 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
#   %convolution_26 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf62
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6025216}, 'kernel_num_gb': 0.004018176, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 501760,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004018176
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/fs/cfsho57emdrazb5msjymwmeazsa7yqz236oiqtyxpnnp42odxyds.py
# Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_74 => add_57, add_58, mul_75, mul_76, mul_77, reciprocal_25, sqrt_25, sub_25, unsqueeze_200, unsqueeze_201, unsqueeze_202, unsqueeze_203, unsqueeze_204, unsqueeze_205, unsqueeze_206, unsqueeze_207
#   out_75 => relu_23
#   out_76 => convolution_26
# Graph fragment:
#   %buf62 : Tensor "f32[10, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf62]
#   %arg131_1 : Tensor "f32[1024, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=arg131_1]
#   %unsqueeze_200 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg127_1, -1), kwargs = {})
#   %unsqueeze_201 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_200, -1), kwargs = {})
#   %sub_25 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %add_57 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_57,), kwargs = {})
#   %reciprocal_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_25,), kwargs = {})
#   %mul_75 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 1), kwargs = {})
#   %unsqueeze_202 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_75, -1), kwargs = {})
#   %unsqueeze_203 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_202, -1), kwargs = {})
#   %mul_76 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %unsqueeze_204 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, -1), kwargs = {})
#   %unsqueeze_205 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_204, -1), kwargs = {})
#   %mul_77 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %unsqueeze_206 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, -1), kwargs = {})
#   %unsqueeze_207 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_206, -1), kwargs = {})
#   %add_58 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
#   %convolution_26 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf63
triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.011083776, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 1024
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1024*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 256, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.011083776
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/g7/cg7osmj2rp4mv6v5pvnl3vkm2pciscgsymvf3fb3hqzbhpdidxgy.py
# Topologically Sorted Source Nodes: [out_74, out_75, out_76, out_77, input_6, out_78, out_79, out_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_6 => add_61, add_62, mul_81, mul_82, mul_83, reciprocal_27, sqrt_27, sub_27, unsqueeze_216, unsqueeze_217, unsqueeze_218, unsqueeze_219, unsqueeze_220, unsqueeze_221, unsqueeze_222, unsqueeze_223
#   out_74 => add_57, add_58, mul_75, mul_76, mul_77, reciprocal_25, sqrt_25, sub_25, unsqueeze_200, unsqueeze_201, unsqueeze_202, unsqueeze_203, unsqueeze_204, unsqueeze_205, unsqueeze_206, unsqueeze_207
#   out_75 => relu_23
#   out_76 => convolution_26
#   out_77 => add_59, add_60, mul_78, mul_79, mul_80, reciprocal_26, sqrt_26, sub_26, unsqueeze_208, unsqueeze_209, unsqueeze_210, unsqueeze_211, unsqueeze_212, unsqueeze_213, unsqueeze_214, unsqueeze_215
#   out_78 => add_63
#   out_79 => relu_24
#   out_80 => convolution_28
# Graph fragment:
#   %buf63 : Tensor "f32[1960, 1024][1024, 1]cuda:0" = PlaceHolder[target=buf63]
#   %arg132_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg132_1]
#   %arg133_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg133_1]
#   %arg134_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg134_1]
#   %arg135_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg135_1]
#   %convolution_27 : Tensor "f32[10, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=convolution_27]
#   %arg137_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg137_1]
#   %arg138_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg138_1]
#   %arg139_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg139_1]
#   %arg140_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg140_1]
#   %add_63 : Tensor "f32[10, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=add_63]
#   %unsqueeze_200 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg127_1, -1), kwargs = {})
#   %unsqueeze_201 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_200, -1), kwargs = {})
#   %sub_25 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %add_57 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_57,), kwargs = {})
#   %reciprocal_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_25,), kwargs = {})
#   %mul_75 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 1), kwargs = {})
#   %unsqueeze_202 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_75, -1), kwargs = {})
#   %unsqueeze_203 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_202, -1), kwargs = {})
#   %mul_76 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %unsqueeze_204 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, -1), kwargs = {})
#   %unsqueeze_205 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_204, -1), kwargs = {})
#   %mul_77 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %unsqueeze_206 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, -1), kwargs = {})
#   %unsqueeze_207 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_206, -1), kwargs = {})
#   %add_58 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
#   %convolution_26 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_208 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg132_1, -1), kwargs = {})
#   %unsqueeze_209 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_208, -1), kwargs = {})
#   %sub_26 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %add_59 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg133_1, 1e-05), kwargs = {})
#   %sqrt_26 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_59,), kwargs = {})
#   %reciprocal_26 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_26,), kwargs = {})
#   %mul_78 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_26, 1), kwargs = {})
#   %unsqueeze_210 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_78, -1), kwargs = {})
#   %unsqueeze_211 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_210, -1), kwargs = {})
#   %mul_79 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %unsqueeze_212 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg134_1, -1), kwargs = {})
#   %unsqueeze_213 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_212, -1), kwargs = {})
#   %mul_80 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %unsqueeze_214 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg135_1, -1), kwargs = {})
#   %unsqueeze_215 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_214, -1), kwargs = {})
#   %add_60 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %unsqueeze_216 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg137_1, -1), kwargs = {})
#   %unsqueeze_217 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_216, -1), kwargs = {})
#   %sub_27 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %add_61 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg138_1, 1e-05), kwargs = {})
#   %sqrt_27 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_61,), kwargs = {})
#   %reciprocal_27 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_27,), kwargs = {})
#   %mul_81 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_27, 1), kwargs = {})
#   %unsqueeze_218 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_81, -1), kwargs = {})
#   %unsqueeze_219 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_218, -1), kwargs = {})
#   %mul_82 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %unsqueeze_220 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg139_1, -1), kwargs = {})
#   %unsqueeze_221 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_220, -1), kwargs = {})
#   %mul_83 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %unsqueeze_222 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg140_1, -1), kwargs = {})
#   %unsqueeze_223 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_222, -1), kwargs = {})
#   %add_62 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %add_63 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %add_62), kwargs = {})
#   %relu_24 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
#   %convolution_28 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_63,%buf66
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26', '''
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 48201728}, 'kernel_num_gb': 0.032145408, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


def get_args():
    arg_0 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.032145408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/5s/c5sc7ggqvdlmxxldwxmb7ljtojvnc4ysgtunyy2bmi7bfyzah4kl.py
# Topologically Sorted Source Nodes: [out_79, out_80], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_79 => relu_24
#   out_80 => convolution_28
# Graph fragment:
#   %buf66 : Tensor "f32[10, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=buf66]
#   %arg141_1 : Tensor "f32[256, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = PlaceHolder[target=arg141_1]
#   %relu_24 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
#   %convolution_28 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf67
triton_tem_fused_convolution_relu_27 = async_compile.triton('triton_tem_fused_convolution_relu_27', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_relu_27', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.011083776, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_relu_27(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 256
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
        xindex = idx_n + 1024*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 124, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_relu_27.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_relu_27.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.011083776
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/fj/cfjdb67mzzbbwvxzlnbyvy4fpsnupbqalistuazbwntxrnpbiqz2.py
# Topologically Sorted Source Nodes: [out_79, out_84, out_85, out_86, out_87, out_88, out_89], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_79 => relu_24
#   out_84 => add_66, add_67, mul_87, mul_88, mul_89, reciprocal_29, sqrt_29, sub_29, unsqueeze_232, unsqueeze_233, unsqueeze_234, unsqueeze_235, unsqueeze_236, unsqueeze_237, unsqueeze_238, unsqueeze_239
#   out_85 => relu_26
#   out_86 => convolution_30
#   out_87 => add_68, add_69, mul_90, mul_91, mul_92, reciprocal_30, sqrt_30, sub_30, unsqueeze_240, unsqueeze_241, unsqueeze_242, unsqueeze_243, unsqueeze_244, unsqueeze_245, unsqueeze_246, unsqueeze_247
#   out_88 => add_70
#   out_89 => relu_27
# Graph fragment:
#   %buf72 : Tensor "f32[1960, 1024][1024, 1]cuda:0" = PlaceHolder[target=buf72]
#   %arg152_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg152_1]
#   %arg153_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %arg154_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg154_1]
#   %arg155_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg155_1]
#   %add_63 : Tensor "f32[10, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=add_63]
#   %relu_24 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
#   %unsqueeze_232 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg147_1, -1), kwargs = {})
#   %unsqueeze_233 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_232, -1), kwargs = {})
#   %sub_29 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %add_66 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg148_1, 1e-05), kwargs = {})
#   %sqrt_29 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_66,), kwargs = {})
#   %reciprocal_29 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_29,), kwargs = {})
#   %mul_87 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_29, 1), kwargs = {})
#   %unsqueeze_234 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_87, -1), kwargs = {})
#   %unsqueeze_235 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_234, -1), kwargs = {})
#   %mul_88 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %unsqueeze_236 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg149_1, -1), kwargs = {})
#   %unsqueeze_237 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_236, -1), kwargs = {})
#   %mul_89 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %unsqueeze_238 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, -1), kwargs = {})
#   %unsqueeze_239 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_238, -1), kwargs = {})
#   %add_67 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
#   %relu_26 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_67,), kwargs = {})
#   %convolution_30 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_240 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg152_1, -1), kwargs = {})
#   %unsqueeze_241 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_240, -1), kwargs = {})
#   %sub_30 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %add_68 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg153_1, 1e-05), kwargs = {})
#   %sqrt_30 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_68,), kwargs = {})
#   %reciprocal_30 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_30,), kwargs = {})
#   %mul_90 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_30, 1), kwargs = {})
#   %unsqueeze_242 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_90, -1), kwargs = {})
#   %unsqueeze_243 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_242, -1), kwargs = {})
#   %mul_91 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %unsqueeze_244 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg154_1, -1), kwargs = {})
#   %unsqueeze_245 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_244, -1), kwargs = {})
#   %mul_92 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %unsqueeze_246 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg155_1, -1), kwargs = {})
#   %unsqueeze_247 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_246, -1), kwargs = {})
#   %add_69 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %add_70 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %relu_24), kwargs = {})
#   %relu_27 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_70,), kwargs = {})
#   return %relu_27
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28', '''
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32129024}, 'kernel_num_gb': 0.024100864, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


def get_args():
    arg_0 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024100864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/at/cattazj6y3lomx2cs2plcgytym4jfww45owcoigeo4wvteka7rhi.py
# Topologically Sorted Source Nodes: [out_94, out_95, out_96, out_97, out_98, out_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_94 => add_73, add_74, mul_96, mul_97, mul_98, reciprocal_32, sqrt_32, sub_32, unsqueeze_256, unsqueeze_257, unsqueeze_258, unsqueeze_259, unsqueeze_260, unsqueeze_261, unsqueeze_262, unsqueeze_263
#   out_95 => relu_29
#   out_96 => convolution_33
#   out_97 => add_75, add_76, mul_100, mul_101, mul_99, reciprocal_33, sqrt_33, sub_33, unsqueeze_264, unsqueeze_265, unsqueeze_266, unsqueeze_267, unsqueeze_268, unsqueeze_269, unsqueeze_270, unsqueeze_271
#   out_98 => add_77
#   out_99 => relu_30
# Graph fragment:
#   %buf79 : Tensor "f32[1960, 1024][1024, 1]cuda:0" = PlaceHolder[target=buf79]
#   %arg167_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg167_1]
#   %arg168_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg168_1]
#   %arg169_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg169_1]
#   %arg170_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg170_1]
#   %relu_27 : Tensor "f32[10, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=relu_27]
#   %unsqueeze_256 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg162_1, -1), kwargs = {})
#   %unsqueeze_257 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_256, -1), kwargs = {})
#   %sub_32 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %add_73 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg163_1, 1e-05), kwargs = {})
#   %sqrt_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_73,), kwargs = {})
#   %reciprocal_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_32,), kwargs = {})
#   %mul_96 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_32, 1), kwargs = {})
#   %unsqueeze_258 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_96, -1), kwargs = {})
#   %unsqueeze_259 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_258, -1), kwargs = {})
#   %mul_97 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %unsqueeze_260 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg164_1, -1), kwargs = {})
#   %unsqueeze_261 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_260, -1), kwargs = {})
#   %mul_98 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %unsqueeze_262 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg165_1, -1), kwargs = {})
#   %unsqueeze_263 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_262, -1), kwargs = {})
#   %add_74 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
#   %relu_29 : Tensor "f32[10, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_74,), kwargs = {})
#   %convolution_33 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_264 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg167_1, -1), kwargs = {})
#   %unsqueeze_265 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_264, -1), kwargs = {})
#   %sub_33 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %add_75 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg168_1, 1e-05), kwargs = {})
#   %sqrt_33 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_75,), kwargs = {})
#   %reciprocal_33 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_33,), kwargs = {})
#   %mul_99 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_33, 1), kwargs = {})
#   %unsqueeze_266 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_99, -1), kwargs = {})
#   %unsqueeze_267 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_266, -1), kwargs = {})
#   %mul_100 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %unsqueeze_268 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg169_1, -1), kwargs = {})
#   %unsqueeze_269 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_268, -1), kwargs = {})
#   %mul_101 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %unsqueeze_270 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg170_1, -1), kwargs = {})
#   %unsqueeze_271 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_270, -1), kwargs = {})
#   %add_76 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %add_77 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %relu_27), kwargs = {})
#   %relu_30 : Tensor "f32[10, 1024, 14, 14][200704, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
#   return %relu_30
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29', '''
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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32129024}, 'kernel_num_gb': 0.024100864, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024100864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/cl/cclm6nd2eiugye7t6fya7xokxf6yxmb32wr5faenezkutrszy3vr.py
# Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_300 => convolution_94
# Graph fragment:
#   %relu_90 : Tensor "f32[10, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=relu_90]
#   %arg471_1 : Tensor "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = PlaceHolder[target=arg471_1]
#   %convolution_94 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_90, %arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf221
triton_tem_fused_convolution_30 = async_compile.triton('triton_tem_fused_convolution_30', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_30', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.014139392, 'kernel_flop': 2055208960, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_30(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 512
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
        xindex = idx_n + 1024*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 124, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_30.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_30.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.014139392
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pa/cpao6zp64qg36h4rjxspmcb6cfljo3zpndstan3cyyia4k4xupta.py
# Topologically Sorted Source Nodes: [out_300, out_301, out_302], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_300 => convolution_94
#   out_301 => add_218, add_219, mul_282, mul_283, mul_284, reciprocal_94, sqrt_94, sub_94, unsqueeze_752, unsqueeze_753, unsqueeze_754, unsqueeze_755, unsqueeze_756, unsqueeze_757, unsqueeze_758, unsqueeze_759
#   out_302 => relu_91
# Graph fragment:
#   %buf221 : Tensor "f32[1960, 512][512, 1]cuda:0" = PlaceHolder[target=buf221]
#   %arg472_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg472_1]
#   %arg473_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg473_1]
#   %arg474_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg474_1]
#   %arg475_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg475_1]
#   %convolution_94 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_90, %arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_752 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg472_1, -1), kwargs = {})
#   %unsqueeze_753 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_752, -1), kwargs = {})
#   %sub_94 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %add_218 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg473_1, 1e-05), kwargs = {})
#   %sqrt_94 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_218,), kwargs = {})
#   %reciprocal_94 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_94,), kwargs = {})
#   %mul_282 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_94, 1), kwargs = {})
#   %unsqueeze_754 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_282, -1), kwargs = {})
#   %unsqueeze_755 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_754, -1), kwargs = {})
#   %mul_283 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %unsqueeze_756 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg474_1, -1), kwargs = {})
#   %unsqueeze_757 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_756, -1), kwargs = {})
#   %mul_284 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %unsqueeze_758 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg475_1, -1), kwargs = {})
#   %unsqueeze_759 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_758, -1), kwargs = {})
#   %add_219 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %relu_91 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_219,), kwargs = {})
#   return %relu_91
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12050432}, 'kernel_num_gb': 0.008036352, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((10, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1003520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008036352
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ug/cug7k3vxvxnwwpf2zw6ybhmmltep4slnp5qdidqj72s6rmm7jipi.py
# Topologically Sorted Source Nodes: [out_300, out_301, out_302, out_303], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_300 => convolution_94
#   out_301 => add_218, add_219, mul_282, mul_283, mul_284, reciprocal_94, sqrt_94, sub_94, unsqueeze_752, unsqueeze_753, unsqueeze_754, unsqueeze_755, unsqueeze_756, unsqueeze_757, unsqueeze_758, unsqueeze_759
#   out_302 => relu_91
#   out_303 => convolution_95
# Graph fragment:
#   %arg476_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg476_1]
#   %convolution_94 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_90, %arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_752 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg472_1, -1), kwargs = {})
#   %unsqueeze_753 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_752, -1), kwargs = {})
#   %sub_94 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %add_218 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg473_1, 1e-05), kwargs = {})
#   %sqrt_94 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_218,), kwargs = {})
#   %reciprocal_94 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_94,), kwargs = {})
#   %mul_282 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_94, 1), kwargs = {})
#   %unsqueeze_754 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_282, -1), kwargs = {})
#   %unsqueeze_755 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_754, -1), kwargs = {})
#   %mul_283 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %unsqueeze_756 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg474_1, -1), kwargs = {})
#   %unsqueeze_757 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_756, -1), kwargs = {})
#   %mul_284 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %unsqueeze_758 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg475_1, -1), kwargs = {})
#   %unsqueeze_759 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_758, -1), kwargs = {})
#   %add_219 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %relu_91 : Tensor "f32[10, 512, 14, 14][100352, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_219,), kwargs = {})
#   %convolution_95 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_91, %arg476_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf223
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32', '''
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}, 'kernel_num_gb': 0.018874368, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 262144, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.018874368
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/he/chekxjfuh44ynbasj2ybvg3qspzwxjaje4w5tkm27nzrazjukbvg.py
# Topologically Sorted Source Nodes: [out_304, out_305, out_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_304 => add_220, add_221, mul_285, mul_286, mul_287, reciprocal_95, sqrt_95, sub_95, unsqueeze_760, unsqueeze_761, unsqueeze_762, unsqueeze_763, unsqueeze_764, unsqueeze_765, unsqueeze_766, unsqueeze_767
#   out_305 => relu_92
#   out_306 => convolution_96
# Graph fragment:
#   %convolution_95 : Tensor "f32[10, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = PlaceHolder[target=convolution_95]
#   %arg477_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg477_1]
#   %arg478_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg478_1]
#   %arg479_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg479_1]
#   %arg480_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg480_1]
#   %unsqueeze_760 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg477_1, -1), kwargs = {})
#   %unsqueeze_761 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_760, -1), kwargs = {})
#   %sub_95 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_761), kwargs = {})
#   %add_220 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg478_1, 1e-05), kwargs = {})
#   %sqrt_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_220,), kwargs = {})
#   %reciprocal_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_95,), kwargs = {})
#   %mul_285 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_95, 1), kwargs = {})
#   %unsqueeze_762 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_285, -1), kwargs = {})
#   %unsqueeze_763 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_762, -1), kwargs = {})
#   %mul_286 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %unsqueeze_764 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg479_1, -1), kwargs = {})
#   %unsqueeze_765 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_764, -1), kwargs = {})
#   %mul_287 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_765), kwargs = {})
#   %unsqueeze_766 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg480_1, -1), kwargs = {})
#   %unsqueeze_767 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_766, -1), kwargs = {})
#   %add_221 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_767), kwargs = {})
#   %relu_92 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
#   %convolution_96 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_92, %arg481_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf225
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3018752}, 'kernel_num_gb': 0.002015232, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 250880,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002015232
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/qh/cqh334mwgcly2lfh5cnrij2wjl7p5jviupyfxblb4osktr62sd46.py
# Topologically Sorted Source Nodes: [out_304, out_305, out_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_304 => add_220, add_221, mul_285, mul_286, mul_287, reciprocal_95, sqrt_95, sub_95, unsqueeze_760, unsqueeze_761, unsqueeze_762, unsqueeze_763, unsqueeze_764, unsqueeze_765, unsqueeze_766, unsqueeze_767
#   out_305 => relu_92
#   out_306 => convolution_96
# Graph fragment:
#   %buf225 : Tensor "f32[10, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = PlaceHolder[target=buf225]
#   %arg481_1 : Tensor "f32[2048, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=arg481_1]
#   %unsqueeze_760 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg477_1, -1), kwargs = {})
#   %unsqueeze_761 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_760, -1), kwargs = {})
#   %sub_95 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_761), kwargs = {})
#   %add_220 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg478_1, 1e-05), kwargs = {})
#   %sqrt_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_220,), kwargs = {})
#   %reciprocal_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_95,), kwargs = {})
#   %mul_285 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_95, 1), kwargs = {})
#   %unsqueeze_762 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_285, -1), kwargs = {})
#   %unsqueeze_763 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_762, -1), kwargs = {})
#   %mul_286 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %unsqueeze_764 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg479_1, -1), kwargs = {})
#   %unsqueeze_765 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_764, -1), kwargs = {})
#   %mul_287 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_765), kwargs = {})
#   %unsqueeze_766 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg480_1, -1), kwargs = {})
#   %unsqueeze_767 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_766, -1), kwargs = {})
#   %add_221 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_767), kwargs = {})
#   %relu_92 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
#   %convolution_96 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_92, %arg481_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf226
triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.009211904, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34(arg_A, arg_B, out_ptr0):
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

    M = 490
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
    arg_0 = rand_strided((10, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 128, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009211904
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/5d/c5d7a7ru6hyaifdqb5du6pwuuc6xqbgjwaevtfqrky5mjbb2qbnl.py
# Topologically Sorted Source Nodes: [out_304, out_305, out_306, out_307, input_8, out_308, out_309, out_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_8 => add_224, add_225, mul_291, mul_292, mul_293, reciprocal_97, sqrt_97, sub_97, unsqueeze_776, unsqueeze_777, unsqueeze_778, unsqueeze_779, unsqueeze_780, unsqueeze_781, unsqueeze_782, unsqueeze_783
#   out_304 => add_220, add_221, mul_285, mul_286, mul_287, reciprocal_95, sqrt_95, sub_95, unsqueeze_760, unsqueeze_761, unsqueeze_762, unsqueeze_763, unsqueeze_764, unsqueeze_765, unsqueeze_766, unsqueeze_767
#   out_305 => relu_92
#   out_306 => convolution_96
#   out_307 => add_222, add_223, mul_288, mul_289, mul_290, reciprocal_96, sqrt_96, sub_96, unsqueeze_768, unsqueeze_769, unsqueeze_770, unsqueeze_771, unsqueeze_772, unsqueeze_773, unsqueeze_774, unsqueeze_775
#   out_308 => add_226
#   out_309 => relu_93
#   out_310 => convolution_98
# Graph fragment:
#   %buf226 : Tensor "f32[490, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf226]
#   %arg482_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg482_1]
#   %arg483_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg483_1]
#   %arg484_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg484_1]
#   %arg485_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg485_1]
#   %convolution_97 : Tensor "f32[10, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=convolution_97]
#   %arg487_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg487_1]
#   %arg488_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg488_1]
#   %arg489_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg489_1]
#   %arg490_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg490_1]
#   %add_226 : Tensor "f32[10, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=add_226]
#   %unsqueeze_760 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg477_1, -1), kwargs = {})
#   %unsqueeze_761 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_760, -1), kwargs = {})
#   %sub_95 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_761), kwargs = {})
#   %add_220 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg478_1, 1e-05), kwargs = {})
#   %sqrt_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_220,), kwargs = {})
#   %reciprocal_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_95,), kwargs = {})
#   %mul_285 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_95, 1), kwargs = {})
#   %unsqueeze_762 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_285, -1), kwargs = {})
#   %unsqueeze_763 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_762, -1), kwargs = {})
#   %mul_286 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %unsqueeze_764 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg479_1, -1), kwargs = {})
#   %unsqueeze_765 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_764, -1), kwargs = {})
#   %mul_287 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_765), kwargs = {})
#   %unsqueeze_766 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg480_1, -1), kwargs = {})
#   %unsqueeze_767 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_766, -1), kwargs = {})
#   %add_221 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_767), kwargs = {})
#   %relu_92 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
#   %convolution_96 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_92, %arg481_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_768 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg482_1, -1), kwargs = {})
#   %unsqueeze_769 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_768, -1), kwargs = {})
#   %sub_96 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_769), kwargs = {})
#   %add_222 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg483_1, 1e-05), kwargs = {})
#   %sqrt_96 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_222,), kwargs = {})
#   %reciprocal_96 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_96,), kwargs = {})
#   %mul_288 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_96, 1), kwargs = {})
#   %unsqueeze_770 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_288, -1), kwargs = {})
#   %unsqueeze_771 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_770, -1), kwargs = {})
#   %mul_289 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %unsqueeze_772 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg484_1, -1), kwargs = {})
#   %unsqueeze_773 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_772, -1), kwargs = {})
#   %mul_290 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %unsqueeze_774 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg485_1, -1), kwargs = {})
#   %unsqueeze_775 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_774, -1), kwargs = {})
#   %add_223 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %unsqueeze_776 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg487_1, -1), kwargs = {})
#   %unsqueeze_777 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_776, -1), kwargs = {})
#   %sub_97 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_97, %unsqueeze_777), kwargs = {})
#   %add_224 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg488_1, 1e-05), kwargs = {})
#   %sqrt_97 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_224,), kwargs = {})
#   %reciprocal_97 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_97,), kwargs = {})
#   %mul_291 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_97, 1), kwargs = {})
#   %unsqueeze_778 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_291, -1), kwargs = {})
#   %unsqueeze_779 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_778, -1), kwargs = {})
#   %mul_292 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %unsqueeze_779), kwargs = {})
#   %unsqueeze_780 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg489_1, -1), kwargs = {})
#   %unsqueeze_781 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_780, -1), kwargs = {})
#   %mul_293 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_292, %unsqueeze_781), kwargs = {})
#   %unsqueeze_782 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg490_1, -1), kwargs = {})
#   %unsqueeze_783 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_782, -1), kwargs = {})
#   %add_225 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_293, %unsqueeze_783), kwargs = {})
#   %add_226 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, %add_225), kwargs = {})
#   %relu_93 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_226,), kwargs = {})
#   %convolution_98 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_93, %arg491_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_226,%buf229
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 24150016}, 'kernel_num_gb': 0.016121856, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


def get_args():
    arg_0 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 1003520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016121856
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/jj/cjjnurxcjwig6ibyhbuzhu4sqdza326yws5gpitmakbfv772ljd6.py
# Topologically Sorted Source Nodes: [out_309, out_310], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_309 => relu_93
#   out_310 => convolution_98
# Graph fragment:
#   %buf229 : Tensor "f32[10, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=buf229]
#   %arg491_1 : Tensor "f32[512, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = PlaceHolder[target=arg491_1]
#   %relu_93 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_226,), kwargs = {})
#   %convolution_98 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_93, %arg491_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf230
triton_tem_fused_convolution_relu_36 = async_compile.triton('triton_tem_fused_convolution_relu_36', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_relu_36', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.009211904, 'kernel_flop': 1027604480, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_relu_36(arg_A, arg_B, out_ptr0):
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

    M = 490
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
    arg_0 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 128, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_relu_36.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_relu_36.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009211904
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/eh/ceh5f3rjbk3tl36bltxenuwli64jbiqci34736jvzlgbs5x67225.py
# Topologically Sorted Source Nodes: [out_309, out_314, out_315, out_316, out_317, out_318, out_319], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out_309 => relu_93
#   out_314 => add_229, add_230, mul_297, mul_298, mul_299, reciprocal_99, sqrt_99, sub_99, unsqueeze_792, unsqueeze_793, unsqueeze_794, unsqueeze_795, unsqueeze_796, unsqueeze_797, unsqueeze_798, unsqueeze_799
#   out_315 => relu_95
#   out_316 => convolution_100
#   out_317 => add_231, add_232, mul_300, mul_301, mul_302, reciprocal_100, sqrt_100, sub_100, unsqueeze_800, unsqueeze_801, unsqueeze_802, unsqueeze_803, unsqueeze_804, unsqueeze_805, unsqueeze_806, unsqueeze_807
#   out_318 => add_233
#   out_319 => relu_96
# Graph fragment:
#   %buf235 : Tensor "f32[490, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf235]
#   %arg502_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg502_1]
#   %arg503_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg503_1]
#   %arg504_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg504_1]
#   %arg505_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg505_1]
#   %add_226 : Tensor "f32[10, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=add_226]
#   %relu_93 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_226,), kwargs = {})
#   %unsqueeze_792 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg497_1, -1), kwargs = {})
#   %unsqueeze_793 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_792, -1), kwargs = {})
#   %sub_99 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_793), kwargs = {})
#   %add_229 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg498_1, 1e-05), kwargs = {})
#   %sqrt_99 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_229,), kwargs = {})
#   %reciprocal_99 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_99,), kwargs = {})
#   %mul_297 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_99, 1), kwargs = {})
#   %unsqueeze_794 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_297, -1), kwargs = {})
#   %unsqueeze_795 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_794, -1), kwargs = {})
#   %mul_298 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %unsqueeze_795), kwargs = {})
#   %unsqueeze_796 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg499_1, -1), kwargs = {})
#   %unsqueeze_797 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_796, -1), kwargs = {})
#   %mul_299 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_298, %unsqueeze_797), kwargs = {})
#   %unsqueeze_798 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg500_1, -1), kwargs = {})
#   %unsqueeze_799 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_798, -1), kwargs = {})
#   %add_230 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_299, %unsqueeze_799), kwargs = {})
#   %relu_95 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_230,), kwargs = {})
#   %convolution_100 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_95, %arg501_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_800 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg502_1, -1), kwargs = {})
#   %unsqueeze_801 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_800, -1), kwargs = {})
#   %sub_100 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %add_231 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg503_1, 1e-05), kwargs = {})
#   %sqrt_100 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_231,), kwargs = {})
#   %reciprocal_100 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_100,), kwargs = {})
#   %mul_300 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_100, 1), kwargs = {})
#   %unsqueeze_802 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_300, -1), kwargs = {})
#   %unsqueeze_803 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_802, -1), kwargs = {})
#   %mul_301 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %unsqueeze_804 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg504_1, -1), kwargs = {})
#   %unsqueeze_805 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_804, -1), kwargs = {})
#   %mul_302 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_805), kwargs = {})
#   %unsqueeze_806 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg505_1, -1), kwargs = {})
#   %unsqueeze_807 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_806, -1), kwargs = {})
#   %add_232 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_807), kwargs = {})
#   %add_233 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %relu_93), kwargs = {})
#   %relu_96 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_233,), kwargs = {})
#   return %relu_96
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37', '''
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
    size_hints={'y': 32768, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16089088, 'x': 0}, 'kernel_num_gb': 0.012075008, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_out_ptr0 + (y0 + 2048*x2 + 100352*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (y0 + 2048*x2 + 100352*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1, 1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1, 1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + 2048*x2 + 100352*y1), tmp20, xmask)


def get_args():
    arg_0 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 20480, 49,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012075008
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/du/cduoj7fltnk3ujwqdr7zsvwdvws4yblo2yrnm7kjmwrbv2f323zn.py
# Topologically Sorted Source Nodes: [out_324, out_325, out_326, out_327, out_328, out_329, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add, aten.mean]
# Source node to ATen node mapping:
#   out_324 => add_236, add_237, mul_306, mul_307, mul_308, reciprocal_102, sqrt_102, sub_102, unsqueeze_816, unsqueeze_817, unsqueeze_818, unsqueeze_819, unsqueeze_820, unsqueeze_821, unsqueeze_822, unsqueeze_823
#   out_325 => relu_98
#   out_326 => convolution_103
#   out_327 => add_238, add_239, mul_309, mul_310, mul_311, reciprocal_103, sqrt_103, sub_103, unsqueeze_824, unsqueeze_825, unsqueeze_826, unsqueeze_827, unsqueeze_828, unsqueeze_829, unsqueeze_830, unsqueeze_831
#   out_328 => add_240
#   out_329 => relu_99
#   x_4 => mean
# Graph fragment:
#   %buf242 : Tensor "f32[490, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf242]
#   %arg517_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg517_1]
#   %arg518_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg518_1]
#   %arg519_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg519_1]
#   %arg520_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg520_1]
#   %relu_96 : Tensor "f32[10, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=relu_96]
#   %buf243 : Tensor "f32[10, 2048, 1, 1][2048, 1, 20480, 20480]cuda:0" = PlaceHolder[target=buf243]
#   %unsqueeze_816 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg512_1, -1), kwargs = {})
#   %unsqueeze_817 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_816, -1), kwargs = {})
#   %sub_102 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %add_236 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg513_1, 1e-05), kwargs = {})
#   %sqrt_102 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_236,), kwargs = {})
#   %reciprocal_102 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_102,), kwargs = {})
#   %mul_306 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_102, 1), kwargs = {})
#   %unsqueeze_818 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_306, -1), kwargs = {})
#   %unsqueeze_819 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_818, -1), kwargs = {})
#   %mul_307 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %unsqueeze_820 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg514_1, -1), kwargs = {})
#   %unsqueeze_821 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_820, -1), kwargs = {})
#   %mul_308 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %unsqueeze_822 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg515_1, -1), kwargs = {})
#   %unsqueeze_823 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_822, -1), kwargs = {})
#   %add_237 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
#   %relu_98 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
#   %convolution_103 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_98, %arg516_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_824 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg517_1, -1), kwargs = {})
#   %unsqueeze_825 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_824, -1), kwargs = {})
#   %sub_103 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %add_238 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg518_1, 1e-05), kwargs = {})
#   %sqrt_103 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_238,), kwargs = {})
#   %reciprocal_103 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_103,), kwargs = {})
#   %mul_309 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_103, 1), kwargs = {})
#   %unsqueeze_826 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_309, -1), kwargs = {})
#   %unsqueeze_827 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_826, -1), kwargs = {})
#   %mul_310 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %unsqueeze_828 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg519_1, -1), kwargs = {})
#   %unsqueeze_829 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_828, -1), kwargs = {})
#   %mul_311 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_829), kwargs = {})
#   %unsqueeze_830 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg520_1, -1), kwargs = {})
#   %unsqueeze_831 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_830, -1), kwargs = {})
#   %add_239 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_831), kwargs = {})
#   %add_240 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_239, %relu_96), kwargs = {})
#   %relu_99 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_240,), kwargs = {})
#   %mean : Tensor "f32[10, 2048, 1, 1][2048, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_99, [-1, -2], True), kwargs = {})
#   return %buf243,%mean
triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38', '''
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
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8224768, 'r0_': 0}, 'kernel_num_gb': 0.008224768, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 20480
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 100352*x1), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + 2048*r0_2 + 100352*x1), r0_mask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1, 1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = tl.full([1, 1], 49.0, tl.float32)
    tmp25 = (tmp23 / tmp24)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, None)


def get_args():
    arg_0 = rand_strided((10, 2048, 1, 1), (2048, 1, 20480, 20480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((490, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 20480, 49,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008224768
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/x6/cx6alq6wajbuu2bsgju36mqtftuqiojr5pmj2a44sy5i3qbbiidf.py
# Topologically Sorted Source Nodes: [out_324, out_325, out_326, out_327, out_328, out_329, x_4, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add, aten.mean, aten.view, aten.addmm, aten.t]
# Source node to ATen node mapping:
#   out_324 => add_236, add_237, mul_306, mul_307, mul_308, reciprocal_102, sqrt_102, sub_102, unsqueeze_816, unsqueeze_817, unsqueeze_818, unsqueeze_819, unsqueeze_820, unsqueeze_821, unsqueeze_822, unsqueeze_823
#   out_325 => relu_98
#   out_326 => convolution_103
#   out_327 => add_238, add_239, mul_309, mul_310, mul_311, reciprocal_103, sqrt_103, sub_103, unsqueeze_824, unsqueeze_825, unsqueeze_826, unsqueeze_827, unsqueeze_828, unsqueeze_829, unsqueeze_830, unsqueeze_831
#   out_328 => add_240
#   out_329 => relu_99
#   x_4 => mean
#   x_5 => view
#   x_6 => addmm_default, constant_pad_nd_default, permute
# Graph fragment:
#   %mean : Tensor "f32[10, 2048, 1, 1][2048, 1, 20480, 20480]cuda:0" = PlaceHolder[target=mean]
#   %arg522_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg522_1]
#   %constant_pad_nd_default : Tensor "f32[12, 2048][2048, 1]cuda:0" = PlaceHolder[target=constant_pad_nd_default]
#   %arg521_1 : Tensor "f32[1000, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg521_1]
#   %unsqueeze_816 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg512_1, -1), kwargs = {})
#   %unsqueeze_817 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_816, -1), kwargs = {})
#   %sub_102 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %add_236 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg513_1, 1e-05), kwargs = {})
#   %sqrt_102 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_236,), kwargs = {})
#   %reciprocal_102 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_102,), kwargs = {})
#   %mul_306 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_102, 1), kwargs = {})
#   %unsqueeze_818 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_306, -1), kwargs = {})
#   %unsqueeze_819 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_818, -1), kwargs = {})
#   %mul_307 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %unsqueeze_820 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg514_1, -1), kwargs = {})
#   %unsqueeze_821 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_820, -1), kwargs = {})
#   %mul_308 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %unsqueeze_822 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg515_1, -1), kwargs = {})
#   %unsqueeze_823 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_822, -1), kwargs = {})
#   %add_237 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
#   %relu_98 : Tensor "f32[10, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
#   %convolution_103 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_98, %arg516_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_824 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg517_1, -1), kwargs = {})
#   %unsqueeze_825 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_824, -1), kwargs = {})
#   %sub_103 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %add_238 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg518_1, 1e-05), kwargs = {})
#   %sqrt_103 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_238,), kwargs = {})
#   %reciprocal_103 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_103,), kwargs = {})
#   %mul_309 : Tensor "f32[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_103, 1), kwargs = {})
#   %unsqueeze_826 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_309, -1), kwargs = {})
#   %unsqueeze_827 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_826, -1), kwargs = {})
#   %mul_310 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %unsqueeze_828 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg519_1, -1), kwargs = {})
#   %unsqueeze_829 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_828, -1), kwargs = {})
#   %mul_311 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_829), kwargs = {})
#   %unsqueeze_830 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg520_1, -1), kwargs = {})
#   %unsqueeze_831 : Tensor "f32[2048, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_830, -1), kwargs = {})
#   %add_239 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_831), kwargs = {})
#   %add_240 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_239, %relu_96), kwargs = {})
#   %relu_99 : Tensor "f32[10, 2048, 7, 7][100352, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_240,), kwargs = {})
#   %mean : Tensor "f32[10, 2048, 1, 1][2048, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_99, [-1, -2], True), kwargs = {})
#   %view : Tensor "f32[10, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [10, 2048]), kwargs = {})
#   %constant_pad_nd_default : Tensor "f32[12, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view, [0, 0, 0, 2]), kwargs = {})
#   %permute : Tensor "f32[2048, 1000][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg521_1, [1, 0]), kwargs = {})
#   %addmm_default : Tensor "f32[12, 1000][1000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg522_1, %constant_pad_nd_default, %permute), kwargs = {})
#   return %constant_pad_nd_default,%addmm_default
triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39', '''
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
triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_B': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.008386304, 'kernel_flop': 49152000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39(in_ptr0, arg_B, in_ptr2, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    B = arg_B

    M = 12
    N = 1000
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
        tmp2 = tl.broadcast_to(idx_m, [BLOCK_M, BLOCK_K])
        tmp3 = tl.full([1], 10, tl.int64)
        tmp4 = tmp2 < tmp3
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 2048*idx_m, [BLOCK_M, BLOCK_K])), tmp4, eviction_policy='evict_last', other=0.0)
        a = tmp5.broadcast_to(xindex.shape)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 2048, 1, 1), (2048, 1, 20480, 20480), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((12, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 32, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008386304
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
        assert_size_stride(arg1_1, (10, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg2_1, (64, ), (1, ))
        assert_size_stride(arg3_1, (64, ), (1, ))
        assert_size_stride(arg4_1, (64, ), (1, ))
        assert_size_stride(arg5_1, (64, ), (1, ))
        assert_size_stride(arg6_1, (64, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg7_1, (64, ), (1, ))
        assert_size_stride(arg8_1, (64, ), (1, ))
        assert_size_stride(arg9_1, (64, ), (1, ))
        assert_size_stride(arg10_1, (64, ), (1, ))
        assert_size_stride(arg11_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg12_1, (64, ), (1, ))
        assert_size_stride(arg13_1, (64, ), (1, ))
        assert_size_stride(arg14_1, (64, ), (1, ))
        assert_size_stride(arg15_1, (64, ), (1, ))
        assert_size_stride(arg16_1, (256, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg17_1, (256, ), (1, ))
        assert_size_stride(arg18_1, (256, ), (1, ))
        assert_size_stride(arg19_1, (256, ), (1, ))
        assert_size_stride(arg20_1, (256, ), (1, ))
        assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg22_1, (256, ), (1, ))
        assert_size_stride(arg23_1, (256, ), (1, ))
        assert_size_stride(arg24_1, (256, ), (1, ))
        assert_size_stride(arg25_1, (256, ), (1, ))
        assert_size_stride(arg26_1, (64, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg27_1, (64, ), (1, ))
        assert_size_stride(arg28_1, (64, ), (1, ))
        assert_size_stride(arg29_1, (64, ), (1, ))
        assert_size_stride(arg30_1, (64, ), (1, ))
        assert_size_stride(arg31_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg32_1, (64, ), (1, ))
        assert_size_stride(arg33_1, (64, ), (1, ))
        assert_size_stride(arg34_1, (64, ), (1, ))
        assert_size_stride(arg35_1, (64, ), (1, ))
        assert_size_stride(arg36_1, (256, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg37_1, (256, ), (1, ))
        assert_size_stride(arg38_1, (256, ), (1, ))
        assert_size_stride(arg39_1, (256, ), (1, ))
        assert_size_stride(arg40_1, (256, ), (1, ))
        assert_size_stride(arg41_1, (64, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg42_1, (64, ), (1, ))
        assert_size_stride(arg43_1, (64, ), (1, ))
        assert_size_stride(arg44_1, (64, ), (1, ))
        assert_size_stride(arg45_1, (64, ), (1, ))
        assert_size_stride(arg46_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg47_1, (64, ), (1, ))
        assert_size_stride(arg48_1, (64, ), (1, ))
        assert_size_stride(arg49_1, (64, ), (1, ))
        assert_size_stride(arg50_1, (64, ), (1, ))
        assert_size_stride(arg51_1, (256, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg52_1, (256, ), (1, ))
        assert_size_stride(arg53_1, (256, ), (1, ))
        assert_size_stride(arg54_1, (256, ), (1, ))
        assert_size_stride(arg55_1, (256, ), (1, ))
        assert_size_stride(arg56_1, (128, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg57_1, (128, ), (1, ))
        assert_size_stride(arg58_1, (128, ), (1, ))
        assert_size_stride(arg59_1, (128, ), (1, ))
        assert_size_stride(arg60_1, (128, ), (1, ))
        assert_size_stride(arg61_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg62_1, (128, ), (1, ))
        assert_size_stride(arg63_1, (128, ), (1, ))
        assert_size_stride(arg64_1, (128, ), (1, ))
        assert_size_stride(arg65_1, (128, ), (1, ))
        assert_size_stride(arg66_1, (512, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg67_1, (512, ), (1, ))
        assert_size_stride(arg68_1, (512, ), (1, ))
        assert_size_stride(arg69_1, (512, ), (1, ))
        assert_size_stride(arg70_1, (512, ), (1, ))
        assert_size_stride(arg71_1, (512, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg72_1, (512, ), (1, ))
        assert_size_stride(arg73_1, (512, ), (1, ))
        assert_size_stride(arg74_1, (512, ), (1, ))
        assert_size_stride(arg75_1, (512, ), (1, ))
        assert_size_stride(arg76_1, (128, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg77_1, (128, ), (1, ))
        assert_size_stride(arg78_1, (128, ), (1, ))
        assert_size_stride(arg79_1, (128, ), (1, ))
        assert_size_stride(arg80_1, (128, ), (1, ))
        assert_size_stride(arg81_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg82_1, (128, ), (1, ))
        assert_size_stride(arg83_1, (128, ), (1, ))
        assert_size_stride(arg84_1, (128, ), (1, ))
        assert_size_stride(arg85_1, (128, ), (1, ))
        assert_size_stride(arg86_1, (512, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg87_1, (512, ), (1, ))
        assert_size_stride(arg88_1, (512, ), (1, ))
        assert_size_stride(arg89_1, (512, ), (1, ))
        assert_size_stride(arg90_1, (512, ), (1, ))
        assert_size_stride(arg91_1, (128, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg92_1, (128, ), (1, ))
        assert_size_stride(arg93_1, (128, ), (1, ))
        assert_size_stride(arg94_1, (128, ), (1, ))
        assert_size_stride(arg95_1, (128, ), (1, ))
        assert_size_stride(arg96_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg97_1, (128, ), (1, ))
        assert_size_stride(arg98_1, (128, ), (1, ))
        assert_size_stride(arg99_1, (128, ), (1, ))
        assert_size_stride(arg100_1, (128, ), (1, ))
        assert_size_stride(arg101_1, (512, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg102_1, (512, ), (1, ))
        assert_size_stride(arg103_1, (512, ), (1, ))
        assert_size_stride(arg104_1, (512, ), (1, ))
        assert_size_stride(arg105_1, (512, ), (1, ))
        assert_size_stride(arg106_1, (128, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg107_1, (128, ), (1, ))
        assert_size_stride(arg108_1, (128, ), (1, ))
        assert_size_stride(arg109_1, (128, ), (1, ))
        assert_size_stride(arg110_1, (128, ), (1, ))
        assert_size_stride(arg111_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg112_1, (128, ), (1, ))
        assert_size_stride(arg113_1, (128, ), (1, ))
        assert_size_stride(arg114_1, (128, ), (1, ))
        assert_size_stride(arg115_1, (128, ), (1, ))
        assert_size_stride(arg116_1, (512, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(arg117_1, (512, ), (1, ))
        assert_size_stride(arg118_1, (512, ), (1, ))
        assert_size_stride(arg119_1, (512, ), (1, ))
        assert_size_stride(arg120_1, (512, ), (1, ))
        assert_size_stride(arg121_1, (256, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg122_1, (256, ), (1, ))
        assert_size_stride(arg123_1, (256, ), (1, ))
        assert_size_stride(arg124_1, (256, ), (1, ))
        assert_size_stride(arg125_1, (256, ), (1, ))
        assert_size_stride(arg126_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg127_1, (256, ), (1, ))
        assert_size_stride(arg128_1, (256, ), (1, ))
        assert_size_stride(arg129_1, (256, ), (1, ))
        assert_size_stride(arg130_1, (256, ), (1, ))
        assert_size_stride(arg131_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg132_1, (1024, ), (1, ))
        assert_size_stride(arg133_1, (1024, ), (1, ))
        assert_size_stride(arg134_1, (1024, ), (1, ))
        assert_size_stride(arg135_1, (1024, ), (1, ))
        assert_size_stride(arg136_1, (1024, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg137_1, (1024, ), (1, ))
        assert_size_stride(arg138_1, (1024, ), (1, ))
        assert_size_stride(arg139_1, (1024, ), (1, ))
        assert_size_stride(arg140_1, (1024, ), (1, ))
        assert_size_stride(arg141_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg142_1, (256, ), (1, ))
        assert_size_stride(arg143_1, (256, ), (1, ))
        assert_size_stride(arg144_1, (256, ), (1, ))
        assert_size_stride(arg145_1, (256, ), (1, ))
        assert_size_stride(arg146_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg147_1, (256, ), (1, ))
        assert_size_stride(arg148_1, (256, ), (1, ))
        assert_size_stride(arg149_1, (256, ), (1, ))
        assert_size_stride(arg150_1, (256, ), (1, ))
        assert_size_stride(arg151_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg152_1, (1024, ), (1, ))
        assert_size_stride(arg153_1, (1024, ), (1, ))
        assert_size_stride(arg154_1, (1024, ), (1, ))
        assert_size_stride(arg155_1, (1024, ), (1, ))
        assert_size_stride(arg156_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg157_1, (256, ), (1, ))
        assert_size_stride(arg158_1, (256, ), (1, ))
        assert_size_stride(arg159_1, (256, ), (1, ))
        assert_size_stride(arg160_1, (256, ), (1, ))
        assert_size_stride(arg161_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg162_1, (256, ), (1, ))
        assert_size_stride(arg163_1, (256, ), (1, ))
        assert_size_stride(arg164_1, (256, ), (1, ))
        assert_size_stride(arg165_1, (256, ), (1, ))
        assert_size_stride(arg166_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg167_1, (1024, ), (1, ))
        assert_size_stride(arg168_1, (1024, ), (1, ))
        assert_size_stride(arg169_1, (1024, ), (1, ))
        assert_size_stride(arg170_1, (1024, ), (1, ))
        assert_size_stride(arg171_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg172_1, (256, ), (1, ))
        assert_size_stride(arg173_1, (256, ), (1, ))
        assert_size_stride(arg174_1, (256, ), (1, ))
        assert_size_stride(arg175_1, (256, ), (1, ))
        assert_size_stride(arg176_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg177_1, (256, ), (1, ))
        assert_size_stride(arg178_1, (256, ), (1, ))
        assert_size_stride(arg179_1, (256, ), (1, ))
        assert_size_stride(arg180_1, (256, ), (1, ))
        assert_size_stride(arg181_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg182_1, (1024, ), (1, ))
        assert_size_stride(arg183_1, (1024, ), (1, ))
        assert_size_stride(arg184_1, (1024, ), (1, ))
        assert_size_stride(arg185_1, (1024, ), (1, ))
        assert_size_stride(arg186_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg187_1, (256, ), (1, ))
        assert_size_stride(arg188_1, (256, ), (1, ))
        assert_size_stride(arg189_1, (256, ), (1, ))
        assert_size_stride(arg190_1, (256, ), (1, ))
        assert_size_stride(arg191_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg192_1, (256, ), (1, ))
        assert_size_stride(arg193_1, (256, ), (1, ))
        assert_size_stride(arg194_1, (256, ), (1, ))
        assert_size_stride(arg195_1, (256, ), (1, ))
        assert_size_stride(arg196_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg197_1, (1024, ), (1, ))
        assert_size_stride(arg198_1, (1024, ), (1, ))
        assert_size_stride(arg199_1, (1024, ), (1, ))
        assert_size_stride(arg200_1, (1024, ), (1, ))
        assert_size_stride(arg201_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg202_1, (256, ), (1, ))
        assert_size_stride(arg203_1, (256, ), (1, ))
        assert_size_stride(arg204_1, (256, ), (1, ))
        assert_size_stride(arg205_1, (256, ), (1, ))
        assert_size_stride(arg206_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg207_1, (256, ), (1, ))
        assert_size_stride(arg208_1, (256, ), (1, ))
        assert_size_stride(arg209_1, (256, ), (1, ))
        assert_size_stride(arg210_1, (256, ), (1, ))
        assert_size_stride(arg211_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg212_1, (1024, ), (1, ))
        assert_size_stride(arg213_1, (1024, ), (1, ))
        assert_size_stride(arg214_1, (1024, ), (1, ))
        assert_size_stride(arg215_1, (1024, ), (1, ))
        assert_size_stride(arg216_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg217_1, (256, ), (1, ))
        assert_size_stride(arg218_1, (256, ), (1, ))
        assert_size_stride(arg219_1, (256, ), (1, ))
        assert_size_stride(arg220_1, (256, ), (1, ))
        assert_size_stride(arg221_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg222_1, (256, ), (1, ))
        assert_size_stride(arg223_1, (256, ), (1, ))
        assert_size_stride(arg224_1, (256, ), (1, ))
        assert_size_stride(arg225_1, (256, ), (1, ))
        assert_size_stride(arg226_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg227_1, (1024, ), (1, ))
        assert_size_stride(arg228_1, (1024, ), (1, ))
        assert_size_stride(arg229_1, (1024, ), (1, ))
        assert_size_stride(arg230_1, (1024, ), (1, ))
        assert_size_stride(arg231_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg232_1, (256, ), (1, ))
        assert_size_stride(arg233_1, (256, ), (1, ))
        assert_size_stride(arg234_1, (256, ), (1, ))
        assert_size_stride(arg235_1, (256, ), (1, ))
        assert_size_stride(arg236_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg237_1, (256, ), (1, ))
        assert_size_stride(arg238_1, (256, ), (1, ))
        assert_size_stride(arg239_1, (256, ), (1, ))
        assert_size_stride(arg240_1, (256, ), (1, ))
        assert_size_stride(arg241_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg242_1, (1024, ), (1, ))
        assert_size_stride(arg243_1, (1024, ), (1, ))
        assert_size_stride(arg244_1, (1024, ), (1, ))
        assert_size_stride(arg245_1, (1024, ), (1, ))
        assert_size_stride(arg246_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg247_1, (256, ), (1, ))
        assert_size_stride(arg248_1, (256, ), (1, ))
        assert_size_stride(arg249_1, (256, ), (1, ))
        assert_size_stride(arg250_1, (256, ), (1, ))
        assert_size_stride(arg251_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg252_1, (256, ), (1, ))
        assert_size_stride(arg253_1, (256, ), (1, ))
        assert_size_stride(arg254_1, (256, ), (1, ))
        assert_size_stride(arg255_1, (256, ), (1, ))
        assert_size_stride(arg256_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg257_1, (1024, ), (1, ))
        assert_size_stride(arg258_1, (1024, ), (1, ))
        assert_size_stride(arg259_1, (1024, ), (1, ))
        assert_size_stride(arg260_1, (1024, ), (1, ))
        assert_size_stride(arg261_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg262_1, (256, ), (1, ))
        assert_size_stride(arg263_1, (256, ), (1, ))
        assert_size_stride(arg264_1, (256, ), (1, ))
        assert_size_stride(arg265_1, (256, ), (1, ))
        assert_size_stride(arg266_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg267_1, (256, ), (1, ))
        assert_size_stride(arg268_1, (256, ), (1, ))
        assert_size_stride(arg269_1, (256, ), (1, ))
        assert_size_stride(arg270_1, (256, ), (1, ))
        assert_size_stride(arg271_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg272_1, (1024, ), (1, ))
        assert_size_stride(arg273_1, (1024, ), (1, ))
        assert_size_stride(arg274_1, (1024, ), (1, ))
        assert_size_stride(arg275_1, (1024, ), (1, ))
        assert_size_stride(arg276_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg277_1, (256, ), (1, ))
        assert_size_stride(arg278_1, (256, ), (1, ))
        assert_size_stride(arg279_1, (256, ), (1, ))
        assert_size_stride(arg280_1, (256, ), (1, ))
        assert_size_stride(arg281_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg282_1, (256, ), (1, ))
        assert_size_stride(arg283_1, (256, ), (1, ))
        assert_size_stride(arg284_1, (256, ), (1, ))
        assert_size_stride(arg285_1, (256, ), (1, ))
        assert_size_stride(arg286_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg287_1, (1024, ), (1, ))
        assert_size_stride(arg288_1, (1024, ), (1, ))
        assert_size_stride(arg289_1, (1024, ), (1, ))
        assert_size_stride(arg290_1, (1024, ), (1, ))
        assert_size_stride(arg291_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg292_1, (256, ), (1, ))
        assert_size_stride(arg293_1, (256, ), (1, ))
        assert_size_stride(arg294_1, (256, ), (1, ))
        assert_size_stride(arg295_1, (256, ), (1, ))
        assert_size_stride(arg296_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg297_1, (256, ), (1, ))
        assert_size_stride(arg298_1, (256, ), (1, ))
        assert_size_stride(arg299_1, (256, ), (1, ))
        assert_size_stride(arg300_1, (256, ), (1, ))
        assert_size_stride(arg301_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg302_1, (1024, ), (1, ))
        assert_size_stride(arg303_1, (1024, ), (1, ))
        assert_size_stride(arg304_1, (1024, ), (1, ))
        assert_size_stride(arg305_1, (1024, ), (1, ))
        assert_size_stride(arg306_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg307_1, (256, ), (1, ))
        assert_size_stride(arg308_1, (256, ), (1, ))
        assert_size_stride(arg309_1, (256, ), (1, ))
        assert_size_stride(arg310_1, (256, ), (1, ))
        assert_size_stride(arg311_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg312_1, (256, ), (1, ))
        assert_size_stride(arg313_1, (256, ), (1, ))
        assert_size_stride(arg314_1, (256, ), (1, ))
        assert_size_stride(arg315_1, (256, ), (1, ))
        assert_size_stride(arg316_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg317_1, (1024, ), (1, ))
        assert_size_stride(arg318_1, (1024, ), (1, ))
        assert_size_stride(arg319_1, (1024, ), (1, ))
        assert_size_stride(arg320_1, (1024, ), (1, ))
        assert_size_stride(arg321_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg322_1, (256, ), (1, ))
        assert_size_stride(arg323_1, (256, ), (1, ))
        assert_size_stride(arg324_1, (256, ), (1, ))
        assert_size_stride(arg325_1, (256, ), (1, ))
        assert_size_stride(arg326_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg327_1, (256, ), (1, ))
        assert_size_stride(arg328_1, (256, ), (1, ))
        assert_size_stride(arg329_1, (256, ), (1, ))
        assert_size_stride(arg330_1, (256, ), (1, ))
        assert_size_stride(arg331_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg332_1, (1024, ), (1, ))
        assert_size_stride(arg333_1, (1024, ), (1, ))
        assert_size_stride(arg334_1, (1024, ), (1, ))
        assert_size_stride(arg335_1, (1024, ), (1, ))
        assert_size_stride(arg336_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg337_1, (256, ), (1, ))
        assert_size_stride(arg338_1, (256, ), (1, ))
        assert_size_stride(arg339_1, (256, ), (1, ))
        assert_size_stride(arg340_1, (256, ), (1, ))
        assert_size_stride(arg341_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg342_1, (256, ), (1, ))
        assert_size_stride(arg343_1, (256, ), (1, ))
        assert_size_stride(arg344_1, (256, ), (1, ))
        assert_size_stride(arg345_1, (256, ), (1, ))
        assert_size_stride(arg346_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg347_1, (1024, ), (1, ))
        assert_size_stride(arg348_1, (1024, ), (1, ))
        assert_size_stride(arg349_1, (1024, ), (1, ))
        assert_size_stride(arg350_1, (1024, ), (1, ))
        assert_size_stride(arg351_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg352_1, (256, ), (1, ))
        assert_size_stride(arg353_1, (256, ), (1, ))
        assert_size_stride(arg354_1, (256, ), (1, ))
        assert_size_stride(arg355_1, (256, ), (1, ))
        assert_size_stride(arg356_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg357_1, (256, ), (1, ))
        assert_size_stride(arg358_1, (256, ), (1, ))
        assert_size_stride(arg359_1, (256, ), (1, ))
        assert_size_stride(arg360_1, (256, ), (1, ))
        assert_size_stride(arg361_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg362_1, (1024, ), (1, ))
        assert_size_stride(arg363_1, (1024, ), (1, ))
        assert_size_stride(arg364_1, (1024, ), (1, ))
        assert_size_stride(arg365_1, (1024, ), (1, ))
        assert_size_stride(arg366_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg367_1, (256, ), (1, ))
        assert_size_stride(arg368_1, (256, ), (1, ))
        assert_size_stride(arg369_1, (256, ), (1, ))
        assert_size_stride(arg370_1, (256, ), (1, ))
        assert_size_stride(arg371_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg372_1, (256, ), (1, ))
        assert_size_stride(arg373_1, (256, ), (1, ))
        assert_size_stride(arg374_1, (256, ), (1, ))
        assert_size_stride(arg375_1, (256, ), (1, ))
        assert_size_stride(arg376_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg377_1, (1024, ), (1, ))
        assert_size_stride(arg378_1, (1024, ), (1, ))
        assert_size_stride(arg379_1, (1024, ), (1, ))
        assert_size_stride(arg380_1, (1024, ), (1, ))
        assert_size_stride(arg381_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg382_1, (256, ), (1, ))
        assert_size_stride(arg383_1, (256, ), (1, ))
        assert_size_stride(arg384_1, (256, ), (1, ))
        assert_size_stride(arg385_1, (256, ), (1, ))
        assert_size_stride(arg386_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg387_1, (256, ), (1, ))
        assert_size_stride(arg388_1, (256, ), (1, ))
        assert_size_stride(arg389_1, (256, ), (1, ))
        assert_size_stride(arg390_1, (256, ), (1, ))
        assert_size_stride(arg391_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg392_1, (1024, ), (1, ))
        assert_size_stride(arg393_1, (1024, ), (1, ))
        assert_size_stride(arg394_1, (1024, ), (1, ))
        assert_size_stride(arg395_1, (1024, ), (1, ))
        assert_size_stride(arg396_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg397_1, (256, ), (1, ))
        assert_size_stride(arg398_1, (256, ), (1, ))
        assert_size_stride(arg399_1, (256, ), (1, ))
        assert_size_stride(arg400_1, (256, ), (1, ))
        assert_size_stride(arg401_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg402_1, (256, ), (1, ))
        assert_size_stride(arg403_1, (256, ), (1, ))
        assert_size_stride(arg404_1, (256, ), (1, ))
        assert_size_stride(arg405_1, (256, ), (1, ))
        assert_size_stride(arg406_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg407_1, (1024, ), (1, ))
        assert_size_stride(arg408_1, (1024, ), (1, ))
        assert_size_stride(arg409_1, (1024, ), (1, ))
        assert_size_stride(arg410_1, (1024, ), (1, ))
        assert_size_stride(arg411_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg412_1, (256, ), (1, ))
        assert_size_stride(arg413_1, (256, ), (1, ))
        assert_size_stride(arg414_1, (256, ), (1, ))
        assert_size_stride(arg415_1, (256, ), (1, ))
        assert_size_stride(arg416_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg417_1, (256, ), (1, ))
        assert_size_stride(arg418_1, (256, ), (1, ))
        assert_size_stride(arg419_1, (256, ), (1, ))
        assert_size_stride(arg420_1, (256, ), (1, ))
        assert_size_stride(arg421_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg422_1, (1024, ), (1, ))
        assert_size_stride(arg423_1, (1024, ), (1, ))
        assert_size_stride(arg424_1, (1024, ), (1, ))
        assert_size_stride(arg425_1, (1024, ), (1, ))
        assert_size_stride(arg426_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg427_1, (256, ), (1, ))
        assert_size_stride(arg428_1, (256, ), (1, ))
        assert_size_stride(arg429_1, (256, ), (1, ))
        assert_size_stride(arg430_1, (256, ), (1, ))
        assert_size_stride(arg431_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg432_1, (256, ), (1, ))
        assert_size_stride(arg433_1, (256, ), (1, ))
        assert_size_stride(arg434_1, (256, ), (1, ))
        assert_size_stride(arg435_1, (256, ), (1, ))
        assert_size_stride(arg436_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg437_1, (1024, ), (1, ))
        assert_size_stride(arg438_1, (1024, ), (1, ))
        assert_size_stride(arg439_1, (1024, ), (1, ))
        assert_size_stride(arg440_1, (1024, ), (1, ))
        assert_size_stride(arg441_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg442_1, (256, ), (1, ))
        assert_size_stride(arg443_1, (256, ), (1, ))
        assert_size_stride(arg444_1, (256, ), (1, ))
        assert_size_stride(arg445_1, (256, ), (1, ))
        assert_size_stride(arg446_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg447_1, (256, ), (1, ))
        assert_size_stride(arg448_1, (256, ), (1, ))
        assert_size_stride(arg449_1, (256, ), (1, ))
        assert_size_stride(arg450_1, (256, ), (1, ))
        assert_size_stride(arg451_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg452_1, (1024, ), (1, ))
        assert_size_stride(arg453_1, (1024, ), (1, ))
        assert_size_stride(arg454_1, (1024, ), (1, ))
        assert_size_stride(arg455_1, (1024, ), (1, ))
        assert_size_stride(arg456_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg457_1, (256, ), (1, ))
        assert_size_stride(arg458_1, (256, ), (1, ))
        assert_size_stride(arg459_1, (256, ), (1, ))
        assert_size_stride(arg460_1, (256, ), (1, ))
        assert_size_stride(arg461_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg462_1, (256, ), (1, ))
        assert_size_stride(arg463_1, (256, ), (1, ))
        assert_size_stride(arg464_1, (256, ), (1, ))
        assert_size_stride(arg465_1, (256, ), (1, ))
        assert_size_stride(arg466_1, (1024, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg467_1, (1024, ), (1, ))
        assert_size_stride(arg468_1, (1024, ), (1, ))
        assert_size_stride(arg469_1, (1024, ), (1, ))
        assert_size_stride(arg470_1, (1024, ), (1, ))
        assert_size_stride(arg471_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg472_1, (512, ), (1, ))
        assert_size_stride(arg473_1, (512, ), (1, ))
        assert_size_stride(arg474_1, (512, ), (1, ))
        assert_size_stride(arg475_1, (512, ), (1, ))
        assert_size_stride(arg476_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg477_1, (512, ), (1, ))
        assert_size_stride(arg478_1, (512, ), (1, ))
        assert_size_stride(arg479_1, (512, ), (1, ))
        assert_size_stride(arg480_1, (512, ), (1, ))
        assert_size_stride(arg481_1, (2048, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg482_1, (2048, ), (1, ))
        assert_size_stride(arg483_1, (2048, ), (1, ))
        assert_size_stride(arg484_1, (2048, ), (1, ))
        assert_size_stride(arg485_1, (2048, ), (1, ))
        assert_size_stride(arg486_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        assert_size_stride(arg487_1, (2048, ), (1, ))
        assert_size_stride(arg488_1, (2048, ), (1, ))
        assert_size_stride(arg489_1, (2048, ), (1, ))
        assert_size_stride(arg490_1, (2048, ), (1, ))
        assert_size_stride(arg491_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
        assert_size_stride(arg492_1, (512, ), (1, ))
        assert_size_stride(arg493_1, (512, ), (1, ))
        assert_size_stride(arg494_1, (512, ), (1, ))
        assert_size_stride(arg495_1, (512, ), (1, ))
        assert_size_stride(arg496_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg497_1, (512, ), (1, ))
        assert_size_stride(arg498_1, (512, ), (1, ))
        assert_size_stride(arg499_1, (512, ), (1, ))
        assert_size_stride(arg500_1, (512, ), (1, ))
        assert_size_stride(arg501_1, (2048, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg502_1, (2048, ), (1, ))
        assert_size_stride(arg503_1, (2048, ), (1, ))
        assert_size_stride(arg504_1, (2048, ), (1, ))
        assert_size_stride(arg505_1, (2048, ), (1, ))
        assert_size_stride(arg506_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
        assert_size_stride(arg507_1, (512, ), (1, ))
        assert_size_stride(arg508_1, (512, ), (1, ))
        assert_size_stride(arg509_1, (512, ), (1, ))
        assert_size_stride(arg510_1, (512, ), (1, ))
        assert_size_stride(arg511_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg512_1, (512, ), (1, ))
        assert_size_stride(arg513_1, (512, ), (1, ))
        assert_size_stride(arg514_1, (512, ), (1, ))
        assert_size_stride(arg515_1, (512, ), (1, ))
        assert_size_stride(arg516_1, (2048, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg517_1, (2048, ), (1, ))
        assert_size_stride(arg518_1, (2048, ), (1, ))
        assert_size_stride(arg519_1, (2048, ), (1, ))
        assert_size_stride(arg520_1, (2048, ), (1, ))
        assert_size_stride(arg521_1, (1000, 2048), (2048, 1))
        assert_size_stride(arg522_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((10, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 30, 50176, stream=stream0)
            del arg1_1
            buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (10, 64, 112, 112), (802816, 1, 7168, 64), 'unknown_op')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 8028160, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((10, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 2007040, stream=stream0)
            buf5 = empty_strided_cuda((31360, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_4.run(buf4, arg6_1, buf5, 245, 1, 1, stream=stream0)
            del arg6_1
            buf6 = reinterpret_tensor(buf5, (10, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [out, out_1, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 2007040, stream=stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            buf7 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(arg11_1, buf7, 4096, 9, stream=stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf8, (10, 64, 56, 56), (200704, 1, 3584, 64), 'unknown_op')
            del buf6
            del buf7
            buf9 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf9, arg12_1, arg13_1, arg14_1, arg15_1, 2007040, stream=stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            buf10 = reinterpret_tensor(buf3, (31360, 256), (256, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf9, arg16_1, buf10, 490, 1, 1, stream=stream0)
            del arg16_1
            del buf9
            buf11 = empty_strided_cuda((31360, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf4, arg21_1, buf11, 490, 1, 1, stream=stream0)
            del arg21_1
            del buf4
            buf12 = reinterpret_tensor(buf10, (10, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf10  # reuse
            buf13 = empty_strided_cuda((10, 256, 56, 56), (802816, 1, 14336, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_4, out_5, out_6, out_7, input_1, input_2, out_8, out_9, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8.run(buf12, arg17_1, arg18_1, arg19_1, arg20_1, buf11, arg22_1, arg23_1, arg24_1, arg25_1, buf13, 8028160, stream=stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del buf11
            buf14 = empty_strided_cuda((31360, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out_9, out_10], Original ATen: [aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_9.run(buf13, arg26_1, buf14, 245, 1, 1, stream=stream0)
            del arg26_1
            buf15 = reinterpret_tensor(buf14, (10, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [out_9, out_10, out_11, out_12], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf15, arg27_1, arg28_1, arg29_1, arg30_1, 2007040, stream=stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            buf16 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out_9, out_10, out_11, out_12, out_13], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(arg31_1, buf16, 4096, 9, stream=stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [out_9, out_10, out_11, out_12, out_13], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf17, (10, 64, 56, 56), (200704, 1, 3584, 64), 'unknown_op')
            del buf15
            buf18 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [out_14, out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf18, arg32_1, arg33_1, arg34_1, arg35_1, 2007040, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            buf19 = reinterpret_tensor(buf13, (31360, 256), (256, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [out_14, out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf18, arg36_1, buf19, 490, 1, 1, stream=stream0)
            del arg36_1
            buf20 = reinterpret_tensor(buf19, (10, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [out_9, out_14, out_15, out_16, out_17, out_18, out_19], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10.run(buf20, arg37_1, arg38_1, arg39_1, arg40_1, buf12, 8028160, stream=stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            buf21 = reinterpret_tensor(buf18, (31360, 64), (64, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_9.run(buf20, arg41_1, buf21, 245, 1, 1, stream=stream0)
            del arg41_1
            buf22 = reinterpret_tensor(buf21, (10, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [out_20, out_21, out_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf22, arg42_1, arg43_1, arg44_1, arg45_1, 2007040, stream=stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            buf23 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [out_20, out_21, out_22, out_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(arg46_1, buf23, 4096, 9, stream=stream0)
            del arg46_1
            # Topologically Sorted Source Nodes: [out_20, out_21, out_22, out_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf24, (10, 64, 56, 56), (200704, 1, 3584, 64), 'unknown_op')
            del buf23
            buf25 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [out_24, out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf25, arg47_1, arg48_1, arg49_1, arg50_1, 2007040, stream=stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            buf26 = reinterpret_tensor(buf12, (31360, 256), (256, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [out_24, out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf25, arg51_1, buf26, 490, 1, 1, stream=stream0)
            del arg51_1
            buf27 = reinterpret_tensor(buf26, (10, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [out_24, out_25, out_26, out_27, out_28, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11.run(buf27, arg52_1, arg53_1, arg54_1, arg55_1, buf20, 8028160, stream=stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            del buf20
            buf28 = empty_strided_cuda((31360, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_12.run(buf27, arg56_1, buf28, 245, 1, 1, stream=stream0)
            del arg56_1
            buf29 = reinterpret_tensor(buf28, (10, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [out_30, out_31, out_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf29, arg57_1, arg58_1, arg59_1, arg60_1, 4014080, stream=stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            buf30 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [out_30, out_31, out_32, out_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg61_1, buf30, 16384, 9, stream=stream0)
            del arg61_1
            # Topologically Sorted Source Nodes: [out_30, out_31, out_32, out_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf31 = extern_kernels.convolution(buf29, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf31, (10, 128, 28, 28), (100352, 1, 3584, 128), 'unknown_op')
            buf32 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf32, arg62_1, arg63_1, arg64_1, arg65_1, 1003520, stream=stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            buf33 = reinterpret_tensor(buf29, (7840, 512), (512, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf32, arg66_1, buf33, 248, 1, 1, stream=stream0)
            del arg66_1
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
            buf34 = extern_kernels.convolution(buf27, arg71_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf34, (10, 512, 28, 28), (401408, 1, 14336, 512), 'unknown_op')
            del arg71_1
            del buf27
            buf35 = reinterpret_tensor(buf33, (10, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf33  # reuse
            buf36 = empty_strided_cuda((10, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_34, out_35, out_36, out_37, input_4, out_38, out_39, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17.run(buf35, arg67_1, arg68_1, arg69_1, arg70_1, buf34, arg72_1, arg73_1, arg74_1, arg75_1, buf36, 4014080, stream=stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            del buf34
            buf37 = reinterpret_tensor(buf32, (7840, 128), (128, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [out_39, out_40], Original ATen: [aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_18.run(buf36, arg76_1, buf37, 123, 1, 1, stream=stream0)
            del arg76_1
            buf38 = reinterpret_tensor(buf37, (10, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [out_39, out_40, out_41, out_42], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf38, arg77_1, arg78_1, arg79_1, arg80_1, 1003520, stream=stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            buf39 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [out_39, out_40, out_41, out_42, out_43], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg81_1, buf39, 16384, 9, stream=stream0)
            del arg81_1
            # Topologically Sorted Source Nodes: [out_39, out_40, out_41, out_42, out_43], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf40, (10, 128, 28, 28), (100352, 1, 3584, 128), 'unknown_op')
            del buf38
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf41, arg82_1, arg83_1, arg84_1, arg85_1, 1003520, stream=stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            buf42 = reinterpret_tensor(buf36, (7840, 512), (512, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf41, arg86_1, buf42, 248, 1, 1, stream=stream0)
            del arg86_1
            buf43 = reinterpret_tensor(buf42, (10, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [out_39, out_44, out_45, out_46, out_47, out_48, out_49], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19.run(buf43, arg87_1, arg88_1, arg89_1, arg90_1, buf35, 4014080, stream=stream0)
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            buf44 = reinterpret_tensor(buf41, (7840, 128), (128, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_18.run(buf43, arg91_1, buf44, 123, 1, 1, stream=stream0)
            del arg91_1
            buf45 = reinterpret_tensor(buf44, (10, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [out_50, out_51, out_52], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf45, arg92_1, arg93_1, arg94_1, arg95_1, 1003520, stream=stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            buf46 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [out_50, out_51, out_52, out_53], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg96_1, buf46, 16384, 9, stream=stream0)
            del arg96_1
            # Topologically Sorted Source Nodes: [out_50, out_51, out_52, out_53], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf47, (10, 128, 28, 28), (100352, 1, 3584, 128), 'unknown_op')
            del buf45
            buf48 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [out_54, out_55, out_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf48, arg97_1, arg98_1, arg99_1, arg100_1, 1003520, stream=stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            buf49 = reinterpret_tensor(buf35, (7840, 512), (512, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [out_54, out_55, out_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf48, arg101_1, buf49, 248, 1, 1, stream=stream0)
            del arg101_1
            buf50 = reinterpret_tensor(buf49, (10, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [out_54, out_55, out_56, out_57, out_58, out_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20.run(buf50, arg102_1, arg103_1, arg104_1, arg105_1, buf43, 4014080, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            del arg105_1
            buf51 = reinterpret_tensor(buf48, (7840, 128), (128, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_18.run(buf50, arg106_1, buf51, 123, 1, 1, stream=stream0)
            del arg106_1
            buf52 = reinterpret_tensor(buf51, (10, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [out_60, out_61, out_62], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf52, arg107_1, arg108_1, arg109_1, arg110_1, 1003520, stream=stream0)
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            buf53 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [out_60, out_61, out_62, out_63], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg111_1, buf53, 16384, 9, stream=stream0)
            del arg111_1
            # Topologically Sorted Source Nodes: [out_60, out_61, out_62, out_63], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf54, (10, 128, 28, 28), (100352, 1, 3584, 128), 'unknown_op')
            del buf53
            buf55 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [out_64, out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf55, arg112_1, arg113_1, arg114_1, arg115_1, 1003520, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            buf56 = reinterpret_tensor(buf43, (7840, 512), (512, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [out_64, out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf55, arg116_1, buf56, 248, 1, 1, stream=stream0)
            del arg116_1
            buf57 = reinterpret_tensor(buf56, (10, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [out_64, out_65, out_66, out_67, out_68, out_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20.run(buf57, arg117_1, arg118_1, arg119_1, arg120_1, buf50, 4014080, stream=stream0)
            del arg117_1
            del arg118_1
            del arg119_1
            del arg120_1
            del buf50
            buf58 = reinterpret_tensor(buf25, (7840, 256), (256, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_21.run(buf57, arg121_1, buf58, 246, 1, 1, stream=stream0)
            del arg121_1
            buf59 = reinterpret_tensor(buf58, (10, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [out_70, out_71, out_72], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf59, arg122_1, arg123_1, arg124_1, arg125_1, 2007040, stream=stream0)
            del arg122_1
            del arg123_1
            del arg124_1
            del arg125_1
            buf60 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_70, out_71, out_72, out_73], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg126_1, buf60, 65536, 9, stream=stream0)
            del arg126_1
            # Topologically Sorted Source Nodes: [out_70, out_71, out_72, out_73], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf61 = extern_kernels.convolution(buf59, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf61, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            buf62 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf62, arg127_1, arg128_1, arg129_1, arg130_1, 501760, stream=stream0)
            del arg127_1
            del arg128_1
            del arg129_1
            del arg130_1
            buf63 = reinterpret_tensor(buf59, (1960, 1024), (1024, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf62, arg131_1, buf63, 256, 1, 1, stream=stream0)
            del arg131_1
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
            buf64 = extern_kernels.convolution(buf57, arg136_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf64, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 'unknown_op')
            del arg136_1
            del buf57
            buf65 = reinterpret_tensor(buf63, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf63  # reuse
            buf66 = reinterpret_tensor(buf22, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [out_74, out_75, out_76, out_77, input_6, out_78, out_79, out_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26.run(buf65, arg132_1, arg133_1, arg134_1, arg135_1, buf64, arg137_1, arg138_1, arg139_1, arg140_1, buf66, 2007040, stream=stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            del arg135_1
            del arg137_1
            del arg138_1
            del arg139_1
            del arg140_1
            del buf64
            buf67 = reinterpret_tensor(buf62, (1960, 256), (256, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [out_79, out_80], Original ATen: [aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf66, arg141_1, buf67, 124, 1, 1, stream=stream0)
            del arg141_1
            buf68 = reinterpret_tensor(buf67, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [out_79, out_80, out_81, out_82], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf68, arg142_1, arg143_1, arg144_1, arg145_1, 501760, stream=stream0)
            del arg142_1
            del arg143_1
            del arg144_1
            del arg145_1
            buf69 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [out_79, out_80, out_81, out_82, out_83], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg146_1, buf69, 65536, 9, stream=stream0)
            del arg146_1
            # Topologically Sorted Source Nodes: [out_79, out_80, out_81, out_82, out_83], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            buf70 = extern_kernels.convolution(buf68, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf70, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf68
            buf71 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [out_84, out_85, out_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf71, arg147_1, arg148_1, arg149_1, arg150_1, 501760, stream=stream0)
            del arg147_1
            del arg148_1
            del arg149_1
            del arg150_1
            buf72 = reinterpret_tensor(buf66, (1960, 1024), (1024, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [out_84, out_85, out_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf71, arg151_1, buf72, 256, 1, 1, stream=stream0)
            del arg151_1
            buf73 = reinterpret_tensor(buf72, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [out_79, out_84, out_85, out_86, out_87, out_88, out_89], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28.run(buf73, arg152_1, arg153_1, arg154_1, arg155_1, buf65, 2007040, stream=stream0)
            del arg152_1
            del arg153_1
            del arg154_1
            del arg155_1
            buf74 = reinterpret_tensor(buf71, (1960, 256), (256, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf73, arg156_1, buf74, 124, 1, 1, stream=stream0)
            del arg156_1
            buf75 = reinterpret_tensor(buf74, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [out_90, out_91, out_92], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf75, arg157_1, arg158_1, arg159_1, arg160_1, 501760, stream=stream0)
            del arg157_1
            del arg158_1
            del arg159_1
            del arg160_1
            buf76 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [out_90, out_91, out_92, out_93], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg161_1, buf76, 65536, 9, stream=stream0)
            del arg161_1
            # Topologically Sorted Source Nodes: [out_90, out_91, out_92, out_93], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf77 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf77, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf75
            buf78 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [out_94, out_95, out_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf78, arg162_1, arg163_1, arg164_1, arg165_1, 501760, stream=stream0)
            del arg162_1
            del arg163_1
            del arg164_1
            del arg165_1
            buf79 = reinterpret_tensor(buf65, (1960, 1024), (1024, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [out_94, out_95, out_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf78, arg166_1, buf79, 256, 1, 1, stream=stream0)
            del arg166_1
            buf80 = reinterpret_tensor(buf79, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [out_94, out_95, out_96, out_97, out_98, out_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf80, arg167_1, arg168_1, arg169_1, arg170_1, buf73, 2007040, stream=stream0)
            del arg167_1
            del arg168_1
            del arg169_1
            del arg170_1
            buf81 = reinterpret_tensor(buf78, (1960, 256), (256, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [out_100], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf80, arg171_1, buf81, 124, 1, 1, stream=stream0)
            del arg171_1
            buf82 = reinterpret_tensor(buf81, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [out_100, out_101, out_102], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf82, arg172_1, arg173_1, arg174_1, arg175_1, 501760, stream=stream0)
            del arg172_1
            del arg173_1
            del arg174_1
            del arg175_1
            buf83 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [out_100, out_101, out_102, out_103], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg176_1, buf83, 65536, 9, stream=stream0)
            del arg176_1
            # Topologically Sorted Source Nodes: [out_100, out_101, out_102, out_103], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf84, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf82
            buf85 = buf84; del buf84  # reuse
            # Topologically Sorted Source Nodes: [out_104, out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf85, arg177_1, arg178_1, arg179_1, arg180_1, 501760, stream=stream0)
            del arg177_1
            del arg178_1
            del arg179_1
            del arg180_1
            buf86 = reinterpret_tensor(buf73, (1960, 1024), (1024, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [out_104, out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf85, arg181_1, buf86, 256, 1, 1, stream=stream0)
            del arg181_1
            buf87 = reinterpret_tensor(buf86, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [out_104, out_105, out_106, out_107, out_108, out_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf87, arg182_1, arg183_1, arg184_1, arg185_1, buf80, 2007040, stream=stream0)
            del arg182_1
            del arg183_1
            del arg184_1
            del arg185_1
            buf88 = reinterpret_tensor(buf85, (1960, 256), (256, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf87, arg186_1, buf88, 124, 1, 1, stream=stream0)
            del arg186_1
            buf89 = reinterpret_tensor(buf88, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [out_110, out_111, out_112], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf89, arg187_1, arg188_1, arg189_1, arg190_1, 501760, stream=stream0)
            del arg187_1
            del arg188_1
            del arg189_1
            del arg190_1
            buf90 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [out_110, out_111, out_112, out_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg191_1, buf90, 65536, 9, stream=stream0)
            del arg191_1
            # Topologically Sorted Source Nodes: [out_110, out_111, out_112, out_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf91 = extern_kernels.convolution(buf89, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf91, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf89
            buf92 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [out_114, out_115, out_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf92, arg192_1, arg193_1, arg194_1, arg195_1, 501760, stream=stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del arg195_1
            buf93 = reinterpret_tensor(buf80, (1960, 1024), (1024, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [out_114, out_115, out_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf92, arg196_1, buf93, 256, 1, 1, stream=stream0)
            del arg196_1
            buf94 = reinterpret_tensor(buf93, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [out_114, out_115, out_116, out_117, out_118, out_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf94, arg197_1, arg198_1, arg199_1, arg200_1, buf87, 2007040, stream=stream0)
            del arg197_1
            del arg198_1
            del arg199_1
            del arg200_1
            buf95 = reinterpret_tensor(buf92, (1960, 256), (256, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf94, arg201_1, buf95, 124, 1, 1, stream=stream0)
            del arg201_1
            buf96 = reinterpret_tensor(buf95, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [out_120, out_121, out_122], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf96, arg202_1, arg203_1, arg204_1, arg205_1, 501760, stream=stream0)
            del arg202_1
            del arg203_1
            del arg204_1
            del arg205_1
            buf97 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [out_120, out_121, out_122, out_123], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg206_1, buf97, 65536, 9, stream=stream0)
            del arg206_1
            # Topologically Sorted Source Nodes: [out_120, out_121, out_122, out_123], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf98, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf96
            buf99 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [out_124, out_125, out_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf99, arg207_1, arg208_1, arg209_1, arg210_1, 501760, stream=stream0)
            del arg207_1
            del arg208_1
            del arg209_1
            del arg210_1
            buf100 = reinterpret_tensor(buf87, (1960, 1024), (1024, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [out_124, out_125, out_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf99, arg211_1, buf100, 256, 1, 1, stream=stream0)
            del arg211_1
            buf101 = reinterpret_tensor(buf100, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [out_124, out_125, out_126, out_127, out_128, out_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf101, arg212_1, arg213_1, arg214_1, arg215_1, buf94, 2007040, stream=stream0)
            del arg212_1
            del arg213_1
            del arg214_1
            del arg215_1
            buf102 = reinterpret_tensor(buf99, (1960, 256), (256, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf101, arg216_1, buf102, 124, 1, 1, stream=stream0)
            del arg216_1
            buf103 = reinterpret_tensor(buf102, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [out_130, out_131, out_132], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf103, arg217_1, arg218_1, arg219_1, arg220_1, 501760, stream=stream0)
            del arg217_1
            del arg218_1
            del arg219_1
            del arg220_1
            buf104 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [out_130, out_131, out_132, out_133], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg221_1, buf104, 65536, 9, stream=stream0)
            del arg221_1
            # Topologically Sorted Source Nodes: [out_130, out_131, out_132, out_133], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf105, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf103
            buf106 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [out_134, out_135, out_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf106, arg222_1, arg223_1, arg224_1, arg225_1, 501760, stream=stream0)
            del arg222_1
            del arg223_1
            del arg224_1
            del arg225_1
            buf107 = reinterpret_tensor(buf94, (1960, 1024), (1024, 1), 0); del buf94  # reuse
            # Topologically Sorted Source Nodes: [out_134, out_135, out_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf106, arg226_1, buf107, 256, 1, 1, stream=stream0)
            del arg226_1
            buf108 = reinterpret_tensor(buf107, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [out_134, out_135, out_136, out_137, out_138, out_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf108, arg227_1, arg228_1, arg229_1, arg230_1, buf101, 2007040, stream=stream0)
            del arg227_1
            del arg228_1
            del arg229_1
            del arg230_1
            buf109 = reinterpret_tensor(buf106, (1960, 256), (256, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf108, arg231_1, buf109, 124, 1, 1, stream=stream0)
            del arg231_1
            buf110 = reinterpret_tensor(buf109, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [out_140, out_141, out_142], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf110, arg232_1, arg233_1, arg234_1, arg235_1, 501760, stream=stream0)
            del arg232_1
            del arg233_1
            del arg234_1
            del arg235_1
            buf111 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [out_140, out_141, out_142, out_143], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg236_1, buf111, 65536, 9, stream=stream0)
            del arg236_1
            # Topologically Sorted Source Nodes: [out_140, out_141, out_142, out_143], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf112, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf110
            buf113 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [out_144, out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf113, arg237_1, arg238_1, arg239_1, arg240_1, 501760, stream=stream0)
            del arg237_1
            del arg238_1
            del arg239_1
            del arg240_1
            buf114 = reinterpret_tensor(buf101, (1960, 1024), (1024, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [out_144, out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf113, arg241_1, buf114, 256, 1, 1, stream=stream0)
            del arg241_1
            buf115 = reinterpret_tensor(buf114, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [out_144, out_145, out_146, out_147, out_148, out_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf115, arg242_1, arg243_1, arg244_1, arg245_1, buf108, 2007040, stream=stream0)
            del arg242_1
            del arg243_1
            del arg244_1
            del arg245_1
            buf116 = reinterpret_tensor(buf113, (1960, 256), (256, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf115, arg246_1, buf116, 124, 1, 1, stream=stream0)
            del arg246_1
            buf117 = reinterpret_tensor(buf116, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [out_150, out_151, out_152], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf117, arg247_1, arg248_1, arg249_1, arg250_1, 501760, stream=stream0)
            del arg247_1
            del arg248_1
            del arg249_1
            del arg250_1
            buf118 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [out_150, out_151, out_152, out_153], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg251_1, buf118, 65536, 9, stream=stream0)
            del arg251_1
            # Topologically Sorted Source Nodes: [out_150, out_151, out_152, out_153], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf119, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf117
            buf120 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [out_154, out_155, out_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf120, arg252_1, arg253_1, arg254_1, arg255_1, 501760, stream=stream0)
            del arg252_1
            del arg253_1
            del arg254_1
            del arg255_1
            buf121 = reinterpret_tensor(buf108, (1960, 1024), (1024, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [out_154, out_155, out_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf120, arg256_1, buf121, 256, 1, 1, stream=stream0)
            del arg256_1
            buf122 = reinterpret_tensor(buf121, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [out_154, out_155, out_156, out_157, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf122, arg257_1, arg258_1, arg259_1, arg260_1, buf115, 2007040, stream=stream0)
            del arg257_1
            del arg258_1
            del arg259_1
            del arg260_1
            buf123 = reinterpret_tensor(buf120, (1960, 256), (256, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [out_160], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf122, arg261_1, buf123, 124, 1, 1, stream=stream0)
            del arg261_1
            buf124 = reinterpret_tensor(buf123, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [out_160, out_161, out_162], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf124, arg262_1, arg263_1, arg264_1, arg265_1, 501760, stream=stream0)
            del arg262_1
            del arg263_1
            del arg264_1
            del arg265_1
            buf125 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [out_160, out_161, out_162, out_163], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg266_1, buf125, 65536, 9, stream=stream0)
            del arg266_1
            # Topologically Sorted Source Nodes: [out_160, out_161, out_162, out_163], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf126 = extern_kernels.convolution(buf124, buf125, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf126, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf124
            buf127 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [out_164, out_165, out_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf127, arg267_1, arg268_1, arg269_1, arg270_1, 501760, stream=stream0)
            del arg267_1
            del arg268_1
            del arg269_1
            del arg270_1
            buf128 = reinterpret_tensor(buf115, (1960, 1024), (1024, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [out_164, out_165, out_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf127, arg271_1, buf128, 256, 1, 1, stream=stream0)
            del arg271_1
            buf129 = reinterpret_tensor(buf128, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [out_164, out_165, out_166, out_167, out_168, out_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf129, arg272_1, arg273_1, arg274_1, arg275_1, buf122, 2007040, stream=stream0)
            del arg272_1
            del arg273_1
            del arg274_1
            del arg275_1
            buf130 = reinterpret_tensor(buf127, (1960, 256), (256, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [out_170], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf129, arg276_1, buf130, 124, 1, 1, stream=stream0)
            del arg276_1
            buf131 = reinterpret_tensor(buf130, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [out_170, out_171, out_172], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf131, arg277_1, arg278_1, arg279_1, arg280_1, 501760, stream=stream0)
            del arg277_1
            del arg278_1
            del arg279_1
            del arg280_1
            buf132 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [out_170, out_171, out_172, out_173], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg281_1, buf132, 65536, 9, stream=stream0)
            del arg281_1
            # Topologically Sorted Source Nodes: [out_170, out_171, out_172, out_173], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf133 = extern_kernels.convolution(buf131, buf132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf133, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf131
            buf134 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [out_174, out_175, out_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf134, arg282_1, arg283_1, arg284_1, arg285_1, 501760, stream=stream0)
            del arg282_1
            del arg283_1
            del arg284_1
            del arg285_1
            buf135 = reinterpret_tensor(buf122, (1960, 1024), (1024, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [out_174, out_175, out_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf134, arg286_1, buf135, 256, 1, 1, stream=stream0)
            del arg286_1
            buf136 = reinterpret_tensor(buf135, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [out_174, out_175, out_176, out_177, out_178, out_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf136, arg287_1, arg288_1, arg289_1, arg290_1, buf129, 2007040, stream=stream0)
            del arg287_1
            del arg288_1
            del arg289_1
            del arg290_1
            buf137 = reinterpret_tensor(buf134, (1960, 256), (256, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf136, arg291_1, buf137, 124, 1, 1, stream=stream0)
            del arg291_1
            buf138 = reinterpret_tensor(buf137, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [out_180, out_181, out_182], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf138, arg292_1, arg293_1, arg294_1, arg295_1, 501760, stream=stream0)
            del arg292_1
            del arg293_1
            del arg294_1
            del arg295_1
            buf139 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [out_180, out_181, out_182, out_183], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg296_1, buf139, 65536, 9, stream=stream0)
            del arg296_1
            # Topologically Sorted Source Nodes: [out_180, out_181, out_182, out_183], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf140, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf138
            buf141 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [out_184, out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf141, arg297_1, arg298_1, arg299_1, arg300_1, 501760, stream=stream0)
            del arg297_1
            del arg298_1
            del arg299_1
            del arg300_1
            buf142 = reinterpret_tensor(buf129, (1960, 1024), (1024, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [out_184, out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf141, arg301_1, buf142, 256, 1, 1, stream=stream0)
            del arg301_1
            buf143 = reinterpret_tensor(buf142, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [out_184, out_185, out_186, out_187, out_188, out_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf143, arg302_1, arg303_1, arg304_1, arg305_1, buf136, 2007040, stream=stream0)
            del arg302_1
            del arg303_1
            del arg304_1
            del arg305_1
            buf144 = reinterpret_tensor(buf141, (1960, 256), (256, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [out_190], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf143, arg306_1, buf144, 124, 1, 1, stream=stream0)
            del arg306_1
            buf145 = reinterpret_tensor(buf144, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [out_190, out_191, out_192], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf145, arg307_1, arg308_1, arg309_1, arg310_1, 501760, stream=stream0)
            del arg307_1
            del arg308_1
            del arg309_1
            del arg310_1
            buf146 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [out_190, out_191, out_192, out_193], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg311_1, buf146, 65536, 9, stream=stream0)
            del arg311_1
            # Topologically Sorted Source Nodes: [out_190, out_191, out_192, out_193], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf147, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf145
            buf148 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [out_194, out_195, out_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf148, arg312_1, arg313_1, arg314_1, arg315_1, 501760, stream=stream0)
            del arg312_1
            del arg313_1
            del arg314_1
            del arg315_1
            buf149 = reinterpret_tensor(buf136, (1960, 1024), (1024, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [out_194, out_195, out_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf148, arg316_1, buf149, 256, 1, 1, stream=stream0)
            del arg316_1
            buf150 = reinterpret_tensor(buf149, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [out_194, out_195, out_196, out_197, out_198, out_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf150, arg317_1, arg318_1, arg319_1, arg320_1, buf143, 2007040, stream=stream0)
            del arg317_1
            del arg318_1
            del arg319_1
            del arg320_1
            buf151 = reinterpret_tensor(buf148, (1960, 256), (256, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [out_200], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf150, arg321_1, buf151, 124, 1, 1, stream=stream0)
            del arg321_1
            buf152 = reinterpret_tensor(buf151, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [out_200, out_201, out_202], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf152, arg322_1, arg323_1, arg324_1, arg325_1, 501760, stream=stream0)
            del arg322_1
            del arg323_1
            del arg324_1
            del arg325_1
            buf153 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [out_200, out_201, out_202, out_203], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg326_1, buf153, 65536, 9, stream=stream0)
            del arg326_1
            # Topologically Sorted Source Nodes: [out_200, out_201, out_202, out_203], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf154, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf152
            buf155 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [out_204, out_205, out_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf155, arg327_1, arg328_1, arg329_1, arg330_1, 501760, stream=stream0)
            del arg327_1
            del arg328_1
            del arg329_1
            del arg330_1
            buf156 = reinterpret_tensor(buf143, (1960, 1024), (1024, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [out_204, out_205, out_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf155, arg331_1, buf156, 256, 1, 1, stream=stream0)
            del arg331_1
            buf157 = reinterpret_tensor(buf156, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [out_204, out_205, out_206, out_207, out_208, out_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf157, arg332_1, arg333_1, arg334_1, arg335_1, buf150, 2007040, stream=stream0)
            del arg332_1
            del arg333_1
            del arg334_1
            del arg335_1
            buf158 = reinterpret_tensor(buf155, (1960, 256), (256, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [out_210], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf157, arg336_1, buf158, 124, 1, 1, stream=stream0)
            del arg336_1
            buf159 = reinterpret_tensor(buf158, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [out_210, out_211, out_212], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf159, arg337_1, arg338_1, arg339_1, arg340_1, 501760, stream=stream0)
            del arg337_1
            del arg338_1
            del arg339_1
            del arg340_1
            buf160 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [out_210, out_211, out_212, out_213], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg341_1, buf160, 65536, 9, stream=stream0)
            del arg341_1
            # Topologically Sorted Source Nodes: [out_210, out_211, out_212, out_213], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf161, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf159
            buf162 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [out_214, out_215, out_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf162, arg342_1, arg343_1, arg344_1, arg345_1, 501760, stream=stream0)
            del arg342_1
            del arg343_1
            del arg344_1
            del arg345_1
            buf163 = reinterpret_tensor(buf150, (1960, 1024), (1024, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [out_214, out_215, out_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf162, arg346_1, buf163, 256, 1, 1, stream=stream0)
            del arg346_1
            buf164 = reinterpret_tensor(buf163, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [out_214, out_215, out_216, out_217, out_218, out_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf164, arg347_1, arg348_1, arg349_1, arg350_1, buf157, 2007040, stream=stream0)
            del arg347_1
            del arg348_1
            del arg349_1
            del arg350_1
            buf165 = reinterpret_tensor(buf162, (1960, 256), (256, 1), 0); del buf162  # reuse
            # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf164, arg351_1, buf165, 124, 1, 1, stream=stream0)
            del arg351_1
            buf166 = reinterpret_tensor(buf165, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [out_220, out_221, out_222], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf166, arg352_1, arg353_1, arg354_1, arg355_1, 501760, stream=stream0)
            del arg352_1
            del arg353_1
            del arg354_1
            del arg355_1
            buf167 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [out_220, out_221, out_222, out_223], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg356_1, buf167, 65536, 9, stream=stream0)
            del arg356_1
            # Topologically Sorted Source Nodes: [out_220, out_221, out_222, out_223], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf168 = extern_kernels.convolution(buf166, buf167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf168, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf166
            buf169 = buf168; del buf168  # reuse
            # Topologically Sorted Source Nodes: [out_224, out_225, out_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf169, arg357_1, arg358_1, arg359_1, arg360_1, 501760, stream=stream0)
            del arg357_1
            del arg358_1
            del arg359_1
            del arg360_1
            buf170 = reinterpret_tensor(buf157, (1960, 1024), (1024, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [out_224, out_225, out_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf169, arg361_1, buf170, 256, 1, 1, stream=stream0)
            del arg361_1
            buf171 = reinterpret_tensor(buf170, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf170  # reuse
            # Topologically Sorted Source Nodes: [out_224, out_225, out_226, out_227, out_228, out_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf171, arg362_1, arg363_1, arg364_1, arg365_1, buf164, 2007040, stream=stream0)
            del arg362_1
            del arg363_1
            del arg364_1
            del arg365_1
            buf172 = reinterpret_tensor(buf169, (1960, 256), (256, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [out_230], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf171, arg366_1, buf172, 124, 1, 1, stream=stream0)
            del arg366_1
            buf173 = reinterpret_tensor(buf172, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [out_230, out_231, out_232], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf173, arg367_1, arg368_1, arg369_1, arg370_1, 501760, stream=stream0)
            del arg367_1
            del arg368_1
            del arg369_1
            del arg370_1
            buf174 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [out_230, out_231, out_232, out_233], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg371_1, buf174, 65536, 9, stream=stream0)
            del arg371_1
            # Topologically Sorted Source Nodes: [out_230, out_231, out_232, out_233], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf175, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf173
            buf176 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [out_234, out_235, out_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf176, arg372_1, arg373_1, arg374_1, arg375_1, 501760, stream=stream0)
            del arg372_1
            del arg373_1
            del arg374_1
            del arg375_1
            buf177 = reinterpret_tensor(buf164, (1960, 1024), (1024, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [out_234, out_235, out_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf176, arg376_1, buf177, 256, 1, 1, stream=stream0)
            del arg376_1
            buf178 = reinterpret_tensor(buf177, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [out_234, out_235, out_236, out_237, out_238, out_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf178, arg377_1, arg378_1, arg379_1, arg380_1, buf171, 2007040, stream=stream0)
            del arg377_1
            del arg378_1
            del arg379_1
            del arg380_1
            buf179 = reinterpret_tensor(buf176, (1960, 256), (256, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [out_240], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf178, arg381_1, buf179, 124, 1, 1, stream=stream0)
            del arg381_1
            buf180 = reinterpret_tensor(buf179, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [out_240, out_241, out_242], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf180, arg382_1, arg383_1, arg384_1, arg385_1, 501760, stream=stream0)
            del arg382_1
            del arg383_1
            del arg384_1
            del arg385_1
            buf181 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [out_240, out_241, out_242, out_243], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg386_1, buf181, 65536, 9, stream=stream0)
            del arg386_1
            # Topologically Sorted Source Nodes: [out_240, out_241, out_242, out_243], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf182, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf180
            buf183 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [out_244, out_245, out_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf183, arg387_1, arg388_1, arg389_1, arg390_1, 501760, stream=stream0)
            del arg387_1
            del arg388_1
            del arg389_1
            del arg390_1
            buf184 = reinterpret_tensor(buf171, (1960, 1024), (1024, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [out_244, out_245, out_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf183, arg391_1, buf184, 256, 1, 1, stream=stream0)
            del arg391_1
            buf185 = reinterpret_tensor(buf184, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [out_244, out_245, out_246, out_247, out_248, out_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf185, arg392_1, arg393_1, arg394_1, arg395_1, buf178, 2007040, stream=stream0)
            del arg392_1
            del arg393_1
            del arg394_1
            del arg395_1
            buf186 = reinterpret_tensor(buf183, (1960, 256), (256, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [out_250], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf185, arg396_1, buf186, 124, 1, 1, stream=stream0)
            del arg396_1
            buf187 = reinterpret_tensor(buf186, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [out_250, out_251, out_252], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf187, arg397_1, arg398_1, arg399_1, arg400_1, 501760, stream=stream0)
            del arg397_1
            del arg398_1
            del arg399_1
            del arg400_1
            buf188 = buf181; del buf181  # reuse
            # Topologically Sorted Source Nodes: [out_250, out_251, out_252, out_253], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg401_1, buf188, 65536, 9, stream=stream0)
            del arg401_1
            # Topologically Sorted Source Nodes: [out_250, out_251, out_252, out_253], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf189 = extern_kernels.convolution(buf187, buf188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf189, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf187
            buf190 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [out_254, out_255, out_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf190, arg402_1, arg403_1, arg404_1, arg405_1, 501760, stream=stream0)
            del arg402_1
            del arg403_1
            del arg404_1
            del arg405_1
            buf191 = reinterpret_tensor(buf178, (1960, 1024), (1024, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [out_254, out_255, out_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf190, arg406_1, buf191, 256, 1, 1, stream=stream0)
            del arg406_1
            buf192 = reinterpret_tensor(buf191, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [out_254, out_255, out_256, out_257, out_258, out_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf192, arg407_1, arg408_1, arg409_1, arg410_1, buf185, 2007040, stream=stream0)
            del arg407_1
            del arg408_1
            del arg409_1
            del arg410_1
            buf193 = reinterpret_tensor(buf190, (1960, 256), (256, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [out_260], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf192, arg411_1, buf193, 124, 1, 1, stream=stream0)
            del arg411_1
            buf194 = reinterpret_tensor(buf193, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [out_260, out_261, out_262], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf194, arg412_1, arg413_1, arg414_1, arg415_1, 501760, stream=stream0)
            del arg412_1
            del arg413_1
            del arg414_1
            del arg415_1
            buf195 = buf188; del buf188  # reuse
            # Topologically Sorted Source Nodes: [out_260, out_261, out_262, out_263], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg416_1, buf195, 65536, 9, stream=stream0)
            del arg416_1
            # Topologically Sorted Source Nodes: [out_260, out_261, out_262, out_263], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf196, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf194
            buf197 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [out_264, out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf197, arg417_1, arg418_1, arg419_1, arg420_1, 501760, stream=stream0)
            del arg417_1
            del arg418_1
            del arg419_1
            del arg420_1
            buf198 = reinterpret_tensor(buf185, (1960, 1024), (1024, 1), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [out_264, out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf197, arg421_1, buf198, 256, 1, 1, stream=stream0)
            del arg421_1
            buf199 = reinterpret_tensor(buf198, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [out_264, out_265, out_266, out_267, out_268, out_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf199, arg422_1, arg423_1, arg424_1, arg425_1, buf192, 2007040, stream=stream0)
            del arg422_1
            del arg423_1
            del arg424_1
            del arg425_1
            buf200 = reinterpret_tensor(buf197, (1960, 256), (256, 1), 0); del buf197  # reuse
            # Topologically Sorted Source Nodes: [out_270], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf199, arg426_1, buf200, 124, 1, 1, stream=stream0)
            del arg426_1
            buf201 = reinterpret_tensor(buf200, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [out_270, out_271, out_272], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf201, arg427_1, arg428_1, arg429_1, arg430_1, 501760, stream=stream0)
            del arg427_1
            del arg428_1
            del arg429_1
            del arg430_1
            buf202 = buf195; del buf195  # reuse
            # Topologically Sorted Source Nodes: [out_270, out_271, out_272, out_273], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg431_1, buf202, 65536, 9, stream=stream0)
            del arg431_1
            # Topologically Sorted Source Nodes: [out_270, out_271, out_272, out_273], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf203 = extern_kernels.convolution(buf201, buf202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf203, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf201
            buf204 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [out_274, out_275, out_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf204, arg432_1, arg433_1, arg434_1, arg435_1, 501760, stream=stream0)
            del arg432_1
            del arg433_1
            del arg434_1
            del arg435_1
            buf205 = reinterpret_tensor(buf192, (1960, 1024), (1024, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [out_274, out_275, out_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf204, arg436_1, buf205, 256, 1, 1, stream=stream0)
            del arg436_1
            buf206 = reinterpret_tensor(buf205, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf205  # reuse
            # Topologically Sorted Source Nodes: [out_274, out_275, out_276, out_277, out_278, out_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf206, arg437_1, arg438_1, arg439_1, arg440_1, buf199, 2007040, stream=stream0)
            del arg437_1
            del arg438_1
            del arg439_1
            del arg440_1
            buf207 = reinterpret_tensor(buf204, (1960, 256), (256, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [out_280], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf206, arg441_1, buf207, 124, 1, 1, stream=stream0)
            del arg441_1
            buf208 = reinterpret_tensor(buf207, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [out_280, out_281, out_282], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf208, arg442_1, arg443_1, arg444_1, arg445_1, 501760, stream=stream0)
            del arg442_1
            del arg443_1
            del arg444_1
            del arg445_1
            buf209 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [out_280, out_281, out_282, out_283], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg446_1, buf209, 65536, 9, stream=stream0)
            del arg446_1
            # Topologically Sorted Source Nodes: [out_280, out_281, out_282, out_283], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf210, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf208
            buf211 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [out_284, out_285, out_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf211, arg447_1, arg448_1, arg449_1, arg450_1, 501760, stream=stream0)
            del arg447_1
            del arg448_1
            del arg449_1
            del arg450_1
            buf212 = reinterpret_tensor(buf199, (1960, 1024), (1024, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [out_284, out_285, out_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf211, arg451_1, buf212, 256, 1, 1, stream=stream0)
            del arg451_1
            buf213 = reinterpret_tensor(buf212, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [out_284, out_285, out_286, out_287, out_288, out_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf213, arg452_1, arg453_1, arg454_1, arg455_1, buf206, 2007040, stream=stream0)
            del arg452_1
            del arg453_1
            del arg454_1
            del arg455_1
            buf214 = reinterpret_tensor(buf211, (1960, 256), (256, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [out_290], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_27.run(buf213, arg456_1, buf214, 124, 1, 1, stream=stream0)
            del arg456_1
            buf215 = reinterpret_tensor(buf214, (10, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf214  # reuse
            # Topologically Sorted Source Nodes: [out_290, out_291, out_292], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf215, arg457_1, arg458_1, arg459_1, arg460_1, 501760, stream=stream0)
            del arg457_1
            del arg458_1
            del arg459_1
            del arg460_1
            buf216 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [out_290, out_291, out_292, out_293], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(arg461_1, buf216, 65536, 9, stream=stream0)
            del arg461_1
            # Topologically Sorted Source Nodes: [out_290, out_291, out_292, out_293], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf217 = extern_kernels.convolution(buf215, buf216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf217, (10, 256, 14, 14), (50176, 1, 3584, 256), 'unknown_op')
            del buf215
            del buf216
            buf218 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [out_294, out_295, out_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf218, arg462_1, arg463_1, arg464_1, arg465_1, 501760, stream=stream0)
            del arg462_1
            del arg463_1
            del arg464_1
            del arg465_1
            buf219 = reinterpret_tensor(buf206, (1960, 1024), (1024, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [out_294, out_295, out_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25.run(buf218, arg466_1, buf219, 256, 1, 1, stream=stream0)
            del arg466_1
            del buf218
            buf220 = reinterpret_tensor(buf219, (10, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [out_294, out_295, out_296, out_297, out_298, out_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29.run(buf220, arg467_1, arg468_1, arg469_1, arg470_1, buf213, 2007040, stream=stream0)
            del arg467_1
            del arg468_1
            del arg469_1
            del arg470_1
            del buf213
            buf221 = reinterpret_tensor(buf55, (1960, 512), (512, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_30.run(buf220, arg471_1, buf221, 124, 1, 1, stream=stream0)
            del arg471_1
            buf222 = reinterpret_tensor(buf221, (10, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [out_300, out_301, out_302], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(buf222, arg472_1, arg473_1, arg474_1, arg475_1, 1003520, stream=stream0)
            del arg472_1
            del arg473_1
            del arg474_1
            del arg475_1
            buf223 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_300, out_301, out_302, out_303], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(arg476_1, buf223, 262144, 9, stream=stream0)
            del arg476_1
            # Topologically Sorted Source Nodes: [out_300, out_301, out_302, out_303], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf224 = extern_kernels.convolution(buf222, buf223, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf224, (10, 512, 7, 7), (25088, 1, 3584, 512), 'unknown_op')
            buf225 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [out_304, out_305, out_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf225, arg477_1, arg478_1, arg479_1, arg480_1, 250880, stream=stream0)
            del arg477_1
            del arg478_1
            del arg479_1
            del arg480_1
            buf226 = reinterpret_tensor(buf222, (490, 2048), (2048, 1), 0); del buf222  # reuse
            # Topologically Sorted Source Nodes: [out_304, out_305, out_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(buf225, arg481_1, buf226, 128, 1, 1, stream=stream0)
            del arg481_1
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
            buf227 = extern_kernels.convolution(buf220, arg486_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf227, (10, 2048, 7, 7), (100352, 1, 14336, 2048), 'unknown_op')
            del arg486_1
            del buf220
            buf228 = reinterpret_tensor(buf226, (10, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf226  # reuse
            buf229 = reinterpret_tensor(buf52, (10, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [out_304, out_305, out_306, out_307, input_8, out_308, out_309, out_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35.run(buf228, arg482_1, arg483_1, arg484_1, arg485_1, buf227, arg487_1, arg488_1, arg489_1, arg490_1, buf229, 1003520, stream=stream0)
            del arg482_1
            del arg483_1
            del arg484_1
            del arg485_1
            del arg487_1
            del arg488_1
            del arg489_1
            del arg490_1
            del buf227
            buf230 = reinterpret_tensor(buf225, (490, 512), (512, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [out_309, out_310], Original ATen: [aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_36.run(buf229, arg491_1, buf230, 128, 1, 1, stream=stream0)
            del arg491_1
            buf231 = reinterpret_tensor(buf230, (10, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [out_309, out_310, out_311, out_312], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf231, arg492_1, arg493_1, arg494_1, arg495_1, 250880, stream=stream0)
            del arg492_1
            del arg493_1
            del arg494_1
            del arg495_1
            buf232 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [out_309, out_310, out_311, out_312, out_313], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(arg496_1, buf232, 262144, 9, stream=stream0)
            del arg496_1
            # Topologically Sorted Source Nodes: [out_309, out_310, out_311, out_312, out_313], Original ATen: [aten.relu, aten.convolution, aten._native_batch_norm_legit_no_training]
            buf233 = extern_kernels.convolution(buf231, buf232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf233, (10, 512, 7, 7), (25088, 1, 3584, 512), 'unknown_op')
            del buf231
            buf234 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [out_314, out_315, out_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf234, arg497_1, arg498_1, arg499_1, arg500_1, 250880, stream=stream0)
            del arg497_1
            del arg498_1
            del arg499_1
            del arg500_1
            buf235 = reinterpret_tensor(buf229, (490, 2048), (2048, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [out_314, out_315, out_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(buf234, arg501_1, buf235, 128, 1, 1, stream=stream0)
            del arg501_1
            buf236 = reinterpret_tensor(buf235, (10, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [out_309, out_314, out_315, out_316, out_317, out_318, out_319], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37.run(buf236, arg502_1, arg503_1, arg504_1, arg505_1, buf228, 20480, 49, stream=stream0)
            del arg502_1
            del arg503_1
            del arg504_1
            del arg505_1
            buf237 = reinterpret_tensor(buf234, (490, 512), (512, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [out_320], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_relu_36.run(buf236, arg506_1, buf237, 128, 1, 1, stream=stream0)
            del arg506_1
            buf238 = reinterpret_tensor(buf237, (10, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [out_320, out_321, out_322], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf238, arg507_1, arg508_1, arg509_1, arg510_1, 250880, stream=stream0)
            del arg507_1
            del arg508_1
            del arg509_1
            del arg510_1
            buf239 = buf232; del buf232  # reuse
            # Topologically Sorted Source Nodes: [out_320, out_321, out_322, out_323], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(arg511_1, buf239, 262144, 9, stream=stream0)
            del arg511_1
            # Topologically Sorted Source Nodes: [out_320, out_321, out_322, out_323], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
            buf240 = extern_kernels.convolution(buf238, buf239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf240, (10, 512, 7, 7), (25088, 1, 3584, 512), 'unknown_op')
            del buf238
            del buf239
            buf241 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [out_324, out_325, out_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf241, arg512_1, arg513_1, arg514_1, arg515_1, 250880, stream=stream0)
            del arg512_1
            del arg513_1
            del arg514_1
            del arg515_1
            buf242 = reinterpret_tensor(buf228, (490, 2048), (2048, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [out_324, out_325, out_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(buf241, arg516_1, buf242, 128, 1, 1, stream=stream0)
            del arg516_1
            del buf241
            buf243 = empty_strided_cuda((10, 2048, 1, 1), (2048, 1, 20480, 20480), torch.float32)
            buf244 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [out_324, out_325, out_326, out_327, out_328, out_329, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38.run(buf244, buf242, arg517_1, arg518_1, arg519_1, arg520_1, buf236, 20480, 49, stream=stream0)
            del arg517_1
            del arg518_1
            del arg519_1
            del arg520_1
            del buf236
            del buf242
            buf246 = empty_strided_cuda((12, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out_324, out_325, out_326, out_327, out_328, out_329, x_4, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add, aten.mean, aten.view, aten.addmm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39.run(arg522_1, arg521_1, buf244, buf246, 32, 1, 1, stream=stream0)
            del arg521_1
            del arg522_1
            del buf244
        return (reinterpret_tensor(buf246, (10, 1000), (1000, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels * expansion, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, layers, num_classes=1000):
        """
        :param block: Type of block to use (BasicBlock or Bottleneck)
        :param layers: List of integers specifying the number of blocks in each layer
        :param num_classes: Number of output classes
        """
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Test code
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [layers, num_classes]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 3584) % 56)
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x5 = xindex // 3584
    x6 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x5), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x5), tmp16, other=float("-inf"))
    tmp18 = tl.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x5), tmp23, other=float("-inf"))
    tmp25 = tl.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x5), tmp30, other=float("-inf"))
    tmp32 = tl.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x5), tmp33, other=float("-inf"))
    tmp35 = tl.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x5), tmp36, other=float("-inf"))
    tmp38 = tl.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x5), tmp43, other=float("-inf"))
    tmp45 = tl.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x5), tmp46, other=float("-inf"))
    tmp48 = tl.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x5), tmp49, other=float("-inf"))
    tmp51 = tl.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 64
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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
        xindex = idx_n + 64*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 64*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_7(arg_A, arg_B, out_ptr0):
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

    M = 31360
    N = 256
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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
        xindex = idx_n + 64*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_relu_9(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 64
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 64*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = tl.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = tl.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tl.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_12(arg_A, arg_B, out_ptr0):
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

    M = 31360
    N = 128
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_16(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 512
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
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_relu_18(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 128
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
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = tl.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = tl.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tl.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_21(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 256
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
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_25(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 1024
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1024*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_relu_27(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 256
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
        xindex = idx_n + 1024*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = tl.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = tl.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tl.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_30(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 512
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
        xindex = idx_n + 1024*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_relu_34(arg_A, arg_B, out_ptr0):
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

    M = 490
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
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp7 / tmp21)
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_relu_36(arg_A, arg_B, out_ptr0):
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

    M = 490
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
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_out_ptr0 + (y0 + 2048*x2 + 100352*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (y0 + 2048*x2 + 100352*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1, 1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([1, 1], 0, tl.int32)
    tmp18 = tl.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = tl.maximum(tmp17, tmp19)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + 2048*x2 + 100352*y1), tmp20, xmask)

''',
    '''@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 20480
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 100352*x1), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + 2048*r0_2 + 100352*x1), r0_mask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1, 1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = tl.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = tl.full([1, 1], 49.0, tl.float32)
    tmp25 = (tmp23 / tmp24)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, None)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_39(in_ptr0, arg_B, in_ptr2, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    B = arg_B

    M = 12
    N = 1000
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
        tmp2 = tl.broadcast_to(idx_m, [BLOCK_M, BLOCK_K])
        tmp3 = tl.full([1], 10, tl.int64)
        tmp4 = tmp2 < tmp3
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 2048*idx_m, [BLOCK_M, BLOCK_K])), tmp4, eviction_policy='evict_last', other=0.0)
        a = tmp5.broadcast_to(xindex.shape)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1]

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
