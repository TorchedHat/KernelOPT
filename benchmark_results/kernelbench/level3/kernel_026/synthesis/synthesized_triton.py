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
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[10, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/uz/cuzclu72hb74ox3yvds6jcu5emtpqa3lorayz4msn4cbalfj7lct.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[24, 3, 3, 3][27, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 5184, 'x': 2592}, 'kernel_num_gb': 5.184e-06, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 72, 9,


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
    num_gb = 5.184e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ea/ceaw3wrl2frwaww2tflkr4xbsuhqjqmaexugjhoikxgzb7rll3lp.py
# Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x => relu
# Graph fragment:
#   %convolution : Tensor "f32[10, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 36127104}, 'kernel_num_gb': 0.024084864, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 24)
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
    arg_0 = rand_strided((10, 24, 112, 112), (301056, 1, 2688, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 3010560,


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
    num_gb = 0.024084864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/hv/chvdmv6ieayw4mfnqrehfqkimq3sgeo7ltvoqdygtpamcaoa6s25.py
# Topologically Sorted Source Nodes: [batch_norm, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x => relu
#   x_1 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[10, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = PlaceHolder[target=relu]
#   %unsqueeze : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[10, 24, 112, 112][301056, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33116160}, 'kernel_num_gb': 0.0301056, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 1344) % 56)
    x1 = ((xindex // 24) % 56)
    x0 = (xindex % 24)
    x5 = xindex // 1344
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
    tmp11 = tl.load(in_ptr0 + ((-2712) + x0 + 48*x1 + 5376*x5), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2688) + x0 + 48*x1 + 5376*x5), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-2664) + x0 + 48*x1 + 5376*x5), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-24) + x0 + 48*x1 + 5376*x5), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 48*x1 + 5376*x5), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (24 + x0 + 48*x1 + 5376*x5), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (2664 + x0 + 48*x1 + 5376*x5), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2688 + x0 + 48*x1 + 5376*x5), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (2712 + x0 + 48*x1 + 5376*x5), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, xmask)


def get_args():
    arg_0 = rand_strided((10, 24, 112, 112), (301056, 1, 2688, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 752640,


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
    num_gb = 0.0301056
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/2s/c2sqoek3rr7vg72tbdkqydwytbtzaisddqw36j5ooykuc23obkca.py
# Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
# Graph fragment:
#   %getitem : Tensor "f32[10, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = PlaceHolder[target=getitem]
#   %arg6_1 : Tensor "f32[60, 8, 1, 1][8, 1, 1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_1 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3), kwargs = {})
#   return %convolution_1
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

num_stages=4,
num_warps=4,
triton_meta={'signature': {'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_4', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.01053888, 'kernel_flop': 30105600, 'config_args': {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 3, 'UNROLL': True, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused_convolution_4(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 3
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 10
    IN_C = 24
    IN_H = 56
    IN_W = 56
    OUT_C = 60
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 75264
    stride_xc = 1
    stride_xh = 1344
    stride_xw = 24
    stride_wc_out = 8
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 56*idx_h + 3136*idx_c + 188160*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 60*idx_w + 3360*idx_h + 188160*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((60, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 3,


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
    num_gb = 0.01053888
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/r6/cr6sx3jtu4lqrpxjkenzwfazuamk6u4lkdqhz5ewun4ugunjpnit.py
# Topologically Sorted Source Nodes: [batch_norm_1, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   out => relu_1
# Graph fragment:
#   %convolution_1 : Tensor "f32[10, 60, 56, 56][188160, 1, 3360, 60]cuda:0" = PlaceHolder[target=convolution_1]
#   %arg7_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %unsqueeze_8 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   return %relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 22580160}, 'kernel_num_gb': 0.01505376, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 60)
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
    arg_0 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1881600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.01505376
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/c2/cc2tmlcnt2lwxoava6jjxcndj7fzm24hdvwqgg7dx7jkwrmdjq5b.py
# Topologically Sorted Source Nodes: [out_1, x_2, transpose, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   out_1 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   transpose => permute
#   x_2 => view
#   x_3 => clone
# Graph fragment:
#   %convolution_2 : Tensor "f32[10, 60, 56, 56][188160, 1, 3360, 60]cuda:0" = PlaceHolder[target=convolution_2]
#   %arg12_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[60][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %unsqueeze_16 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %view : Tensor "f32[10, 3, 20, 56, 56][188160, 62720, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_5, [10, 3, 20, 56, 56]), kwargs = {})
#   %permute : Tensor "f32[10, 20, 3, 56, 56][188160, 3136, 62720, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1, 3, 4]), kwargs = {})
#   %clone : Tensor "f32[10, 20, 3, 56, 56][188160, 9408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6', '''
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
    size_hints={'y': 1024, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 7527360, 'x': 15052800}, 'kernel_num_gb': 0.01505376, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 600
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 60
    y4 = (yindex % 60)
    y0 = (yindex % 20)
    y1 = ((yindex // 20) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 60*x3 + 188160*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 188160*y2), tmp15, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 20, 3, 56, 56), (188160, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 600, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.01505376
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/m7/cm7tv24i2r7gkxqqg4ynfumszgazkpzzi5exsxrtzdwcgy3hwbgd.py
# Topologically Sorted Source Nodes: [out_1, x_2, transpose, x_3, x_4, conv2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
#   out_1 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   transpose => permute
#   x_2 => view
#   x_3 => clone
#   x_4 => view_1
# Graph fragment:
#   %clone : Tensor "f32[10, 20, 3, 56, 56][188160, 9408, 3136, 56, 1]cuda:0" = PlaceHolder[target=clone]
#   %unsqueeze_16 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[60, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %view : Tensor "f32[10, 3, 20, 56, 56][188160, 62720, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_5, [10, 3, 20, 56, 56]), kwargs = {})
#   %permute : Tensor "f32[10, 20, 3, 56, 56][188160, 3136, 62720, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1, 3, 4]), kwargs = {})
#   %clone : Tensor "f32[10, 20, 3, 56, 56][188160, 9408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1 : Tensor "f32[10, 60, 56, 56][188160, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [10, -1, 56, 56]), kwargs = {})
#   %convolution_3 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_1, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3), kwargs = {})
#   return %buf9
triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7', '''
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
    size_hints={'y': 1024, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 15052800, 'x': 7526400}, 'kernel_num_gb': 0.0150528, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 600
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 60)
    y1 = yindex // 60
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 60*x2 + 188160*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 20, 3, 56, 56), (188160, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 600, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0150528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ax/cax3zk3b7js2gsbaqsyt4ekk5ct323oyarrbpdhsjv4grc6ufhlw.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution_4
# Graph fragment:
#   %getitem : Tensor "f32[10, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = PlaceHolder[target=getitem]
#   %arg21_1 : Tensor "f32[240, 24, 1, 1][24, 1, 1, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %convolution_4 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf11
triton_tem_fused_convolution_8 = async_compile.triton('triton_tem_fused_convolution_8', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_8', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0331392, 'kernel_flop': 361267200, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_8(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
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
    N = 240
    K = 24
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 24
    stride_ak = 1
    stride_bk = 1
    stride_bn = 24

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
        xindex = idx_n + 24*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 240*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 24*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 240*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 490, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0331392
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/3z/c3zxikwgltp5rm2bpzxzjdpunmk2y3mfwtrbwxo46h2oep75dk54.py
# Topologically Sorted Source Nodes: [batch_norm_3, out_2, input_1, input_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   batch_norm_3 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   input_1 => convolution_4
#   input_2 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   out_2 => relu_2
#   out_3 => add_10
# Graph fragment:
#   %convolution_3 : Tensor "f32[10, 240, 56, 56][752640, 1, 13440, 240]cuda:0" = PlaceHolder[target=convolution_3]
#   %arg17_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %buf11 : Tensor "f32[31360, 240][240, 1]cuda:0" = PlaceHolder[target=buf11]
#   %arg22_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %unsqueeze_24 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_2 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   %convolution_4 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_2, %add_9), kwargs = {})
#   return %add_10
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 120430080}, 'kernel_num_gb': 0.09032448, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), xmask)
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = tl.sqrt_rn(tmp22)
    tmp24 = (tmp7 / tmp23)
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, xmask)


def get_args():
    arg_0 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((31360, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 7526400,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.09032448
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/qz/cqz2z6ppqqsyhdhcv2leh75xlq4gj63l54rnmzfez5rqlrhwzpay.py
# Topologically Sorted Source Nodes: [batch_norm_7, out_6, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_7 => add_15, add_16, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   out_6 => relu_4
#   out_7 => add_17
# Graph fragment:
#   %convolution_7 : Tensor "f32[10, 240, 56, 56][752640, 1, 13440, 240]cuda:0" = PlaceHolder[target=convolution_7]
#   %arg37_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %add_10 : Tensor "f32[10, 240, 56, 56][752640, 1, 13440, 240]cuda:0" = PlaceHolder[target=add_10]
#   %unsqueeze_56 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_15 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_15,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_16 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_4 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %add_17 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_4, %add_10), kwargs = {})
#   return %add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 120426240}, 'kernel_num_gb': 0.09032064, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), xmask)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 7526400,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.09032064
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/rj/crjlyv3ejb4cavzsof7owh2lnyn2ddixjcfpqz46cc5x3qtihkmy.py
# Topologically Sorted Source Nodes: [batch_norm_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_25, add_26, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   out_12 => relu_7
# Graph fragment:
#   %convolution_11 : Tensor "f32[10, 120, 56, 56][376320, 1, 6720, 120]cuda:0" = PlaceHolder[target=convolution_11]
#   %arg57_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %arg59_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %unsqueeze_88 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_25 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_25,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_26 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_7 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   return %relu_7
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 45160320}, 'kernel_num_gb': 0.03010752, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3763200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
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
    arg_0 = rand_strided((10, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 3763200,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.03010752
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ry/cryfdvuwakireje2xmxnahr35n5cscgt5spw3l6znh24woyzic5e.py
# Topologically Sorted Source Nodes: [out_13, x_11, transpose_3, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   out_13 => add_27, add_28, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   transpose_3 => permute_3
#   x_11 => view_6
#   x_12 => clone_3
# Graph fragment:
#   %convolution_12 : Tensor "f32[10, 120, 56, 56][376320, 1, 6720, 120]cuda:0" = PlaceHolder[target=convolution_12]
#   %arg62_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %arg63_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f32[120][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %unsqueeze_96 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_27 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %view_6 : Tensor "f32[10, 3, 40, 56, 56][376320, 125440, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_28, [10, 3, 40, 56, 56]), kwargs = {})
#   %permute_3 : Tensor "f32[10, 40, 3, 56, 56][376320, 3136, 125440, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3, 4]), kwargs = {})
#   %clone_3 : Tensor "f32[10, 40, 3, 56, 56][376320, 9408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12', '''
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
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 15054720, 'x': 30105600}, 'kernel_num_gb': 0.03010752, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1200
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 120
    y4 = (yindex % 120)
    y0 = (yindex % 40)
    y1 = ((yindex // 40) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 120*x3 + 376320*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 376320*y2), tmp15, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1200, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.03010752
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/f3/cf3m32gajqhudz2x7r57zcbkvqtvt7d7zanhyatcqzklv7isplnu.py
# Topologically Sorted Source Nodes: [out_13, x_11, transpose_3, x_12, x_13, conv2d_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_13 => convolution_13
#   out_13 => add_27, add_28, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   transpose_3 => permute_3
#   x_11 => view_6
#   x_12 => clone_3
#   x_13 => view_7
# Graph fragment:
#   %clone_3 : Tensor "f32[10, 40, 3, 56, 56][376320, 9408, 3136, 56, 1]cuda:0" = PlaceHolder[target=clone_3]
#   %unsqueeze_96 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_27 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[120, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %view_6 : Tensor "f32[10, 3, 40, 56, 56][376320, 125440, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_28, [10, 3, 40, 56, 56]), kwargs = {})
#   %permute_3 : Tensor "f32[10, 40, 3, 56, 56][376320, 3136, 125440, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3, 4]), kwargs = {})
#   %clone_3 : Tensor "f32[10, 40, 3, 56, 56][376320, 9408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_7 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [10, -1, 56, 56]), kwargs = {})
#   %convolution_13 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_7, %arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3), kwargs = {})
#   return %buf31
triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13', '''
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
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 30105600, 'x': 15052800}, 'kernel_num_gb': 0.0301056, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1200
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 120)
    y1 = yindex // 120
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 120*x2 + 376320*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1200, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0301056
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/od/coddzurvdpbfy6tom54v4o35hken6wrhdlwuoow6f36bvevpbvdg.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_3 => convolution_14
# Graph fragment:
#   %add_24 : Tensor "f32[10, 240, 56, 56][752640, 1, 13440, 240]cuda:0" = PlaceHolder[target=add_24]
#   %arg71_1 : Tensor "f32[480, 240, 1, 1][240, 1, 1, 1]cuda:0" = PlaceHolder[target=arg71_1]
#   %convolution_14 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_24, %arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf33
triton_tem_fused_convolution_14 = async_compile.triton('triton_tem_fused_convolution_14', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_14', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0907776, 'kernel_flop': 7225344000, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_14(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
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
    N = 480
    K = 240
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 240
    stride_ak = 1
    stride_bk = 1
    stride_bn = 240

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
        xindex = idx_n + 240*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 480*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 480*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1960, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0907776
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/3v/c3v7vk5ocoaonm36zvrhjh4b2ru5qk7oq24gr525dviljqvsczjl.py
# Topologically Sorted Source Nodes: [batch_norm_13, out_14, input_3, input_4, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   batch_norm_13 => add_29, add_30, mul_39, mul_40, mul_41, reciprocal_13, sqrt_13, sub_13, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111
#   input_3 => convolution_14
#   input_4 => add_31, add_32, mul_42, mul_43, mul_44, reciprocal_14, sqrt_14, sub_14, unsqueeze_112, unsqueeze_113, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119
#   out_14 => relu_8
#   out_15 => add_33
# Graph fragment:
#   %convolution_13 : Tensor "f32[10, 480, 56, 56][1505280, 1, 26880, 480]cuda:0" = PlaceHolder[target=convolution_13]
#   %arg67_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg67_1]
#   %arg68_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg68_1]
#   %arg69_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %arg70_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %buf33 : Tensor "f32[31360, 480][480, 1]cuda:0" = PlaceHolder[target=buf33]
#   %arg72_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %arg73_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %arg74_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %arg75_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %unsqueeze_104 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg67_1, -1), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_104, -1), kwargs = {})
#   %sub_13 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %add_29 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg68_1, 1e-05), kwargs = {})
#   %sqrt_13 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_29,), kwargs = {})
#   %reciprocal_13 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_13,), kwargs = {})
#   %mul_39 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 1), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_39, -1), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, -1), kwargs = {})
#   %mul_40 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg69_1, -1), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_108, -1), kwargs = {})
#   %mul_41 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg70_1, -1), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_110, -1), kwargs = {})
#   %add_30 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_8 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_30,), kwargs = {})
#   %convolution_14 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_24, %arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg72_1, -1), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, -1), kwargs = {})
#   %sub_14 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %add_31 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg73_1, 1e-05), kwargs = {})
#   %sqrt_14 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_31,), kwargs = {})
#   %reciprocal_14 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_14,), kwargs = {})
#   %mul_42 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, 1), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_42, -1), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, -1), kwargs = {})
#   %mul_43 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg74_1, -1), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, -1), kwargs = {})
#   %mul_44 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg75_1, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, -1), kwargs = {})
#   %add_32 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_33 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_8, %add_32), kwargs = {})
#   return %add_33
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 240860160}, 'kernel_num_gb': 0.18064896, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15052800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 480)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = tl.sqrt_rn(tmp22)
    tmp24 = (tmp7 / tmp23)
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, None)


def get_args():
    arg_0 = rand_strided((10, 480, 56, 56), (1505280, 1, 26880, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((31360, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 15052800,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.18064896
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/rc/crckgi5cpsas2ervrlfpxnlals7kb5hr7oussxlcjik223j2o6fq.py
# Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_15 => convolution_15
# Graph fragment:
#   %add_33 : Tensor "f32[10, 480, 56, 56][1505280, 1, 26880, 480]cuda:0" = PlaceHolder[target=add_33]
#   %arg76_1 : Tensor "f32[120, 160, 1, 1][160, 1, 1, 1]cuda:0" = PlaceHolder[target=arg76_1]
#   %convolution_15 : Tensor "f32[10, 120, 56, 56][376320, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_33, %arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3), kwargs = {})
#   return %convolution_15
triton_tem_fused_convolution_16 = async_compile.triton('triton_tem_fused_convolution_16', '''
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
triton_meta={'signature': {'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_16', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0753408, 'kernel_flop': 1204224000, 'config_args': {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 3, 'UNROLL': True, 'ALLOW_TF32': False, 'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused_convolution_16(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 3
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 10
    IN_C = 480
    IN_H = 56
    IN_W = 56
    OUT_C = 120
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 1505280
    stride_xc = 1
    stride_xh = 26880
    stride_xw = 480
    stride_wc_out = 160
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 56*idx_h + 3136*idx_c + 376320*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 120*idx_w + 6720*idx_h + 376320*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 480, 56, 56), (1505280, 1, 26880, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 123, 2, 3,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_16.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_16.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0753408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/g7/cg77k2c7gauha3ddhyhgifovk23qrdmmjt2qvckzshkowdnr2r3e.py
# Topologically Sorted Source Nodes: [batch_norm_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_17 => add_38, add_39, mul_51, mul_52, mul_53, reciprocal_17, sqrt_17, sub_17, unsqueeze_136, unsqueeze_137, unsqueeze_138, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143
#   out_18 => relu_10
#   out_19 => add_40
# Graph fragment:
#   %convolution_17 : Tensor "f32[10, 480, 56, 56][1505280, 1, 26880, 480]cuda:0" = PlaceHolder[target=convolution_17]
#   %arg87_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg88_1]
#   %arg89_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg89_1]
#   %arg90_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %add_33 : Tensor "f32[10, 480, 56, 56][1505280, 1, 26880, 480]cuda:0" = PlaceHolder[target=add_33]
#   %unsqueeze_136 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg87_1, -1), kwargs = {})
#   %unsqueeze_137 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_136, -1), kwargs = {})
#   %sub_17 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %add_38 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg88_1, 1e-05), kwargs = {})
#   %sqrt_17 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %reciprocal_17 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_17,), kwargs = {})
#   %mul_51 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 1), kwargs = {})
#   %unsqueeze_138 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_51, -1), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_138, -1), kwargs = {})
#   %mul_52 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg89_1, -1), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, -1), kwargs = {})
#   %mul_53 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg90_1, -1), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, -1), kwargs = {})
#   %add_39 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %relu_10 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_39,), kwargs = {})
#   %add_40 : Tensor "f32[10, 480, 56, 56][1505280, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_10, %add_33), kwargs = {})
#   return %add_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 240852480}, 'kernel_num_gb': 0.18064128, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15052800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 480)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 480, 56, 56), (1505280, 1, 26880, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 480, 56, 56), (1505280, 1, 26880, 480), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 15052800,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.18064128
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/l7/cl7pzx24mqcmwyftxbx5a4uq32hy4yqe5sqldn5ga5hr3vgxltlh.py
# Topologically Sorted Source Nodes: [batch_norm_33, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_33 => add_76, add_77, mul_100, mul_101, mul_99, reciprocal_33, sqrt_33, sub_33, unsqueeze_264, unsqueeze_265, unsqueeze_266, unsqueeze_267, unsqueeze_268, unsqueeze_269, unsqueeze_270, unsqueeze_271
#   out_40 => relu_21
# Graph fragment:
#   %convolution_33 : Tensor "f32[10, 240, 56, 56][752640, 1, 13440, 240]cuda:0" = PlaceHolder[target=convolution_33]
#   %arg167_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg167_1]
#   %arg168_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg168_1]
#   %arg169_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg169_1]
#   %arg170_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg170_1]
#   %unsqueeze_264 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg167_1, -1), kwargs = {})
#   %unsqueeze_265 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_264, -1), kwargs = {})
#   %sub_33 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %add_76 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg168_1, 1e-05), kwargs = {})
#   %sqrt_33 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_76,), kwargs = {})
#   %reciprocal_33 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_33,), kwargs = {})
#   %mul_99 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_33, 1), kwargs = {})
#   %unsqueeze_266 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_99, -1), kwargs = {})
#   %unsqueeze_267 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_266, -1), kwargs = {})
#   %mul_100 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %unsqueeze_268 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg169_1, -1), kwargs = {})
#   %unsqueeze_269 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_268, -1), kwargs = {})
#   %mul_101 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %unsqueeze_270 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg170_1, -1), kwargs = {})
#   %unsqueeze_271 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_270, -1), kwargs = {})
#   %add_77 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %relu_21 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
#   return %relu_21
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 90320640}, 'kernel_num_gb': 0.06021504, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
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
    arg_0 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 7526400,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.06021504
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/rn/crninlxiih2d7soqnwdfj2rf4indb5ftnygfqrf64mitfpa3keyi.py
# Topologically Sorted Source Nodes: [out_41, x_32, transpose_10, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   out_41 => add_78, add_79, mul_102, mul_103, mul_104, reciprocal_34, sqrt_34, sub_34, unsqueeze_272, unsqueeze_273, unsqueeze_274, unsqueeze_275, unsqueeze_276, unsqueeze_277, unsqueeze_278, unsqueeze_279
#   transpose_10 => permute_10
#   x_32 => view_20
#   x_33 => clone_10
# Graph fragment:
#   %convolution_34 : Tensor "f32[10, 240, 56, 56][752640, 1, 13440, 240]cuda:0" = PlaceHolder[target=convolution_34]
#   %arg172_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg172_1]
#   %arg173_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg173_1]
#   %arg174_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg174_1]
#   %arg175_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg175_1]
#   %unsqueeze_272 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg172_1, -1), kwargs = {})
#   %unsqueeze_273 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_272, -1), kwargs = {})
#   %sub_34 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_273), kwargs = {})
#   %add_78 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg173_1, 1e-05), kwargs = {})
#   %sqrt_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_78,), kwargs = {})
#   %reciprocal_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_34,), kwargs = {})
#   %mul_102 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_34, 1), kwargs = {})
#   %unsqueeze_274 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_102, -1), kwargs = {})
#   %unsqueeze_275 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_274, -1), kwargs = {})
#   %mul_103 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %unsqueeze_276 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg174_1, -1), kwargs = {})
#   %unsqueeze_277 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_276, -1), kwargs = {})
#   %mul_104 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %unsqueeze_278 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg175_1, -1), kwargs = {})
#   %unsqueeze_279 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_278, -1), kwargs = {})
#   %add_79 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %view_20 : Tensor "f32[10, 3, 80, 56, 56][752640, 250880, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_79, [10, 3, 80, 56, 56]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 80, 3, 56, 56][752640, 3136, 250880, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_20, [0, 2, 1, 3, 4]), kwargs = {})
#   %clone_10 : Tensor "f32[10, 80, 3, 56, 56][752640, 9408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_10
triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19', '''
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
    size_hints={'y': 4096, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 30109440, 'x': 60211200}, 'kernel_num_gb': 0.06021504, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 240
    y4 = (yindex % 240)
    y0 = (yindex % 80)
    y1 = ((yindex // 80) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 240*x3 + 752640*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 752640*y2), tmp15, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 80, 3, 56, 56), (752640, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2400, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.06021504
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/vs/cvs6pay6wnl4cwcos2d6y2xext2koqtgdq2knvqdmzv4ftcbqggw.py
# Topologically Sorted Source Nodes: [out_41, x_32, transpose_10, x_33, x_34, conv2d_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_35 => convolution_35
#   out_41 => add_78, add_79, mul_102, mul_103, mul_104, reciprocal_34, sqrt_34, sub_34, unsqueeze_272, unsqueeze_273, unsqueeze_274, unsqueeze_275, unsqueeze_276, unsqueeze_277, unsqueeze_278, unsqueeze_279
#   transpose_10 => permute_10
#   x_32 => view_20
#   x_33 => clone_10
#   x_34 => view_21
# Graph fragment:
#   %clone_10 : Tensor "f32[10, 80, 3, 56, 56][752640, 9408, 3136, 56, 1]cuda:0" = PlaceHolder[target=clone_10]
#   %unsqueeze_272 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg172_1, -1), kwargs = {})
#   %unsqueeze_273 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_272, -1), kwargs = {})
#   %sub_34 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_273), kwargs = {})
#   %add_78 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg173_1, 1e-05), kwargs = {})
#   %sqrt_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_78,), kwargs = {})
#   %reciprocal_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_34,), kwargs = {})
#   %mul_102 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_34, 1), kwargs = {})
#   %unsqueeze_274 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_102, -1), kwargs = {})
#   %unsqueeze_275 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_274, -1), kwargs = {})
#   %mul_103 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %unsqueeze_276 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg174_1, -1), kwargs = {})
#   %unsqueeze_277 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_276, -1), kwargs = {})
#   %mul_104 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %unsqueeze_278 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg175_1, -1), kwargs = {})
#   %unsqueeze_279 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_278, -1), kwargs = {})
#   %add_79 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %view_20 : Tensor "f32[10, 3, 80, 56, 56][752640, 250880, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_79, [10, 3, 80, 56, 56]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 80, 3, 56, 56][752640, 3136, 250880, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_20, [0, 2, 1, 3, 4]), kwargs = {})
#   %clone_10 : Tensor "f32[10, 80, 3, 56, 56][752640, 9408, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_21 : Tensor "f32[10, 240, 56, 56][752640, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [10, -1, 56, 56]), kwargs = {})
#   %convolution_35 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_21, %arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3), kwargs = {})
#   return %buf81
triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20', '''
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
    size_hints={'y': 4096, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 60211200, 'x': 30105600}, 'kernel_num_gb': 0.0602112, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 240)
    y1 = yindex // 240
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 240*x2 + 752640*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 80, 3, 56, 56), (752640, 9408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 2400, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0602112
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/xt/cxtr7fnnutehzekjvx6ihs6x2ji2qldlyn6kl32hivzjha7hligu.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_5 => convolution_36
# Graph fragment:
#   %add_75 : Tensor "f32[10, 480, 56, 56][1505280, 1, 26880, 480]cuda:0" = PlaceHolder[target=add_75]
#   %arg181_1 : Tensor "f32[960, 480, 1, 1][480, 1, 1, 1]cuda:0" = PlaceHolder[target=arg181_1]
#   %convolution_36 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf83
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_21', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.1824768, 'kernel_flop': 28901376000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_21(arg_A, arg_B, out_ptr0):
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
    N = 960
    K = 480
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 480
    stride_ak = 1
    stride_bk = 1
    stride_bn = 480

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
        xindex = idx_n + 480*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 960*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 960*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 480, 56, 56), (1505280, 1, 26880, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((960, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 3675, 1, 1,


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
    num_gb = 0.1824768
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bh/cbh5xojn6ocjwvvs4xlkyiecy2yrbwx4k7b42l4rutqiga6dzj6c.py
# Topologically Sorted Source Nodes: [batch_norm_35, out_42, input_5, input_6, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   batch_norm_35 => add_80, add_81, mul_105, mul_106, mul_107, reciprocal_35, sqrt_35, sub_35, unsqueeze_280, unsqueeze_281, unsqueeze_282, unsqueeze_283, unsqueeze_284, unsqueeze_285, unsqueeze_286, unsqueeze_287
#   input_5 => convolution_36
#   input_6 => add_82, add_83, mul_108, mul_109, mul_110, reciprocal_36, sqrt_36, sub_36, unsqueeze_288, unsqueeze_289, unsqueeze_290, unsqueeze_291, unsqueeze_292, unsqueeze_293, unsqueeze_294, unsqueeze_295
#   out_42 => relu_22
#   out_43 => add_84
# Graph fragment:
#   %convolution_35 : Tensor "f32[10, 960, 56, 56][3010560, 1, 53760, 960]cuda:0" = PlaceHolder[target=convolution_35]
#   %arg177_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg177_1]
#   %arg178_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg178_1]
#   %arg179_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg179_1]
#   %arg180_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg180_1]
#   %buf83 : Tensor "f32[31360, 960][960, 1]cuda:0" = PlaceHolder[target=buf83]
#   %arg182_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg182_1]
#   %arg183_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg183_1]
#   %arg184_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg184_1]
#   %arg185_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg185_1]
#   %unsqueeze_280 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg177_1, -1), kwargs = {})
#   %unsqueeze_281 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_280, -1), kwargs = {})
#   %sub_35 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %add_80 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg178_1, 1e-05), kwargs = {})
#   %sqrt_35 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_80,), kwargs = {})
#   %reciprocal_35 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_35,), kwargs = {})
#   %mul_105 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_35, 1), kwargs = {})
#   %unsqueeze_282 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_105, -1), kwargs = {})
#   %unsqueeze_283 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_282, -1), kwargs = {})
#   %mul_106 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %unsqueeze_284 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg179_1, -1), kwargs = {})
#   %unsqueeze_285 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_284, -1), kwargs = {})
#   %mul_107 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_285), kwargs = {})
#   %unsqueeze_286 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg180_1, -1), kwargs = {})
#   %unsqueeze_287 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_286, -1), kwargs = {})
#   %add_81 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_287), kwargs = {})
#   %relu_22 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_81,), kwargs = {})
#   %convolution_36 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_75, %arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_288 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg182_1, -1), kwargs = {})
#   %unsqueeze_289 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_288, -1), kwargs = {})
#   %sub_36 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_289), kwargs = {})
#   %add_82 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg183_1, 1e-05), kwargs = {})
#   %sqrt_36 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_82,), kwargs = {})
#   %reciprocal_36 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_36,), kwargs = {})
#   %mul_108 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_36, 1), kwargs = {})
#   %unsqueeze_290 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_108, -1), kwargs = {})
#   %unsqueeze_291 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_290, -1), kwargs = {})
#   %mul_109 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %unsqueeze_292 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg184_1, -1), kwargs = {})
#   %unsqueeze_293 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_292, -1), kwargs = {})
#   %mul_110 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %unsqueeze_294 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg185_1, -1), kwargs = {})
#   %unsqueeze_295 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_294, -1), kwargs = {})
#   %add_83 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
#   %add_84 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_22, %add_83), kwargs = {})
#   return %add_84
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 481720320}, 'kernel_num_gb': 0.36129792, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 960)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = tl.sqrt_rn(tmp22)
    tmp24 = (tmp7 / tmp23)
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, None)


def get_args():
    arg_0 = rand_strided((10, 960, 56, 56), (3010560, 1, 53760, 960), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((31360, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 30105600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.36129792
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/6l/c6ldffo5giaw6bpxbryggmy4ww5snfxtbxybbx2qfl3z4kmme5li.py
# Topologically Sorted Source Nodes: [batch_norm_39, out_46, out_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_39 => add_89, add_90, mul_117, mul_118, mul_119, reciprocal_39, sqrt_39, sub_39, unsqueeze_312, unsqueeze_313, unsqueeze_314, unsqueeze_315, unsqueeze_316, unsqueeze_317, unsqueeze_318, unsqueeze_319
#   out_46 => relu_24
#   out_47 => add_91
# Graph fragment:
#   %convolution_39 : Tensor "f32[10, 960, 56, 56][3010560, 1, 53760, 960]cuda:0" = PlaceHolder[target=convolution_39]
#   %arg197_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg197_1]
#   %arg198_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg198_1]
#   %arg199_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg199_1]
#   %arg200_1 : Tensor "f32[960][1]cuda:0" = PlaceHolder[target=arg200_1]
#   %add_84 : Tensor "f32[10, 960, 56, 56][3010560, 1, 53760, 960]cuda:0" = PlaceHolder[target=add_84]
#   %unsqueeze_312 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg197_1, -1), kwargs = {})
#   %unsqueeze_313 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_312, -1), kwargs = {})
#   %sub_39 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %add_89 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg198_1, 1e-05), kwargs = {})
#   %sqrt_39 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_89,), kwargs = {})
#   %reciprocal_39 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_39,), kwargs = {})
#   %mul_117 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_39, 1), kwargs = {})
#   %unsqueeze_314 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_117, -1), kwargs = {})
#   %unsqueeze_315 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_314, -1), kwargs = {})
#   %mul_118 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %unsqueeze_316 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg199_1, -1), kwargs = {})
#   %unsqueeze_317 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_316, -1), kwargs = {})
#   %mul_119 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %unsqueeze_318 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg200_1, -1), kwargs = {})
#   %unsqueeze_319 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_318, -1), kwargs = {})
#   %add_90 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_24 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_90,), kwargs = {})
#   %add_91 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_24, %add_84), kwargs = {})
#   return %add_91
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 481704960}, 'kernel_num_gb': 0.36128256, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 960)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 960, 56, 56), (3010560, 1, 53760, 960), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 960, 56, 56), (3010560, 1, 53760, 960), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 30105600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.36128256
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ku/ckuokdzvxqdcdj2khkp3pzsnjflje7ljeeicligulhcoudvs323v.py
# Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution]
# Source node to ATen node mapping:
#   batch_norm_42 => add_96, add_97, mul_126, mul_127, mul_128, reciprocal_42, sqrt_42, sub_42, unsqueeze_336, unsqueeze_337, unsqueeze_338, unsqueeze_339, unsqueeze_340, unsqueeze_341, unsqueeze_342, unsqueeze_343
#   conv2d_43 => convolution_43
#   out_50 => relu_26
#   out_51 => add_98
# Graph fragment:
#   %buf98 : Tensor "f32[10, 960, 56, 56][3010560, 1, 53760, 960]cuda:0" = PlaceHolder[target=buf98]
#   %arg216_1 : Tensor "f32[1024, 960, 1, 1][960, 1, 1, 1]cuda:0" = PlaceHolder[target=arg216_1]
#   %unsqueeze_336 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg212_1, -1), kwargs = {})
#   %unsqueeze_337 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_336, -1), kwargs = {})
#   %sub_42 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %add_96 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg213_1, 1e-05), kwargs = {})
#   %sqrt_42 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_96,), kwargs = {})
#   %reciprocal_42 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_42,), kwargs = {})
#   %mul_126 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_42, 1), kwargs = {})
#   %unsqueeze_338 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_126, -1), kwargs = {})
#   %unsqueeze_339 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_338, -1), kwargs = {})
#   %mul_127 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %unsqueeze_340 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg214_1, -1), kwargs = {})
#   %unsqueeze_341 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_340, -1), kwargs = {})
#   %mul_128 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %unsqueeze_342 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg215_1, -1), kwargs = {})
#   %unsqueeze_343 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_342, -1), kwargs = {})
#   %add_97 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %relu_26 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_97,), kwargs = {})
#   %add_98 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_26, %add_91), kwargs = {})
#   %convolution_43 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_98, %arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf99
triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.25280512, 'kernel_flop': 61656268800, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24(arg_A, arg_B, out_ptr0):
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

    M = 31360
    N = 1024
    K = 960
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 960
    stride_ak = 1
    stride_bk = 1
    stride_bn = 960

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
        xindex = idx_n + 960*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1024*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 960*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 960, 56, 56), (3010560, 1, 53760, 960), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 3920, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.25280512
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/iv/civklii6knhzpzhqbzomwszsnsc6luljyazqqydpradet4e6xfpv.py
# Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43, batch_norm_43, x_41, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   batch_norm_42 => add_96, add_97, mul_126, mul_127, mul_128, reciprocal_42, sqrt_42, sub_42, unsqueeze_336, unsqueeze_337, unsqueeze_338, unsqueeze_339, unsqueeze_340, unsqueeze_341, unsqueeze_342, unsqueeze_343
#   batch_norm_43 => add_100, add_99, mul_129, mul_130, mul_131, reciprocal_43, sqrt_43, sub_43, unsqueeze_344, unsqueeze_345, unsqueeze_346, unsqueeze_347, unsqueeze_348, unsqueeze_349, unsqueeze_350, unsqueeze_351
#   conv2d_43 => convolution_43
#   out_50 => relu_26
#   out_51 => add_98
#   x_41 => relu_27
#   x_42 => mean
# Graph fragment:
#   %buf99 : Tensor "f32[31360, 1024][1024, 1]cuda:0" = PlaceHolder[target=buf99]
#   %arg217_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg217_1]
#   %arg218_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg218_1]
#   %arg219_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg219_1]
#   %arg220_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg220_1]
#   %buf100 : Tensor "f32[10, 1024, 1, 1][1024, 1, 10240, 10240]cuda:0" = PlaceHolder[target=buf100]
#   %unsqueeze_336 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg212_1, -1), kwargs = {})
#   %unsqueeze_337 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_336, -1), kwargs = {})
#   %sub_42 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %add_96 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg213_1, 1e-05), kwargs = {})
#   %sqrt_42 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_96,), kwargs = {})
#   %reciprocal_42 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_42,), kwargs = {})
#   %mul_126 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_42, 1), kwargs = {})
#   %unsqueeze_338 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_126, -1), kwargs = {})
#   %unsqueeze_339 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_338, -1), kwargs = {})
#   %mul_127 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %unsqueeze_340 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg214_1, -1), kwargs = {})
#   %unsqueeze_341 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_340, -1), kwargs = {})
#   %mul_128 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %unsqueeze_342 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg215_1, -1), kwargs = {})
#   %unsqueeze_343 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_342, -1), kwargs = {})
#   %add_97 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %relu_26 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_97,), kwargs = {})
#   %add_98 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_26, %add_91), kwargs = {})
#   %convolution_43 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_98, %arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_344 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg217_1, -1), kwargs = {})
#   %unsqueeze_345 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_344, -1), kwargs = {})
#   %sub_43 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %add_99 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg218_1, 1e-05), kwargs = {})
#   %sqrt_43 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_99,), kwargs = {})
#   %reciprocal_43 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_43,), kwargs = {})
#   %mul_129 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_43, 1), kwargs = {})
#   %unsqueeze_346 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_129, -1), kwargs = {})
#   %unsqueeze_347 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_346, -1), kwargs = {})
#   %mul_130 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %unsqueeze_348 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg219_1, -1), kwargs = {})
#   %unsqueeze_349 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_348, -1), kwargs = {})
#   %mul_131 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %unsqueeze_350 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg220_1, -1), kwargs = {})
#   %unsqueeze_351 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_350, -1), kwargs = {})
#   %add_100 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_27 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_100,), kwargs = {})
#   %mean : Tensor "f32[10, 1024, 1, 1][1024, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_27, [-1, -2], True), kwargs = {})
#   return %buf100,%mean
triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25', '''
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
    size_hints={'x': 16384, 'r0_': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 128548864, 'r0_': 0}, 'kernel_num_gb': 0.128548864, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10240
    r0_numel = 3136
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 3211264*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.full([1, 1], 3136.0, tl.float32)
    tmp22 = (tmp19 / tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)


def get_args():
    arg_0 = rand_strided((10, 1024, 1, 1), (1024, 1, 10240, 10240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((31360, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 10240, 3136,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.128548864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/gm/cgmcnuvwojy4cpz4r2ix7nrh53ttr2jouzkv45xoy4peafyzysiy.py
# Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43, batch_norm_43, x_41, x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution, aten.mean, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   batch_norm_42 => add_96, add_97, mul_126, mul_127, mul_128, reciprocal_42, sqrt_42, sub_42, unsqueeze_336, unsqueeze_337, unsqueeze_338, unsqueeze_339, unsqueeze_340, unsqueeze_341, unsqueeze_342, unsqueeze_343
#   batch_norm_43 => add_100, add_99, mul_129, mul_130, mul_131, reciprocal_43, sqrt_43, sub_43, unsqueeze_344, unsqueeze_345, unsqueeze_346, unsqueeze_347, unsqueeze_348, unsqueeze_349, unsqueeze_350, unsqueeze_351
#   conv2d_43 => convolution_43
#   out_50 => relu_26
#   out_51 => add_98
#   x_41 => relu_27
#   x_42 => mean
#   x_43 => view_26
#   x_44 => addmm, permute_13
# Graph fragment:
#   %arg222_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg222_1]
#   %mean : Tensor "f32[10, 1024, 1, 1][1024, 1, 10240, 10240]cuda:0" = PlaceHolder[target=mean]
#   %arg221_1 : Tensor "f32[1000, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg221_1]
#   %unsqueeze_336 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg212_1, -1), kwargs = {})
#   %unsqueeze_337 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_336, -1), kwargs = {})
#   %sub_42 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %add_96 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg213_1, 1e-05), kwargs = {})
#   %sqrt_42 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_96,), kwargs = {})
#   %reciprocal_42 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_42,), kwargs = {})
#   %mul_126 : Tensor "f32[960][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_42, 1), kwargs = {})
#   %unsqueeze_338 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_126, -1), kwargs = {})
#   %unsqueeze_339 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_338, -1), kwargs = {})
#   %mul_127 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %unsqueeze_340 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg214_1, -1), kwargs = {})
#   %unsqueeze_341 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_340, -1), kwargs = {})
#   %mul_128 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %unsqueeze_342 : Tensor "f32[960, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg215_1, -1), kwargs = {})
#   %unsqueeze_343 : Tensor "f32[960, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_342, -1), kwargs = {})
#   %add_97 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %relu_26 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_97,), kwargs = {})
#   %add_98 : Tensor "f32[10, 960, 56, 56][3010560, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_26, %add_91), kwargs = {})
#   %convolution_43 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_98, %arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_344 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg217_1, -1), kwargs = {})
#   %unsqueeze_345 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_344, -1), kwargs = {})
#   %sub_43 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %add_99 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg218_1, 1e-05), kwargs = {})
#   %sqrt_43 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_99,), kwargs = {})
#   %reciprocal_43 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_43,), kwargs = {})
#   %mul_129 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_43, 1), kwargs = {})
#   %unsqueeze_346 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_129, -1), kwargs = {})
#   %unsqueeze_347 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_346, -1), kwargs = {})
#   %mul_130 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %unsqueeze_348 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg219_1, -1), kwargs = {})
#   %unsqueeze_349 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_348, -1), kwargs = {})
#   %mul_131 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %unsqueeze_350 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg220_1, -1), kwargs = {})
#   %unsqueeze_351 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_350, -1), kwargs = {})
#   %add_100 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_27 : Tensor "f32[10, 1024, 56, 56][3211264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_100,), kwargs = {})
#   %mean : Tensor "f32[10, 1024, 1, 1][1024, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_27, [-1, -2], True), kwargs = {})
#   %view_26 : Tensor "f32[10, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [10, -1]), kwargs = {})
#   %permute_13 : Tensor "f32[1024, 1000][1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg221_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "f32[10, 1000][1000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg222_1, %view_26, %permute_13), kwargs = {})
#   return %addmm
triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00421696, 'kernel_flop': 20480000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26(in_ptr0, arg_A, arg_B, out_ptr0):
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
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_1 = rand_strided((10, 1024, 1, 1), (1024, 1, 10240, 10240), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 32, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00421696
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1 = args
        args.clear()
        assert_size_stride(arg0_1, (24, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (10, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg2_1, (24, ), (1, ))
        assert_size_stride(arg3_1, (24, ), (1, ))
        assert_size_stride(arg4_1, (24, ), (1, ))
        assert_size_stride(arg5_1, (24, ), (1, ))
        assert_size_stride(arg6_1, (60, 8, 1, 1), (8, 1, 1, 1))
        assert_size_stride(arg7_1, (60, ), (1, ))
        assert_size_stride(arg8_1, (60, ), (1, ))
        assert_size_stride(arg9_1, (60, ), (1, ))
        assert_size_stride(arg10_1, (60, ), (1, ))
        assert_size_stride(arg11_1, (60, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg12_1, (60, ), (1, ))
        assert_size_stride(arg13_1, (60, ), (1, ))
        assert_size_stride(arg14_1, (60, ), (1, ))
        assert_size_stride(arg15_1, (60, ), (1, ))
        assert_size_stride(arg16_1, (240, 20, 1, 1), (20, 1, 1, 1))
        assert_size_stride(arg17_1, (240, ), (1, ))
        assert_size_stride(arg18_1, (240, ), (1, ))
        assert_size_stride(arg19_1, (240, ), (1, ))
        assert_size_stride(arg20_1, (240, ), (1, ))
        assert_size_stride(arg21_1, (240, 24, 1, 1), (24, 1, 1, 1))
        assert_size_stride(arg22_1, (240, ), (1, ))
        assert_size_stride(arg23_1, (240, ), (1, ))
        assert_size_stride(arg24_1, (240, ), (1, ))
        assert_size_stride(arg25_1, (240, ), (1, ))
        assert_size_stride(arg26_1, (60, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg27_1, (60, ), (1, ))
        assert_size_stride(arg28_1, (60, ), (1, ))
        assert_size_stride(arg29_1, (60, ), (1, ))
        assert_size_stride(arg30_1, (60, ), (1, ))
        assert_size_stride(arg31_1, (60, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg32_1, (60, ), (1, ))
        assert_size_stride(arg33_1, (60, ), (1, ))
        assert_size_stride(arg34_1, (60, ), (1, ))
        assert_size_stride(arg35_1, (60, ), (1, ))
        assert_size_stride(arg36_1, (240, 20, 1, 1), (20, 1, 1, 1))
        assert_size_stride(arg37_1, (240, ), (1, ))
        assert_size_stride(arg38_1, (240, ), (1, ))
        assert_size_stride(arg39_1, (240, ), (1, ))
        assert_size_stride(arg40_1, (240, ), (1, ))
        assert_size_stride(arg41_1, (60, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg42_1, (60, ), (1, ))
        assert_size_stride(arg43_1, (60, ), (1, ))
        assert_size_stride(arg44_1, (60, ), (1, ))
        assert_size_stride(arg45_1, (60, ), (1, ))
        assert_size_stride(arg46_1, (60, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg47_1, (60, ), (1, ))
        assert_size_stride(arg48_1, (60, ), (1, ))
        assert_size_stride(arg49_1, (60, ), (1, ))
        assert_size_stride(arg50_1, (60, ), (1, ))
        assert_size_stride(arg51_1, (240, 20, 1, 1), (20, 1, 1, 1))
        assert_size_stride(arg52_1, (240, ), (1, ))
        assert_size_stride(arg53_1, (240, ), (1, ))
        assert_size_stride(arg54_1, (240, ), (1, ))
        assert_size_stride(arg55_1, (240, ), (1, ))
        assert_size_stride(arg56_1, (120, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg57_1, (120, ), (1, ))
        assert_size_stride(arg58_1, (120, ), (1, ))
        assert_size_stride(arg59_1, (120, ), (1, ))
        assert_size_stride(arg60_1, (120, ), (1, ))
        assert_size_stride(arg61_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg62_1, (120, ), (1, ))
        assert_size_stride(arg63_1, (120, ), (1, ))
        assert_size_stride(arg64_1, (120, ), (1, ))
        assert_size_stride(arg65_1, (120, ), (1, ))
        assert_size_stride(arg66_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg67_1, (480, ), (1, ))
        assert_size_stride(arg68_1, (480, ), (1, ))
        assert_size_stride(arg69_1, (480, ), (1, ))
        assert_size_stride(arg70_1, (480, ), (1, ))
        assert_size_stride(arg71_1, (480, 240, 1, 1), (240, 1, 1, 1))
        assert_size_stride(arg72_1, (480, ), (1, ))
        assert_size_stride(arg73_1, (480, ), (1, ))
        assert_size_stride(arg74_1, (480, ), (1, ))
        assert_size_stride(arg75_1, (480, ), (1, ))
        assert_size_stride(arg76_1, (120, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg77_1, (120, ), (1, ))
        assert_size_stride(arg78_1, (120, ), (1, ))
        assert_size_stride(arg79_1, (120, ), (1, ))
        assert_size_stride(arg80_1, (120, ), (1, ))
        assert_size_stride(arg81_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg82_1, (120, ), (1, ))
        assert_size_stride(arg83_1, (120, ), (1, ))
        assert_size_stride(arg84_1, (120, ), (1, ))
        assert_size_stride(arg85_1, (120, ), (1, ))
        assert_size_stride(arg86_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg87_1, (480, ), (1, ))
        assert_size_stride(arg88_1, (480, ), (1, ))
        assert_size_stride(arg89_1, (480, ), (1, ))
        assert_size_stride(arg90_1, (480, ), (1, ))
        assert_size_stride(arg91_1, (120, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg92_1, (120, ), (1, ))
        assert_size_stride(arg93_1, (120, ), (1, ))
        assert_size_stride(arg94_1, (120, ), (1, ))
        assert_size_stride(arg95_1, (120, ), (1, ))
        assert_size_stride(arg96_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg97_1, (120, ), (1, ))
        assert_size_stride(arg98_1, (120, ), (1, ))
        assert_size_stride(arg99_1, (120, ), (1, ))
        assert_size_stride(arg100_1, (120, ), (1, ))
        assert_size_stride(arg101_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg102_1, (480, ), (1, ))
        assert_size_stride(arg103_1, (480, ), (1, ))
        assert_size_stride(arg104_1, (480, ), (1, ))
        assert_size_stride(arg105_1, (480, ), (1, ))
        assert_size_stride(arg106_1, (120, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg107_1, (120, ), (1, ))
        assert_size_stride(arg108_1, (120, ), (1, ))
        assert_size_stride(arg109_1, (120, ), (1, ))
        assert_size_stride(arg110_1, (120, ), (1, ))
        assert_size_stride(arg111_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg112_1, (120, ), (1, ))
        assert_size_stride(arg113_1, (120, ), (1, ))
        assert_size_stride(arg114_1, (120, ), (1, ))
        assert_size_stride(arg115_1, (120, ), (1, ))
        assert_size_stride(arg116_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg117_1, (480, ), (1, ))
        assert_size_stride(arg118_1, (480, ), (1, ))
        assert_size_stride(arg119_1, (480, ), (1, ))
        assert_size_stride(arg120_1, (480, ), (1, ))
        assert_size_stride(arg121_1, (120, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg122_1, (120, ), (1, ))
        assert_size_stride(arg123_1, (120, ), (1, ))
        assert_size_stride(arg124_1, (120, ), (1, ))
        assert_size_stride(arg125_1, (120, ), (1, ))
        assert_size_stride(arg126_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg127_1, (120, ), (1, ))
        assert_size_stride(arg128_1, (120, ), (1, ))
        assert_size_stride(arg129_1, (120, ), (1, ))
        assert_size_stride(arg130_1, (120, ), (1, ))
        assert_size_stride(arg131_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg132_1, (480, ), (1, ))
        assert_size_stride(arg133_1, (480, ), (1, ))
        assert_size_stride(arg134_1, (480, ), (1, ))
        assert_size_stride(arg135_1, (480, ), (1, ))
        assert_size_stride(arg136_1, (120, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg137_1, (120, ), (1, ))
        assert_size_stride(arg138_1, (120, ), (1, ))
        assert_size_stride(arg139_1, (120, ), (1, ))
        assert_size_stride(arg140_1, (120, ), (1, ))
        assert_size_stride(arg141_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg142_1, (120, ), (1, ))
        assert_size_stride(arg143_1, (120, ), (1, ))
        assert_size_stride(arg144_1, (120, ), (1, ))
        assert_size_stride(arg145_1, (120, ), (1, ))
        assert_size_stride(arg146_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg147_1, (480, ), (1, ))
        assert_size_stride(arg148_1, (480, ), (1, ))
        assert_size_stride(arg149_1, (480, ), (1, ))
        assert_size_stride(arg150_1, (480, ), (1, ))
        assert_size_stride(arg151_1, (120, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg152_1, (120, ), (1, ))
        assert_size_stride(arg153_1, (120, ), (1, ))
        assert_size_stride(arg154_1, (120, ), (1, ))
        assert_size_stride(arg155_1, (120, ), (1, ))
        assert_size_stride(arg156_1, (120, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg157_1, (120, ), (1, ))
        assert_size_stride(arg158_1, (120, ), (1, ))
        assert_size_stride(arg159_1, (120, ), (1, ))
        assert_size_stride(arg160_1, (120, ), (1, ))
        assert_size_stride(arg161_1, (480, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg162_1, (480, ), (1, ))
        assert_size_stride(arg163_1, (480, ), (1, ))
        assert_size_stride(arg164_1, (480, ), (1, ))
        assert_size_stride(arg165_1, (480, ), (1, ))
        assert_size_stride(arg166_1, (240, 160, 1, 1), (160, 1, 1, 1))
        assert_size_stride(arg167_1, (240, ), (1, ))
        assert_size_stride(arg168_1, (240, ), (1, ))
        assert_size_stride(arg169_1, (240, ), (1, ))
        assert_size_stride(arg170_1, (240, ), (1, ))
        assert_size_stride(arg171_1, (240, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg172_1, (240, ), (1, ))
        assert_size_stride(arg173_1, (240, ), (1, ))
        assert_size_stride(arg174_1, (240, ), (1, ))
        assert_size_stride(arg175_1, (240, ), (1, ))
        assert_size_stride(arg176_1, (960, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg177_1, (960, ), (1, ))
        assert_size_stride(arg178_1, (960, ), (1, ))
        assert_size_stride(arg179_1, (960, ), (1, ))
        assert_size_stride(arg180_1, (960, ), (1, ))
        assert_size_stride(arg181_1, (960, 480, 1, 1), (480, 1, 1, 1))
        assert_size_stride(arg182_1, (960, ), (1, ))
        assert_size_stride(arg183_1, (960, ), (1, ))
        assert_size_stride(arg184_1, (960, ), (1, ))
        assert_size_stride(arg185_1, (960, ), (1, ))
        assert_size_stride(arg186_1, (240, 320, 1, 1), (320, 1, 1, 1))
        assert_size_stride(arg187_1, (240, ), (1, ))
        assert_size_stride(arg188_1, (240, ), (1, ))
        assert_size_stride(arg189_1, (240, ), (1, ))
        assert_size_stride(arg190_1, (240, ), (1, ))
        assert_size_stride(arg191_1, (240, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg192_1, (240, ), (1, ))
        assert_size_stride(arg193_1, (240, ), (1, ))
        assert_size_stride(arg194_1, (240, ), (1, ))
        assert_size_stride(arg195_1, (240, ), (1, ))
        assert_size_stride(arg196_1, (960, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg197_1, (960, ), (1, ))
        assert_size_stride(arg198_1, (960, ), (1, ))
        assert_size_stride(arg199_1, (960, ), (1, ))
        assert_size_stride(arg200_1, (960, ), (1, ))
        assert_size_stride(arg201_1, (240, 320, 1, 1), (320, 1, 1, 1))
        assert_size_stride(arg202_1, (240, ), (1, ))
        assert_size_stride(arg203_1, (240, ), (1, ))
        assert_size_stride(arg204_1, (240, ), (1, ))
        assert_size_stride(arg205_1, (240, ), (1, ))
        assert_size_stride(arg206_1, (240, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg207_1, (240, ), (1, ))
        assert_size_stride(arg208_1, (240, ), (1, ))
        assert_size_stride(arg209_1, (240, ), (1, ))
        assert_size_stride(arg210_1, (240, ), (1, ))
        assert_size_stride(arg211_1, (960, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg212_1, (960, ), (1, ))
        assert_size_stride(arg213_1, (960, ), (1, ))
        assert_size_stride(arg214_1, (960, ), (1, ))
        assert_size_stride(arg215_1, (960, ), (1, ))
        assert_size_stride(arg216_1, (1024, 960, 1, 1), (960, 1, 1, 1))
        assert_size_stride(arg217_1, (1024, ), (1, ))
        assert_size_stride(arg218_1, (1024, ), (1, ))
        assert_size_stride(arg219_1, (1024, ), (1, ))
        assert_size_stride(arg220_1, (1024, ), (1, ))
        assert_size_stride(arg221_1, (1000, 1024), (1024, 1))
        assert_size_stride(arg222_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((10, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 30, 50176, stream=stream0)
            del arg1_1
            buf1 = empty_strided_cuda((24, 3, 3, 3), (27, 1, 9, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 72, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (10, 24, 112, 112), (301056, 1, 2688, 24), 'unknown_op')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3010560, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((10, 24, 56, 56), (75264, 1, 1344, 24), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 752640, stream=stream0)
            del buf3
            buf5 = empty_strided_cuda((10, 60, 56, 56), (188160, 1, 3360, 60), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_4.run(buf4, arg6_1, buf5, 245, 1, 3, stream=stream0)
            del arg6_1
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_1, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 1881600, stream=stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            # Topologically Sorted Source Nodes: [batch_norm_1, out, conv2d_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf7 = extern_kernels.convolution(buf6, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
            assert_size_stride(buf7, (10, 60, 56, 56), (188160, 1, 3360, 60), 'unknown_op')
            del arg11_1
            buf8 = reinterpret_tensor(buf6, (10, 20, 3, 56, 56), (188160, 9408, 3136, 56, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [out_1, x_2, transpose, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, buf8, 600, 3136, stream=stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            buf9 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [out_1, x_2, transpose, x_3, x_4, conv2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7.run(buf8, buf9, 600, 3136, stream=stream0)
            del buf8
            # Topologically Sorted Source Nodes: [out_1, x_2, transpose, x_3, x_4, conv2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf10 = extern_kernels.convolution(buf9, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf10, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg16_1
            del buf9
            buf11 = empty_strided_cuda((31360, 240), (240, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_8.run(buf4, arg21_1, buf11, 490, 1, 1, stream=stream0)
            del arg21_1
            del buf4
            buf12 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_3, out_2, input_1, input_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.run(buf12, arg17_1, arg18_1, arg19_1, arg20_1, buf11, arg22_1, arg23_1, arg24_1, arg25_1, 7526400, stream=stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del buf11
            # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
            buf13 = extern_kernels.convolution(buf12, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf13, (10, 60, 56, 56), (188160, 1, 3360, 60), 'unknown_op')
            del arg26_1
            buf14 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_5, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf14, arg27_1, arg28_1, arg29_1, arg30_1, 1881600, stream=stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [batch_norm_5, out_4, conv2d_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf15 = extern_kernels.convolution(buf14, arg31_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
            assert_size_stride(buf15, (10, 60, 56, 56), (188160, 1, 3360, 60), 'unknown_op')
            del arg31_1
            buf16 = reinterpret_tensor(buf14, (10, 20, 3, 56, 56), (188160, 9408, 3136, 56, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [out_5, x_5, transpose_1, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6.run(buf15, arg32_1, arg33_1, arg34_1, arg35_1, buf16, 600, 3136, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            buf17 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [out_5, x_5, transpose_1, x_6, x_7, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7.run(buf16, buf17, 600, 3136, stream=stream0)
            del buf16
            # Topologically Sorted Source Nodes: [out_5, x_5, transpose_1, x_6, x_7, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf18 = extern_kernels.convolution(buf17, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf18, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg36_1
            del buf17
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_7, out_6, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf19, arg37_1, arg38_1, arg39_1, arg40_1, buf12, 7526400, stream=stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            del buf12
            # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
            buf20 = extern_kernels.convolution(buf19, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf20, (10, 60, 56, 56), (188160, 1, 3360, 60), 'unknown_op')
            del arg41_1
            buf21 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_8, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf21, arg42_1, arg43_1, arg44_1, arg45_1, 1881600, stream=stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            # Topologically Sorted Source Nodes: [batch_norm_8, out_8, conv2d_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf22 = extern_kernels.convolution(buf21, arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
            assert_size_stride(buf22, (10, 60, 56, 56), (188160, 1, 3360, 60), 'unknown_op')
            del arg46_1
            buf23 = reinterpret_tensor(buf21, (10, 20, 3, 56, 56), (188160, 9408, 3136, 56, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [out_9, x_8, transpose_2, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6.run(buf22, arg47_1, arg48_1, arg49_1, arg50_1, buf23, 600, 3136, stream=stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            buf24 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [out_9, x_8, transpose_2, x_9, x_10, conv2d_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7.run(buf23, buf24, 600, 3136, stream=stream0)
            del buf23
            # Topologically Sorted Source Nodes: [out_9, x_8, transpose_2, x_9, x_10, conv2d_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf25 = extern_kernels.convolution(buf24, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf25, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg51_1
            del buf24
            buf26 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_10, out_10, out_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf26, arg52_1, arg53_1, arg54_1, arg55_1, buf19, 7526400, stream=stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            del buf19
            # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
            buf27 = extern_kernels.convolution(buf26, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf27, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg56_1
            buf28 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf28, arg57_1, arg58_1, arg59_1, arg60_1, 3763200, stream=stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            # Topologically Sorted Source Nodes: [batch_norm_11, out_12, conv2d_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf29 = extern_kernels.convolution(buf28, arg61_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf29, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg61_1
            buf30 = reinterpret_tensor(buf28, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [out_13, x_11, transpose_3, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf29, arg62_1, arg63_1, arg64_1, arg65_1, buf30, 1200, 3136, stream=stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            buf31 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [out_13, x_11, transpose_3, x_12, x_13, conv2d_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf30, buf31, 1200, 3136, stream=stream0)
            del buf30
            # Topologically Sorted Source Nodes: [out_13, x_11, transpose_3, x_12, x_13, conv2d_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf32 = extern_kernels.convolution(buf31, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf32, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg66_1
            buf33 = empty_strided_cuda((31360, 480), (480, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_14.run(buf26, arg71_1, buf33, 1960, 1, 1, stream=stream0)
            del arg71_1
            del buf26
            buf34 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_13, out_14, input_3, input_4, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf34, arg67_1, arg68_1, arg69_1, arg70_1, buf33, arg72_1, arg73_1, arg74_1, arg75_1, 15052800, stream=stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            del buf33
            buf35 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_16.run(buf34, arg76_1, buf35, 123, 2, 3, stream=stream0)
            del arg76_1
            buf36 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf36, arg77_1, arg78_1, arg79_1, arg80_1, 3763200, stream=stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            # Topologically Sorted Source Nodes: [batch_norm_15, out_16, conv2d_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf37 = extern_kernels.convolution(buf36, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf37, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg81_1
            buf38 = reinterpret_tensor(buf36, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [out_17, x_14, transpose_4, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf37, arg82_1, arg83_1, arg84_1, arg85_1, buf38, 1200, 3136, stream=stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            buf39 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [out_17, x_14, transpose_4, x_15, x_16, conv2d_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf38, buf39, 1200, 3136, stream=stream0)
            del buf38
            # Topologically Sorted Source Nodes: [out_17, x_14, transpose_4, x_15, x_16, conv2d_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf40 = extern_kernels.convolution(buf39, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf40, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg86_1
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf41, arg87_1, arg88_1, arg89_1, arg90_1, buf34, 15052800, stream=stream0)
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            del buf34
            buf42 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_16.run(buf41, arg91_1, buf42, 123, 2, 3, stream=stream0)
            del arg91_1
            buf43 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_18, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf43, arg92_1, arg93_1, arg94_1, arg95_1, 3763200, stream=stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            # Topologically Sorted Source Nodes: [batch_norm_18, out_20, conv2d_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf44 = extern_kernels.convolution(buf43, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf44, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg96_1
            buf45 = reinterpret_tensor(buf43, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [out_21, x_17, transpose_5, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf44, arg97_1, arg98_1, arg99_1, arg100_1, buf45, 1200, 3136, stream=stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            buf46 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [out_21, x_17, transpose_5, x_18, x_19, conv2d_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf45, buf46, 1200, 3136, stream=stream0)
            del buf45
            # Topologically Sorted Source Nodes: [out_21, x_17, transpose_5, x_18, x_19, conv2d_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf47 = extern_kernels.convolution(buf46, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf47, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg101_1
            buf48 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_20, out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf48, arg102_1, arg103_1, arg104_1, arg105_1, buf41, 15052800, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            del arg105_1
            del buf41
            buf49 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_16.run(buf48, arg106_1, buf49, 123, 2, 3, stream=stream0)
            del arg106_1
            buf50 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_21, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf50, arg107_1, arg108_1, arg109_1, arg110_1, 3763200, stream=stream0)
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [batch_norm_21, out_24, conv2d_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf51 = extern_kernels.convolution(buf50, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf51, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg111_1
            buf52 = reinterpret_tensor(buf50, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [out_25, x_20, transpose_6, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf51, arg112_1, arg113_1, arg114_1, arg115_1, buf52, 1200, 3136, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            buf53 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [out_25, x_20, transpose_6, x_21, x_22, conv2d_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf52, buf53, 1200, 3136, stream=stream0)
            del buf52
            # Topologically Sorted Source Nodes: [out_25, x_20, transpose_6, x_21, x_22, conv2d_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf54 = extern_kernels.convolution(buf53, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf54, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg116_1
            buf55 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_23, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf55, arg117_1, arg118_1, arg119_1, arg120_1, buf48, 15052800, stream=stream0)
            del arg117_1
            del arg118_1
            del arg119_1
            del arg120_1
            del buf48
            buf56 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_16.run(buf55, arg121_1, buf56, 123, 2, 3, stream=stream0)
            del arg121_1
            buf57 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_24, out_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf57, arg122_1, arg123_1, arg124_1, arg125_1, 3763200, stream=stream0)
            del arg122_1
            del arg123_1
            del arg124_1
            del arg125_1
            # Topologically Sorted Source Nodes: [batch_norm_24, out_28, conv2d_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf58 = extern_kernels.convolution(buf57, arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf58, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg126_1
            buf59 = reinterpret_tensor(buf57, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [out_29, x_23, transpose_7, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf58, arg127_1, arg128_1, arg129_1, arg130_1, buf59, 1200, 3136, stream=stream0)
            del arg127_1
            del arg128_1
            del arg129_1
            del arg130_1
            buf60 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [out_29, x_23, transpose_7, x_24, x_25, conv2d_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf59, buf60, 1200, 3136, stream=stream0)
            del buf59
            # Topologically Sorted Source Nodes: [out_29, x_23, transpose_7, x_24, x_25, conv2d_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf61 = extern_kernels.convolution(buf60, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf61, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg131_1
            buf62 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_26, out_30, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf62, arg132_1, arg133_1, arg134_1, arg135_1, buf55, 15052800, stream=stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            del arg135_1
            del buf55
            buf63 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_16.run(buf62, arg136_1, buf63, 123, 2, 3, stream=stream0)
            del arg136_1
            buf64 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_27, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf64, arg137_1, arg138_1, arg139_1, arg140_1, 3763200, stream=stream0)
            del arg137_1
            del arg138_1
            del arg139_1
            del arg140_1
            # Topologically Sorted Source Nodes: [batch_norm_27, out_32, conv2d_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf65 = extern_kernels.convolution(buf64, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf65, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg141_1
            buf66 = reinterpret_tensor(buf64, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [out_33, x_26, transpose_8, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf65, arg142_1, arg143_1, arg144_1, arg145_1, buf66, 1200, 3136, stream=stream0)
            del arg142_1
            del arg143_1
            del arg144_1
            del arg145_1
            buf67 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [out_33, x_26, transpose_8, x_27, x_28, conv2d_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf66, buf67, 1200, 3136, stream=stream0)
            del buf66
            # Topologically Sorted Source Nodes: [out_33, x_26, transpose_8, x_27, x_28, conv2d_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf68 = extern_kernels.convolution(buf67, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf68, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg146_1
            buf69 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_29, out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf69, arg147_1, arg148_1, arg149_1, arg150_1, buf62, 15052800, stream=stream0)
            del arg147_1
            del arg148_1
            del arg149_1
            del arg150_1
            del buf62
            buf70 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_16.run(buf69, arg151_1, buf70, 123, 2, 3, stream=stream0)
            del arg151_1
            buf71 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_30, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf71, arg152_1, arg153_1, arg154_1, arg155_1, 3763200, stream=stream0)
            del arg152_1
            del arg153_1
            del arg154_1
            del arg155_1
            # Topologically Sorted Source Nodes: [batch_norm_30, out_36, conv2d_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf72 = extern_kernels.convolution(buf71, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
            assert_size_stride(buf72, (10, 120, 56, 56), (376320, 1, 6720, 120), 'unknown_op')
            del arg156_1
            buf73 = reinterpret_tensor(buf71, (10, 40, 3, 56, 56), (376320, 9408, 3136, 56, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [out_37, x_29, transpose_9, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12.run(buf72, arg157_1, arg158_1, arg159_1, arg160_1, buf73, 1200, 3136, stream=stream0)
            del arg157_1
            del arg158_1
            del arg159_1
            del arg160_1
            buf74 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [out_37, x_29, transpose_9, x_30, x_31, conv2d_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13.run(buf73, buf74, 1200, 3136, stream=stream0)
            del buf73
            # Topologically Sorted Source Nodes: [out_37, x_29, transpose_9, x_30, x_31, conv2d_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf75 = extern_kernels.convolution(buf74, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf75, (10, 480, 56, 56), (1505280, 1, 26880, 480), 'unknown_op')
            del arg161_1
            del buf74
            buf76 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_32, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf76, arg162_1, arg163_1, arg164_1, arg165_1, buf69, 15052800, stream=stream0)
            del arg162_1
            del arg163_1
            del arg164_1
            del arg165_1
            del buf69
            # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
            buf77 = extern_kernels.convolution(buf76, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf77, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg166_1
            buf78 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_33, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf78, arg167_1, arg168_1, arg169_1, arg170_1, 7526400, stream=stream0)
            del arg167_1
            del arg168_1
            del arg169_1
            del arg170_1
            # Topologically Sorted Source Nodes: [batch_norm_33, out_40, conv2d_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf79 = extern_kernels.convolution(buf78, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
            assert_size_stride(buf79, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg171_1
            buf80 = reinterpret_tensor(buf78, (10, 80, 3, 56, 56), (752640, 9408, 3136, 56, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [out_41, x_32, transpose_10, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19.run(buf79, arg172_1, arg173_1, arg174_1, arg175_1, buf80, 2400, 3136, stream=stream0)
            del arg172_1
            del arg173_1
            del arg174_1
            del arg175_1
            buf81 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [out_41, x_32, transpose_10, x_33, x_34, conv2d_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20.run(buf80, buf81, 2400, 3136, stream=stream0)
            del buf80
            # Topologically Sorted Source Nodes: [out_41, x_32, transpose_10, x_33, x_34, conv2d_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf82 = extern_kernels.convolution(buf81, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf82, (10, 960, 56, 56), (3010560, 1, 53760, 960), 'unknown_op')
            del arg176_1
            del buf81
            buf83 = empty_strided_cuda((31360, 960), (960, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_21.run(buf76, arg181_1, buf83, 3675, 1, 1, stream=stream0)
            del arg181_1
            del buf76
            buf84 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_35, out_42, input_5, input_6, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22.run(buf84, arg177_1, arg178_1, arg179_1, arg180_1, buf83, arg182_1, arg183_1, arg184_1, arg185_1, 30105600, stream=stream0)
            del arg177_1
            del arg178_1
            del arg179_1
            del arg180_1
            del arg182_1
            del arg183_1
            del arg184_1
            del arg185_1
            del buf83
            # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
            buf85 = extern_kernels.convolution(buf84, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf85, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg186_1
            buf86 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_37, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf86, arg187_1, arg188_1, arg189_1, arg190_1, 7526400, stream=stream0)
            del arg187_1
            del arg188_1
            del arg189_1
            del arg190_1
            # Topologically Sorted Source Nodes: [batch_norm_37, out_44, conv2d_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf87 = extern_kernels.convolution(buf86, arg191_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
            assert_size_stride(buf87, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg191_1
            buf88 = reinterpret_tensor(buf86, (10, 80, 3, 56, 56), (752640, 9408, 3136, 56, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [out_45, x_35, transpose_11, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19.run(buf87, arg192_1, arg193_1, arg194_1, arg195_1, buf88, 2400, 3136, stream=stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del arg195_1
            buf89 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [out_45, x_35, transpose_11, x_36, x_37, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20.run(buf88, buf89, 2400, 3136, stream=stream0)
            del buf88
            # Topologically Sorted Source Nodes: [out_45, x_35, transpose_11, x_36, x_37, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf90 = extern_kernels.convolution(buf89, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf90, (10, 960, 56, 56), (3010560, 1, 53760, 960), 'unknown_op')
            del arg196_1
            del buf89
            buf91 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_39, out_46, out_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf91, arg197_1, arg198_1, arg199_1, arg200_1, buf84, 30105600, stream=stream0)
            del arg197_1
            del arg198_1
            del arg199_1
            del arg200_1
            del buf84
            # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
            buf92 = extern_kernels.convolution(buf91, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf92, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg201_1
            buf93 = buf92; del buf92  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_40, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf93, arg202_1, arg203_1, arg204_1, arg205_1, 7526400, stream=stream0)
            del arg202_1
            del arg203_1
            del arg204_1
            del arg205_1
            # Topologically Sorted Source Nodes: [batch_norm_40, out_48, conv2d_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf94 = extern_kernels.convolution(buf93, arg206_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
            assert_size_stride(buf94, (10, 240, 56, 56), (752640, 1, 13440, 240), 'unknown_op')
            del arg206_1
            buf95 = reinterpret_tensor(buf93, (10, 80, 3, 56, 56), (752640, 9408, 3136, 56, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [out_49, x_38, transpose_12, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19.run(buf94, arg207_1, arg208_1, arg209_1, arg210_1, buf95, 2400, 3136, stream=stream0)
            del arg207_1
            del arg208_1
            del arg209_1
            del arg210_1
            buf96 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [out_49, x_38, transpose_12, x_39, x_40, conv2d_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20.run(buf95, buf96, 2400, 3136, stream=stream0)
            del buf95
            # Topologically Sorted Source Nodes: [out_49, x_38, transpose_12, x_39, x_40, conv2d_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.transpose, aten.clone, aten.convolution]
            buf97 = extern_kernels.convolution(buf96, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
            assert_size_stride(buf97, (10, 960, 56, 56), (3010560, 1, 53760, 960), 'unknown_op')
            del arg211_1
            del buf96
            buf98 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf98, arg212_1, arg213_1, arg214_1, arg215_1, buf91, 30105600, stream=stream0)
            del arg212_1
            del arg213_1
            del arg214_1
            del arg215_1
            del buf91
            buf99 = empty_strided_cuda((31360, 1024), (1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf98, arg216_1, buf99, 3920, 1, 1, stream=stream0)
            del arg216_1
            del buf98
            buf100 = empty_strided_cuda((10, 1024, 1, 1), (1024, 1, 10240, 10240), torch.float32)
            buf101 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43, batch_norm_43, x_41, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution, aten.mean]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25.run(buf101, buf99, arg217_1, arg218_1, arg219_1, arg220_1, 10240, 3136, stream=stream0)
            del arg217_1
            del arg218_1
            del arg219_1
            del arg220_1
            del buf99
            buf102 = empty_strided_cuda((10, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [batch_norm_42, out_50, out_51, conv2d_43, batch_norm_43, x_41, x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.convolution, aten.mean, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26.run(arg222_1, buf101, arg221_1, buf102, 32, 1, 1, stream=stream0)
            del arg221_1
            del arg222_1
            del buf101
        return (buf102, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((60, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((240, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((60, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((60, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((120, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((480, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((960, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((240, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((240, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ShuffleNetUnit, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch_size, -1, height, width)
        
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet.

        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]'''

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
    ynumel = 72
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 24)
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
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 1344) % 56)
    x1 = ((xindex // 24) % 56)
    x0 = (xindex % 24)
    x5 = xindex // 1344
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
    tmp11 = tl.load(in_ptr0 + ((-2712) + x0 + 48*x1 + 5376*x5), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2688) + x0 + 48*x1 + 5376*x5), tmp16 & xmask, other=float("-inf"))
    tmp18 = tl.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-2664) + x0 + 48*x1 + 5376*x5), tmp23 & xmask, other=float("-inf"))
    tmp25 = tl.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-24) + x0 + 48*x1 + 5376*x5), tmp30 & xmask, other=float("-inf"))
    tmp32 = tl.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 48*x1 + 5376*x5), tmp33 & xmask, other=float("-inf"))
    tmp35 = tl.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (24 + x0 + 48*x1 + 5376*x5), tmp36 & xmask, other=float("-inf"))
    tmp38 = tl.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (2664 + x0 + 48*x1 + 5376*x5), tmp43 & xmask, other=float("-inf"))
    tmp45 = tl.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2688 + x0 + 48*x1 + 5376*x5), tmp46 & xmask, other=float("-inf"))
    tmp48 = tl.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (2712 + x0 + 48*x1 + 5376*x5), tmp49 & xmask, other=float("-inf"))
    tmp51 = tl.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, xmask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_4(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 3
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 10
    IN_C = 24
    IN_H = 56
    IN_W = 56
    OUT_C = 60
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 75264
    stride_xc = 1
    stride_xh = 1344
    stride_xw = 24
    stride_wc_out = 8
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 56*idx_h + 3136*idx_c + 188160*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 60*idx_w + 3360*idx_h + 188160*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 60)
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
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 600
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 60
    y4 = (yindex % 60)
    y0 = (yindex % 20)
    y1 = ((yindex // 20) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 60*x3 + 188160*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 188160*y2), tmp15, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 600
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 60)
    y1 = yindex // 60
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 60*x2 + 188160*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_8(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
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
    N = 240
    K = 24
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 24
    stride_ak = 1
    stride_bk = 1
    stride_bn = 24

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
        xindex = idx_n + 24*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 240*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 24*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 240*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), xmask)
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = tl.sqrt_rn(tmp22)
    tmp24 = (tmp7 / tmp23)
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), xmask)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3763200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
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
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1200
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 120
    y4 = (yindex % 120)
    y0 = (yindex % 40)
    y1 = ((yindex // 40) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 120*x3 + 376320*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 376320*y2), tmp15, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1200
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 120)
    y1 = yindex // 120
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 120*x2 + 376320*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_14(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
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
    N = 480
    K = 240
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 240
    stride_ak = 1
    stride_bk = 1
    stride_bn = 240

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
        xindex = idx_n + 240*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 480*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 480*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15052800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 480)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = tl.sqrt_rn(tmp22)
    tmp24 = (tmp7 / tmp23)
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, None)

''',
    '''@triton.jit
def triton_tem_fused_convolution_16(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 3
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 10
    IN_C = 480
    IN_H = 56
    IN_W = 56
    OUT_C = 120
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 1505280
    stride_xc = 1
    stride_xh = 26880
    stride_xw = 480
    stride_wc_out = 160
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 56*idx_h + 3136*idx_c + 376320*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 120*idx_w + 6720*idx_h + 376320*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15052800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 480)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7526400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
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
def triton_poi_fused__native_batch_norm_legit_no_training_clone_transpose_view_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 240
    y4 = (yindex % 240)
    y0 = (yindex % 80)
    y1 = ((yindex // 80) % 3)
    tmp0 = tl.load(in_ptr0 + (y4 + 240*x3 + 752640*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y4), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + 3136*y1 + 9408*y0 + 752640*y2), tmp15, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 240)
    y1 = yindex // 240
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 240*x2 + 752640*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_21(arg_A, arg_B, out_ptr0):
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
    N = 960
    K = 480
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 480
    stride_ak = 1
    stride_bk = 1
    stride_bn = 480

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
        xindex = idx_n + 480*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 960*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 960*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 960)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = tl.sqrt_rn(tmp22)
    tmp24 = (tmp7 / tmp23)
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tl.store(in_out_ptr0 + (x2), tmp31, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 960)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x2), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_24(arg_A, arg_B, out_ptr0):
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

    M = 31360
    N = 1024
    K = 960
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 960
    stride_ak = 1
    stride_bk = 1
    stride_bn = 960

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
        xindex = idx_n + 960*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1024*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 960*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_red_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10240
    r0_numel = 3136
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 3211264*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = tl.maximum(tmp16, tmp15)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.full([1, 1], 3136.0, tl.float32)
    tmp22 = (tmp19 / tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_add_addmm_convolution_mean_relu_t_view_26(in_ptr0, arg_A, arg_B, out_ptr0):
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
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg0_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((60, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((240, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((60, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((60, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((120, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((480, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((120, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((120,), (1,), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((480, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((960, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((240, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((240, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1]

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
