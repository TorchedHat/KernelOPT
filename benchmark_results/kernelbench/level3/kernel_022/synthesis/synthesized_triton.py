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


# kernel path: /workspace/tmp/torchinductor_user/du/cducrdlblbsf4gq6eunicd6n4g434tsltmrvdpkdnowcimnjc5dq.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[10, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /workspace/tmp/torchinductor_user/3h/c3hauzphdunabpllro25vrvmm3cieyjubfnyjbt5hifazuo5moju.py
# Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 3, 3, 3][27, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 6912, 'x': 3456}, 'kernel_num_gb': 6.912e-06, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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
    arg_0 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 96, 9,


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
    num_gb = 6.912e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/g6/cg6z5wrmoxgidrbhztetiyonxx4r7vczu2xtmlhj4chlgojxf35r.py
# Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x => relu
# Graph fragment:
#   %convolution : Tensor "f32[10, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 48169472}, 'kernel_num_gb': 0.032113152, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
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
    arg_0 = rand_strided((10, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4014080,


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
    num_gb = 0.032113152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/op/coplp5ssmp4j44t4ifta33iluchyt7tvvukxsufyhmngvmjp5ays.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_3 => clamp_max, clamp_min
#   input_4 => convolution_2
# Graph fragment:
#   %convolution_1 : Tensor "f32[10, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = PlaceHolder[target=convolution_1]
#   %arg7_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %unsqueeze_8 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
#   %convolution_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 48169472}, 'kernel_num_gb': 0.032113152, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4014080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.032113152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/w4/cw4rbitvyf4i6mpf2lqvv5x36tpoxjiz7hp5q6exikuxgezwhxra.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_3 => clamp_max, clamp_min
#   input_4 => convolution_2
# Graph fragment:
#   %buf5 : Tensor "f32[10, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = PlaceHolder[target=buf5]
#   %arg11_1 : Tensor "f32[16, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %unsqueeze_8 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
#   %convolution_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf6
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.024086528, 'kernel_flop': 128450560, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 125440
    N = 16
    K = 32
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 32
    stride_ak = 1
    stride_bk = 1
    stride_bn = 32

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
        xindex = idx_n + 32*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 16*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((125440, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 980, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024086528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/hx/chxk6qxolqa4agal3z4ijcxdnfiz7d7cpuxtwejjxei4dagg2l5a.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_3 => clamp_max, clamp_min
#   input_4 => convolution_2
#   input_5 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   input_6 => convolution_3
# Graph fragment:
#   %buf6 : Tensor "f32[125440, 16][16, 1]cuda:0" = PlaceHolder[target=buf6]
#   %arg12_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %unsqueeze_8 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
#   %convolution_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %convolution_3 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 24084736}, 'kernel_num_gb': 0.016056576, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)


def get_args():
    arg_0 = rand_strided((10, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 2007040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016056576
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/fj/cfjn5yxjpm2fnbbbb3xxfryhvzrvy76dr3rakmpkilv5avpsgd5y.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_3 => clamp_max, clamp_min
#   input_4 => convolution_2
#   input_5 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   input_6 => convolution_3
# Graph fragment:
#   %buf7 : Tensor "f32[10, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = PlaceHolder[target=buf7]
#   %arg16_1 : Tensor "f32[96, 16, 1, 1][16, 1, 1, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %unsqueeze_8 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
#   %convolution_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %convolution_3 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf8
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6', '''
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
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.056203264, 'kernel_flop': 385351680, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 125440
    N = 96
    K = 16
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 16
    stride_ak = 1
    stride_bk = 1
    stride_bn = 16

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
        xindex = idx_n + 16*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 96*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 16*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 96*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((125440, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1960, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.056203264
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/tf/ctfkvgihvx5cpidhijdwrnh33fcizijjw77ssofn64qqrvlxwog3.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_3 => clamp_max, clamp_min
#   input_4 => convolution_2
#   input_5 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   input_6 => convolution_3
#   input_7 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   input_8 => clamp_max_1, clamp_min_1
# Graph fragment:
#   %buf8 : Tensor "f32[125440, 96][96, 1]cuda:0" = PlaceHolder[target=buf8]
#   %arg17_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %unsqueeze_8 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max : Tensor "f32[10, 32, 112, 112][401408, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
#   %convolution_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[10, 16, 112, 112][200704, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %convolution_3 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %clamp_min_1 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0.0), kwargs = {})
#   %clamp_max_1 : Tensor "f32[10, 96, 112, 112][1204224, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
#   return %clamp_max_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 144508416}, 'kernel_num_gb': 0.096339456, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12042240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 12042240,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.096339456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/z4/cz4og6lzx3h3wgdxdp4czfyxf6gwgeqcoz3xp6547bucynlsoemb.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_10 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_11 => clamp_max_2, clamp_min_2
#   input_12 => convolution_5
# Graph fragment:
#   %convolution_4 : Tensor "f32[10, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = PlaceHolder[target=convolution_4]
#   %arg22_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %unsqueeze_32 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %clamp_min_2 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_9, 0.0), kwargs = {})
#   %clamp_max_2 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6.0), kwargs = {})
#   %convolution_5 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf11
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 36128256}, 'kernel_num_gb': 0.024086016, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((10, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 3010560,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024086016
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/wn/cwnbjbezue2vpb2lb4lxh3cpk5pgmvp4dg44yg5cpyxxgitzhtrp.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_10 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_11 => clamp_max_2, clamp_min_2
#   input_12 => convolution_5
# Graph fragment:
#   %buf11 : Tensor "f32[10, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = PlaceHolder[target=buf11]
#   %arg26_1 : Tensor "f32[24, 96, 1, 1][96, 1, 1, 1]cuda:0" = PlaceHolder[target=arg26_1]
#   %unsqueeze_32 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %clamp_min_2 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_9, 0.0), kwargs = {})
#   %clamp_max_2 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6.0), kwargs = {})
#   %convolution_5 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf12
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.015062016, 'kernel_flop': 144506880, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 24
    K = 96
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 96
    stride_ak = 1
    stride_bk = 1
    stride_bn = 96

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
        xindex = idx_n + 96*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 24*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 24*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.015062016
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/us/cuswxo5jnl3yc6zqfr26de2xfszly6y4pl2ncwzw6rj6eh7wr4oe.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_10 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_11 => clamp_max_2, clamp_min_2
#   input_12 => convolution_5
#   input_13 => add_10, add_11, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
# Graph fragment:
#   %buf12 : Tensor "f32[31360, 24][24, 1]cuda:0" = PlaceHolder[target=buf12]
#   %arg27_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %arg29_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %unsqueeze_32 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[96][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[96, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[96, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %clamp_min_2 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_9, 0.0), kwargs = {})
#   %clamp_max_2 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6.0), kwargs = {})
#   %convolution_5 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg27_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_5 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %add_10 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_10,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg29_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg30_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_11 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   return %add_11
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9032064}, 'kernel_num_gb': 0.006021504, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)


def get_args():
    arg_0 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006021504
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/z3/cz3v62wf2ch3smujjnhncb7efd662wru6shmefn3m7y4bjhig77h.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_14 => convolution_6
# Graph fragment:
#   %add_11 : Tensor "f32[10, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = PlaceHolder[target=add_11]
#   %arg31_1 : Tensor "f32[144, 24, 1, 1][24, 1, 1, 1]cuda:0" = PlaceHolder[target=arg31_1]
#   %convolution_6 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_11, %arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf14
triton_tem_fused_convolution_11 = async_compile.triton('triton_tem_fused_convolution_11', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_11', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.021087744, 'kernel_flop': 216760320, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_11(arg_A, arg_B, out_ptr0):
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
    N = 144
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
        xindex = idx_n + 144*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 24*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 144*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 735, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.021087744
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/fa/cfaae7jtwckbxy6tngf5xjw26poj3jg4pbnzdgmvwkf24teworxm.py
# Topologically Sorted Source Nodes: [input_14, input_15, input_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_14 => convolution_6
#   input_15 => add_12, add_13, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   input_16 => clamp_max_3, clamp_min_3
# Graph fragment:
#   %buf14 : Tensor "f32[31360, 144][144, 1]cuda:0" = PlaceHolder[target=buf14]
#   %arg32_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %arg33_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg33_1]
#   %arg34_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_6 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_11, %arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_48 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_12 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg33_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg34_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_13 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %clamp_min_3 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_13, 0.0), kwargs = {})
#   %clamp_max_3 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6.0), kwargs = {})
#   return %clamp_max_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 54192384}, 'kernel_num_gb': 0.036129024, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4515840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4515840,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.036129024
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/6l/c6l7v6k3f5gsmwffhmmepi542cwqrcb2gtdjqq3mbalrergooqy4.py
# Topologically Sorted Source Nodes: [input_18, input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_18 => add_14, add_15, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   input_19 => clamp_max_4, clamp_min_4
#   input_20 => convolution_8
# Graph fragment:
#   %buf17 : Tensor "f32[10, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = PlaceHolder[target=buf17]
#   %arg41_1 : Tensor "f32[24, 144, 1, 1][144, 1, 1, 1]cuda:0" = PlaceHolder[target=arg41_1]
#   %unsqueeze_56 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_14 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_15 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %clamp_min_4 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_15, 0.0), kwargs = {})
#   %clamp_max_4 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6.0), kwargs = {})
#   %convolution_8 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_4, %arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf18
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.021087744, 'kernel_flop': 216760320, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 24
    K = 144
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 144
    stride_ak = 1
    stride_bk = 1
    stride_bn = 144

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
        xindex = idx_n + 144*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 24*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 24*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.021087744
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/3o/c3obqf2xs5ix4nl4k2nk7cmmjgjbpvps3hh77xvp445d75leulwd.py
# Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_1, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_18 => add_14, add_15, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   input_19 => clamp_max_4, clamp_min_4
#   input_20 => convolution_8
#   input_21 => add_16, add_17, mul_24, mul_25, mul_26, reciprocal_8, sqrt_8, sub_8, unsqueeze_64, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, unsqueeze_71
#   input_22 => convolution_9
#   x_1 => add_18
# Graph fragment:
#   %buf18 : Tensor "f32[31360, 24][24, 1]cuda:0" = PlaceHolder[target=buf18]
#   %arg42_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %arg43_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %arg44_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %add_11 : Tensor "f32[10, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = PlaceHolder[target=add_11]
#   %unsqueeze_56 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_14 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_15 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %clamp_min_4 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_15, 0.0), kwargs = {})
#   %clamp_max_4 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6.0), kwargs = {})
#   %convolution_8 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_4, %arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %sub_8 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %add_16 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg43_1, 1e-05), kwargs = {})
#   %sqrt_8 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_8 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_8,), kwargs = {})
#   %mul_24 : Tensor "f32[24][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_24, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %mul_25 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg44_1, -1), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, -1), kwargs = {})
#   %mul_26 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[24, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg45_1, -1), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[24, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, -1), kwargs = {})
#   %add_17 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_18 : Tensor "f32[10, 24, 56, 56][75264, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %add_11), kwargs = {})
#   %convolution_9 : Tensor "f32[10, 144, 56, 56][451584, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf19
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12042624}, 'kernel_num_gb': 0.009032064, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009032064
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/2m/c2mfcx5rtxclxjdbk22db43tismqu2t2eozsqem6ct5lgxjnsvpu.py
# Topologically Sorted Source Nodes: [input_26, input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_26 => add_21, add_22, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   input_27 => clamp_max_6, clamp_min_6
#   input_28 => convolution_11
# Graph fragment:
#   %convolution_10 : Tensor "f32[10, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = PlaceHolder[target=convolution_10]
#   %arg52_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %arg53_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %arg54_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %arg55_1 : Tensor "f32[144][1]cuda:0" = PlaceHolder[target=arg55_1]
#   %unsqueeze_80 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_21 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_21,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_22 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %clamp_min_6 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_22, 0.0), kwargs = {})
#   %clamp_max_6 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6.0), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_6, %arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf23
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 13549824}, 'kernel_num_gb': 0.009033984, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1128960,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009033984
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/td/ctdslmldyiwq6gbplolrqkhsbnu37eqxv5hhpczbsywiypxqmqc5.py
# Topologically Sorted Source Nodes: [input_26, input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_26 => add_21, add_22, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   input_27 => clamp_max_6, clamp_min_6
#   input_28 => convolution_11
# Graph fragment:
#   %buf23 : Tensor "f32[10, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = PlaceHolder[target=buf23]
#   %arg56_1 : Tensor "f32[40, 144, 1, 1][144, 1, 1, 1]cuda:0" = PlaceHolder[target=arg56_1]
#   %unsqueeze_80 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_21 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_21,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_22 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %clamp_min_6 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_22, 0.0), kwargs = {})
#   %clamp_max_6 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6.0), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_6, %arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf24
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00579328, 'kernel_flop': 90316800, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 40
    K = 144
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 144
    stride_ak = 1
    stride_bk = 1
    stride_bn = 144

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
        xindex = idx_n + 144*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 40*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 40*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 123, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00579328
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/jk/cjkpl4gefgdo4yhnmwdfe6g2lsy5l27vzjkwkck42jqrdpijtu5l.py
# Topologically Sorted Source Nodes: [input_26, input_27, input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_26 => add_21, add_22, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   input_27 => clamp_max_6, clamp_min_6
#   input_28 => convolution_11
#   input_29 => add_23, add_24, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
# Graph fragment:
#   %buf24 : Tensor "f32[7840, 40][40, 1]cuda:0" = PlaceHolder[target=buf24]
#   %arg57_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %arg59_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %unsqueeze_80 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_21 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_21,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[144, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_22 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %clamp_min_6 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_22, 0.0), kwargs = {})
#   %clamp_max_6 : Tensor "f32[10, 144, 28, 28][112896, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6.0), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_6, %arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_88 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_23 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_23,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_24 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   return %add_24
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3763840}, 'kernel_num_gb': 0.00250944, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)


def get_args():
    arg_0 = rand_strided((10, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 313600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00250944
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/px/cpxggummytedj74qil6xrw4qm6pvo7dxbhvue34z2d6dpy763l2q.py
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_30 => convolution_12
# Graph fragment:
#   %add_24 : Tensor "f32[10, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = PlaceHolder[target=add_24]
#   %arg61_1 : Tensor "f32[240, 40, 1, 1][40, 1, 1, 1]cuda:0" = PlaceHolder[target=arg61_1]
#   %convolution_12 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_24, %arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf26
triton_tem_fused_convolution_18 = async_compile.triton('triton_tem_fused_convolution_18', '''
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
num_warps=8,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_18', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0088192, 'kernel_flop': 150528000, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_18(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 240
    K = 40
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 40
    stride_ak = 1
    stride_bk = 1
    stride_bn = 40

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
        xindex = idx_n + 40*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 240*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 40*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


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
    arg_0 = rand_strided((10, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 124, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0088192
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/3a/c3alwek6w3qs3uhit6cff25udawsyganvc5foaixbucd35hv2bfp.py
# Topologically Sorted Source Nodes: [input_30, input_31, input_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_30 => convolution_12
#   input_31 => add_25, add_26, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   input_32 => clamp_max_7, clamp_min_7
# Graph fragment:
#   %buf26 : Tensor "f32[7840, 240][240, 1]cuda:0" = PlaceHolder[target=buf26]
#   %arg62_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %arg63_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %convolution_12 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_24, %arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_96 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_25 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_25,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_26 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %clamp_min_7 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_26, 0.0), kwargs = {})
#   %clamp_max_7 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 6.0), kwargs = {})
#   return %clamp_max_7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 22583040}, 'kernel_num_gb': 0.01505664, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1881600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.01505664
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/iy/ciy573fyy5odczkv3pv77dkffhhntbebcjsugnfoy3oombkxqruh.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_34 => add_27, add_28, mul_39, mul_40, mul_41, reciprocal_13, sqrt_13, sub_13, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111
#   input_35 => clamp_max_8, clamp_min_8
#   input_36 => convolution_14
# Graph fragment:
#   %buf29 : Tensor "f32[10, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = PlaceHolder[target=buf29]
#   %arg71_1 : Tensor "f32[40, 240, 1, 1][240, 1, 1, 1]cuda:0" = PlaceHolder[target=arg71_1]
#   %unsqueeze_104 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg67_1, -1), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_104, -1), kwargs = {})
#   %sub_13 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %add_27 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg68_1, 1e-05), kwargs = {})
#   %sqrt_13 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_13 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_13,), kwargs = {})
#   %mul_39 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 1), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_39, -1), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, -1), kwargs = {})
#   %mul_40 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg69_1, -1), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_108, -1), kwargs = {})
#   %mul_41 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg70_1, -1), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_110, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %clamp_min_8 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_28, 0.0), kwargs = {})
#   %clamp_max_8 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_8, 6.0), kwargs = {})
#   %convolution_14 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_8, %arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf30
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0088192, 'kernel_flop': 150528000, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 40
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
        xindex = idx_n + 40*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 40*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 123, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0088192
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/hk/chk3jfludzy2mym4xr3rlkceuwr2mgevcpgbnilju2niofjcko6k.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36, input_37, x_2, input_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_34 => add_27, add_28, mul_39, mul_40, mul_41, reciprocal_13, sqrt_13, sub_13, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111
#   input_35 => clamp_max_8, clamp_min_8
#   input_36 => convolution_14
#   input_37 => add_29, add_30, mul_42, mul_43, mul_44, reciprocal_14, sqrt_14, sub_14, unsqueeze_112, unsqueeze_113, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119
#   input_38 => convolution_15
#   x_2 => add_31
# Graph fragment:
#   %buf30 : Tensor "f32[7840, 40][40, 1]cuda:0" = PlaceHolder[target=buf30]
#   %arg72_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %arg73_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %arg74_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %arg75_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %add_24 : Tensor "f32[10, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = PlaceHolder[target=add_24]
#   %unsqueeze_104 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg67_1, -1), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_104, -1), kwargs = {})
#   %sub_13 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %add_27 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg68_1, 1e-05), kwargs = {})
#   %sqrt_13 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_13 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_13,), kwargs = {})
#   %mul_39 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 1), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_39, -1), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, -1), kwargs = {})
#   %mul_40 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg69_1, -1), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_108, -1), kwargs = {})
#   %mul_41 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg70_1, -1), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_110, -1), kwargs = {})
#   %add_28 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %clamp_min_8 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_28, 0.0), kwargs = {})
#   %clamp_max_8 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_8, 6.0), kwargs = {})
#   %convolution_14 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_8, %arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg72_1, -1), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, -1), kwargs = {})
#   %sub_14 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %add_29 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg73_1, 1e-05), kwargs = {})
#   %sqrt_14 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_29,), kwargs = {})
#   %reciprocal_14 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_14,), kwargs = {})
#   %mul_42 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, 1), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_42, -1), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, -1), kwargs = {})
#   %mul_43 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg74_1, -1), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, -1), kwargs = {})
#   %mul_44 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg75_1, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[40, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, -1), kwargs = {})
#   %add_30 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_31 : Tensor "f32[10, 40, 28, 28][31360, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_24), kwargs = {})
#   %convolution_15 : Tensor "f32[10, 240, 28, 28][188160, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_31, %arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf31
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5018240}, 'kernel_num_gb': 0.00376384, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 313600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00376384
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/e4/ce4qgosnd6effozl2ui5svf3siyflvlnj6wfmmpgxgqafwly4mvd.py
# Topologically Sorted Source Nodes: [input_42, input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_42 => add_34, add_35, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135
#   input_43 => clamp_max_10, clamp_min_10
#   input_44 => convolution_17
# Graph fragment:
#   %convolution_16 : Tensor "f32[10, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = PlaceHolder[target=convolution_16]
#   %arg82_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg82_1]
#   %arg83_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %arg84_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg84_1]
#   %arg85_1 : Tensor "f32[240][1]cuda:0" = PlaceHolder[target=arg85_1]
#   %unsqueeze_128 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, -1), kwargs = {})
#   %sub_16 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %add_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_34,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, -1), kwargs = {})
#   %mul_49 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_132, -1), kwargs = {})
#   %mul_50 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, -1), kwargs = {})
#   %add_35 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %clamp_min_10 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 0.0), kwargs = {})
#   %clamp_max_10 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6.0), kwargs = {})
#   %convolution_17 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_10, %arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf35
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5648640}, 'kernel_num_gb': 0.00376704, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 470400
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 470400,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00376704
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/5m/c5m4zy3gdgegxofs2ljwljtvijiy6kvzzi3p7ejhf2vj6zmka26e.py
# Topologically Sorted Source Nodes: [input_42, input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_42 => add_34, add_35, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135
#   input_43 => clamp_max_10, clamp_min_10
#   input_44 => convolution_17
# Graph fragment:
#   %buf35 : Tensor "f32[10, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = PlaceHolder[target=buf35]
#   %arg86_1 : Tensor "f32[80, 240, 1, 1][240, 1, 1, 1]cuda:0" = PlaceHolder[target=arg86_1]
#   %unsqueeze_128 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, -1), kwargs = {})
#   %sub_16 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %add_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_34,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, -1), kwargs = {})
#   %mul_49 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_132, -1), kwargs = {})
#   %mul_50 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, -1), kwargs = {})
#   %add_35 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %clamp_min_10 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 0.0), kwargs = {})
#   %clamp_max_10 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6.0), kwargs = {})
#   %convolution_17 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_10, %arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf36
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0025856, 'kernel_flop': 75264000, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 80
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
        xindex = idx_n + 80*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 80*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 186, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0025856
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/mp/cmpzq5whkvwyl5nb624h44ltbrdlhuk3zejhqumfdov53vjxihdl.py
# Topologically Sorted Source Nodes: [input_42, input_43, input_44, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_42 => add_34, add_35, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135
#   input_43 => clamp_max_10, clamp_min_10
#   input_44 => convolution_17
#   input_45 => add_36, add_37, mul_51, mul_52, mul_53, reciprocal_17, sqrt_17, sub_17, unsqueeze_136, unsqueeze_137, unsqueeze_138, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143
# Graph fragment:
#   %buf36 : Tensor "f32[1960, 80][80, 1]cuda:0" = PlaceHolder[target=buf36]
#   %arg87_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg88_1]
#   %arg89_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg89_1]
#   %arg90_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %unsqueeze_128 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, -1), kwargs = {})
#   %sub_16 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %add_34 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_34,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[240][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, -1), kwargs = {})
#   %mul_49 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_132, -1), kwargs = {})
#   %mul_50 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[240, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[240, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, -1), kwargs = {})
#   %add_35 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %clamp_min_10 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 0.0), kwargs = {})
#   %clamp_max_10 : Tensor "f32[10, 240, 14, 14][47040, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6.0), kwargs = {})
#   %convolution_17 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_10, %arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_136 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg87_1, -1), kwargs = {})
#   %unsqueeze_137 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_136, -1), kwargs = {})
#   %sub_17 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %add_36 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg88_1, 1e-05), kwargs = {})
#   %sqrt_17 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_36,), kwargs = {})
#   %reciprocal_17 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_17,), kwargs = {})
#   %mul_51 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 1), kwargs = {})
#   %unsqueeze_138 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_51, -1), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_138, -1), kwargs = {})
#   %mul_52 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg89_1, -1), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, -1), kwargs = {})
#   %mul_53 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg90_1, -1), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, -1), kwargs = {})
#   %add_37 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   return %add_37
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1882880}, 'kernel_num_gb': 0.00125568, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)


def get_args():
    arg_0 = rand_strided((10, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 156800,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00125568
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/dt/cdtcgytbr7hroim34nvaffoblafpqkkttfeqmnzwf5rv22lfuvlg.py
# Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_46 => convolution_18
# Graph fragment:
#   %add_37 : Tensor "f32[10, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = PlaceHolder[target=add_37]
#   %arg91_1 : Tensor "f32[480, 80, 1, 1][80, 1, 1, 1]cuda:0" = PlaceHolder[target=arg91_1]
#   %convolution_18 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_37, %arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf38
triton_tem_fused_convolution_25 = async_compile.triton('triton_tem_fused_convolution_25', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_25', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.004544, 'kernel_flop': 150528000, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_25(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 480
    K = 80
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 80
    stride_ak = 1
    stride_bk = 1
    stride_bn = 80

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
        xindex = idx_n + 80*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 480*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 80*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


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
    arg_0 = rand_strided((10, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 128, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_25.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_25.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004544
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/zh/czhmzszitdxaeyleuxlslw2o2l3aai575mqmhulxzgmru3mnsf7w.py
# Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_46 => convolution_18
#   input_47 => add_38, add_39, mul_54, mul_55, mul_56, reciprocal_18, sqrt_18, sub_18, unsqueeze_144, unsqueeze_145, unsqueeze_146, unsqueeze_147, unsqueeze_148, unsqueeze_149, unsqueeze_150, unsqueeze_151
#   input_48 => clamp_max_11, clamp_min_11
# Graph fragment:
#   %buf38 : Tensor "f32[1960, 480][480, 1]cuda:0" = PlaceHolder[target=buf38]
#   %arg92_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg92_1]
#   %arg93_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg93_1]
#   %arg94_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg94_1]
#   %arg95_1 : Tensor "f32[480][1]cuda:0" = PlaceHolder[target=arg95_1]
#   %convolution_18 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_37, %arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_144 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg92_1, -1), kwargs = {})
#   %unsqueeze_145 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_144, -1), kwargs = {})
#   %sub_18 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %add_38 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg93_1, 1e-05), kwargs = {})
#   %sqrt_18 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %reciprocal_18 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_18,), kwargs = {})
#   %mul_54 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_18, 1), kwargs = {})
#   %unsqueeze_146 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_54, -1), kwargs = {})
#   %unsqueeze_147 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_146, -1), kwargs = {})
#   %mul_55 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %unsqueeze_148 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg94_1, -1), kwargs = {})
#   %unsqueeze_149 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_148, -1), kwargs = {})
#   %mul_56 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %unsqueeze_150 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg95_1, -1), kwargs = {})
#   %unsqueeze_151 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_150, -1), kwargs = {})
#   %add_39 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %clamp_min_11 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_39, 0.0), kwargs = {})
#   %clamp_max_11 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 6.0), kwargs = {})
#   return %clamp_max_11
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11297280}, 'kernel_num_gb': 0.00753408, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 480)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 940800,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00753408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ig/cigw7lqexygrfvms76vmtkka5j2ak663kv34adtuwa6gtnq73xhx.py
# Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_50 => add_40, add_41, mul_57, mul_58, mul_59, reciprocal_19, sqrt_19, sub_19, unsqueeze_152, unsqueeze_153, unsqueeze_154, unsqueeze_155, unsqueeze_156, unsqueeze_157, unsqueeze_158, unsqueeze_159
#   input_51 => clamp_max_12, clamp_min_12
#   input_52 => convolution_20
# Graph fragment:
#   %buf41 : Tensor "f32[10, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = PlaceHolder[target=buf41]
#   %arg101_1 : Tensor "f32[80, 480, 1, 1][480, 1, 1, 1]cuda:0" = PlaceHolder[target=arg101_1]
#   %unsqueeze_152 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg97_1, -1), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, -1), kwargs = {})
#   %sub_19 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %add_40 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg98_1, 1e-05), kwargs = {})
#   %sqrt_19 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_40,), kwargs = {})
#   %reciprocal_19 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_19,), kwargs = {})
#   %mul_57 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_19, 1), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_57, -1), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, -1), kwargs = {})
#   %mul_58 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg99_1, -1), kwargs = {})
#   %unsqueeze_157 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_156, -1), kwargs = {})
#   %mul_59 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %unsqueeze_158 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg100_1, -1), kwargs = {})
#   %unsqueeze_159 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_158, -1), kwargs = {})
#   %add_41 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %clamp_min_12 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_41, 0.0), kwargs = {})
#   %clamp_max_12 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6.0), kwargs = {})
#   %convolution_20 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_12, %arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf42
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27', '''
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
num_warps=8,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.004544, 'kernel_flop': 150528000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 80
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
        xindex = idx_n + 80*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 80*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 124, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004544
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/sk/csk3j5d3z4nf36wl35yuxisowvrrsyigbb64xffjd7bhosk3czli.py
# Topologically Sorted Source Nodes: [input_50, input_51, input_52, input_53, x_3, input_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_50 => add_40, add_41, mul_57, mul_58, mul_59, reciprocal_19, sqrt_19, sub_19, unsqueeze_152, unsqueeze_153, unsqueeze_154, unsqueeze_155, unsqueeze_156, unsqueeze_157, unsqueeze_158, unsqueeze_159
#   input_51 => clamp_max_12, clamp_min_12
#   input_52 => convolution_20
#   input_53 => add_42, add_43, mul_60, mul_61, mul_62, reciprocal_20, sqrt_20, sub_20, unsqueeze_160, unsqueeze_161, unsqueeze_162, unsqueeze_163, unsqueeze_164, unsqueeze_165, unsqueeze_166, unsqueeze_167
#   input_54 => convolution_21
#   x_3 => add_44
# Graph fragment:
#   %buf42 : Tensor "f32[1960, 80][80, 1]cuda:0" = PlaceHolder[target=buf42]
#   %arg102_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg102_1]
#   %arg103_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg103_1]
#   %arg104_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg104_1]
#   %arg105_1 : Tensor "f32[80][1]cuda:0" = PlaceHolder[target=arg105_1]
#   %add_37 : Tensor "f32[10, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = PlaceHolder[target=add_37]
#   %unsqueeze_152 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg97_1, -1), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, -1), kwargs = {})
#   %sub_19 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %add_40 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg98_1, 1e-05), kwargs = {})
#   %sqrt_19 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_40,), kwargs = {})
#   %reciprocal_19 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_19,), kwargs = {})
#   %mul_57 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_19, 1), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_57, -1), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, -1), kwargs = {})
#   %mul_58 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg99_1, -1), kwargs = {})
#   %unsqueeze_157 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_156, -1), kwargs = {})
#   %mul_59 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %unsqueeze_158 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg100_1, -1), kwargs = {})
#   %unsqueeze_159 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_158, -1), kwargs = {})
#   %add_41 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %clamp_min_12 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_41, 0.0), kwargs = {})
#   %clamp_max_12 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6.0), kwargs = {})
#   %convolution_20 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_12, %arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_160 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg102_1, -1), kwargs = {})
#   %unsqueeze_161 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_160, -1), kwargs = {})
#   %sub_20 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %add_42 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg103_1, 1e-05), kwargs = {})
#   %sqrt_20 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_42,), kwargs = {})
#   %reciprocal_20 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_20,), kwargs = {})
#   %mul_60 : Tensor "f32[80][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_20, 1), kwargs = {})
#   %unsqueeze_162 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_60, -1), kwargs = {})
#   %unsqueeze_163 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_162, -1), kwargs = {})
#   %mul_61 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %unsqueeze_164 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg104_1, -1), kwargs = {})
#   %unsqueeze_165 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_164, -1), kwargs = {})
#   %mul_62 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %unsqueeze_166 : Tensor "f32[80, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg105_1, -1), kwargs = {})
#   %unsqueeze_167 : Tensor "f32[80, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_166, -1), kwargs = {})
#   %add_43 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_44 : Tensor "f32[10, 80, 14, 14][15680, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %add_37), kwargs = {})
#   %convolution_21 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf43
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2510080}, 'kernel_num_gb': 0.00188288, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 156800,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00188288
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/gq/cgqpbovtuwh4nm3kxy2u7cp7rnx5xnhfzetudtyg3m4jftb7izne.py
# Topologically Sorted Source Nodes: [input_58, input_59, input_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_58 => add_47, add_48, mul_66, mul_67, mul_68, reciprocal_22, sqrt_22, sub_22, unsqueeze_176, unsqueeze_177, unsqueeze_178, unsqueeze_179, unsqueeze_180, unsqueeze_181, unsqueeze_182, unsqueeze_183
#   input_59 => clamp_max_14, clamp_min_14
#   input_60 => convolution_23
# Graph fragment:
#   %buf47 : Tensor "f32[10, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = PlaceHolder[target=buf47]
#   %arg116_1 : Tensor "f32[112, 480, 1, 1][480, 1, 1, 1]cuda:0" = PlaceHolder[target=arg116_1]
#   %unsqueeze_176 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg112_1, -1), kwargs = {})
#   %unsqueeze_177 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_176, -1), kwargs = {})
#   %sub_22 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %add_47 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg113_1, 1e-05), kwargs = {})
#   %sqrt_22 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_47,), kwargs = {})
#   %reciprocal_22 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_22,), kwargs = {})
#   %mul_66 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_22, 1), kwargs = {})
#   %unsqueeze_178 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_66, -1), kwargs = {})
#   %unsqueeze_179 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_178, -1), kwargs = {})
#   %mul_67 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %unsqueeze_180 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg114_1, -1), kwargs = {})
#   %unsqueeze_181 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_180, -1), kwargs = {})
#   %mul_68 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %unsqueeze_182 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg115_1, -1), kwargs = {})
#   %unsqueeze_183 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_182, -1), kwargs = {})
#   %add_48 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %clamp_min_14 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_48, 0.0), kwargs = {})
#   %clamp_max_14 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6.0), kwargs = {})
#   %convolution_23 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_14, %arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf48
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29', '''
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
num_warps=8,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00485632, 'kernel_flop': 210739200, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 112
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
        xindex = idx_n + 112*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 112*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 112), (112, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 124, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00485632
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/hi/chinjt3qxbbnworlqpddmc4uyt4b33jubsljysooj7x54c4j5af6.py
# Topologically Sorted Source Nodes: [input_58, input_59, input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_58 => add_47, add_48, mul_66, mul_67, mul_68, reciprocal_22, sqrt_22, sub_22, unsqueeze_176, unsqueeze_177, unsqueeze_178, unsqueeze_179, unsqueeze_180, unsqueeze_181, unsqueeze_182, unsqueeze_183
#   input_59 => clamp_max_14, clamp_min_14
#   input_60 => convolution_23
#   input_61 => add_49, add_50, mul_69, mul_70, mul_71, reciprocal_23, sqrt_23, sub_23, unsqueeze_184, unsqueeze_185, unsqueeze_186, unsqueeze_187, unsqueeze_188, unsqueeze_189, unsqueeze_190, unsqueeze_191
# Graph fragment:
#   %buf48 : Tensor "f32[1960, 112][112, 1]cuda:0" = PlaceHolder[target=buf48]
#   %arg117_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg117_1]
#   %arg118_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %arg119_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg119_1]
#   %arg120_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg120_1]
#   %unsqueeze_176 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg112_1, -1), kwargs = {})
#   %unsqueeze_177 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_176, -1), kwargs = {})
#   %sub_22 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %add_47 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg113_1, 1e-05), kwargs = {})
#   %sqrt_22 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_47,), kwargs = {})
#   %reciprocal_22 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_22,), kwargs = {})
#   %mul_66 : Tensor "f32[480][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_22, 1), kwargs = {})
#   %unsqueeze_178 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_66, -1), kwargs = {})
#   %unsqueeze_179 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_178, -1), kwargs = {})
#   %mul_67 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %unsqueeze_180 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg114_1, -1), kwargs = {})
#   %unsqueeze_181 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_180, -1), kwargs = {})
#   %mul_68 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %unsqueeze_182 : Tensor "f32[480, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg115_1, -1), kwargs = {})
#   %unsqueeze_183 : Tensor "f32[480, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_182, -1), kwargs = {})
#   %add_48 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %clamp_min_14 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_48, 0.0), kwargs = {})
#   %clamp_max_14 : Tensor "f32[10, 480, 14, 14][94080, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6.0), kwargs = {})
#   %convolution_23 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_14, %arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_184 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg117_1, -1), kwargs = {})
#   %unsqueeze_185 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_184, -1), kwargs = {})
#   %sub_23 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %add_49 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg118_1, 1e-05), kwargs = {})
#   %sqrt_23 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_49,), kwargs = {})
#   %reciprocal_23 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_23,), kwargs = {})
#   %mul_69 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_23, 1), kwargs = {})
#   %unsqueeze_186 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_69, -1), kwargs = {})
#   %unsqueeze_187 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_186, -1), kwargs = {})
#   %mul_70 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %unsqueeze_188 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg119_1, -1), kwargs = {})
#   %unsqueeze_189 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_188, -1), kwargs = {})
#   %mul_71 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_189), kwargs = {})
#   %unsqueeze_190 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg120_1, -1), kwargs = {})
#   %unsqueeze_191 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_190, -1), kwargs = {})
#   %add_50 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_191), kwargs = {})
#   return %add_50
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2636032}, 'kernel_num_gb': 0.001757952, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 219520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 112)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)


def get_args():
    arg_0 = rand_strided((10, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 219520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001757952
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/p3/cp3p4y6blyefttsx4gphyod7v7yjlr5yokeuocomsmok3fvd2ki4.py
# Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_62 => convolution_24
# Graph fragment:
#   %add_50 : Tensor "f32[10, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = PlaceHolder[target=add_50]
#   %arg121_1 : Tensor "f32[672, 112, 1, 1][112, 1, 1, 1]cuda:0" = PlaceHolder[target=arg121_1]
#   %convolution_24 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_50, %arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf50
triton_tem_fused_convolution_31 = async_compile.triton('triton_tem_fused_convolution_31', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_31', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.006447616, 'kernel_flop': 295034880, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_31(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 672
    K = 112
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 112
    stride_ak = 1
    stride_bk = 1
    stride_bn = 112

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
        xindex = idx_n + 112*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 672*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 112*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 672*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 672), (672, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 176, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_31.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_31.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006447616
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/pn/cpn6hnolspo5go4idczeyy7knkagrbvxexyv22rl42fclzzk7giq.py
# Topologically Sorted Source Nodes: [input_62, input_63, input_64], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_62 => convolution_24
#   input_63 => add_51, add_52, mul_72, mul_73, mul_74, reciprocal_24, sqrt_24, sub_24, unsqueeze_192, unsqueeze_193, unsqueeze_194, unsqueeze_195, unsqueeze_196, unsqueeze_197, unsqueeze_198, unsqueeze_199
#   input_64 => clamp_max_15, clamp_min_15
# Graph fragment:
#   %buf50 : Tensor "f32[1960, 672][672, 1]cuda:0" = PlaceHolder[target=buf50]
#   %arg122_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg122_1]
#   %arg123_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %arg124_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg124_1]
#   %arg125_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %convolution_24 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_50, %arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_192 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg122_1, -1), kwargs = {})
#   %unsqueeze_193 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_192, -1), kwargs = {})
#   %sub_24 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %add_51 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, 1e-05), kwargs = {})
#   %sqrt_24 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_51,), kwargs = {})
#   %reciprocal_24 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_24,), kwargs = {})
#   %mul_72 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_24, 1), kwargs = {})
#   %unsqueeze_194 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_72, -1), kwargs = {})
#   %unsqueeze_195 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_194, -1), kwargs = {})
#   %mul_73 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %unsqueeze_196 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg124_1, -1), kwargs = {})
#   %unsqueeze_197 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_196, -1), kwargs = {})
#   %mul_74 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %unsqueeze_198 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg125_1, -1), kwargs = {})
#   %unsqueeze_199 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_198, -1), kwargs = {})
#   %add_52 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %clamp_min_15 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_52, 0.0), kwargs = {})
#   %clamp_max_15 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6.0), kwargs = {})
#   return %clamp_max_15
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15816192}, 'kernel_num_gb': 0.010547712, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1317120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 672)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1317120,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.010547712
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/nf/cnfoalfppwtmmw3anq5jksln3pirzw6jvo7fls4nvioqg2pk57ci.py
# Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_66 => add_53, add_54, mul_75, mul_76, mul_77, reciprocal_25, sqrt_25, sub_25, unsqueeze_200, unsqueeze_201, unsqueeze_202, unsqueeze_203, unsqueeze_204, unsqueeze_205, unsqueeze_206, unsqueeze_207
#   input_67 => clamp_max_16, clamp_min_16
#   input_68 => convolution_26
# Graph fragment:
#   %buf53 : Tensor "f32[10, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = PlaceHolder[target=buf53]
#   %arg131_1 : Tensor "f32[112, 672, 1, 1][672, 1, 1, 1]cuda:0" = PlaceHolder[target=arg131_1]
#   %unsqueeze_200 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg127_1, -1), kwargs = {})
#   %unsqueeze_201 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_200, -1), kwargs = {})
#   %sub_25 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %add_53 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_25 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_53,), kwargs = {})
#   %reciprocal_25 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_25,), kwargs = {})
#   %mul_75 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 1), kwargs = {})
#   %unsqueeze_202 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_75, -1), kwargs = {})
#   %unsqueeze_203 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_202, -1), kwargs = {})
#   %mul_76 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %unsqueeze_204 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, -1), kwargs = {})
#   %unsqueeze_205 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_204, -1), kwargs = {})
#   %mul_77 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %unsqueeze_206 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, -1), kwargs = {})
#   %unsqueeze_207 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_206, -1), kwargs = {})
#   %add_54 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %clamp_min_16 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_54, 0.0), kwargs = {})
#   %clamp_max_16 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6.0), kwargs = {})
#   %convolution_26 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_16, %arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf54
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33', '''
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
num_warps=8,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.006447616, 'kernel_flop': 295034880, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 112
    K = 672
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 672
    stride_ak = 1
    stride_bk = 1
    stride_bn = 672

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
        xindex = idx_n + 672*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 112*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 112*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 112), (112, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 124, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006447616
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/dv/cdvk5igdaoz5ubamrmc6oixc7txfrh746lw2qzr3dppkhnbdbard.py
# Topologically Sorted Source Nodes: [input_66, input_67, input_68, input_69, x_4, input_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_66 => add_53, add_54, mul_75, mul_76, mul_77, reciprocal_25, sqrt_25, sub_25, unsqueeze_200, unsqueeze_201, unsqueeze_202, unsqueeze_203, unsqueeze_204, unsqueeze_205, unsqueeze_206, unsqueeze_207
#   input_67 => clamp_max_16, clamp_min_16
#   input_68 => convolution_26
#   input_69 => add_55, add_56, mul_78, mul_79, mul_80, reciprocal_26, sqrt_26, sub_26, unsqueeze_208, unsqueeze_209, unsqueeze_210, unsqueeze_211, unsqueeze_212, unsqueeze_213, unsqueeze_214, unsqueeze_215
#   input_70 => convolution_27
#   x_4 => add_57
# Graph fragment:
#   %buf54 : Tensor "f32[1960, 112][112, 1]cuda:0" = PlaceHolder[target=buf54]
#   %arg132_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg132_1]
#   %arg133_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg133_1]
#   %arg134_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg134_1]
#   %arg135_1 : Tensor "f32[112][1]cuda:0" = PlaceHolder[target=arg135_1]
#   %add_50 : Tensor "f32[10, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = PlaceHolder[target=add_50]
#   %unsqueeze_200 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg127_1, -1), kwargs = {})
#   %unsqueeze_201 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_200, -1), kwargs = {})
#   %sub_25 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %add_53 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_25 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_53,), kwargs = {})
#   %reciprocal_25 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_25,), kwargs = {})
#   %mul_75 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 1), kwargs = {})
#   %unsqueeze_202 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_75, -1), kwargs = {})
#   %unsqueeze_203 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_202, -1), kwargs = {})
#   %mul_76 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %unsqueeze_204 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, -1), kwargs = {})
#   %unsqueeze_205 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_204, -1), kwargs = {})
#   %mul_77 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %unsqueeze_206 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, -1), kwargs = {})
#   %unsqueeze_207 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_206, -1), kwargs = {})
#   %add_54 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %clamp_min_16 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_54, 0.0), kwargs = {})
#   %clamp_max_16 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6.0), kwargs = {})
#   %convolution_26 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_16, %arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_208 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg132_1, -1), kwargs = {})
#   %unsqueeze_209 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_208, -1), kwargs = {})
#   %sub_26 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %add_55 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg133_1, 1e-05), kwargs = {})
#   %sqrt_26 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_55,), kwargs = {})
#   %reciprocal_26 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_26,), kwargs = {})
#   %mul_78 : Tensor "f32[112][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_26, 1), kwargs = {})
#   %unsqueeze_210 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_78, -1), kwargs = {})
#   %unsqueeze_211 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_210, -1), kwargs = {})
#   %mul_79 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %unsqueeze_212 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg134_1, -1), kwargs = {})
#   %unsqueeze_213 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_212, -1), kwargs = {})
#   %mul_80 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %unsqueeze_214 : Tensor "f32[112, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg135_1, -1), kwargs = {})
#   %unsqueeze_215 : Tensor "f32[112, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_214, -1), kwargs = {})
#   %add_56 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %add_57 : Tensor "f32[10, 112, 14, 14][21952, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %add_50), kwargs = {})
#   %convolution_27 : Tensor "f32[10, 672, 14, 14][131712, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_57, %arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf55
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3514112}, 'kernel_num_gb': 0.002636032, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 219520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 112)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 219520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002636032
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/3u/c3ud4tdxladfgie3oarmmsgxnvhg6kuwh2tphjo27dtdsfdtjore.py
# Topologically Sorted Source Nodes: [input_74, input_75, input_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_74 => add_60, add_61, mul_84, mul_85, mul_86, reciprocal_28, sqrt_28, sub_28, unsqueeze_224, unsqueeze_225, unsqueeze_226, unsqueeze_227, unsqueeze_228, unsqueeze_229, unsqueeze_230, unsqueeze_231
#   input_75 => clamp_max_18, clamp_min_18
#   input_76 => convolution_29
# Graph fragment:
#   %convolution_28 : Tensor "f32[10, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = PlaceHolder[target=convolution_28]
#   %arg142_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg142_1]
#   %arg143_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg143_1]
#   %arg144_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg144_1]
#   %arg145_1 : Tensor "f32[672][1]cuda:0" = PlaceHolder[target=arg145_1]
#   %unsqueeze_224 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg142_1, -1), kwargs = {})
#   %unsqueeze_225 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_224, -1), kwargs = {})
#   %sub_28 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %add_60 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg143_1, 1e-05), kwargs = {})
#   %sqrt_28 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_60,), kwargs = {})
#   %reciprocal_28 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_28,), kwargs = {})
#   %mul_84 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_28, 1), kwargs = {})
#   %unsqueeze_226 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_84, -1), kwargs = {})
#   %unsqueeze_227 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_226, -1), kwargs = {})
#   %mul_85 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %unsqueeze_228 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg144_1, -1), kwargs = {})
#   %unsqueeze_229 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_228, -1), kwargs = {})
#   %mul_86 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %unsqueeze_230 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg145_1, -1), kwargs = {})
#   %unsqueeze_231 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_230, -1), kwargs = {})
#   %add_61 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %clamp_min_18 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_61, 0.0), kwargs = {})
#   %clamp_max_18 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
#   %convolution_29 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_18, %arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf59
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3962112}, 'kernel_num_gb': 0.002644992, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 329280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 672)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 329280,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002644992
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/kx/ckxj5d5jyooolfik664t7o7e4f3ur2r5oajka4fhnztao6tydbbk.py
# Topologically Sorted Source Nodes: [input_74, input_75, input_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_74 => add_60, add_61, mul_84, mul_85, mul_86, reciprocal_28, sqrt_28, sub_28, unsqueeze_224, unsqueeze_225, unsqueeze_226, unsqueeze_227, unsqueeze_228, unsqueeze_229, unsqueeze_230, unsqueeze_231
#   input_75 => clamp_max_18, clamp_min_18
#   input_76 => convolution_29
# Graph fragment:
#   %buf59 : Tensor "f32[10, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = PlaceHolder[target=buf59]
#   %arg146_1 : Tensor "f32[192, 672, 1, 1][672, 1, 1, 1]cuda:0" = PlaceHolder[target=arg146_1]
#   %unsqueeze_224 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg142_1, -1), kwargs = {})
#   %unsqueeze_225 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_224, -1), kwargs = {})
#   %sub_28 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %add_60 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg143_1, 1e-05), kwargs = {})
#   %sqrt_28 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_60,), kwargs = {})
#   %reciprocal_28 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_28,), kwargs = {})
#   %mul_84 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_28, 1), kwargs = {})
#   %unsqueeze_226 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_84, -1), kwargs = {})
#   %unsqueeze_227 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_226, -1), kwargs = {})
#   %mul_85 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %unsqueeze_228 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg144_1, -1), kwargs = {})
#   %unsqueeze_229 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_228, -1), kwargs = {})
#   %mul_86 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %unsqueeze_230 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg145_1, -1), kwargs = {})
#   %unsqueeze_231 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_230, -1), kwargs = {})
#   %add_61 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %clamp_min_18 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_61, 0.0), kwargs = {})
#   %clamp_max_18 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
#   %convolution_29 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_18, %arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf60
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.002209536, 'kernel_flop': 126443520, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 192
    K = 672
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 672
    stride_ak = 1
    stride_bk = 1
    stride_bn = 672

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
        xindex = idx_n + 672*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 96, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002209536
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/cx/ccxhffukp3faa56sndxr27qv4ton6jnxpklw3azj7nwx3ordw2us.py
# Topologically Sorted Source Nodes: [input_74, input_75, input_76, input_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_74 => add_60, add_61, mul_84, mul_85, mul_86, reciprocal_28, sqrt_28, sub_28, unsqueeze_224, unsqueeze_225, unsqueeze_226, unsqueeze_227, unsqueeze_228, unsqueeze_229, unsqueeze_230, unsqueeze_231
#   input_75 => clamp_max_18, clamp_min_18
#   input_76 => convolution_29
#   input_77 => add_62, add_63, mul_87, mul_88, mul_89, reciprocal_29, sqrt_29, sub_29, unsqueeze_232, unsqueeze_233, unsqueeze_234, unsqueeze_235, unsqueeze_236, unsqueeze_237, unsqueeze_238, unsqueeze_239
# Graph fragment:
#   %buf60 : Tensor "f32[490, 192][192, 1]cuda:0" = PlaceHolder[target=buf60]
#   %arg147_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg147_1]
#   %arg148_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg148_1]
#   %arg149_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %arg150_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg150_1]
#   %unsqueeze_224 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg142_1, -1), kwargs = {})
#   %unsqueeze_225 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_224, -1), kwargs = {})
#   %sub_28 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %add_60 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg143_1, 1e-05), kwargs = {})
#   %sqrt_28 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_60,), kwargs = {})
#   %reciprocal_28 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_28,), kwargs = {})
#   %mul_84 : Tensor "f32[672][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_28, 1), kwargs = {})
#   %unsqueeze_226 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_84, -1), kwargs = {})
#   %unsqueeze_227 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_226, -1), kwargs = {})
#   %mul_85 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %unsqueeze_228 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg144_1, -1), kwargs = {})
#   %unsqueeze_229 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_228, -1), kwargs = {})
#   %mul_86 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %unsqueeze_230 : Tensor "f32[672, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg145_1, -1), kwargs = {})
#   %unsqueeze_231 : Tensor "f32[672, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_230, -1), kwargs = {})
#   %add_61 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %clamp_min_18 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_61, 0.0), kwargs = {})
#   %clamp_max_18 : Tensor "f32[10, 672, 7, 7][32928, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
#   %convolution_29 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_18, %arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_232 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg147_1, -1), kwargs = {})
#   %unsqueeze_233 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_232, -1), kwargs = {})
#   %sub_29 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %add_62 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg148_1, 1e-05), kwargs = {})
#   %sqrt_29 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_62,), kwargs = {})
#   %reciprocal_29 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_29,), kwargs = {})
#   %mul_87 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_29, 1), kwargs = {})
#   %unsqueeze_234 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_87, -1), kwargs = {})
#   %unsqueeze_235 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_234, -1), kwargs = {})
#   %mul_88 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %unsqueeze_236 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg149_1, -1), kwargs = {})
#   %unsqueeze_237 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_236, -1), kwargs = {})
#   %mul_89 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %unsqueeze_238 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, -1), kwargs = {})
#   %unsqueeze_239 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_238, -1), kwargs = {})
#   %add_63 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
#   return %add_63
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37', '''
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1132032}, 'kernel_num_gb': 0.000755712, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)


def get_args():
    arg_0 = rand_strided((10, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 94080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000755712
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ed/cedhzux6acbfye53y5omy3jufrim7dxem3trbtdn5xdj25aphfjm.py
# Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_78 => convolution_30
# Graph fragment:
#   %add_63 : Tensor "f32[10, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = PlaceHolder[target=add_63]
#   %arg151_1 : Tensor "f32[1152, 192, 1, 1][192, 1, 1, 1]cuda:0" = PlaceHolder[target=arg151_1]
#   %convolution_30 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_63, %arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf62
triton_tem_fused_convolution_38 = async_compile.triton('triton_tem_fused_convolution_38', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_38', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.003518976, 'kernel_flop': 216760320, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_convolution_38(arg_A, arg_B, out_ptr0):
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
    N = 1152
    K = 192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 192
    stride_ak = 1
    stride_bk = 1
    stride_bn = 192

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
        xindex = idx_n + 192*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1152*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1152*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 72, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_38.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_38.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.003518976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/p5/cp56td2vpphocrrd64yvz4jfjzjm2adjwitzra62kyarqfbvsyxk.py
# Topologically Sorted Source Nodes: [input_78, input_79, input_80], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_78 => convolution_30
#   input_79 => add_64, add_65, mul_90, mul_91, mul_92, reciprocal_30, sqrt_30, sub_30, unsqueeze_240, unsqueeze_241, unsqueeze_242, unsqueeze_243, unsqueeze_244, unsqueeze_245, unsqueeze_246, unsqueeze_247
#   input_80 => clamp_max_19, clamp_min_19
# Graph fragment:
#   %buf62 : Tensor "f32[490, 1152][1152, 1]cuda:0" = PlaceHolder[target=buf62]
#   %arg152_1 : Tensor "f32[1152][1]cuda:0" = PlaceHolder[target=arg152_1]
#   %arg153_1 : Tensor "f32[1152][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %arg154_1 : Tensor "f32[1152][1]cuda:0" = PlaceHolder[target=arg154_1]
#   %arg155_1 : Tensor "f32[1152][1]cuda:0" = PlaceHolder[target=arg155_1]
#   %convolution_30 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_63, %arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_240 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg152_1, -1), kwargs = {})
#   %unsqueeze_241 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_240, -1), kwargs = {})
#   %sub_30 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %add_64 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg153_1, 1e-05), kwargs = {})
#   %sqrt_30 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_64,), kwargs = {})
#   %reciprocal_30 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_30,), kwargs = {})
#   %mul_90 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_30, 1), kwargs = {})
#   %unsqueeze_242 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_90, -1), kwargs = {})
#   %unsqueeze_243 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_242, -1), kwargs = {})
#   %mul_91 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %unsqueeze_244 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg154_1, -1), kwargs = {})
#   %unsqueeze_245 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_244, -1), kwargs = {})
#   %mul_92 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %unsqueeze_246 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg155_1, -1), kwargs = {})
#   %unsqueeze_247 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_246, -1), kwargs = {})
#   %add_65 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %clamp_min_19 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_65, 0.0), kwargs = {})
#   %clamp_max_19 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6.0), kwargs = {})
#   return %clamp_max_19
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6792192}, 'kernel_num_gb': 0.004534272, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 564480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1152)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 564480,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004534272
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ty/ctyp2dquiaset6lfyp6qrpt2khctakeyn6sbrknptkmmz3ve2w3z.py
# Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_82 => add_66, add_67, mul_93, mul_94, mul_95, reciprocal_31, sqrt_31, sub_31, unsqueeze_248, unsqueeze_249, unsqueeze_250, unsqueeze_251, unsqueeze_252, unsqueeze_253, unsqueeze_254, unsqueeze_255
#   input_83 => clamp_max_20, clamp_min_20
#   input_84 => convolution_32
# Graph fragment:
#   %buf65 : Tensor "f32[10, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = PlaceHolder[target=buf65]
#   %arg161_1 : Tensor "f32[192, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = PlaceHolder[target=arg161_1]
#   %unsqueeze_248 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg157_1, -1), kwargs = {})
#   %unsqueeze_249 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_248, -1), kwargs = {})
#   %sub_31 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %add_66 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg158_1, 1e-05), kwargs = {})
#   %sqrt_31 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_66,), kwargs = {})
#   %reciprocal_31 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_31,), kwargs = {})
#   %mul_93 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_31, 1), kwargs = {})
#   %unsqueeze_250 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_93, -1), kwargs = {})
#   %unsqueeze_251 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_250, -1), kwargs = {})
#   %mul_94 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %unsqueeze_252 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg159_1, -1), kwargs = {})
#   %unsqueeze_253 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_252, -1), kwargs = {})
#   %mul_95 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_253), kwargs = {})
#   %unsqueeze_254 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg160_1, -1), kwargs = {})
#   %unsqueeze_255 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_254, -1), kwargs = {})
#   %add_67 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_255), kwargs = {})
#   %clamp_min_20 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_67, 0.0), kwargs = {})
#   %clamp_max_20 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_20, 6.0), kwargs = {})
#   %convolution_32 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_20, %arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf66
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.003518976, 'kernel_flop': 216760320, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 192
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1152

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
        xindex = idx_n + 1152*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 96, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.003518976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/sm/csmuk3pqm2elrntffnbruxjcs2o3pesxaegvqm3tajpdfi6m45bs.py
# Topologically Sorted Source Nodes: [input_82, input_83, input_84, input_85, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_82 => add_66, add_67, mul_93, mul_94, mul_95, reciprocal_31, sqrt_31, sub_31, unsqueeze_248, unsqueeze_249, unsqueeze_250, unsqueeze_251, unsqueeze_252, unsqueeze_253, unsqueeze_254, unsqueeze_255
#   input_83 => clamp_max_20, clamp_min_20
#   input_84 => convolution_32
#   input_85 => add_68, add_69, mul_96, mul_97, mul_98, reciprocal_32, sqrt_32, sub_32, unsqueeze_256, unsqueeze_257, unsqueeze_258, unsqueeze_259, unsqueeze_260, unsqueeze_261, unsqueeze_262, unsqueeze_263
#   x_5 => add_70
# Graph fragment:
#   %buf66 : Tensor "f32[490, 192][192, 1]cuda:0" = PlaceHolder[target=buf66]
#   %arg162_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg162_1]
#   %arg163_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg163_1]
#   %arg164_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg164_1]
#   %arg165_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg165_1]
#   %add_63 : Tensor "f32[10, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = PlaceHolder[target=add_63]
#   %unsqueeze_248 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg157_1, -1), kwargs = {})
#   %unsqueeze_249 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_248, -1), kwargs = {})
#   %sub_31 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %add_66 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg158_1, 1e-05), kwargs = {})
#   %sqrt_31 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_66,), kwargs = {})
#   %reciprocal_31 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_31,), kwargs = {})
#   %mul_93 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_31, 1), kwargs = {})
#   %unsqueeze_250 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_93, -1), kwargs = {})
#   %unsqueeze_251 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_250, -1), kwargs = {})
#   %mul_94 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %unsqueeze_252 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg159_1, -1), kwargs = {})
#   %unsqueeze_253 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_252, -1), kwargs = {})
#   %mul_95 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_253), kwargs = {})
#   %unsqueeze_254 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg160_1, -1), kwargs = {})
#   %unsqueeze_255 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_254, -1), kwargs = {})
#   %add_67 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_255), kwargs = {})
#   %clamp_min_20 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_67, 0.0), kwargs = {})
#   %clamp_max_20 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_20, 6.0), kwargs = {})
#   %convolution_32 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_20, %arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_256 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg162_1, -1), kwargs = {})
#   %unsqueeze_257 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_256, -1), kwargs = {})
#   %sub_32 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %add_68 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg163_1, 1e-05), kwargs = {})
#   %sqrt_32 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_68,), kwargs = {})
#   %reciprocal_32 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_32,), kwargs = {})
#   %mul_96 : Tensor "f32[192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_32, 1), kwargs = {})
#   %unsqueeze_258 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_96, -1), kwargs = {})
#   %unsqueeze_259 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_258, -1), kwargs = {})
#   %mul_97 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %unsqueeze_260 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg164_1, -1), kwargs = {})
#   %unsqueeze_261 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_260, -1), kwargs = {})
#   %mul_98 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %unsqueeze_262 : Tensor "f32[192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg165_1, -1), kwargs = {})
#   %unsqueeze_263 : Tensor "f32[192, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_262, -1), kwargs = {})
#   %add_69 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
#   %add_70 : Tensor "f32[10, 192, 7, 7][9408, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %add_63), kwargs = {})
#   return %add_70
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41', '''
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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1508352}, 'kernel_num_gb': 0.001132032, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def get_args():
    arg_0 = rand_strided((10, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 94080,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001132032
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ry/cryrtagabta6j3ynm7fkcaubts3e2uxs2ocsqhzzcxdeyvuv6gn7.py
# Topologically Sorted Source Nodes: [input_98, input_99, input_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   input_100 => convolution_38
#   input_98 => add_80, add_81, mul_111, mul_112, mul_113, reciprocal_37, sqrt_37, sub_37, unsqueeze_296, unsqueeze_297, unsqueeze_298, unsqueeze_299, unsqueeze_300, unsqueeze_301, unsqueeze_302, unsqueeze_303
#   input_99 => clamp_max_24, clamp_min_24
# Graph fragment:
#   %buf77 : Tensor "f32[10, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = PlaceHolder[target=buf77]
#   %arg191_1 : Tensor "f32[320, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = PlaceHolder[target=arg191_1]
#   %unsqueeze_296 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg187_1, -1), kwargs = {})
#   %unsqueeze_297 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_296, -1), kwargs = {})
#   %sub_37 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %add_80 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg188_1, 1e-05), kwargs = {})
#   %sqrt_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_80,), kwargs = {})
#   %reciprocal_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_37,), kwargs = {})
#   %mul_111 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_37, 1), kwargs = {})
#   %unsqueeze_298 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_111, -1), kwargs = {})
#   %unsqueeze_299 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_298, -1), kwargs = {})
#   %mul_112 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %unsqueeze_300 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg189_1, -1), kwargs = {})
#   %unsqueeze_301 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_300, -1), kwargs = {})
#   %mul_113 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %unsqueeze_302 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg190_1, -1), kwargs = {})
#   %unsqueeze_303 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_302, -1), kwargs = {})
#   %add_81 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %clamp_min_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_81, 0.0), kwargs = {})
#   %clamp_max_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
#   %convolution_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf78
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.00435968, 'kernel_flop': 361267200, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42(arg_A, arg_B, out_ptr0):
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
    N = 320
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1152

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
        xindex = idx_n + 1152*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 320*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 320*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 80, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00435968
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/se/cse2imzdgj4qcwblgzimdttrz2yjwwlemf7fohrppc6vgo2dnj5g.py
# Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_39 => convolution_39
#   input_100 => convolution_38
#   input_101 => add_82, add_83, mul_114, mul_115, mul_116, reciprocal_38, sqrt_38, sub_38, unsqueeze_304, unsqueeze_305, unsqueeze_306, unsqueeze_307, unsqueeze_308, unsqueeze_309, unsqueeze_310, unsqueeze_311
#   input_98 => add_80, add_81, mul_111, mul_112, mul_113, reciprocal_37, sqrt_37, sub_37, unsqueeze_296, unsqueeze_297, unsqueeze_298, unsqueeze_299, unsqueeze_300, unsqueeze_301, unsqueeze_302, unsqueeze_303
#   input_99 => clamp_max_24, clamp_min_24
# Graph fragment:
#   %buf78 : Tensor "f32[490, 320][320, 1]cuda:0" = PlaceHolder[target=buf78]
#   %arg192_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg192_1]
#   %arg193_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg193_1]
#   %arg194_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg194_1]
#   %arg195_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg195_1]
#   %unsqueeze_296 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg187_1, -1), kwargs = {})
#   %unsqueeze_297 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_296, -1), kwargs = {})
#   %sub_37 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %add_80 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg188_1, 1e-05), kwargs = {})
#   %sqrt_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_80,), kwargs = {})
#   %reciprocal_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_37,), kwargs = {})
#   %mul_111 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_37, 1), kwargs = {})
#   %unsqueeze_298 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_111, -1), kwargs = {})
#   %unsqueeze_299 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_298, -1), kwargs = {})
#   %mul_112 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %unsqueeze_300 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg189_1, -1), kwargs = {})
#   %unsqueeze_301 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_300, -1), kwargs = {})
#   %mul_113 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %unsqueeze_302 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg190_1, -1), kwargs = {})
#   %unsqueeze_303 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_302, -1), kwargs = {})
#   %add_81 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %clamp_min_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_81, 0.0), kwargs = {})
#   %clamp_max_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
#   %convolution_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_304 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg192_1, -1), kwargs = {})
#   %unsqueeze_305 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_304, -1), kwargs = {})
#   %sub_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %add_82 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg193_1, 1e-05), kwargs = {})
#   %sqrt_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_82,), kwargs = {})
#   %reciprocal_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_38,), kwargs = {})
#   %mul_114 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_38, 1), kwargs = {})
#   %unsqueeze_306 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_114, -1), kwargs = {})
#   %unsqueeze_307 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_306, -1), kwargs = {})
#   %mul_115 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %unsqueeze_308 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg194_1, -1), kwargs = {})
#   %unsqueeze_309 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_308, -1), kwargs = {})
#   %mul_116 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %unsqueeze_310 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg195_1, -1), kwargs = {})
#   %unsqueeze_311 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_310, -1), kwargs = {})
#   %add_83 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %convolution_39 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_83, %arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf79
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1886720}, 'kernel_num_gb': 0.00125952, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 320)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)


def get_args():
    arg_0 = rand_strided((10, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 156800,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00125952
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/pm/cpmdvroepf2mdzkajkoi4bkglzmudz5dkt25s7rrtnxzic4kw4wh.py
# Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_39 => convolution_39
#   input_100 => convolution_38
#   input_101 => add_82, add_83, mul_114, mul_115, mul_116, reciprocal_38, sqrt_38, sub_38, unsqueeze_304, unsqueeze_305, unsqueeze_306, unsqueeze_307, unsqueeze_308, unsqueeze_309, unsqueeze_310, unsqueeze_311
#   input_98 => add_80, add_81, mul_111, mul_112, mul_113, reciprocal_37, sqrt_37, sub_37, unsqueeze_296, unsqueeze_297, unsqueeze_298, unsqueeze_299, unsqueeze_300, unsqueeze_301, unsqueeze_302, unsqueeze_303
#   input_99 => clamp_max_24, clamp_min_24
# Graph fragment:
#   %buf79 : Tensor "f32[10, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = PlaceHolder[target=buf79]
#   %arg196_1 : Tensor "f32[1280, 320, 1, 1][320, 1, 1, 1]cuda:0" = PlaceHolder[target=arg196_1]
#   %unsqueeze_296 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg187_1, -1), kwargs = {})
#   %unsqueeze_297 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_296, -1), kwargs = {})
#   %sub_37 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %add_80 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg188_1, 1e-05), kwargs = {})
#   %sqrt_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_80,), kwargs = {})
#   %reciprocal_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_37,), kwargs = {})
#   %mul_111 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_37, 1), kwargs = {})
#   %unsqueeze_298 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_111, -1), kwargs = {})
#   %unsqueeze_299 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_298, -1), kwargs = {})
#   %mul_112 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %unsqueeze_300 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg189_1, -1), kwargs = {})
#   %unsqueeze_301 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_300, -1), kwargs = {})
#   %mul_113 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %unsqueeze_302 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg190_1, -1), kwargs = {})
#   %unsqueeze_303 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_302, -1), kwargs = {})
#   %add_81 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %clamp_min_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_81, 0.0), kwargs = {})
#   %clamp_max_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
#   %convolution_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_304 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg192_1, -1), kwargs = {})
#   %unsqueeze_305 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_304, -1), kwargs = {})
#   %sub_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %add_82 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg193_1, 1e-05), kwargs = {})
#   %sqrt_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_82,), kwargs = {})
#   %reciprocal_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_38,), kwargs = {})
#   %mul_114 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_38, 1), kwargs = {})
#   %unsqueeze_306 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_114, -1), kwargs = {})
#   %unsqueeze_307 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_306, -1), kwargs = {})
#   %mul_115 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %unsqueeze_308 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg194_1, -1), kwargs = {})
#   %unsqueeze_309 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_308, -1), kwargs = {})
#   %mul_116 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %unsqueeze_310 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg195_1, -1), kwargs = {})
#   %unsqueeze_311 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_310, -1), kwargs = {})
#   %add_83 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %convolution_39 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_83, %arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf80
triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0047744, 'kernel_flop': 401408000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44(arg_A, arg_B, out_ptr0):
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
    N = 1280
    K = 320
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 320
    stride_ak = 1
    stride_bk = 1
    stride_bn = 320

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
        xindex = idx_n + 320*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1280*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 320*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1280*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 80, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0047744
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/r6/cr67ynstkell2jxkebtjgc6worbktawfevxpb5sma7nmpmzdgohm.py
# Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39, batch_norm_39, x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   batch_norm_39 => add_84, add_85, mul_117, mul_118, mul_119, reciprocal_39, sqrt_39, sub_39, unsqueeze_312, unsqueeze_313, unsqueeze_314, unsqueeze_315, unsqueeze_316, unsqueeze_317, unsqueeze_318, unsqueeze_319
#   conv2d_39 => convolution_39
#   input_100 => convolution_38
#   input_101 => add_82, add_83, mul_114, mul_115, mul_116, reciprocal_38, sqrt_38, sub_38, unsqueeze_304, unsqueeze_305, unsqueeze_306, unsqueeze_307, unsqueeze_308, unsqueeze_309, unsqueeze_310, unsqueeze_311
#   input_98 => add_80, add_81, mul_111, mul_112, mul_113, reciprocal_37, sqrt_37, sub_37, unsqueeze_296, unsqueeze_297, unsqueeze_298, unsqueeze_299, unsqueeze_300, unsqueeze_301, unsqueeze_302, unsqueeze_303
#   input_99 => clamp_max_24, clamp_min_24
#   x_7 => relu_1
#   x_8 => mean
# Graph fragment:
#   %buf80 : Tensor "f32[490, 1280][1280, 1]cuda:0" = PlaceHolder[target=buf80]
#   %arg197_1 : Tensor "f32[1280][1]cuda:0" = PlaceHolder[target=arg197_1]
#   %arg198_1 : Tensor "f32[1280][1]cuda:0" = PlaceHolder[target=arg198_1]
#   %arg199_1 : Tensor "f32[1280][1]cuda:0" = PlaceHolder[target=arg199_1]
#   %arg200_1 : Tensor "f32[1280][1]cuda:0" = PlaceHolder[target=arg200_1]
#   %buf81 : Tensor "f32[10, 1280, 1, 1][1280, 1, 12800, 12800]cuda:0" = PlaceHolder[target=buf81]
#   %unsqueeze_296 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg187_1, -1), kwargs = {})
#   %unsqueeze_297 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_296, -1), kwargs = {})
#   %sub_37 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %add_80 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg188_1, 1e-05), kwargs = {})
#   %sqrt_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_80,), kwargs = {})
#   %reciprocal_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_37,), kwargs = {})
#   %mul_111 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_37, 1), kwargs = {})
#   %unsqueeze_298 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_111, -1), kwargs = {})
#   %unsqueeze_299 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_298, -1), kwargs = {})
#   %mul_112 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %unsqueeze_300 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg189_1, -1), kwargs = {})
#   %unsqueeze_301 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_300, -1), kwargs = {})
#   %mul_113 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %unsqueeze_302 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg190_1, -1), kwargs = {})
#   %unsqueeze_303 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_302, -1), kwargs = {})
#   %add_81 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %clamp_min_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_81, 0.0), kwargs = {})
#   %clamp_max_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
#   %convolution_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_304 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg192_1, -1), kwargs = {})
#   %unsqueeze_305 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_304, -1), kwargs = {})
#   %sub_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %add_82 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg193_1, 1e-05), kwargs = {})
#   %sqrt_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_82,), kwargs = {})
#   %reciprocal_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_38,), kwargs = {})
#   %mul_114 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_38, 1), kwargs = {})
#   %unsqueeze_306 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_114, -1), kwargs = {})
#   %unsqueeze_307 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_306, -1), kwargs = {})
#   %mul_115 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %unsqueeze_308 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg194_1, -1), kwargs = {})
#   %unsqueeze_309 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_308, -1), kwargs = {})
#   %mul_116 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %unsqueeze_310 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg195_1, -1), kwargs = {})
#   %unsqueeze_311 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_310, -1), kwargs = {})
#   %add_83 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %convolution_39 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_83, %arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_312 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg197_1, -1), kwargs = {})
#   %unsqueeze_313 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_312, -1), kwargs = {})
#   %sub_39 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %add_84 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg198_1, 1e-05), kwargs = {})
#   %sqrt_39 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_84,), kwargs = {})
#   %reciprocal_39 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_39,), kwargs = {})
#   %mul_117 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_39, 1), kwargs = {})
#   %unsqueeze_314 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_117, -1), kwargs = {})
#   %unsqueeze_315 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_314, -1), kwargs = {})
#   %mul_118 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %unsqueeze_316 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg199_1, -1), kwargs = {})
#   %unsqueeze_317 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_316, -1), kwargs = {})
#   %mul_119 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %unsqueeze_318 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg200_1, -1), kwargs = {})
#   %unsqueeze_319 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_318, -1), kwargs = {})
#   %add_85 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_1 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
#   %mean : Tensor "f32[10, 1280, 1, 1][1280, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_1, [-1, -2], True), kwargs = {})
#   return %buf81,%mean
triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45', '''
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
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2631680, 'r0_': 0}, 'kernel_num_gb': 0.00263168, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12800
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
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
    r0_2 = r0_index
    x0 = (xindex % 1280)
    x1 = xindex // 1280
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1280*r0_2 + 62720*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tl.full([1, 1], 49.0, tl.float32)
    tmp23 = (tmp21 / tmp22)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)


def get_args():
    arg_0 = rand_strided((10, 1280, 1, 1), (1280, 1, 12800, 12800), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((490, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12800, 49,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00263168
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/2q/c2qymau6rxh25csjqc55nzkg5xa74jasutbjbynfeivqt7uhptp5.py
# Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39, batch_norm_39, x_7, x_8, x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.relu, aten.mean, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   batch_norm_39 => add_84, add_85, mul_117, mul_118, mul_119, reciprocal_39, sqrt_39, sub_39, unsqueeze_312, unsqueeze_313, unsqueeze_314, unsqueeze_315, unsqueeze_316, unsqueeze_317, unsqueeze_318, unsqueeze_319
#   conv2d_39 => convolution_39
#   input_100 => convolution_38
#   input_101 => add_82, add_83, mul_114, mul_115, mul_116, reciprocal_38, sqrt_38, sub_38, unsqueeze_304, unsqueeze_305, unsqueeze_306, unsqueeze_307, unsqueeze_308, unsqueeze_309, unsqueeze_310, unsqueeze_311
#   input_98 => add_80, add_81, mul_111, mul_112, mul_113, reciprocal_37, sqrt_37, sub_37, unsqueeze_296, unsqueeze_297, unsqueeze_298, unsqueeze_299, unsqueeze_300, unsqueeze_301, unsqueeze_302, unsqueeze_303
#   input_99 => clamp_max_24, clamp_min_24
#   x_10 => addmm, permute
#   x_7 => relu_1
#   x_8 => mean
#   x_9 => view
# Graph fragment:
#   %arg202_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg202_1]
#   %mean : Tensor "f32[10, 1280, 1, 1][1280, 1, 12800, 12800]cuda:0" = PlaceHolder[target=mean]
#   %arg201_1 : Tensor "f32[1000, 1280][1280, 1]cuda:0" = PlaceHolder[target=arg201_1]
#   %unsqueeze_296 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg187_1, -1), kwargs = {})
#   %unsqueeze_297 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_296, -1), kwargs = {})
#   %sub_37 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_297), kwargs = {})
#   %add_80 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg188_1, 1e-05), kwargs = {})
#   %sqrt_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_80,), kwargs = {})
#   %reciprocal_37 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_37,), kwargs = {})
#   %mul_111 : Tensor "f32[1152][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_37, 1), kwargs = {})
#   %unsqueeze_298 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_111, -1), kwargs = {})
#   %unsqueeze_299 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_298, -1), kwargs = {})
#   %mul_112 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %unsqueeze_300 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg189_1, -1), kwargs = {})
#   %unsqueeze_301 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_300, -1), kwargs = {})
#   %mul_113 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %unsqueeze_302 : Tensor "f32[1152, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg190_1, -1), kwargs = {})
#   %unsqueeze_303 : Tensor "f32[1152, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_302, -1), kwargs = {})
#   %add_81 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %clamp_min_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_81, 0.0), kwargs = {})
#   %clamp_max_24 : Tensor "f32[10, 1152, 7, 7][56448, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
#   %convolution_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_24, %arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_304 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg192_1, -1), kwargs = {})
#   %unsqueeze_305 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_304, -1), kwargs = {})
#   %sub_38 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %add_82 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg193_1, 1e-05), kwargs = {})
#   %sqrt_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_82,), kwargs = {})
#   %reciprocal_38 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_38,), kwargs = {})
#   %mul_114 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_38, 1), kwargs = {})
#   %unsqueeze_306 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_114, -1), kwargs = {})
#   %unsqueeze_307 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_306, -1), kwargs = {})
#   %mul_115 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %unsqueeze_308 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg194_1, -1), kwargs = {})
#   %unsqueeze_309 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_308, -1), kwargs = {})
#   %mul_116 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %unsqueeze_310 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg195_1, -1), kwargs = {})
#   %unsqueeze_311 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_310, -1), kwargs = {})
#   %add_83 : Tensor "f32[10, 320, 7, 7][15680, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %convolution_39 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_83, %arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_312 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg197_1, -1), kwargs = {})
#   %unsqueeze_313 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_312, -1), kwargs = {})
#   %sub_39 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %add_84 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg198_1, 1e-05), kwargs = {})
#   %sqrt_39 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_84,), kwargs = {})
#   %reciprocal_39 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_39,), kwargs = {})
#   %mul_117 : Tensor "f32[1280][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_39, 1), kwargs = {})
#   %unsqueeze_314 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_117, -1), kwargs = {})
#   %unsqueeze_315 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_314, -1), kwargs = {})
#   %mul_118 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %unsqueeze_316 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg199_1, -1), kwargs = {})
#   %unsqueeze_317 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_316, -1), kwargs = {})
#   %mul_119 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %unsqueeze_318 : Tensor "f32[1280, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg200_1, -1), kwargs = {})
#   %unsqueeze_319 : Tensor "f32[1280, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_318, -1), kwargs = {})
#   %add_85 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_1 : Tensor "f32[10, 1280, 7, 7][62720, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_85,), kwargs = {})
#   %mean : Tensor "f32[10, 1280, 1, 1][1280, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_1, [-1, -2], True), kwargs = {})
#   %view : Tensor "f32[10, 1280][1280, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [10, -1]), kwargs = {})
#   %permute : Tensor "f32[1280, 1000][1, 1280]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg201_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "f32[10, 1000][1000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg202_1, %view, %permute), kwargs = {})
#   return %addmm
triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.0052512, 'kernel_flop': 25600000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46(in_ptr0, arg_A, arg_B, out_ptr0):
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
    K = 1280
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1280
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1280

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
        xindex = idx_n + 1280*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1280*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_1 = rand_strided((10, 1280, 1, 1), (1280, 1, 12800, 12800), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 32, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0052512
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1 = args
        args.clear()
        assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
        assert_size_stride(arg1_1, (10, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg2_1, (32, ), (1, ))
        assert_size_stride(arg3_1, (32, ), (1, ))
        assert_size_stride(arg4_1, (32, ), (1, ))
        assert_size_stride(arg5_1, (32, ), (1, ))
        assert_size_stride(arg6_1, (32, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg7_1, (32, ), (1, ))
        assert_size_stride(arg8_1, (32, ), (1, ))
        assert_size_stride(arg9_1, (32, ), (1, ))
        assert_size_stride(arg10_1, (32, ), (1, ))
        assert_size_stride(arg11_1, (16, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(arg12_1, (16, ), (1, ))
        assert_size_stride(arg13_1, (16, ), (1, ))
        assert_size_stride(arg14_1, (16, ), (1, ))
        assert_size_stride(arg15_1, (16, ), (1, ))
        assert_size_stride(arg16_1, (96, 16, 1, 1), (16, 1, 1, 1))
        assert_size_stride(arg17_1, (96, ), (1, ))
        assert_size_stride(arg18_1, (96, ), (1, ))
        assert_size_stride(arg19_1, (96, ), (1, ))
        assert_size_stride(arg20_1, (96, ), (1, ))
        assert_size_stride(arg21_1, (96, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg22_1, (96, ), (1, ))
        assert_size_stride(arg23_1, (96, ), (1, ))
        assert_size_stride(arg24_1, (96, ), (1, ))
        assert_size_stride(arg25_1, (96, ), (1, ))
        assert_size_stride(arg26_1, (24, 96, 1, 1), (96, 1, 1, 1))
        assert_size_stride(arg27_1, (24, ), (1, ))
        assert_size_stride(arg28_1, (24, ), (1, ))
        assert_size_stride(arg29_1, (24, ), (1, ))
        assert_size_stride(arg30_1, (24, ), (1, ))
        assert_size_stride(arg31_1, (144, 24, 1, 1), (24, 1, 1, 1))
        assert_size_stride(arg32_1, (144, ), (1, ))
        assert_size_stride(arg33_1, (144, ), (1, ))
        assert_size_stride(arg34_1, (144, ), (1, ))
        assert_size_stride(arg35_1, (144, ), (1, ))
        assert_size_stride(arg36_1, (144, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg37_1, (144, ), (1, ))
        assert_size_stride(arg38_1, (144, ), (1, ))
        assert_size_stride(arg39_1, (144, ), (1, ))
        assert_size_stride(arg40_1, (144, ), (1, ))
        assert_size_stride(arg41_1, (24, 144, 1, 1), (144, 1, 1, 1))
        assert_size_stride(arg42_1, (24, ), (1, ))
        assert_size_stride(arg43_1, (24, ), (1, ))
        assert_size_stride(arg44_1, (24, ), (1, ))
        assert_size_stride(arg45_1, (24, ), (1, ))
        assert_size_stride(arg46_1, (144, 24, 1, 1), (24, 1, 1, 1))
        assert_size_stride(arg47_1, (144, ), (1, ))
        assert_size_stride(arg48_1, (144, ), (1, ))
        assert_size_stride(arg49_1, (144, ), (1, ))
        assert_size_stride(arg50_1, (144, ), (1, ))
        assert_size_stride(arg51_1, (144, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg52_1, (144, ), (1, ))
        assert_size_stride(arg53_1, (144, ), (1, ))
        assert_size_stride(arg54_1, (144, ), (1, ))
        assert_size_stride(arg55_1, (144, ), (1, ))
        assert_size_stride(arg56_1, (40, 144, 1, 1), (144, 1, 1, 1))
        assert_size_stride(arg57_1, (40, ), (1, ))
        assert_size_stride(arg58_1, (40, ), (1, ))
        assert_size_stride(arg59_1, (40, ), (1, ))
        assert_size_stride(arg60_1, (40, ), (1, ))
        assert_size_stride(arg61_1, (240, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg62_1, (240, ), (1, ))
        assert_size_stride(arg63_1, (240, ), (1, ))
        assert_size_stride(arg64_1, (240, ), (1, ))
        assert_size_stride(arg65_1, (240, ), (1, ))
        assert_size_stride(arg66_1, (240, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg67_1, (240, ), (1, ))
        assert_size_stride(arg68_1, (240, ), (1, ))
        assert_size_stride(arg69_1, (240, ), (1, ))
        assert_size_stride(arg70_1, (240, ), (1, ))
        assert_size_stride(arg71_1, (40, 240, 1, 1), (240, 1, 1, 1))
        assert_size_stride(arg72_1, (40, ), (1, ))
        assert_size_stride(arg73_1, (40, ), (1, ))
        assert_size_stride(arg74_1, (40, ), (1, ))
        assert_size_stride(arg75_1, (40, ), (1, ))
        assert_size_stride(arg76_1, (240, 40, 1, 1), (40, 1, 1, 1))
        assert_size_stride(arg77_1, (240, ), (1, ))
        assert_size_stride(arg78_1, (240, ), (1, ))
        assert_size_stride(arg79_1, (240, ), (1, ))
        assert_size_stride(arg80_1, (240, ), (1, ))
        assert_size_stride(arg81_1, (240, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg82_1, (240, ), (1, ))
        assert_size_stride(arg83_1, (240, ), (1, ))
        assert_size_stride(arg84_1, (240, ), (1, ))
        assert_size_stride(arg85_1, (240, ), (1, ))
        assert_size_stride(arg86_1, (80, 240, 1, 1), (240, 1, 1, 1))
        assert_size_stride(arg87_1, (80, ), (1, ))
        assert_size_stride(arg88_1, (80, ), (1, ))
        assert_size_stride(arg89_1, (80, ), (1, ))
        assert_size_stride(arg90_1, (80, ), (1, ))
        assert_size_stride(arg91_1, (480, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg92_1, (480, ), (1, ))
        assert_size_stride(arg93_1, (480, ), (1, ))
        assert_size_stride(arg94_1, (480, ), (1, ))
        assert_size_stride(arg95_1, (480, ), (1, ))
        assert_size_stride(arg96_1, (480, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg97_1, (480, ), (1, ))
        assert_size_stride(arg98_1, (480, ), (1, ))
        assert_size_stride(arg99_1, (480, ), (1, ))
        assert_size_stride(arg100_1, (480, ), (1, ))
        assert_size_stride(arg101_1, (80, 480, 1, 1), (480, 1, 1, 1))
        assert_size_stride(arg102_1, (80, ), (1, ))
        assert_size_stride(arg103_1, (80, ), (1, ))
        assert_size_stride(arg104_1, (80, ), (1, ))
        assert_size_stride(arg105_1, (80, ), (1, ))
        assert_size_stride(arg106_1, (480, 80, 1, 1), (80, 1, 1, 1))
        assert_size_stride(arg107_1, (480, ), (1, ))
        assert_size_stride(arg108_1, (480, ), (1, ))
        assert_size_stride(arg109_1, (480, ), (1, ))
        assert_size_stride(arg110_1, (480, ), (1, ))
        assert_size_stride(arg111_1, (480, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg112_1, (480, ), (1, ))
        assert_size_stride(arg113_1, (480, ), (1, ))
        assert_size_stride(arg114_1, (480, ), (1, ))
        assert_size_stride(arg115_1, (480, ), (1, ))
        assert_size_stride(arg116_1, (112, 480, 1, 1), (480, 1, 1, 1))
        assert_size_stride(arg117_1, (112, ), (1, ))
        assert_size_stride(arg118_1, (112, ), (1, ))
        assert_size_stride(arg119_1, (112, ), (1, ))
        assert_size_stride(arg120_1, (112, ), (1, ))
        assert_size_stride(arg121_1, (672, 112, 1, 1), (112, 1, 1, 1))
        assert_size_stride(arg122_1, (672, ), (1, ))
        assert_size_stride(arg123_1, (672, ), (1, ))
        assert_size_stride(arg124_1, (672, ), (1, ))
        assert_size_stride(arg125_1, (672, ), (1, ))
        assert_size_stride(arg126_1, (672, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg127_1, (672, ), (1, ))
        assert_size_stride(arg128_1, (672, ), (1, ))
        assert_size_stride(arg129_1, (672, ), (1, ))
        assert_size_stride(arg130_1, (672, ), (1, ))
        assert_size_stride(arg131_1, (112, 672, 1, 1), (672, 1, 1, 1))
        assert_size_stride(arg132_1, (112, ), (1, ))
        assert_size_stride(arg133_1, (112, ), (1, ))
        assert_size_stride(arg134_1, (112, ), (1, ))
        assert_size_stride(arg135_1, (112, ), (1, ))
        assert_size_stride(arg136_1, (672, 112, 1, 1), (112, 1, 1, 1))
        assert_size_stride(arg137_1, (672, ), (1, ))
        assert_size_stride(arg138_1, (672, ), (1, ))
        assert_size_stride(arg139_1, (672, ), (1, ))
        assert_size_stride(arg140_1, (672, ), (1, ))
        assert_size_stride(arg141_1, (672, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg142_1, (672, ), (1, ))
        assert_size_stride(arg143_1, (672, ), (1, ))
        assert_size_stride(arg144_1, (672, ), (1, ))
        assert_size_stride(arg145_1, (672, ), (1, ))
        assert_size_stride(arg146_1, (192, 672, 1, 1), (672, 1, 1, 1))
        assert_size_stride(arg147_1, (192, ), (1, ))
        assert_size_stride(arg148_1, (192, ), (1, ))
        assert_size_stride(arg149_1, (192, ), (1, ))
        assert_size_stride(arg150_1, (192, ), (1, ))
        assert_size_stride(arg151_1, (1152, 192, 1, 1), (192, 1, 1, 1))
        assert_size_stride(arg152_1, (1152, ), (1, ))
        assert_size_stride(arg153_1, (1152, ), (1, ))
        assert_size_stride(arg154_1, (1152, ), (1, ))
        assert_size_stride(arg155_1, (1152, ), (1, ))
        assert_size_stride(arg156_1, (1152, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg157_1, (1152, ), (1, ))
        assert_size_stride(arg158_1, (1152, ), (1, ))
        assert_size_stride(arg159_1, (1152, ), (1, ))
        assert_size_stride(arg160_1, (1152, ), (1, ))
        assert_size_stride(arg161_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
        assert_size_stride(arg162_1, (192, ), (1, ))
        assert_size_stride(arg163_1, (192, ), (1, ))
        assert_size_stride(arg164_1, (192, ), (1, ))
        assert_size_stride(arg165_1, (192, ), (1, ))
        assert_size_stride(arg166_1, (1152, 192, 1, 1), (192, 1, 1, 1))
        assert_size_stride(arg167_1, (1152, ), (1, ))
        assert_size_stride(arg168_1, (1152, ), (1, ))
        assert_size_stride(arg169_1, (1152, ), (1, ))
        assert_size_stride(arg170_1, (1152, ), (1, ))
        assert_size_stride(arg171_1, (1152, 1, 5, 5), (25, 25, 5, 1))
        assert_size_stride(arg172_1, (1152, ), (1, ))
        assert_size_stride(arg173_1, (1152, ), (1, ))
        assert_size_stride(arg174_1, (1152, ), (1, ))
        assert_size_stride(arg175_1, (1152, ), (1, ))
        assert_size_stride(arg176_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
        assert_size_stride(arg177_1, (192, ), (1, ))
        assert_size_stride(arg178_1, (192, ), (1, ))
        assert_size_stride(arg179_1, (192, ), (1, ))
        assert_size_stride(arg180_1, (192, ), (1, ))
        assert_size_stride(arg181_1, (1152, 192, 1, 1), (192, 1, 1, 1))
        assert_size_stride(arg182_1, (1152, ), (1, ))
        assert_size_stride(arg183_1, (1152, ), (1, ))
        assert_size_stride(arg184_1, (1152, ), (1, ))
        assert_size_stride(arg185_1, (1152, ), (1, ))
        assert_size_stride(arg186_1, (1152, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(arg187_1, (1152, ), (1, ))
        assert_size_stride(arg188_1, (1152, ), (1, ))
        assert_size_stride(arg189_1, (1152, ), (1, ))
        assert_size_stride(arg190_1, (1152, ), (1, ))
        assert_size_stride(arg191_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
        assert_size_stride(arg192_1, (320, ), (1, ))
        assert_size_stride(arg193_1, (320, ), (1, ))
        assert_size_stride(arg194_1, (320, ), (1, ))
        assert_size_stride(arg195_1, (320, ), (1, ))
        assert_size_stride(arg196_1, (1280, 320, 1, 1), (320, 1, 1, 1))
        assert_size_stride(arg197_1, (1280, ), (1, ))
        assert_size_stride(arg198_1, (1280, ), (1, ))
        assert_size_stride(arg199_1, (1280, ), (1, ))
        assert_size_stride(arg200_1, (1280, ), (1, ))
        assert_size_stride(arg201_1, (1000, 1280), (1280, 1))
        assert_size_stride(arg202_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((10, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 30, 50176, stream=stream0)
            del arg1_1
            buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (10, 32, 112, 112), (401408, 1, 3584, 32), 'unknown_op')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [batch_norm, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 4014080, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            # Topologically Sorted Source Nodes: [batch_norm, x, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
            assert_size_stride(buf4, (10, 32, 112, 112), (401408, 1, 3584, 32), 'unknown_op')
            del arg6_1
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 4014080, stream=stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            buf6 = empty_strided_cuda((125440, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4.run(buf5, arg11_1, buf6, 980, 1, 1, stream=stream0)
            del arg11_1
            del buf5
            buf7 = reinterpret_tensor(buf6, (10, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, 2007040, stream=stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            buf8 = empty_strided_cuda((125440, 96), (96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6.run(buf7, arg16_1, buf8, 1960, 1, 1, stream=stream0)
            del arg16_1
            del buf7
            buf9 = reinterpret_tensor(buf8, (10, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7.run(buf9, arg17_1, arg18_1, arg19_1, arg20_1, 12042240, stream=stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
            assert_size_stride(buf10, (10, 96, 56, 56), (301056, 1, 5376, 96), 'unknown_op')
            del arg21_1
            del buf9
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8.run(buf11, arg22_1, arg23_1, arg24_1, arg25_1, 3010560, stream=stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            buf12 = empty_strided_cuda((31360, 24), (24, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9.run(buf11, arg26_1, buf12, 245, 1, 1, stream=stream0)
            del arg26_1
            del buf11
            buf13 = reinterpret_tensor(buf12, (10, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [input_10, input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10.run(buf13, arg27_1, arg28_1, arg29_1, arg30_1, 752640, stream=stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            buf14 = empty_strided_cuda((31360, 144), (144, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_11.run(buf13, arg31_1, buf14, 735, 1, 1, stream=stream0)
            del arg31_1
            buf15 = reinterpret_tensor(buf14, (10, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [input_14, input_15, input_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.run(buf15, arg32_1, arg33_1, arg34_1, arg35_1, 4515840, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            # Topologically Sorted Source Nodes: [input_14, input_15, input_16, input_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            buf16 = extern_kernels.convolution(buf15, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
            assert_size_stride(buf16, (10, 144, 56, 56), (451584, 1, 8064, 144), 'unknown_op')
            del arg36_1
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.run(buf17, arg37_1, arg38_1, arg39_1, arg40_1, 4515840, stream=stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            buf18 = empty_strided_cuda((31360, 24), (24, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13.run(buf17, arg41_1, buf18, 245, 1, 1, stream=stream0)
            del arg41_1
            buf19 = reinterpret_tensor(buf18, (10, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_1, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14.run(buf19, arg42_1, arg43_1, arg44_1, arg45_1, buf13, 752640, stream=stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            del buf13
            buf20 = reinterpret_tensor(buf17, (31360, 144), (144, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_1, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_11.run(buf19, arg46_1, buf20, 735, 1, 1, stream=stream0)
            del arg46_1
            del buf19
            buf21 = reinterpret_tensor(buf20, (10, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_1, input_22, input_23, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.run(buf21, arg47_1, arg48_1, arg49_1, arg50_1, 4515840, stream=stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            # Topologically Sorted Source Nodes: [input_18, input_19, input_20, input_21, x_1, input_22, input_23, input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            buf22 = extern_kernels.convolution(buf21, arg51_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
            assert_size_stride(buf22, (10, 144, 28, 28), (112896, 1, 4032, 144), 'unknown_op')
            del arg51_1
            del buf21
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [input_26, input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15.run(buf23, arg52_1, arg53_1, arg54_1, arg55_1, 1128960, stream=stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            buf24 = empty_strided_cuda((7840, 40), (40, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_26, input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16.run(buf23, arg56_1, buf24, 123, 1, 1, stream=stream0)
            del arg56_1
            del buf23
            buf25 = reinterpret_tensor(buf24, (10, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [input_26, input_27, input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17.run(buf25, arg57_1, arg58_1, arg59_1, arg60_1, 313600, stream=stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            buf26 = empty_strided_cuda((7840, 240), (240, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_18.run(buf25, arg61_1, buf26, 124, 1, 1, stream=stream0)
            del arg61_1
            buf27 = reinterpret_tensor(buf26, (10, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [input_30, input_31, input_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19.run(buf27, arg62_1, arg63_1, arg64_1, arg65_1, 1881600, stream=stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            # Topologically Sorted Source Nodes: [input_30, input_31, input_32, input_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            buf28 = extern_kernels.convolution(buf27, arg66_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
            assert_size_stride(buf28, (10, 240, 28, 28), (188160, 1, 6720, 240), 'unknown_op')
            del arg66_1
            del buf27
            buf29 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19.run(buf29, arg67_1, arg68_1, arg69_1, arg70_1, 1881600, stream=stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            buf30 = empty_strided_cuda((7840, 40), (40, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20.run(buf29, arg71_1, buf30, 123, 1, 1, stream=stream0)
            del arg71_1
            buf31 = reinterpret_tensor(buf30, (10, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36, input_37, x_2, input_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21.run(buf31, arg72_1, arg73_1, arg74_1, arg75_1, buf25, 313600, stream=stream0)
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            del buf25
            buf32 = reinterpret_tensor(buf29, (7840, 240), (240, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36, input_37, x_2, input_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_18.run(buf31, arg76_1, buf32, 124, 1, 1, stream=stream0)
            del arg76_1
            del buf31
            buf33 = reinterpret_tensor(buf32, (10, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36, input_37, x_2, input_38, input_39, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19.run(buf33, arg77_1, arg78_1, arg79_1, arg80_1, 1881600, stream=stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36, input_37, x_2, input_38, input_39, input_40, input_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            buf34 = extern_kernels.convolution(buf33, arg81_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
            assert_size_stride(buf34, (10, 240, 14, 14), (47040, 1, 3360, 240), 'unknown_op')
            del arg81_1
            del buf33
            buf35 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [input_42, input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22.run(buf35, arg82_1, arg83_1, arg84_1, arg85_1, 470400, stream=stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            buf36 = empty_strided_cuda((1960, 80), (80, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_42, input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23.run(buf35, arg86_1, buf36, 186, 1, 1, stream=stream0)
            del arg86_1
            del buf35
            buf37 = reinterpret_tensor(buf36, (10, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [input_42, input_43, input_44, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24.run(buf37, arg87_1, arg88_1, arg89_1, arg90_1, 156800, stream=stream0)
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            buf38 = empty_strided_cuda((1960, 480), (480, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_25.run(buf37, arg91_1, buf38, 128, 1, 1, stream=stream0)
            del arg91_1
            buf39 = reinterpret_tensor(buf38, (10, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26.run(buf39, arg92_1, arg93_1, arg94_1, arg95_1, 940800, stream=stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            # Topologically Sorted Source Nodes: [input_46, input_47, input_48, input_49], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            buf40 = extern_kernels.convolution(buf39, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
            assert_size_stride(buf40, (10, 480, 14, 14), (94080, 1, 6720, 480), 'unknown_op')
            del arg96_1
            del buf39
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26.run(buf41, arg97_1, arg98_1, arg99_1, arg100_1, 940800, stream=stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            buf42 = empty_strided_cuda((1960, 80), (80, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27.run(buf41, arg101_1, buf42, 124, 1, 1, stream=stream0)
            del arg101_1
            buf43 = reinterpret_tensor(buf42, (10, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [input_50, input_51, input_52, input_53, x_3, input_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28.run(buf43, arg102_1, arg103_1, arg104_1, arg105_1, buf37, 156800, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            del arg105_1
            del buf37
            buf44 = reinterpret_tensor(buf41, (1960, 480), (480, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [input_50, input_51, input_52, input_53, x_3, input_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_25.run(buf43, arg106_1, buf44, 128, 1, 1, stream=stream0)
            del arg106_1
            buf45 = reinterpret_tensor(buf44, (10, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [input_50, input_51, input_52, input_53, x_3, input_54, input_55, input_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26.run(buf45, arg107_1, arg108_1, arg109_1, arg110_1, 940800, stream=stream0)
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [input_50, input_51, input_52, input_53, x_3, input_54, input_55, input_56, input_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            buf46 = extern_kernels.convolution(buf45, arg111_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
            assert_size_stride(buf46, (10, 480, 14, 14), (94080, 1, 6720, 480), 'unknown_op')
            del arg111_1
            del buf45
            buf47 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [input_58, input_59, input_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26.run(buf47, arg112_1, arg113_1, arg114_1, arg115_1, 940800, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            buf48 = empty_strided_cuda((1960, 112), (112, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_58, input_59, input_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29.run(buf47, arg116_1, buf48, 124, 1, 1, stream=stream0)
            del arg116_1
            del buf47
            buf49 = reinterpret_tensor(buf48, (10, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [input_58, input_59, input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30.run(buf49, arg117_1, arg118_1, arg119_1, arg120_1, 219520, stream=stream0)
            del arg117_1
            del arg118_1
            del arg119_1
            del arg120_1
            buf50 = empty_strided_cuda((1960, 672), (672, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_31.run(buf49, arg121_1, buf50, 176, 1, 1, stream=stream0)
            del arg121_1
            buf51 = reinterpret_tensor(buf50, (10, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [input_62, input_63, input_64], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.run(buf51, arg122_1, arg123_1, arg124_1, arg125_1, 1317120, stream=stream0)
            del arg122_1
            del arg123_1
            del arg124_1
            del arg125_1
            # Topologically Sorted Source Nodes: [input_62, input_63, input_64, input_65], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            buf52 = extern_kernels.convolution(buf51, arg126_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
            assert_size_stride(buf52, (10, 672, 14, 14), (131712, 1, 9408, 672), 'unknown_op')
            del arg126_1
            del buf51
            buf53 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.run(buf53, arg127_1, arg128_1, arg129_1, arg130_1, 1317120, stream=stream0)
            del arg127_1
            del arg128_1
            del arg129_1
            del arg130_1
            buf54 = empty_strided_cuda((1960, 112), (112, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33.run(buf53, arg131_1, buf54, 124, 1, 1, stream=stream0)
            del arg131_1
            buf55 = reinterpret_tensor(buf54, (10, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [input_66, input_67, input_68, input_69, x_4, input_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34.run(buf55, arg132_1, arg133_1, arg134_1, arg135_1, buf49, 219520, stream=stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            del arg135_1
            del buf49
            buf56 = reinterpret_tensor(buf53, (1960, 672), (672, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [input_66, input_67, input_68, input_69, x_4, input_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_31.run(buf55, arg136_1, buf56, 176, 1, 1, stream=stream0)
            del arg136_1
            del buf55
            buf57 = reinterpret_tensor(buf56, (10, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [input_66, input_67, input_68, input_69, x_4, input_70, input_71, input_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.run(buf57, arg137_1, arg138_1, arg139_1, arg140_1, 1317120, stream=stream0)
            del arg137_1
            del arg138_1
            del arg139_1
            del arg140_1
            # Topologically Sorted Source Nodes: [input_66, input_67, input_68, input_69, x_4, input_70, input_71, input_72, input_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            buf58 = extern_kernels.convolution(buf57, arg141_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
            assert_size_stride(buf58, (10, 672, 7, 7), (32928, 1, 4704, 672), 'unknown_op')
            del arg141_1
            del buf57
            buf59 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [input_74, input_75, input_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35.run(buf59, arg142_1, arg143_1, arg144_1, arg145_1, 329280, stream=stream0)
            del arg142_1
            del arg143_1
            del arg144_1
            del arg145_1
            buf60 = empty_strided_cuda((490, 192), (192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_74, input_75, input_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36.run(buf59, arg146_1, buf60, 96, 1, 1, stream=stream0)
            del arg146_1
            del buf59
            buf61 = reinterpret_tensor(buf60, (10, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [input_74, input_75, input_76, input_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37.run(buf61, arg147_1, arg148_1, arg149_1, arg150_1, 94080, stream=stream0)
            del arg147_1
            del arg148_1
            del arg149_1
            del arg150_1
            buf62 = empty_strided_cuda((490, 1152), (1152, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_38.run(buf61, arg151_1, buf62, 72, 1, 1, stream=stream0)
            del arg151_1
            buf63 = reinterpret_tensor(buf62, (10, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [input_78, input_79, input_80], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(buf63, arg152_1, arg153_1, arg154_1, arg155_1, 564480, stream=stream0)
            del arg152_1
            del arg153_1
            del arg154_1
            del arg155_1
            # Topologically Sorted Source Nodes: [input_78, input_79, input_80, input_81], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            buf64 = extern_kernels.convolution(buf63, arg156_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
            assert_size_stride(buf64, (10, 1152, 7, 7), (56448, 1, 8064, 1152), 'unknown_op')
            del arg156_1
            del buf63
            buf65 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(buf65, arg157_1, arg158_1, arg159_1, arg160_1, 564480, stream=stream0)
            del arg157_1
            del arg158_1
            del arg159_1
            del arg160_1
            buf66 = empty_strided_cuda((490, 192), (192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40.run(buf65, arg161_1, buf66, 96, 1, 1, stream=stream0)
            del arg161_1
            buf67 = reinterpret_tensor(buf66, (10, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [input_82, input_83, input_84, input_85, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41.run(buf67, arg162_1, arg163_1, arg164_1, arg165_1, buf61, 94080, stream=stream0)
            del arg162_1
            del arg163_1
            del arg164_1
            del arg165_1
            buf68 = reinterpret_tensor(buf65, (490, 1152), (1152, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_38.run(buf67, arg166_1, buf68, 72, 1, 1, stream=stream0)
            del arg166_1
            buf69 = reinterpret_tensor(buf68, (10, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [input_86, input_87, input_88], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(buf69, arg167_1, arg168_1, arg169_1, arg170_1, 564480, stream=stream0)
            del arg167_1
            del arg168_1
            del arg169_1
            del arg170_1
            # Topologically Sorted Source Nodes: [input_86, input_87, input_88, input_89], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
            buf70 = extern_kernels.convolution(buf69, arg171_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
            assert_size_stride(buf70, (10, 1152, 7, 7), (56448, 1, 8064, 1152), 'unknown_op')
            del arg171_1
            del buf69
            buf71 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [input_90, input_91, input_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(buf71, arg172_1, arg173_1, arg174_1, arg175_1, 564480, stream=stream0)
            del arg172_1
            del arg173_1
            del arg174_1
            del arg175_1
            buf72 = reinterpret_tensor(buf61, (490, 192), (192, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [input_90, input_91, input_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40.run(buf71, arg176_1, buf72, 96, 1, 1, stream=stream0)
            del arg176_1
            buf73 = reinterpret_tensor(buf72, (10, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [input_90, input_91, input_92, input_93, x_6, input_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41.run(buf73, arg177_1, arg178_1, arg179_1, arg180_1, buf67, 94080, stream=stream0)
            del arg177_1
            del arg178_1
            del arg179_1
            del arg180_1
            del buf67
            buf74 = reinterpret_tensor(buf71, (490, 1152), (1152, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [input_90, input_91, input_92, input_93, x_6, input_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_tem_fused_convolution_38.run(buf73, arg181_1, buf74, 72, 1, 1, stream=stream0)
            del arg181_1
            del buf73
            buf75 = reinterpret_tensor(buf74, (10, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [input_90, input_91, input_92, input_93, x_6, input_94, input_95, input_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(buf75, arg182_1, arg183_1, arg184_1, arg185_1, 564480, stream=stream0)
            del arg182_1
            del arg183_1
            del arg184_1
            del arg185_1
            # Topologically Sorted Source Nodes: [input_90, input_91, input_92, input_93, x_6, input_94, input_95, input_96, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.add]
            buf76 = extern_kernels.convolution(buf75, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
            assert_size_stride(buf76, (10, 1152, 7, 7), (56448, 1, 8064, 1152), 'unknown_op')
            del arg186_1
            del buf75
            buf77 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [input_98, input_99, input_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39.run(buf77, arg187_1, arg188_1, arg189_1, arg190_1, 564480, stream=stream0)
            del arg187_1
            del arg188_1
            del arg189_1
            del arg190_1
            buf78 = reinterpret_tensor(buf43, (490, 320), (320, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [input_98, input_99, input_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42.run(buf77, arg191_1, buf78, 80, 1, 1, stream=stream0)
            del arg191_1
            del buf77
            buf79 = reinterpret_tensor(buf78, (10, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43.run(buf79, arg192_1, arg193_1, arg194_1, arg195_1, 156800, stream=stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del arg195_1
            buf80 = empty_strided_cuda((490, 1280), (1280, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44.run(buf79, arg196_1, buf80, 80, 1, 1, stream=stream0)
            del arg196_1
            del buf79
            buf81 = empty_strided_cuda((10, 1280, 1, 1), (1280, 1, 12800, 12800), torch.float32)
            buf82 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39, batch_norm_39, x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.relu, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45.run(buf82, buf80, arg197_1, arg198_1, arg199_1, arg200_1, 12800, 49, stream=stream0)
            del arg197_1
            del arg198_1
            del arg199_1
            del arg200_1
            del buf80
            buf83 = empty_strided_cuda((10, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_98, input_99, input_100, input_101, conv2d_39, batch_norm_39, x_7, x_8, x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution, aten.relu, aten.mean, aten.view, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46.run(arg202_1, buf82, arg201_1, buf83, 32, 1, 1, stream=stream0)
            del arg201_1
            del arg202_1
            del buf82
        return (buf83, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in PyTorch.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB0 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

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
    ynumel = 96
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
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
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
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 32)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 125440
    N = 16
    K = 32
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 32
    stride_ak = 1
    stride_bk = 1
    stride_bn = 32

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
        xindex = idx_n + 32*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 16*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 125440
    N = 96
    K = 16
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 16
    stride_ak = 1
    stride_bk = 1
    stride_bn = 16

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
        xindex = idx_n + 16*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 96*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 16*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 96*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12042240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 24
    K = 96
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 96
    stride_ak = 1
    stride_bk = 1
    stride_bn = 96

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
        xindex = idx_n + 96*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 24*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 24*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_11(arg_A, arg_B, out_ptr0):
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
    N = 144
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
        xindex = idx_n + 144*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 24*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 144*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4515840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 31360
    N = 24
    K = 144
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 144
    stride_ak = 1
    stride_bk = 1
    stride_bn = 144

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
        xindex = idx_n + 144*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 24*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 24*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 40
    K = 144
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 144
    stride_ak = 1
    stride_bk = 1
    stride_bn = 144

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
        xindex = idx_n + 144*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 40*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 40*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_18(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 240
    K = 40
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 40
    stride_ak = 1
    stride_bk = 1
    stride_bn = 40

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
        xindex = idx_n + 40*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 240*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 40*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


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
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
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

    M = 7840
    N = 40
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
        xindex = idx_n + 40*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 40*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 470400
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 80
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
        xindex = idx_n + 80*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 80*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_25(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 480
    K = 80
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 80
    stride_ak = 1
    stride_bk = 1
    stride_bn = 80

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
        xindex = idx_n + 80*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 480*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 80*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


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
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 480)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 80
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
        xindex = idx_n + 80*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 80*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_29(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 112
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
        xindex = idx_n + 112*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 112*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 219520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 112)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_31(arg_A, arg_B, out_ptr0):
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

    M = 1960
    N = 672
    K = 112
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 112
    stride_ak = 1
    stride_bk = 1
    stride_bn = 112

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
        xindex = idx_n + 112*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 672*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 112*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 672*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1317120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 672)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 112
    K = 672
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 672
    stride_ak = 1
    stride_bk = 1
    stride_bn = 672

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
        xindex = idx_n + 672*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 112*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 112*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 219520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 112)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 329280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 672)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 192
    K = 672
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 672
    stride_ak = 1
    stride_bk = 1
    stride_bn = 672

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
        xindex = idx_n + 672*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)

''',
    '''@triton.jit
def triton_tem_fused_convolution_38(arg_A, arg_B, out_ptr0):
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
    N = 1152
    K = 192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 192
    stride_ak = 1
    stride_bk = 1
    stride_bn = 192

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
        xindex = idx_n + 192*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1152*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1152*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 564480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1152)
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
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 192
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1152

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
        xindex = idx_n + 1152*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_42(arg_A, arg_B, out_ptr0):
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
    N = 320
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1152

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
        xindex = idx_n + 1152*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 320*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 320*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 320)
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44(arg_A, arg_B, out_ptr0):
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
    N = 1280
    K = 320
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 320
    stride_ak = 1
    stride_bk = 1
    stride_bn = 320

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
        xindex = idx_n + 320*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1280*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 320*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1280*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12800
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
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
    r0_2 = r0_index
    x0 = (xindex % 1280)
    x1 = xindex // 1280
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1280*r0_2 + 62720*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tl.full([1, 1], 49.0, tl.float32)
    tmp23 = (tmp21 / tmp22)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)

''',
    '''@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46(in_ptr0, arg_A, arg_B, out_ptr0):
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
    K = 1280
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1280
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1280

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
        xindex = idx_n + 1280*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1280*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((40,), (1,), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((112,), (1,), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((672,), (1,), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((320,), (1,), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1]

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
