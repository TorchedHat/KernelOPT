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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ve/cveobhbgctxnpnub3q4ldnvelhasoluyhcrxnlrv6u65ydqbqpqp.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[512, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 32768, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 268435456, 'x': 134217728}, 'kernel_num_gb': 0.268435456, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 65536*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((512, 64, 32, 32), (65536, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 32768, 1024,


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
    num_gb = 0.268435456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4g/c4gphst5wkxcyzeq3p4wyqz67t6ejj747vfz6g47poj5dysgduda.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 128, 5, 5][3200, 25, 5, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 8192, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1638400, 'x': 819200}, 'kernel_num_gb': 0.0016384, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 3200*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((64, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 128, 5, 5), (3200, 1, 640, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 8192, 25,


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
    num_gb = 0.0016384
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/om/comnt3uw556yu4edar3pbtoidqufxpwebfm7unplhmfiitsrjobe.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.tanh]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => tanh
# Graph fragment:
#   %buf2 : Tensor "f32[512, 128, 34, 34][147968, 1, 4352, 128]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %tanh : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_1,), kwargs = {})
#   return %tanh
triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2', '''
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
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 909117952}, 'kernel_num_gb': 0.606079488, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75759616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.tanh(tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)


def get_args():
    arg_0 = rand_strided((512, 128, 34, 34), (147968, 1, 4352, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 75759616,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.606079488
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ig/ciger3jvjp3227syv2bdb6zybym5akkyffzvxhn5t7k5ucj4wjj7.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.tanh, aten.max_pool2d_with_indices, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => tanh
#   x_3 => _low_memory_max_pool_with_offsets
#   x_4 => var_mean, view
# Graph fragment:
#   %tanh : Tensor "f32[512, 128, 34, 34][147968, 1, 4352, 128]cuda:0" = PlaceHolder[target=tanh]
#   %convolution : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %tanh : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%tanh, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %view : Tensor "f32[512, 8, 16, 289][36992, 4624, 289, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [512, 8, 16, 289]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_3,%buf5
triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3', '''
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
    size_hints={'x': 4096, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 65536, 'r0_': 303038464}, 'kernel_num_gb': 0.303071232, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4624
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (128 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (4352 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (4480 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = triton_helpers.maximum(tmp0, tmp1)
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp12, None)


def get_args():
    arg_0 = rand_strided((512, 128, 34, 34), (147968, 1, 4352, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 4096, 4624,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.303071232
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ak/cakrl4yrhfkp36gylqm3klzwo7hkazpbupwo566wvhsrsp5y2w24.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.tanh, aten.max_pool2d_with_indices, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => tanh
#   x_3 => _low_memory_max_pool_with_offsets
#   x_4 => add_2, add_3, mul_3, mul_4, rsqrt, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_8, unsqueeze_9, var_mean, view, view_1
# Graph fragment:
#   %tanh : Tensor "f32[512, 128, 34, 34][147968, 1, 4352, 128]cuda:0" = PlaceHolder[target=tanh]
#   %getitem_3 : Tensor "f32[512, 8, 1, 1][8, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf5 : Tensor "f32[512, 8, 1, 1][8, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf5]
#   %arg7_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convolution : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %tanh : Tensor "f32[512, 128, 34, 34][147968, 1156, 34, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%tanh, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %view : Tensor "f32[512, 8, 16, 289][36992, 4624, 289, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [512, 8, 16, 289]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[512, 8, 16, 289][36992, 4624, 289, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_3), kwargs = {})
#   %add_2 : Tensor "f32[512, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[512, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_3 : Tensor "f32[512, 8, 16, 289][36992, 4624, 289, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[512, 128, 17, 17][36992, 289, 17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [512, 128, 17, 17]), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 2), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 3), kwargs = {})
#   %mul_4 : Tensor "f32[512, 128, 17, 17][36992, 289, 17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_10), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 2), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 3), kwargs = {})
#   %add_3 : Tensor "f32[512, 128, 17, 17][36992, 289, 17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   return %add_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4', '''
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
    size_hints={'y': 262144, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 151519232, 'x': 303072256}, 'kernel_num_gb': 0.378831872, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147968
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 17)
    y4 = yindex // 17
    y2 = yindex // 289
    y5 = (yindex % 289)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4352 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4480 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (8*y2 + (x3 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (8*y2 + (x3 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = tmp6 - tmp7
    tmp10 = tl.full([1, 1], 4624.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y5 + 289*x3 + 36992*y2), tmp19, xmask & ymask)


def get_args():
    arg_0 = rand_strided((512, 128, 34, 34), (147968, 1, 4352, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((512, 128, 17, 17), (36992, 289, 17, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 147968, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.378831872
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 128, 5, 5), (3200, 25, 5, 1))
        assert_size_stride(arg1_1, (128, ), (1, ))
        assert_size_stride(arg2_1, (512, 64, 32, 32), (65536, 1024, 32, 1))
        assert_size_stride(arg3_1, (128, ), (1, ))
        assert_size_stride(arg4_1, (128, ), (1, ))
        assert_size_stride(arg5_1, (128, ), (1, ))
        assert_size_stride(arg6_1, (128, ), (1, ))
        assert_size_stride(arg7_1, (128, ), (1, ))
        assert_size_stride(arg8_1, (128, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((512, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 32768, 1024, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 8192, 25, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (512, 128, 34, 34), (147968, 1, 4352, 128), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2.run(buf3, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, 75759616, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf4 = empty_strided_cuda((512, 8, 1, 1), (8, 1, 4096, 4096), torch.float32)
            buf5 = empty_strided_cuda((512, 8, 1, 1), (8, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.tanh, aten.max_pool2d_with_indices, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3.run(buf3, buf4, buf5, 4096, 4624, stream=stream0)
            buf7 = empty_strided_cuda((512, 128, 17, 17), (36992, 289, 17, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.tanh, aten.max_pool2d_with_indices, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4.run(buf3, buf4, buf5, arg7_1, arg8_1, buf7, 147968, 128, stream=stream0)
            del arg7_1
            del arg8_1
            del buf3
            del buf4
            del buf5
        return (buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 64, 32, 32), (65536, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 65536*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 3200*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75759616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.tanh(tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)

''',
    '''@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4624
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (128 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (4352 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (4480 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.maximum(tmp0, tmp1)
        tmp4 = tl.maximum(tmp2, tmp3)
        tmp6 = tl.maximum(tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp12, None)

''',
    '''@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147968
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 17)
    y4 = yindex // 17
    y2 = yindex // 289
    y5 = (yindex % 289)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4352 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4480 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (8*y2 + (x3 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (8*y2 + (x3 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tmp6 - tmp7
    tmp10 = tl.full([1, 1], 4624.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y5 + 289*x3 + 36992*y2), tmp19, xmask & ymask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((64, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 64, 32, 32), (65536, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]

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
