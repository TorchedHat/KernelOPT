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


# kernel path: /workspace/tmp/torchinductor_user/rw/crw5i634rrnmyzcofjubzr3yogt5l6fpsawoe5r2asp2ry45uhuh.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f32[8, 8, 64, 512][262144, 32768, 512, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    size_hints={'y': 64, 'x': 32768}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 8388608}, 'kernel_num_gb': 0.016777216, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 32768*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 262144*y1), tmp0, ymask)


def get_args():
    arg_0 = rand_strided((8, 8, 64, 512), (262144, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 8, 64, 512), (262144, 1, 4096, 8), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 64, 32768,


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
    num_gb = 0.016777216
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/yg/cyggg4ke2deuarwqt76km52xh63wigodq5qcjvs7frv4x4cwuitz.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 8, 3, 3][72, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /workspace/tmp/torchinductor_user/ux/cuxiru5c3bkyg62ldildy5s5uqlc2dnfk5bj7ynpqzx4gmnx62l5.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   input_3 => div
# Graph fragment:
#   %buf2 : Tensor "f32[8, 64, 64, 512][2097152, 1, 32768, 64]cuda:0" = PlaceHolder[target=buf2]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %add_1 : Tensor "f32[8, 64, 64, 512][2097152, 1, 32768, 64]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_42 : Tensor "f32[8, 64, 64, 1][4096, 1, 64, 32768]cuda:0" = PlaceHolder[target=getitem_42]
#   %getitem_43 : Tensor "f32[8, 64, 64, 1][4096, 1, 64, 32768]cuda:0" = PlaceHolder[target=getitem_43]
#   %convolution : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %prepare_softmax_online_default_17 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_1, -1), kwargs = {})
#   %sub_tensor_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_42), kwargs = {})
#   %exp_default_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_17,), kwargs = {})
#   %div : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_17, %getitem_43), kwargs = {})
#   return %add_1,%getitem_42,%getitem_43,%div
triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2', '''
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
    size_hints={'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 201327872, 'r0_': 0}, 'kernel_num_gb': 0.134219008, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = libdevice.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), tmp24, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 64, 64, 512), (2097152, 1, 32768, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.134219008
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/my/cmysrr3kwr2jttyn6mni3cxp5xo5ceqhlcruml5x4rlbrzlhuhk5.py
# Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   input_3 => div
#   input_4 => convolution_1
# Graph fragment:
#   %arg7_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %sub_tensor_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_42), kwargs = {})
#   %exp_default_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_17,), kwargs = {})
#   %div : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_17, %getitem_43), kwargs = {})
#   %convolution_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf7
triton_poi_fused__softmax_convolution_exp_sub_3 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_3', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 294912, 'x': 147456}, 'kernel_num_gb': 0.000294912, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        triton_poi_fused__softmax_convolution_exp_sub_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000294912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/tx/ctxnvg7mdz6mdoribt6uzl7nfzqxtsdcknv4vwdfjuognglcdyof.py
# Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_3 => div
#   input_4 => convolution_1
#   input_5 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_2, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_6 => div_1
# Graph fragment:
#   %buf8 : Tensor "f32[8, 64, 64, 512][2097152, 1, 32768, 64]cuda:0" = PlaceHolder[target=buf8]
#   %arg8_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %arg12_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_3 : Tensor "f32[8, 64, 64, 512][2097152, 1, 32768, 64]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_40 : Tensor "f32[8, 64, 64, 1][4096, 1, 64, 32768]cuda:0" = PlaceHolder[target=getitem_40]
#   %getitem_41 : Tensor "f32[8, 64, 64, 1][4096, 1, 64, 32768]cuda:0" = PlaceHolder[target=getitem_41]
#   %sub_tensor_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_42), kwargs = {})
#   %exp_default_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_17,), kwargs = {})
#   %div : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_17, %getitem_43), kwargs = {})
#   %convolution_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg7_1, %arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_2 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg10_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg11_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %prepare_softmax_online_default_16 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_3, -1), kwargs = {})
#   %sub_tensor_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_40), kwargs = {})
#   %exp_default_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_16,), kwargs = {})
#   %div_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_16, %getitem_41), kwargs = {})
#   return %add_3,%getitem_40,%getitem_41,%div_1
triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4', '''
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
    size_hints={'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 67110144, 'r0_': 134217728}, 'kernel_num_gb': 0.201327872, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 64) % 64)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = libdevice.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 512*x3 + 32768*x0 + 4194304*x4), tmp24, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 64, 64, 512), (2097152, 1, 32768, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32768, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.201327872
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/2o/c2opoldplclzugwldhpp34vzl5y6q5ae77vcljusuarva24vvhum.py
# Topologically Sorted Source Nodes: [input_6, max_pool2d], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_6 => div_1
#   max_pool2d => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %div_1 : Tensor "f32[8, 64, 64, 512][4194304, 32768, 512, 1]cuda:0" = PlaceHolder[target=div_1]
#   %sub_tensor_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_40), kwargs = {})
#   %exp_default_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_16,), kwargs = {})
#   %div_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_16, %getitem_41), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5 = async_compile.triton('triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5', '''
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
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 33554432, 'x': 0}, 'kernel_num_gb': 0.08388608, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = (xindex % 256)
    x3 = xindex // 256
    y0 = (yindex % 64)
    y1 = yindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (513 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 64*x4 + 524288*y1), tmp6, ymask)


def get_args():
    arg_0 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 64, 32, 256), (524288, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 512, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.08388608
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/zv/czvqsad57lb2h3cctdoe4kyspwkxpjxifbkvrafddl4gxjgmmalg.py
# Topologically Sorted Source Nodes: [input_6, max_pool2d, input_7], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
# Source node to ATen node mapping:
#   input_6 => div_1
#   input_7 => convolution_2
#   max_pool2d => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %arg13_1 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %sub_tensor_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_40), kwargs = {})
#   %exp_default_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_16,), kwargs = {})
#   %div_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_16, %getitem_41), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf14
triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6', '''
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
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 589824, 'x': 294912}, 'kernel_num_gb': 0.000589824, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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
    arg_0 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 8192, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000589824
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/c7/cc77icbv6ofn4de7li7u2dz3tmh2yvmylbblhcu5ubg3v5t5hd6g.py
# Topologically Sorted Source Nodes: [input_6, max_pool2d, input_7, input_8, input_9], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_6 => div_1
#   input_7 => convolution_2
#   input_8 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_4, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   input_9 => div_2
#   max_pool2d => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %buf15 : Tensor "f32[8, 128, 32, 256][1048576, 1, 32768, 128]cuda:0" = PlaceHolder[target=buf15]
#   %arg14_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %arg17_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %add_5 : Tensor "f32[8, 128, 32, 256][1048576, 1, 32768, 128]cuda:0" = PlaceHolder[target=add_5]
#   %getitem_38 : Tensor "f32[8, 128, 32, 1][4096, 1, 128, 32768]cuda:0" = PlaceHolder[target=getitem_38]
#   %getitem_39 : Tensor "f32[8, 128, 32, 1][4096, 1, 128, 32768]cuda:0" = PlaceHolder[target=getitem_39]
#   %sub_tensor_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_40), kwargs = {})
#   %exp_default_16 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_16,), kwargs = {})
#   %div_1 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_16, %getitem_41), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg13_1, %arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_4 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg16_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg18_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %prepare_softmax_online_default_15 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_5, -1), kwargs = {})
#   %sub_tensor_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_38), kwargs = {})
#   %exp_default_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_15,), kwargs = {})
#   %div_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_15, %getitem_39), kwargs = {})
#   return %add_5,%getitem_38,%getitem_39,%div_2
triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7', '''
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
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 100665856, 'r0_': 0}, 'kernel_num_gb': 0.067111424, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = libdevice.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), tmp24, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 128, 32, 256), (1048576, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 256,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.067111424
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/qz/cqzi2siopousukdgjov5mw3fbj2lskuyvvqwr5pbd25alckee2xr.py
# Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   input_10 => convolution_3
#   input_9 => div_2
# Graph fragment:
#   %arg19_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %sub_tensor_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_38), kwargs = {})
#   %exp_default_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_15,), kwargs = {})
#   %div_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_15, %getitem_39), kwargs = {})
#   %convolution_3 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_2, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf20
triton_poi_fused__softmax_convolution_exp_sub_8 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_8', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 589824}, 'kernel_num_gb': 0.001179648, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        triton_poi_fused__softmax_convolution_exp_sub_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001179648
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/cu/ccugwmyky6mucfo7xrks5mf3rr255x4rxitjg66slrutaeuqj6kz.py
# Topologically Sorted Source Nodes: [input_9, input_10, input_11, input_12], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_10 => convolution_3
#   input_11 => add_6, add_7, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_6, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   input_12 => div_3
#   input_9 => div_2
# Graph fragment:
#   %buf21 : Tensor "f32[8, 128, 32, 256][1048576, 1, 32768, 128]cuda:0" = PlaceHolder[target=buf21]
#   %arg20_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %arg21_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %arg22_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %add_7 : Tensor "f32[8, 128, 32, 256][1048576, 1, 32768, 128]cuda:0" = PlaceHolder[target=add_7]
#   %getitem_36 : Tensor "f32[8, 128, 32, 1][4096, 1, 128, 32768]cuda:0" = PlaceHolder[target=getitem_36]
#   %getitem_37 : Tensor "f32[8, 128, 32, 1][4096, 1, 128, 32768]cuda:0" = PlaceHolder[target=getitem_37]
#   %sub_tensor_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_38), kwargs = {})
#   %exp_default_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_15,), kwargs = {})
#   %div_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_15, %getitem_39), kwargs = {})
#   %convolution_3 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_2, %arg19_1, %arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg21_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_6 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg22_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg23_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %prepare_softmax_online_default_14 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_7, -1), kwargs = {})
#   %sub_tensor_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_36), kwargs = {})
#   %exp_default_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_14,), kwargs = {})
#   %div_3 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_14, %getitem_37), kwargs = {})
#   return %add_7,%getitem_36,%getitem_37,%div_3
triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9', '''
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
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33556992, 'r0_': 67108864}, 'kernel_num_gb': 0.100665856, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 128) % 32)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = libdevice.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 256*x3 + 8192*x0 + 2097152*x4), tmp24, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 128, 32, 256), (1048576, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32768, 256,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.100665856
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/x7/cx76cckgsd74wnvxqkxtiwvm5sojn26k6anlk5a5zqfprr3br5f2.py
# Topologically Sorted Source Nodes: [input_12, max_pool2d_1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_12 => div_3
#   max_pool2d_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %div_3 : Tensor "f32[8, 128, 32, 256][2097152, 8192, 256, 1]cuda:0" = PlaceHolder[target=div_3]
#   %sub_tensor_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_36), kwargs = {})
#   %exp_default_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_14,), kwargs = {})
#   %div_3 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_14, %getitem_37), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10 = async_compile.triton('triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10', '''
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
    size_hints={'y': 1024, 'x': 2048}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 0}, 'kernel_num_gb': 0.04194304, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 128)
    x3 = xindex // 128
    y0 = (yindex % 128)
    y1 = yindex // 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (257 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 128*x4 + 262144*y1), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 128, 16, 128), (262144, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1024, 2048,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.04194304
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/65/c65hkltmlfuurla7ocuenikw4nbp4p67vgglrulwsgfjoo67vy4b.py
# Topologically Sorted Source Nodes: [input_12, max_pool2d_1, input_13], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
# Source node to ATen node mapping:
#   input_12 => div_3
#   input_13 => convolution_4
#   max_pool2d_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %arg25_1 : Tensor "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %sub_tensor_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_36), kwargs = {})
#   %exp_default_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_14,), kwargs = {})
#   %div_3 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_14, %getitem_37), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf27
triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11', '''
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
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}, 'kernel_num_gb': 0.002359296, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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
    arg_0 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 32768, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002359296
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/rb/crbvrsrfp3obkyxtb5cxxrp4jgtohtju3uhyn354aeyxhbh2alod.py
# Topologically Sorted Source Nodes: [input_12, max_pool2d_1, input_13, input_14, input_15], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_12 => div_3
#   input_13 => convolution_4
#   input_14 => add_8, add_9, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_8, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_15 => div_4
#   max_pool2d_1 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %buf28 : Tensor "f32[8, 256, 16, 128][524288, 1, 32768, 256]cuda:0" = PlaceHolder[target=buf28]
#   %arg26_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %arg27_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %arg29_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %add_9 : Tensor "f32[8, 256, 16, 128][524288, 1, 32768, 256]cuda:0" = PlaceHolder[target=add_9]
#   %getitem_34 : Tensor "f32[8, 256, 16, 1][4096, 1, 256, 32768]cuda:0" = PlaceHolder[target=getitem_34]
#   %getitem_35 : Tensor "f32[8, 256, 16, 1][4096, 1, 256, 32768]cuda:0" = PlaceHolder[target=getitem_35]
#   %sub_tensor_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_36), kwargs = {})
#   %exp_default_14 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_14,), kwargs = {})
#   %div_3 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_14, %getitem_37), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_3, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %arg25_1, %arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg27_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_8 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg29_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg30_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %prepare_softmax_online_default_13 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_9, -1), kwargs = {})
#   %sub_tensor_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_34), kwargs = {})
#   %exp_default_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_13,), kwargs = {})
#   %div_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_13, %getitem_35), kwargs = {})
#   return %add_9,%getitem_34,%getitem_35,%div_4
triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12', '''
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
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50336768, 'r0_': 0}, 'kernel_num_gb': 0.033559552, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = libdevice.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp24, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 256, 16, 128), (524288, 1, 32768, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033559552
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/rn/crndqkyib4mfqud464hjylpqd3r6xxm6q5lzooh4m4p4zrik6ubc.py
# Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   input_15 => div_4
#   input_16 => convolution_5
# Graph fragment:
#   %arg31_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg31_1]
#   %sub_tensor_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_34), kwargs = {})
#   %exp_default_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_13,), kwargs = {})
#   %div_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_13, %getitem_35), kwargs = {})
#   %convolution_5 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_4, %arg31_1, %arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf33
triton_poi_fused__softmax_convolution_exp_sub_13 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_13', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}, 'kernel_num_gb': 0.004718592, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        triton_poi_fused__softmax_convolution_exp_sub_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004718592
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/bx/cbx2avxgvcf7u2hcjhm4uqyqsmu52hvgse2vj4yv3sonhhnnorkv.py
# Topologically Sorted Source Nodes: [input_15, input_16, input_17, input_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_15 => div_4
#   input_16 => convolution_5
#   input_17 => add_10, add_11, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_10, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   input_18 => div_5
# Graph fragment:
#   %buf34 : Tensor "f32[8, 256, 16, 128][524288, 1, 32768, 256]cuda:0" = PlaceHolder[target=buf34]
#   %arg32_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %arg33_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg33_1]
#   %arg34_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg36_1]
#   %add_11 : Tensor "f32[8, 256, 16, 128][524288, 1, 32768, 256]cuda:0" = PlaceHolder[target=add_11]
#   %getitem_32 : Tensor "f32[8, 256, 16, 1][4096, 1, 256, 32768]cuda:0" = PlaceHolder[target=getitem_32]
#   %getitem_33 : Tensor "f32[8, 256, 16, 1][4096, 1, 256, 32768]cuda:0" = PlaceHolder[target=getitem_33]
#   %sub_tensor_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_34), kwargs = {})
#   %exp_default_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_13,), kwargs = {})
#   %div_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_13, %getitem_35), kwargs = {})
#   %convolution_5 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_4, %arg31_1, %arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg33_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_10 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %add_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg34_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_10,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg36_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_11 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %prepare_softmax_online_default_12 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_11, -1), kwargs = {})
#   %sub_tensor_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_32), kwargs = {})
#   %exp_default_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_12,), kwargs = {})
#   %div_5 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_12, %getitem_33), kwargs = {})
#   return %add_11,%getitem_32,%getitem_33,%div_5
triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14', '''
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
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16782336, 'r0_': 33554432}, 'kernel_num_gb': 0.050336768, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 256) % 16)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = libdevice.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 128*x3 + 2048*x0 + 1048576*x4), tmp24, r0_mask)


def get_args():
    arg_0 = rand_strided((8, 256, 16, 128), (524288, 1, 32768, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32768, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.050336768
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/hl/chlz2bobs2fnw5jamxcofoe5qx7ktq353z25rqma5rqsmtteh3yd.py
# Topologically Sorted Source Nodes: [input_18, max_pool2d_2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_18 => div_5
#   max_pool2d_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %div_5 : Tensor "f32[8, 256, 16, 128][1048576, 2048, 128, 1]cuda:0" = PlaceHolder[target=div_5]
#   %sub_tensor_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_32), kwargs = {})
#   %exp_default_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_12,), kwargs = {})
#   %div_5 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_12, %getitem_33), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_4
triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15 = async_compile.triton('triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15', '''
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
    size_hints={'y': 2048, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8388608, 'x': 0}, 'kernel_num_gb': 0.02097152, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 64)
    x3 = xindex // 64
    y0 = (yindex % 256)
    y1 = yindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 256*x4 + 131072*y1), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 256, 8, 64), (131072, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 2048, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.02097152
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/gc/cgcjgffw7zdvk6h75leklw5viwl374jn7mmn2mb4b6hgwterxefl.py
# Topologically Sorted Source Nodes: [input_18, max_pool2d_2, input_19], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
# Source node to ATen node mapping:
#   input_18 => div_5
#   input_19 => convolution_6
#   max_pool2d_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %arg37_1 : Tensor "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg37_1]
#   %sub_tensor_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_32), kwargs = {})
#   %exp_default_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_12,), kwargs = {})
#   %div_5 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_12, %getitem_33), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf40
triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}, 'kernel_num_gb': 0.009437184, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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
    arg_0 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 131072, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009437184
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ca/ccaatuxblczybglj37a2q76kxn2yhjfltjxzxhhkobgldo3z4py2.py
# Topologically Sorted Source Nodes: [input_18, max_pool2d_2, input_19, input_20, input_21], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_18 => div_5
#   input_19 => convolution_6
#   input_20 => add_12, add_13, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_12, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   input_21 => div_6
#   max_pool2d_2 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %buf41 : Tensor "f32[8, 512, 8, 64][262144, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf41]
#   %arg38_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %arg41_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %arg42_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %add_13 : Tensor "f32[8, 512, 8, 64][262144, 1, 32768, 512]cuda:0" = PlaceHolder[target=add_13]
#   %getitem_30 : Tensor "f32[8, 512, 8, 1][4096, 1, 512, 32768]cuda:0" = PlaceHolder[target=getitem_30]
#   %getitem_31 : Tensor "f32[8, 512, 8, 1][4096, 1, 512, 32768]cuda:0" = PlaceHolder[target=getitem_31]
#   %sub_tensor_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_32), kwargs = {})
#   %exp_default_12 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_12,), kwargs = {})
#   %div_5 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_12, %getitem_33), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_5, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_48 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_12 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_12 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg40_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg41_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_13 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %prepare_softmax_online_default_11 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_13, -1), kwargs = {})
#   %sub_tensor_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_30), kwargs = {})
#   %exp_default_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_11,), kwargs = {})
#   %div_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_11, %getitem_31), kwargs = {})
#   return %add_13,%getitem_30,%getitem_31,%div_6
triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25176064, 'r0_': 0}, 'kernel_num_gb': 0.016787456, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = tmp18 - tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tmp17 - tmp22
    tmp29 = libdevice.exp(tmp28)
    tmp30 = (tmp29 / tmp27)
    tl.store(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), tmp30, None)


def get_args():
    arg_0 = rand_strided((8, 512, 8, 64), (262144, 1, 32768, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016787456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/k4/ck4ni5csu2nj2cbxt5jh3vmbejeqhsb7mny27oksq2r4o5ty55vo.py
# Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   input_21 => div_6
#   input_22 => convolution_7
# Graph fragment:
#   %arg43_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg43_1]
#   %sub_tensor_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_30), kwargs = {})
#   %exp_default_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_11,), kwargs = {})
#   %div_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_11, %getitem_31), kwargs = {})
#   %convolution_7 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_6, %arg43_1, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf46
triton_poi_fused__softmax_convolution_exp_sub_18 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_18', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}, 'kernel_num_gb': 0.018874368, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        triton_poi_fused__softmax_convolution_exp_sub_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.018874368
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/q4/cq4dz5wzgygo7tdbwei2tlue5g6wpmvzvmcoi7mj6gmdcjv2deli.py
# Topologically Sorted Source Nodes: [input_21, input_22, input_23, input_24], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_21 => div_6
#   input_22 => convolution_7
#   input_23 => add_14, add_15, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_14, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   input_24 => div_7
# Graph fragment:
#   %buf47 : Tensor "f32[8, 512, 8, 64][262144, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf47]
#   %arg44_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %arg47_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %arg48_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg48_1]
#   %add_15 : Tensor "f32[8, 512, 8, 64][262144, 1, 32768, 512]cuda:0" = PlaceHolder[target=add_15]
#   %getitem_28 : Tensor "f32[8, 512, 8, 1][4096, 1, 512, 32768]cuda:0" = PlaceHolder[target=getitem_28]
#   %getitem_29 : Tensor "f32[8, 512, 8, 1][4096, 1, 512, 32768]cuda:0" = PlaceHolder[target=getitem_29]
#   %sub_tensor_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_30), kwargs = {})
#   %exp_default_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_11,), kwargs = {})
#   %div_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_11, %getitem_31), kwargs = {})
#   %convolution_7 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_6, %arg43_1, %arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg45_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_14 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_14 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg46_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg47_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg48_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_15 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %prepare_softmax_online_default_10 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_15, -1), kwargs = {})
#   %sub_tensor_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_28), kwargs = {})
#   %exp_default_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_10,), kwargs = {})
#   %div_7 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_10, %getitem_29), kwargs = {})
#   return %add_15,%getitem_28,%getitem_29,%div_7
triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8398848, 'r0_': 16777216}, 'kernel_num_gb': 0.025176064, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x5 = xindex
    x3 = ((xindex // 512) % 8)
    x4 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = tmp18 - tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tmp17 - tmp22
    tmp29 = libdevice.exp(tmp28)
    tmp30 = (tmp29 / tmp27)
    tl.store(out_ptr2 + (r0_2 + 64*x3 + 512*x0 + 524288*x4), tmp30, None)


def get_args():
    arg_0 = rand_strided((8, 512, 8, 64), (262144, 1, 32768, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8, 512, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32768, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.025176064
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/p3/cp3q4zd7hxubeu5tchjts2e6oh3xotl2u34rua2gdkqr52ys5nkg.py
# Topologically Sorted Source Nodes: [input_24, max_pool2d_3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_24 => div_7
#   max_pool2d_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %div_7 : Tensor "f32[8, 512, 8, 64][524288, 512, 64, 1]cuda:0" = PlaceHolder[target=div_7]
#   %sub_tensor_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_28), kwargs = {})
#   %exp_default_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_10,), kwargs = {})
#   %div_7 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_10, %getitem_29), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_6
triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20 = async_compile.triton('triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20', '''
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
    size_hints={'y': 4096, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4194304, 'x': 0}, 'kernel_num_gb': 0.01048576, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 32)
    x3 = xindex // 32
    y0 = (yindex % 512)
    y1 = yindex // 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 512*x4 + 65536*y1), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((8, 512, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 512, 4, 32), (65536, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 4096, 128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.01048576
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ck/cckvjulgnymiivpy42n6jnjfzcf4sj4n3tqoh2tzdgzw7h3tp65i.py
# Topologically Sorted Source Nodes: [input_24, max_pool2d_3, input_25], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
# Source node to ATen node mapping:
#   input_24 => div_7
#   input_25 => convolution_8
#   max_pool2d_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %arg49_1 : Tensor "f32[1024, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg49_1]
#   %sub_tensor_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_28), kwargs = {})
#   %exp_default_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_10,), kwargs = {})
#   %div_7 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_10, %getitem_29), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg49_1, %arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf53
triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21', '''
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
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 37748736, 'x': 18874368}, 'kernel_num_gb': 0.037748736, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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
    arg_0 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 524288, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.037748736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/fx/cfxygflqdozxgn5q4h25kd2seorl5vwqnfn3engkgpbrexn65uu3.py
# Topologically Sorted Source Nodes: [input_24, max_pool2d_3, input_25, input_26, input_27], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_24 => div_7
#   input_25 => convolution_8
#   input_26 => add_16, add_17, mul_24, mul_25, mul_26, reciprocal_8, sqrt_8, sub_16, unsqueeze_64, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, unsqueeze_71
#   input_27 => div_8
#   max_pool2d_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %buf54 : Tensor "f32[8, 1024, 4, 32][131072, 1, 32768, 1024]cuda:0" = PlaceHolder[target=buf54]
#   %arg50_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg50_1]
#   %arg51_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %arg52_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %arg53_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %arg54_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %add_17 : Tensor "f32[8, 1024, 4, 32][131072, 1, 32768, 1024]cuda:0" = PlaceHolder[target=add_17]
#   %getitem_26 : Tensor "f32[8, 1024, 4, 1][4096, 1, 1024, 32768]cuda:0" = PlaceHolder[target=getitem_26]
#   %getitem_27 : Tensor "f32[8, 1024, 4, 1][4096, 1, 1024, 32768]cuda:0" = PlaceHolder[target=getitem_27]
#   %sub_tensor_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_28), kwargs = {})
#   %exp_default_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_10,), kwargs = {})
#   %div_7 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_10, %getitem_29), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %arg49_1, %arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg51_1, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %sub_16 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %add_16 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg52_1, 1e-05), kwargs = {})
#   %sqrt_8 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_8 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_8,), kwargs = {})
#   %mul_24 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_24, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %mul_25 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg53_1, -1), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, -1), kwargs = {})
#   %mul_26 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[1024, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, -1), kwargs = {})
#   %add_17 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %prepare_softmax_online_default_9 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_17, -1), kwargs = {})
#   %sub_tensor_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_17, %getitem_26), kwargs = {})
#   %exp_default_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_9,), kwargs = {})
#   %div_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_9, %getitem_27), kwargs = {})
#   return %add_17,%getitem_26,%getitem_27,%div_8
triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22', '''
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
    size_hints={'x': 32768, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12603392, 'r0_': 0}, 'kernel_num_gb': 0.008409088, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 1024*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = tmp18 - tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tmp17 - tmp22
    tmp29 = libdevice.exp(tmp28)
    tmp30 = (tmp29 / tmp27)
    tl.store(in_out_ptr0 + (x0 + 1024*r0_2 + 32768*x1), tmp30, None)


def get_args():
    arg_0 = rand_strided((8, 1024, 4, 32), (131072, 1, 32768, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 32,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008409088
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/q5/cq5a4bxuvoh2tux43kmzdlsxpiivdltgjiewthhbhwelyo636jzw.py
# Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   input_27 => div_8
#   input_28 => convolution_9
# Graph fragment:
#   %arg55_1 : Tensor "f32[1024, 1024, 3, 3][9216, 9, 3, 1]cuda:0" = PlaceHolder[target=arg55_1]
#   %sub_tensor_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_17, %getitem_26), kwargs = {})
#   %exp_default_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_9,), kwargs = {})
#   %div_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_9, %getitem_27), kwargs = {})
#   %convolution_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_8, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf59
triton_poi_fused__softmax_convolution_exp_sub_23 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_23', '''
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
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 75497472, 'x': 37748736}, 'kernel_num_gb': 0.075497472, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 1024, 3, 3), (9216, 1, 3072, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1048576, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_23.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_23.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.075497472
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/6f/c6f5ypjegspaqjbvn2fajbe24rx3losdfvb4t4dm6q74orfrlb6b.py
# Topologically Sorted Source Nodes: [input_30, dec4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec4 => convolution_10
#   input_30 => div_9
# Graph fragment:
#   %arg61_1 : Tensor "f32[1024, 512, 2, 2][2048, 4, 2, 1]cuda:0" = PlaceHolder[target=arg61_1]
#   %sub_tensor_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %getitem_24), kwargs = {})
#   %exp_default_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_8,), kwargs = {})
#   %div_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_8, %getitem_25), kwargs = {})
#   %convolution_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_9, %arg61_1, %arg62_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf65
triton_poi_fused__softmax_convolution_exp_sub_24 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_24', '''
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
    size_hints={'y': 524288, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16777216, 'x': 4194304}, 'kernel_num_gb': 0.016777216, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 2048*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 524288, 4,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_24.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_24.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016777216
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/jt/cjtkbek7vmy3rdeopb6k2d433xapqizvjc5d2cosajfkpxrhplfd.py
# Topologically Sorted Source Nodes: [input_30, dec4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec4 => convolution_10
#   input_30 => div_9
# Graph fragment:
#   %buf66 : Tensor "f32[8, 512, 8, 64][262144, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf66]
#   %arg62_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %sub_tensor_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %getitem_24), kwargs = {})
#   %exp_default_8 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_8,), kwargs = {})
#   %div_9 : Tensor "f32[8, 1024, 4, 32][131072, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_8, %getitem_25), kwargs = {})
#   %convolution_10 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_9, %arg61_1, %arg62_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %convolution_10
triton_poi_fused__softmax_convolution_exp_sub_25 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_25', '''
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
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8390656, 'x': 16777216}, 'kernel_num_gb': 0.016779264, 'kernel_flop': 4294967296},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_25(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 262144*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512*y0 + 524288*y1), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((8, 512, 8, 64), (262144, 1, 32768, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 512, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 4096, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_25.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_25.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016779264
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ci/ccirekqswshv6f5ukbfy3lz4vyazwdd5qpd7ceihdxc2vtu26vpa.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_31 => convolution_11
# Graph fragment:
#   %cat : Tensor "f32[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = PlaceHolder[target=cat]
#   %convolution_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg63_1, %arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf69
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
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
    size_hints={'y': 8192, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 33554432, 'x': 16777216}, 'kernel_num_gb': 0.033554432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 512*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 524288*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((8, 1024, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 1024, 8, 64), (524288, 1, 65536, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 8192, 512,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_26.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033554432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ni/cnix4ood734uxtfdd7y5koohf7hgbwstwcvvprvp6iyglf72sw4o.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_31 => convolution_11
# Graph fragment:
#   %arg63_1 : Tensor "f32[512, 1024, 3, 3][9216, 9, 3, 1]cuda:0" = PlaceHolder[target=arg63_1]
#   %convolution_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg63_1, %arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf70
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
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
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 37748736, 'x': 18874368}, 'kernel_num_gb': 0.037748736, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 1024, 3, 3), (9216, 1, 3072, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 524288, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_27.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.037748736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/gy/cgytr66o5eawjgpe7ud6bhepmdcs4v34ji3jc22z4qu3wpkuw2er.py
# Topologically Sorted Source Nodes: [input_36, dec3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec3 => convolution_13
#   input_36 => div_11
# Graph fragment:
#   %arg75_1 : Tensor "f32[512, 256, 2, 2][1024, 4, 2, 1]cuda:0" = PlaceHolder[target=arg75_1]
#   %sub_tensor_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_23, %getitem_20), kwargs = {})
#   %exp_default_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_6,), kwargs = {})
#   %div_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_6, %getitem_21), kwargs = {})
#   %convolution_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_11, %arg75_1, %arg76_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf82
triton_poi_fused__softmax_convolution_exp_sub_28 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_28', '''
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
    size_hints={'y': 131072, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4194304, 'x': 1048576}, 'kernel_num_gb': 0.004194304, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 1024*y1), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 131072, 4,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_28.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_28.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004194304
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/43/c43kxkntd5s5zpo4n3h5n5o3gg2v74rsfmcznjgo7g7qb3xggape.py
# Topologically Sorted Source Nodes: [input_36, dec3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec3 => convolution_13
#   input_36 => div_11
# Graph fragment:
#   %buf83 : Tensor "f32[8, 256, 16, 128][524288, 1, 32768, 256]cuda:0" = PlaceHolder[target=buf83]
#   %arg76_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg76_1]
#   %sub_tensor_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_23, %getitem_20), kwargs = {})
#   %exp_default_6 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_6,), kwargs = {})
#   %div_11 : Tensor "f32[8, 512, 8, 64][262144, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_6, %getitem_21), kwargs = {})
#   %convolution_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_11, %arg75_1, %arg76_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %convolution_13
triton_poi_fused__softmax_convolution_exp_sub_29 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_29', '''
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
    size_hints={'y': 2048, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 16778240, 'x': 33554432}, 'kernel_num_gb': 0.033555456, 'kernel_flop': 4294967296},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_29(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 524288*y1), xmask)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 2048*y0 + 1048576*y1), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((8, 256, 16, 128), (524288, 1, 32768, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 2048, 2048,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_29.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_29.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.033555456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/ke/ckeyqqo4brrmxtcfzcqjbw5rl2mxlgc2o7e6tu4x7vnp3mzp3cak.py
# Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_37 => convolution_14
# Graph fragment:
#   %cat_1 : Tensor "f32[8, 512, 16, 128][1048576, 2048, 128, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %convolution_14 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %arg77_1, %arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf86
triton_poi_fused_convolution_30 = async_compile.triton('triton_poi_fused_convolution_30', '''
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
    size_hints={'y': 4096, 'x': 2048}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 67108864, 'x': 33554432}, 'kernel_num_gb': 0.067108864, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 2048*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 1048576*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((8, 512, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 512, 16, 128), (1048576, 1, 65536, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 4096, 2048,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_30.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_30.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.067108864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/wx/cwx3owdbcbv2ed5dhj44eahh4hxn3vws3rfnojiuepof5bpv5dh2.py
# Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_37 => convolution_14
# Graph fragment:
#   %arg77_1 : Tensor "f32[256, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg77_1]
#   %convolution_14 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %arg77_1, %arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf87
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}, 'kernel_num_gb': 0.009437184, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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
    arg_0 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 131072, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_31.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009437184
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/m5/cm5nr5sclipbfnaz4arbwo6xrpagnvvilyba6cobl4cp5mxsysvp.py
# Topologically Sorted Source Nodes: [input_42, dec2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec2 => convolution_16
#   input_42 => div_13
# Graph fragment:
#   %arg89_1 : Tensor "f32[256, 128, 2, 2][512, 4, 2, 1]cuda:0" = PlaceHolder[target=arg89_1]
#   %sub_tensor_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_27, %getitem_16), kwargs = {})
#   %exp_default_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_4,), kwargs = {})
#   %div_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_4, %getitem_17), kwargs = {})
#   %convolution_16 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_13, %arg89_1, %arg90_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf99
triton_poi_fused__softmax_convolution_exp_sub_32 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_32', '''
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
    size_hints={'y': 32768, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048576, 'x': 262144}, 'kernel_num_gb': 0.001048576, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 128, 2, 2), (512, 1, 256, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 32768, 4,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_32.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_32.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001048576
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/dw/cdwuiz7rhrmumgljlkr4zcspbm3vhftbsrqtokihro2jt64nmz5i.py
# Topologically Sorted Source Nodes: [input_42, dec2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec2 => convolution_16
#   input_42 => div_13
# Graph fragment:
#   %buf100 : Tensor "f32[8, 128, 32, 256][1048576, 1, 32768, 128]cuda:0" = PlaceHolder[target=buf100]
#   %arg90_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %sub_tensor_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_27, %getitem_16), kwargs = {})
#   %exp_default_4 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_4,), kwargs = {})
#   %div_13 : Tensor "f32[8, 256, 16, 128][524288, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_4, %getitem_17), kwargs = {})
#   %convolution_16 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_13, %arg89_1, %arg90_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %convolution_16
triton_poi_fused__softmax_convolution_exp_sub_33 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_33', '''
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
    size_hints={'y': 1024, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 33554944, 'x': 67108864}, 'kernel_num_gb': 0.067109376, 'kernel_flop': 4294967296},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_33(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 1048576*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 8192*y0 + 2097152*y1), tmp2, None)


def get_args():
    arg_0 = rand_strided((8, 128, 32, 256), (1048576, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1024, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_33.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_33.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.067109376
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/3t/c3twut7hjmm76coqrqjzstu2mpsfwnb3matz3q3xg6h7pvi3sz3d.py
# Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_43 => convolution_17
# Graph fragment:
#   %cat_2 : Tensor "f32[8, 256, 32, 256][2097152, 8192, 256, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %convolution_17 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg91_1, %arg92_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf103
triton_poi_fused_convolution_34 = async_compile.triton('triton_poi_fused_convolution_34', '''
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
    size_hints={'y': 2048, 'x': 8192}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 134217728, 'x': 67108864}, 'kernel_num_gb': 0.134217728, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 8192*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2097152*y1), tmp0, None)


def get_args():
    arg_0 = rand_strided((8, 256, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 256, 32, 256), (2097152, 1, 65536, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 2048, 8192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_34.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.134217728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/iu/ciuckikk5h7uikwmzcqou44wfopu5xojcjfrzug7aiepyxlb74y3.py
# Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_43 => convolution_17
# Graph fragment:
#   %arg91_1 : Tensor "f32[128, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg91_1]
#   %convolution_17 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg91_1, %arg92_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf104
triton_poi_fused_convolution_35 = async_compile.triton('triton_poi_fused_convolution_35', '''
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
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}, 'kernel_num_gb': 0.002359296, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 32768, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_35.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_35.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.002359296
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/pm/cpmryoow6w3yshzohl54tyt7w6t3oxqdrxfnltnr4v6lb2e6leyy.py
# Topologically Sorted Source Nodes: [input_48, dec1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec1 => convolution_19
#   input_48 => div_15
# Graph fragment:
#   %arg103_1 : Tensor "f32[128, 64, 2, 2][256, 4, 2, 1]cuda:0" = PlaceHolder[target=arg103_1]
#   %sub_tensor_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_31, %getitem_12), kwargs = {})
#   %exp_default_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_2,), kwargs = {})
#   %div_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_2, %getitem_13), kwargs = {})
#   %convolution_19 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_15, %arg103_1, %arg104_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf116
triton_poi_fused__softmax_convolution_exp_sub_36 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_36', '''
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
    size_hints={'y': 8192, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 262144, 'x': 65536}, 'kernel_num_gb': 0.000262144, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 8192, 4,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_36.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_36.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000262144
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/xd/cxdt5z2tr434vqgxns3lh3taxho57qtjwamz37wfr6xfzyy26ixs.py
# Topologically Sorted Source Nodes: [input_48, dec1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   dec1 => convolution_19
#   input_48 => div_15
# Graph fragment:
#   %buf117 : Tensor "f32[8, 64, 64, 512][2097152, 1, 32768, 64]cuda:0" = PlaceHolder[target=buf117]
#   %arg104_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg104_1]
#   %sub_tensor_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_31, %getitem_12), kwargs = {})
#   %exp_default_2 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_2,), kwargs = {})
#   %div_15 : Tensor "f32[8, 128, 32, 256][1048576, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_2, %getitem_13), kwargs = {})
#   %convolution_19 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_15, %arg103_1, %arg104_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %convolution_19
triton_poi_fused__softmax_convolution_exp_sub_37 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_37', '''
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
    size_hints={'y': 512, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 67109120, 'x': 134217728}, 'kernel_num_gb': 0.134217984, 'kernel_flop': 4294967296},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_37(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 2097152*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 32768*y0 + 4194304*y1), tmp2, ymask)


def get_args():
    arg_0 = rand_strided((8, 64, 64, 512), (2097152, 1, 32768, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 512, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_37.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_37.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.134217984
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/e5/ce5dqz4ac7664fk5zjfwhdozrevg5fihsboiaetsbb5ur7moy6xg.py
# Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_49 => convolution_20
# Graph fragment:
#   %cat_3 : Tensor "f32[8, 128, 64, 512][4194304, 32768, 512, 1]cuda:0" = PlaceHolder[target=cat_3]
#   %convolution_20 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg105_1, %arg106_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf120
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
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
    size_hints={'y': 1024, 'x': 32768}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 268435456, 'x': 134217728}, 'kernel_num_gb': 0.268435456, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 32768*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 4194304*y1), tmp0, None)


def get_args():
    arg_0 = rand_strided((8, 128, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 128, 64, 512), (4194304, 1, 65536, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1024, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_38.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.268435456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/nb/cnbnpkmp65hmle5fijlwlnpi4dgdq26o4ax4e6fzjpkvlyd7ravw.py
# Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_49 => convolution_20
# Graph fragment:
#   %arg105_1 : Tensor "f32[64, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg105_1]
#   %convolution_20 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg105_1, %arg106_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf121
triton_poi_fused_convolution_39 = async_compile.triton('triton_poi_fused_convolution_39', '''
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
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 589824, 'x': 294912}, 'kernel_num_gb': 0.000589824, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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
    arg_0 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 8192, 9,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_39.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_convolution_39.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000589824
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/uc/cuc6j2ymwyewxbereg2f2mbxkmw3ealk2tuo4h6lojvhgf43ynou.py
# Topologically Sorted Source Nodes: [input_54, conv2d_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_18 => convolution_22
#   input_54 => div_17
# Graph fragment:
#   %arg118_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %buf132 : Tensor "f32[8, 64, 64, 512][2097152, 1, 32768, 64]cuda:0" = PlaceHolder[target=buf132]
#   %arg117_1 : Tensor "f32[4, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg117_1]
#   %sub_tensor : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_35, %getitem_8), kwargs = {})
#   %exp_default : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_9), kwargs = {})
#   %convolution_22 : Tensor "f32[8, 4, 64, 512][131072, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_17, %arg117_1, %arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf133
triton_tem_fused__softmax_convolution_exp_sub_40 = async_compile.triton('triton_tem_fused__softmax_convolution_exp_sub_40', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__softmax_convolution_exp_sub_40', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.075498496, 'kernel_flop': 134217728, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__softmax_convolution_exp_sub_40(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 262144
    N = 4
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
        xindex = idx_n + 4*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def get_args():
    arg_0 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 64, 64, 512), (2097152, 1, 32768, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((262144, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 2048, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__softmax_convolution_exp_sub_40.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__softmax_convolution_exp_sub_40.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.075498496
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/qu/cqurygc3rimsrbsio5fqjzfubpvh3nslq5tioo5r4yziwmam2sq6.py
# Topologically Sorted Source Nodes: [input_54, conv2d_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_18 => convolution_22
#   input_54 => div_17
# Graph fragment:
#   %buf133 : Tensor "f32[262144, 4][4, 1]cuda:0" = PlaceHolder[target=buf133]
#   %sub_tensor : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_35, %getitem_8), kwargs = {})
#   %exp_default : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_17 : Tensor "f32[8, 64, 64, 512][2097152, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_9), kwargs = {})
#   %convolution_22 : Tensor "f32[8, 4, 64, 512][131072, 32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_17, %arg117_1, %arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_22
triton_poi_fused__softmax_convolution_exp_sub_41 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_sub_41', '''
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
    size_hints={'y': 32, 'x': 32768}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_sub_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 4194304, 'x': 8388608}, 'kernel_num_gb': 0.008388608, 'kernel_flop': 134217728},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 131072*y1), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 32768*y3), tmp0, ymask)


def get_args():
    arg_0 = rand_strided((262144, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 4, 64, 512), (131072, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 32, 32768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_sub_41.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_sub_41.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.008388608
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1 = args
        args.clear()
        assert_size_stride(arg0_1, (64, 8, 3, 3), (72, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (8, 8, 64, 512), (262144, 32768, 512, 1))
        assert_size_stride(arg3_1, (64, ), (1, ))
        assert_size_stride(arg4_1, (64, ), (1, ))
        assert_size_stride(arg5_1, (64, ), (1, ))
        assert_size_stride(arg6_1, (64, ), (1, ))
        assert_size_stride(arg7_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg8_1, (64, ), (1, ))
        assert_size_stride(arg9_1, (64, ), (1, ))
        assert_size_stride(arg10_1, (64, ), (1, ))
        assert_size_stride(arg11_1, (64, ), (1, ))
        assert_size_stride(arg12_1, (64, ), (1, ))
        assert_size_stride(arg13_1, (128, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg14_1, (128, ), (1, ))
        assert_size_stride(arg15_1, (128, ), (1, ))
        assert_size_stride(arg16_1, (128, ), (1, ))
        assert_size_stride(arg17_1, (128, ), (1, ))
        assert_size_stride(arg18_1, (128, ), (1, ))
        assert_size_stride(arg19_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg20_1, (128, ), (1, ))
        assert_size_stride(arg21_1, (128, ), (1, ))
        assert_size_stride(arg22_1, (128, ), (1, ))
        assert_size_stride(arg23_1, (128, ), (1, ))
        assert_size_stride(arg24_1, (128, ), (1, ))
        assert_size_stride(arg25_1, (256, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg26_1, (256, ), (1, ))
        assert_size_stride(arg27_1, (256, ), (1, ))
        assert_size_stride(arg28_1, (256, ), (1, ))
        assert_size_stride(arg29_1, (256, ), (1, ))
        assert_size_stride(arg30_1, (256, ), (1, ))
        assert_size_stride(arg31_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg32_1, (256, ), (1, ))
        assert_size_stride(arg33_1, (256, ), (1, ))
        assert_size_stride(arg34_1, (256, ), (1, ))
        assert_size_stride(arg35_1, (256, ), (1, ))
        assert_size_stride(arg36_1, (256, ), (1, ))
        assert_size_stride(arg37_1, (512, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg38_1, (512, ), (1, ))
        assert_size_stride(arg39_1, (512, ), (1, ))
        assert_size_stride(arg40_1, (512, ), (1, ))
        assert_size_stride(arg41_1, (512, ), (1, ))
        assert_size_stride(arg42_1, (512, ), (1, ))
        assert_size_stride(arg43_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg44_1, (512, ), (1, ))
        assert_size_stride(arg45_1, (512, ), (1, ))
        assert_size_stride(arg46_1, (512, ), (1, ))
        assert_size_stride(arg47_1, (512, ), (1, ))
        assert_size_stride(arg48_1, (512, ), (1, ))
        assert_size_stride(arg49_1, (1024, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg50_1, (1024, ), (1, ))
        assert_size_stride(arg51_1, (1024, ), (1, ))
        assert_size_stride(arg52_1, (1024, ), (1, ))
        assert_size_stride(arg53_1, (1024, ), (1, ))
        assert_size_stride(arg54_1, (1024, ), (1, ))
        assert_size_stride(arg55_1, (1024, 1024, 3, 3), (9216, 9, 3, 1))
        assert_size_stride(arg56_1, (1024, ), (1, ))
        assert_size_stride(arg57_1, (1024, ), (1, ))
        assert_size_stride(arg58_1, (1024, ), (1, ))
        assert_size_stride(arg59_1, (1024, ), (1, ))
        assert_size_stride(arg60_1, (1024, ), (1, ))
        assert_size_stride(arg61_1, (1024, 512, 2, 2), (2048, 4, 2, 1))
        assert_size_stride(arg62_1, (512, ), (1, ))
        assert_size_stride(arg63_1, (512, 1024, 3, 3), (9216, 9, 3, 1))
        assert_size_stride(arg64_1, (512, ), (1, ))
        assert_size_stride(arg65_1, (512, ), (1, ))
        assert_size_stride(arg66_1, (512, ), (1, ))
        assert_size_stride(arg67_1, (512, ), (1, ))
        assert_size_stride(arg68_1, (512, ), (1, ))
        assert_size_stride(arg69_1, (512, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg70_1, (512, ), (1, ))
        assert_size_stride(arg71_1, (512, ), (1, ))
        assert_size_stride(arg72_1, (512, ), (1, ))
        assert_size_stride(arg73_1, (512, ), (1, ))
        assert_size_stride(arg74_1, (512, ), (1, ))
        assert_size_stride(arg75_1, (512, 256, 2, 2), (1024, 4, 2, 1))
        assert_size_stride(arg76_1, (256, ), (1, ))
        assert_size_stride(arg77_1, (256, 512, 3, 3), (4608, 9, 3, 1))
        assert_size_stride(arg78_1, (256, ), (1, ))
        assert_size_stride(arg79_1, (256, ), (1, ))
        assert_size_stride(arg80_1, (256, ), (1, ))
        assert_size_stride(arg81_1, (256, ), (1, ))
        assert_size_stride(arg82_1, (256, ), (1, ))
        assert_size_stride(arg83_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg84_1, (256, ), (1, ))
        assert_size_stride(arg85_1, (256, ), (1, ))
        assert_size_stride(arg86_1, (256, ), (1, ))
        assert_size_stride(arg87_1, (256, ), (1, ))
        assert_size_stride(arg88_1, (256, ), (1, ))
        assert_size_stride(arg89_1, (256, 128, 2, 2), (512, 4, 2, 1))
        assert_size_stride(arg90_1, (128, ), (1, ))
        assert_size_stride(arg91_1, (128, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg92_1, (128, ), (1, ))
        assert_size_stride(arg93_1, (128, ), (1, ))
        assert_size_stride(arg94_1, (128, ), (1, ))
        assert_size_stride(arg95_1, (128, ), (1, ))
        assert_size_stride(arg96_1, (128, ), (1, ))
        assert_size_stride(arg97_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg98_1, (128, ), (1, ))
        assert_size_stride(arg99_1, (128, ), (1, ))
        assert_size_stride(arg100_1, (128, ), (1, ))
        assert_size_stride(arg101_1, (128, ), (1, ))
        assert_size_stride(arg102_1, (128, ), (1, ))
        assert_size_stride(arg103_1, (128, 64, 2, 2), (256, 4, 2, 1))
        assert_size_stride(arg104_1, (64, ), (1, ))
        assert_size_stride(arg105_1, (64, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg106_1, (64, ), (1, ))
        assert_size_stride(arg107_1, (64, ), (1, ))
        assert_size_stride(arg108_1, (64, ), (1, ))
        assert_size_stride(arg109_1, (64, ), (1, ))
        assert_size_stride(arg110_1, (64, ), (1, ))
        assert_size_stride(arg111_1, (64, 64, 3, 3), (576, 9, 3, 1))
        assert_size_stride(arg112_1, (64, ), (1, ))
        assert_size_stride(arg113_1, (64, ), (1, ))
        assert_size_stride(arg114_1, (64, ), (1, ))
        assert_size_stride(arg115_1, (64, ), (1, ))
        assert_size_stride(arg116_1, (64, ), (1, ))
        assert_size_stride(arg117_1, (4, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(arg118_1, (4, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((8, 8, 64, 512), (262144, 1, 4096, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 64, 32768, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((64, 8, 3, 3), (72, 1, 24, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 512, 9, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (8, 64, 64, 512), (2097152, 1, 32768, 64), 'unknown_op')
            del buf1
            buf3 = buf2; del buf2  # reuse
            buf6 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2.run(buf6, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, 32768, 512, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf7 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_3.run(arg7_1, buf7, 4096, 9, stream=stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf8, (8, 64, 64, 512), (2097152, 1, 32768, 64), 'unknown_op')
            del buf6
            del buf7
            buf9 = buf8; del buf8  # reuse
            buf119 = empty_strided_cuda((8, 128, 64, 512), (4194304, 32768, 512, 1), torch.float32)
            buf12 = reinterpret_tensor(buf119, (8, 64, 64, 512), (4194304, 32768, 512, 1), 2097152)  # alias
            # Topologically Sorted Source Nodes: [input_3, input_4, input_5, input_6], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4.run(buf9, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, buf12, 32768, 512, stream=stream0)
            del arg10_1
            del arg11_1
            del arg12_1
            del arg8_1
            del arg9_1
            buf13 = empty_strided_cuda((8, 64, 32, 256), (524288, 1, 16384, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_6, max_pool2d], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5.run(buf12, buf13, 512, 8192, stream=stream0)
            buf14 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_6, max_pool2d, input_7], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6.run(arg13_1, buf14, 8192, 9, stream=stream0)
            del arg13_1
            # Topologically Sorted Source Nodes: [input_6, max_pool2d, input_7], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            buf15 = extern_kernels.convolution(buf13, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf15, (8, 128, 32, 256), (1048576, 1, 32768, 128), 'unknown_op')
            del buf13
            del buf14
            buf16 = buf15; del buf15  # reuse
            buf19 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_6, max_pool2d, input_7, input_8, input_9], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7.run(buf19, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, 32768, 256, stream=stream0)
            del arg14_1
            del arg15_1
            del arg16_1
            del arg17_1
            del arg18_1
            buf20 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_8.run(arg19_1, buf20, 16384, 9, stream=stream0)
            del arg19_1
            # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf21, (8, 128, 32, 256), (1048576, 1, 32768, 128), 'unknown_op')
            del buf20
            buf22 = buf21; del buf21  # reuse
            buf102 = reinterpret_tensor(buf9, (8, 256, 32, 256), (2097152, 8192, 256, 1), 0); del buf9  # reuse
            buf25 = reinterpret_tensor(buf102, (8, 128, 32, 256), (2097152, 8192, 256, 1), 1048576)  # alias
            # Topologically Sorted Source Nodes: [input_9, input_10, input_11, input_12], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9.run(buf22, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, buf25, 32768, 256, stream=stream0)
            del arg20_1
            del arg21_1
            del arg22_1
            del arg23_1
            del arg24_1
            buf26 = reinterpret_tensor(buf0, (8, 128, 16, 128), (262144, 1, 16384, 128), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [input_12, max_pool2d_1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10.run(buf25, buf26, 1024, 2048, stream=stream0)
            buf27 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_12, max_pool2d_1, input_13], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11.run(arg25_1, buf27, 32768, 9, stream=stream0)
            del arg25_1
            # Topologically Sorted Source Nodes: [input_12, max_pool2d_1, input_13], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            buf28 = extern_kernels.convolution(buf26, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf28, (8, 256, 16, 128), (524288, 1, 32768, 256), 'unknown_op')
            del buf26
            del buf27
            buf29 = buf28; del buf28  # reuse
            buf32 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [input_12, max_pool2d_1, input_13, input_14, input_15], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12.run(buf32, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, 32768, 128, stream=stream0)
            del arg26_1
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            buf33 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_13.run(arg31_1, buf33, 65536, 9, stream=stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf34, (8, 256, 16, 128), (524288, 1, 32768, 256), 'unknown_op')
            buf35 = buf34; del buf34  # reuse
            buf85 = reinterpret_tensor(buf22, (8, 512, 16, 128), (1048576, 2048, 128, 1), 0); del buf22  # reuse
            buf38 = reinterpret_tensor(buf85, (8, 256, 16, 128), (1048576, 2048, 128, 1), 524288)  # alias
            # Topologically Sorted Source Nodes: [input_15, input_16, input_17, input_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14.run(buf35, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, buf38, 32768, 128, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            del arg36_1
            buf39 = empty_strided_cuda((8, 256, 8, 64), (131072, 1, 16384, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_18, max_pool2d_2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15.run(buf38, buf39, 2048, 512, stream=stream0)
            buf40 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_18, max_pool2d_2, input_19], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16.run(arg37_1, buf40, 131072, 9, stream=stream0)
            del arg37_1
            # Topologically Sorted Source Nodes: [input_18, max_pool2d_2, input_19], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            buf41 = extern_kernels.convolution(buf39, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf41, (8, 512, 8, 64), (262144, 1, 32768, 512), 'unknown_op')
            del buf39
            buf42 = buf41; del buf41  # reuse
            buf45 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [input_18, max_pool2d_2, input_19, input_20, input_21], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17.run(buf45, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, 32768, 64, stream=stream0)
            del arg38_1
            del arg39_1
            del arg40_1
            del arg41_1
            del arg42_1
            buf46 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_18.run(arg43_1, buf46, 262144, 9, stream=stream0)
            del arg43_1
            # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf47, (8, 512, 8, 64), (262144, 1, 32768, 512), 'unknown_op')
            del buf45
            buf48 = buf47; del buf47  # reuse
            buf68 = reinterpret_tensor(buf35, (8, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf35  # reuse
            buf51 = reinterpret_tensor(buf68, (8, 512, 8, 64), (524288, 512, 64, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [input_21, input_22, input_23, input_24], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19.run(buf48, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, buf51, 32768, 64, stream=stream0)
            del arg44_1
            del arg45_1
            del arg46_1
            del arg47_1
            del arg48_1
            buf52 = empty_strided_cuda((8, 512, 4, 32), (65536, 1, 16384, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_24, max_pool2d_3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20.run(buf51, buf52, 4096, 128, stream=stream0)
            buf53 = empty_strided_cuda((1024, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_24, max_pool2d_3, input_25], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21.run(arg49_1, buf53, 524288, 9, stream=stream0)
            del arg49_1
            # Topologically Sorted Source Nodes: [input_24, max_pool2d_3, input_25], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution]
            buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf54, (8, 1024, 4, 32), (131072, 1, 32768, 1024), 'unknown_op')
            buf55 = buf54; del buf54  # reuse
            buf58 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [input_24, max_pool2d_3, input_25, input_26, input_27], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.max_pool2d_with_indices, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22.run(buf58, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, 32768, 32, stream=stream0)
            del arg50_1
            del arg51_1
            del arg52_1
            del arg53_1
            del arg54_1
            buf59 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
            # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_23.run(arg55_1, buf59, 1048576, 9, stream=stream0)
            del arg55_1
            # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf60 = extern_kernels.convolution(buf58, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf60, (8, 1024, 4, 32), (131072, 1, 32768, 1024), 'unknown_op')
            del buf58
            del buf59
            buf61 = buf60; del buf60  # reuse
            buf64 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [input_27, input_28, input_29, input_30], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22.run(buf64, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, 32768, 32, stream=stream0)
            del arg56_1
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            buf65 = reinterpret_tensor(buf48, (1024, 512, 2, 2), (2048, 1, 1024, 512), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [input_30, dec4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_24.run(arg61_1, buf65, 524288, 4, stream=stream0)
            del arg61_1
            # Topologically Sorted Source Nodes: [input_30, dec4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf66 = extern_kernels.convolution(buf64, buf65, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf66, (8, 512, 8, 64), (262144, 1, 32768, 512), 'torch.ops.aten.convolution.default')
            del buf64
            del buf65
            buf67 = reinterpret_tensor(buf68, (8, 512, 8, 64), (524288, 512, 64, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_30, dec4], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_25.run(buf66, arg62_1, buf67, 4096, 512, stream=stream0)
            del arg62_1
            del buf66
            buf69 = reinterpret_tensor(buf32, (8, 1024, 8, 64), (524288, 1, 65536, 1024), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_26.run(buf68, buf69, 8192, 512, stream=stream0)
            del buf51
            del buf67
            del buf68
            buf70 = reinterpret_tensor(buf53, (512, 1024, 3, 3), (9216, 1, 3072, 1024), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_27.run(arg63_1, buf70, 524288, 9, stream=stream0)
            del arg63_1
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
            buf71 = extern_kernels.convolution(buf69, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf71, (8, 512, 8, 64), (262144, 1, 32768, 512), 'unknown_op')
            del buf69
            del buf70
            buf72 = buf71; del buf71  # reuse
            buf75 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [input_31, input_32, input_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17.run(buf75, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, 32768, 64, stream=stream0)
            del arg64_1
            del arg65_1
            del arg66_1
            del arg67_1
            del arg68_1
            buf76 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_18.run(arg69_1, buf76, 262144, 9, stream=stream0)
            del arg69_1
            # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf77 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf77, (8, 512, 8, 64), (262144, 1, 32768, 512), 'unknown_op')
            del buf75
            del buf76
            buf78 = buf77; del buf77  # reuse
            buf81 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [input_33, input_34, input_35, input_36], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17.run(buf81, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, 32768, 64, stream=stream0)
            del arg70_1
            del arg71_1
            del arg72_1
            del arg73_1
            del arg74_1
            buf82 = reinterpret_tensor(buf52, (512, 256, 2, 2), (1024, 1, 512, 256), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [input_36, dec3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_28.run(arg75_1, buf82, 131072, 4, stream=stream0)
            del arg75_1
            # Topologically Sorted Source Nodes: [input_36, dec3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf83 = extern_kernels.convolution(buf81, buf82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf83, (8, 256, 16, 128), (524288, 1, 32768, 256), 'torch.ops.aten.convolution.default')
            del buf81
            del buf82
            buf84 = reinterpret_tensor(buf85, (8, 256, 16, 128), (1048576, 2048, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_36, dec3], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_29.run(buf83, arg76_1, buf84, 2048, 2048, stream=stream0)
            del arg76_1
            del buf83
            buf86 = reinterpret_tensor(buf19, (8, 512, 16, 128), (1048576, 1, 65536, 512), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_30.run(buf85, buf86, 4096, 2048, stream=stream0)
            del buf38
            del buf84
            del buf85
            buf87 = reinterpret_tensor(buf40, (256, 512, 3, 3), (4608, 1, 1536, 512), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_31.run(arg77_1, buf87, 131072, 9, stream=stream0)
            del arg77_1
            # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
            buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf88, (8, 256, 16, 128), (524288, 1, 32768, 256), 'unknown_op')
            del buf86
            del buf87
            buf89 = buf88; del buf88  # reuse
            buf92 = buf89; del buf89  # reuse
            # Topologically Sorted Source Nodes: [input_37, input_38, input_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12.run(buf92, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, 32768, 128, stream=stream0)
            del arg78_1
            del arg79_1
            del arg80_1
            del arg81_1
            del arg82_1
            buf93 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_13.run(arg83_1, buf93, 65536, 9, stream=stream0)
            del arg83_1
            # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf94 = extern_kernels.convolution(buf92, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf94, (8, 256, 16, 128), (524288, 1, 32768, 256), 'unknown_op')
            del buf92
            del buf93
            buf95 = buf94; del buf94  # reuse
            buf98 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [input_39, input_40, input_41, input_42], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12.run(buf98, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, 32768, 128, stream=stream0)
            del arg84_1
            del arg85_1
            del arg86_1
            del arg87_1
            del arg88_1
            buf99 = empty_strided_cuda((256, 128, 2, 2), (512, 1, 256, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_42, dec2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_32.run(arg89_1, buf99, 32768, 4, stream=stream0)
            del arg89_1
            # Topologically Sorted Source Nodes: [input_42, dec2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf100 = extern_kernels.convolution(buf98, buf99, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf100, (8, 128, 32, 256), (1048576, 1, 32768, 128), 'torch.ops.aten.convolution.default')
            del buf98
            del buf99
            buf101 = reinterpret_tensor(buf102, (8, 128, 32, 256), (2097152, 8192, 256, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_42, dec2], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_33.run(buf100, arg90_1, buf101, 1024, 8192, stream=stream0)
            del arg90_1
            del buf100
            buf103 = empty_strided_cuda((8, 256, 32, 256), (2097152, 1, 65536, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_34.run(buf102, buf103, 2048, 8192, stream=stream0)
            del buf101
            del buf102
            del buf25
            buf104 = empty_strided_cuda((128, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_35.run(arg91_1, buf104, 32768, 9, stream=stream0)
            del arg91_1
            # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
            buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf105, (8, 128, 32, 256), (1048576, 1, 32768, 128), 'unknown_op')
            del buf103
            del buf104
            buf106 = buf105; del buf105  # reuse
            buf109 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [input_43, input_44, input_45], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7.run(buf109, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, 32768, 256, stream=stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            del arg96_1
            buf110 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_8.run(arg97_1, buf110, 16384, 9, stream=stream0)
            del arg97_1
            # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf111 = extern_kernels.convolution(buf109, buf110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf111, (8, 128, 32, 256), (1048576, 1, 32768, 128), 'unknown_op')
            del buf109
            del buf110
            buf112 = buf111; del buf111  # reuse
            buf115 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [input_45, input_46, input_47, input_48], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7.run(buf115, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, 32768, 256, stream=stream0)
            del arg100_1
            del arg101_1
            del arg102_1
            del arg98_1
            del arg99_1
            buf116 = empty_strided_cuda((128, 64, 2, 2), (256, 1, 128, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_48, dec1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_36.run(arg103_1, buf116, 8192, 4, stream=stream0)
            del arg103_1
            # Topologically Sorted Source Nodes: [input_48, dec1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf117 = extern_kernels.convolution(buf115, buf116, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf117, (8, 64, 64, 512), (2097152, 1, 32768, 64), 'torch.ops.aten.convolution.default')
            del buf115
            del buf116
            buf118 = reinterpret_tensor(buf119, (8, 64, 64, 512), (4194304, 32768, 512, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [input_48, dec1], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_37.run(buf117, arg104_1, buf118, 512, 32768, stream=stream0)
            del arg104_1
            del buf117
            buf120 = empty_strided_cuda((8, 128, 64, 512), (4194304, 1, 65536, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_38.run(buf119, buf120, 1024, 32768, stream=stream0)
            del buf118
            del buf119
            del buf12
            buf121 = empty_strided_cuda((64, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_39.run(arg105_1, buf121, 8192, 9, stream=stream0)
            del arg105_1
            # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
            buf122 = extern_kernels.convolution(buf120, buf121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf122, (8, 64, 64, 512), (2097152, 1, 32768, 64), 'unknown_op')
            del buf120
            del buf121
            buf123 = buf122; del buf122  # reuse
            buf126 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [input_49, input_50, input_51], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2.run(buf126, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, 32768, 512, stream=stream0)
            del arg106_1
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            buf127 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_3.run(arg111_1, buf127, 4096, 9, stream=stream0)
            del arg111_1
            # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            buf128 = extern_kernels.convolution(buf126, buf127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf128, (8, 64, 64, 512), (2097152, 1, 32768, 64), 'unknown_op')
            del buf126
            del buf127
            buf129 = buf128; del buf128  # reuse
            buf132 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [input_51, input_52, input_53, input_54, conv2d_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution, aten._native_batch_norm_legit_no_training, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2.run(buf132, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, 32768, 512, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            del arg116_1
            buf133 = empty_strided_cuda((262144, 4), (4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_54, conv2d_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_tem_fused__softmax_convolution_exp_sub_40.run(arg118_1, buf132, arg117_1, buf133, 2048, 1, 1, stream=stream0)
            del arg117_1
            del arg118_1
            del buf132
            buf134 = empty_strided_cuda((8, 4, 64, 512), (131072, 32768, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_54, conv2d_18], Original ATen: [aten.sub, aten.exp, aten._softmax, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_sub_41.run(buf133, buf134, 32, 32768, stream=stream0)
            del buf133
        return (buf134, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 8, 64, 512), (262144, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

# U-Net Implementation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of base features (will be doubled in each layer)
        """
        super(Model, self).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)
    
batch_size = 8
in_channels = 8
out_channels = 4
height = 64
width = 512
features = 64
# Test code for UNet
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, features]
'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 32768*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 8*x2 + 262144*y1), tmp0, ymask)

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
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), tmp24, r0_mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 64) % 64)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 64*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 512*x3 + 32768*x0 + 4194304*x4), tmp24, r0_mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = (xindex % 256)
    x3 = xindex // 256
    y0 = (yindex % 64)
    y1 = yindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (513 + 2*x2 + 1024*x3 + 32768*y0 + 4194304*y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 64*x4 + 524288*y1), tmp6, ymask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), tmp24, r0_mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 128) % 32)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 256*x3 + 8192*x0 + 2097152*x4), tmp24, r0_mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 128)
    x3 = xindex // 128
    y0 = (yindex % 128)
    y1 = yindex // 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (257 + 2*x2 + 512*x3 + 8192*y0 + 2097152*y1), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 128*x4 + 262144*y1), tmp6, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp24, r0_mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 256) % 16)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 128*x3 + 2048*x0 + 1048576*x4), tmp24, r0_mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 64)
    x3 = xindex // 64
    y0 = (yindex % 256)
    y1 = yindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 2*x2 + 256*x3 + 2048*y0 + 1048576*y1), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 256*x4 + 131072*y1), tmp6, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = tmp18 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tmp17 - tmp22
    tmp29 = tl.exp(tmp28)
    tmp30 = (tmp29 / tmp27)
    tl.store(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), tmp30, None)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x5 = xindex
    x3 = ((xindex // 512) % 8)
    x4 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = tmp18 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tmp17 - tmp22
    tmp29 = tl.exp(tmp28)
    tmp30 = (tmp29 / tmp27)
    tl.store(out_ptr2 + (r0_2 + 64*x3 + 512*x0 + 524288*x4), tmp30, None)

''',
    '''@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 32)
    x3 = xindex // 32
    y0 = (yindex % 512)
    y1 = yindex // 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x2 + 128*x3 + 512*y0 + 524288*y1), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 512*x4 + 65536*y1), tmp6, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 1024*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp22 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = tmp18 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp28 = tmp17 - tmp22
    tmp29 = tl.exp(tmp28)
    tmp30 = (tmp29 / tmp27)
    tl.store(in_out_ptr0 + (x0 + 1024*r0_2 + 32768*x1), tmp30, None)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 2048*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_25(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 262144*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512*y0 + 524288*y1), tmp2, xmask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 512*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 524288*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 1024*y1), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_29(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 524288*y1), xmask)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 2048*y0 + 1048576*y1), tmp2, xmask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 2048*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 1048576*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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
def triton_poi_fused__softmax_convolution_exp_sub_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 512*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_33(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 1048576*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 8192*y0 + 2097152*y1), tmp2, None)

''',
    '''@triton.jit
def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 8192*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2097152*y1), tmp0, None)

''',
    '''@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_37(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 2097152*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 32768*y0 + 4194304*y1), tmp2, ymask)

''',
    '''@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 32768*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 4194304*y1), tmp0, None)

''',
    '''@triton.jit
def triton_poi_fused_convolution_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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
def triton_tem_fused__softmax_convolution_exp_sub_40(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 262144
    N = 4
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
        xindex = idx_n + 4*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 131072*y1), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 32768*y3), tmp0, ymask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 8, 64, 512), (262144, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1]

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
