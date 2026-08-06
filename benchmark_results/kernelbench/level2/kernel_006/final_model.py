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


# kernel path: /workspace/tmp/torchinductor_user/aa/caakb6myjzhjsf5qtxqmc6d3ydlxdvgtivglzl5yqbppr3hgwlfy.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
# Graph fragment:
#   %buf0 : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_4 : Tensor "f32[128, 1, 14, 30, 30][12608, 1613824, 900, 30, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %getitem_5 : Tensor "f32[128, 1, 14, 30, 30][12608, 1613824, 900, 30, 1]cuda:0" = PlaceHolder[target=getitem_5]
#   %convolution : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convolution, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_4), kwargs = {})
#   %exp_default : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_5), kwargs = {})
#   return %getitem_4,%getitem_5,%div
triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0 = async_compile.triton('triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0', '''

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
    size_hints={'x': 2097152, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 309657600, 'r0_': 64}, 'kernel_num_gb': 0.206438464, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    r0_numel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D [XBLOCK]
    xmask = xindex < xnumel
    x0 = xindex % 12600   # spatial position (stride-1 in output)
    x1 = xindex // 12600  # batch index

    # Pre-load all 16 bias values as scalars ONCE - only 64 bytes total
    bias0  = tl.load(in_ptr1 + 0,  mask=None, eviction_policy='evict_last')
    bias1  = tl.load(in_ptr1 + 1,  mask=None, eviction_policy='evict_last')
    bias2  = tl.load(in_ptr1 + 2,  mask=None, eviction_policy='evict_last')
    bias3  = tl.load(in_ptr1 + 3,  mask=None, eviction_policy='evict_last')
    bias4  = tl.load(in_ptr1 + 4,  mask=None, eviction_policy='evict_last')
    bias5  = tl.load(in_ptr1 + 5,  mask=None, eviction_policy='evict_last')
    bias6  = tl.load(in_ptr1 + 6,  mask=None, eviction_policy='evict_last')
    bias7  = tl.load(in_ptr1 + 7,  mask=None, eviction_policy='evict_last')
    bias8  = tl.load(in_ptr1 + 8,  mask=None, eviction_policy='evict_last')
    bias9  = tl.load(in_ptr1 + 9,  mask=None, eviction_policy='evict_last')
    bias10 = tl.load(in_ptr1 + 10, mask=None, eviction_policy='evict_last')
    bias11 = tl.load(in_ptr1 + 11, mask=None, eviction_policy='evict_last')
    bias12 = tl.load(in_ptr1 + 12, mask=None, eviction_policy='evict_last')
    bias13 = tl.load(in_ptr1 + 13, mask=None, eviction_policy='evict_last')
    bias14 = tl.load(in_ptr1 + 14, mask=None, eviction_policy='evict_last')
    bias15 = tl.load(in_ptr1 + 15, mask=None, eviction_policy='evict_last')
    bias_cache = [bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7,
                  bias8, bias9, bias10, bias11, bias12, bias13, bias14, bias15]

    # Pass 1: find max over r0=0..15 in registers (no shared memory)
    running_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        val = val + bias_cache[r0]
        running_max = tl.maximum(running_max, val)

    # Pass 2: compute sum of exp(val - max) in registers
    exp_sum = tl.zeros([XBLOCK], tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        val = val + bias_cache[r0]
        exp_val = tl.exp(val - running_max)
        exp_sum = exp_sum + tl.where(xmask, exp_val, 0.0)

    # Pass 3: write normalized output
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        val = val + bias_cache[r0]
        out_val = tl.exp(val - running_max) / exp_sum
        tl.store(out_ptr2 + (x0 + 12608*r0 + 201728*x1), tl.where(xmask, out_val, 0.0), mask=xmask)

def get_args():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201600, 12600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1612800, 16,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.206438464
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/vl/cvl2yg3jtx7km6drosab63z2zy2up64drrjyq7nqnjg6tqjqjkjd.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.sub, aten.exp, aten._softmax, aten.max_pool3d_with_indices]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
#   x_2 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %div : Tensor "f32[128, 16, 14, 30, 30][201728, 12608, 900, 30, 1]cuda:0" = PlaceHolder[target=div]
#   %convolution : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_4), kwargs = {})
#   %exp_default : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_5), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1', '''

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 25804800}, 'kernel_num_gb': 0.1161216, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3225600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex % 15)            # output W [0..14] innermost/fastest
    x2 = ((xindex // 15) % 15)    # output H [0..14]
    x1 = ((xindex // 225) % 7)    # output D [0..6]
    x0 = xindex // 1575            # batch*channel [0..2047] outermost/slowest
    x4 = (xindex % 1575)          # within-tile offset = x3 + 15*x2 + 225*x1
    base = 2*x3 + 60*x2 + 1800*x1 + 12608*x0
    tmp0 = tl.load(in_ptr0 + base,        mask=xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + base + 1,    mask=xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + base + 30,   mask=xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + base + 31,   mask=xmask, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr0 + base + 900,  mask=xmask, eviction_policy='evict_first')
    tmp9 = tl.load(in_ptr0 + base + 901,  mask=xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + base + 930, mask=xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + base + 931, mask=xmask, eviction_policy='evict_first')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4 + 1600*x0), tmp14, mask=xmask)

def get_args():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 3225600,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.1161216
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /workspace/tmp/torchinductor_user/cu/ccu5orutquuohl7y73vpt5aojfvv2oiaxvmh5mmymmtzcjqj5yy3.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.sub, aten.exp, aten._softmax, aten.max_pool3d_with_indices]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
#   x_2 => _low_memory_max_pool_with_offsets
#   x_3 => _low_memory_max_pool_with_offsets_1, getitem_2
# Graph fragment:
#   %getitem : Tensor "f32[128, 16, 7, 15, 15][25600, 1600, 225, 15, 1]cuda:0" = PlaceHolder[target=getitem]
#   %convolution : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_4), kwargs = {})
#   %exp_default : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 16, 14, 30, 30][201600, 12600, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_5), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%div, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%getitem, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), kwargs = {})
#   %getitem_2 : Tensor "f32[128, 16, 3, 7, 7][2352, 147, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_1, 0), kwargs = {})
#   return %getitem_2
triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2 = async_compile.triton('triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2408448}, 'kernel_num_gb': 0.010838016, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 7)
    x2 = ((xindex // 49) % 3)
    x3 = xindex // 147
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (15 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (225 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (226 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (240 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (241 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4), tmp14, xmask)


def get_args():
    arg_0 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 3, 7, 7), (2352, 147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 301056,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.010838016
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
        assert_size_stride(arg0_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (16, ), (1, ))
        assert_size_stride(arg2_1, (128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (128, 16, 14, 30, 30), (201600, 12600, 900, 30, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf3 = empty_strided_cuda((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0.run(buf0, arg1_1, buf3, 1612800, 16, stream=stream0)
            del arg1_1
            del buf0
            buf4 = empty_strided_cuda((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.sub, aten.exp, aten._softmax, aten.max_pool3d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1.run(buf3, buf4, 3225600, stream=stream0)
            del buf3
            buf5 = empty_strided_cuda((128, 16, 3, 7, 7), (2352, 147, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.convolution, aten.sub, aten.exp, aten._softmax, aten.max_pool3d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2.run(buf4, buf5, 301056, stream=stream0)
            del buf4
        return (buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16, 3, 3, 3, 3), (81, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_2 = r0_index
    x0 = (xindex % 12600)
    x1 = xindex // 12600
    tmp0 = tl.load(in_ptr0 + (x0 + 12600*r0_2 + 201600*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tmp3 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = tmp2 - tmp8
    tmp16 = tl.exp(tmp15)
    tmp17 = (tmp16 / tmp14)
    tl.store(out_ptr2 + (x0 + 12608*r0_2 + 201728*x1), tmp17, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3225600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 15)
    x1 = ((xindex // 15) % 15)
    x2 = ((xindex // 225) % 7)
    x3 = xindex // 1575
    x4 = (xindex % 1575)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (30 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (31 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (900 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (901 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (930 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (931 + 2*x0 + 60*x1 + 1800*x2 + 12608*x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4 + 1600*x3), tmp14, xmask)

''',
    '''@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 7)
    x2 = ((xindex // 49) % 3)
    x3 = xindex // 147
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (15 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (225 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (226 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (240 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (241 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4), tmp14, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((16, 3, 3, 3, 3), (81, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
