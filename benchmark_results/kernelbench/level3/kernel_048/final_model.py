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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/sw/cswzdbtdkvvble3e4bp6ly6yejmj4xj34escalj3xoreusqcpid3.py
# Topologically Sorted Source Nodes: [A_blocks, A_blocks_1, A_cumsum], Original ATen: [aten.view, aten.permute, aten.cumsum]
# Source node to ATen node mapping:
#   A_blocks => view_1
#   A_blocks_1 => permute
#   A_cumsum => cumsum
# Graph fragment:
#   %arg0_1 : Tensor "f32[2048, 128, 8][1024, 8, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view_1 : Tensor "f32[2048, 2, 64, 8][1024, 512, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2048, 2, 64, 8]), kwargs = {})
#   %permute : Tensor "f32[2048, 8, 2, 64][1024, 1, 512, 8]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 3, 1, 2]), kwargs = {})
#   %cumsum : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.cumsum.default](args = (%permute, -1), kwargs = {})
#   return %cumsum
triton_per_fused_cumsum_permute_view_0 = async_compile.triton('triton_per_fused_cumsum_permute_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_permute_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8388608, 'r0_': 16777216}, 'kernel_num_gb': 0.016777216, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_cumsum_permute_view_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 8)
    x1 = xindex // 8
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*r0_2 + 512*x1), None)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp3, = tl.associative_scan((tmp2,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r0_2 + 64*x3), tmp3, None)


def get_args():
    arg_0 = rand_strided((2048, 128, 8), (1024, 8, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 32768, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_permute_view_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_cumsum_permute_view_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016777216
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/kc/ckc6n7ogac4qi5zkx2orpdmwvmpin7mnpzxaosybudyv47oksavd.py
# Topologically Sorted Source Nodes: [getitem_4, pad, x_cumsum_1], Original ATen: [aten.select, aten.constant_pad_nd, aten.cumsum]
# Source node to ATen node mapping:
#   getitem_4 => select
#   pad => constant_pad_nd
#   x_cumsum_1 => cumsum_2
# Graph fragment:
#   %cumsum : Tensor "f32[2048, 8, 2, 64][1024, 64, 512, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %select : Tensor "f32[2048, 8, 2][1024, 128, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cumsum, 3, -1), kwargs = {})
#   %constant_pad_nd : Tensor "f32[2048, 8, 3][24, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%select, [1, 0], 0.0), kwargs = {})
#   %cumsum_2 : Tensor "f32[2048, 8, 3][24, 3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%constant_pad_nd, -1), kwargs = {})
#   return %cumsum_2
triton_per_fused_constant_pad_nd_cumsum_select_1 = async_compile.triton('triton_per_fused_constant_pad_nd_cumsum_select_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_cumsum_select_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 147456}, 'kernel_num_gb': 0.000393216, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_constant_pad_nd_cumsum_select_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
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
    x0 = (xindex % 8)
    x1 = xindex // 8
    x3 = xindex
    tmp0 = (-1) + r0_2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-449) + 64*x0 + 512*r0_2 + 1024*x1), r0_mask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp6, = tl.associative_scan((tmp5,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r0_2 + 3*x3), tmp6, r0_mask)


def get_args():
    arg_0 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3), (24, 3, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 16384, 3,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_constant_pad_nd_cumsum_select_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_constant_pad_nd_cumsum_select_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000393216
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/j2/cj2lw7elwmbzltcrl2zbdurp44kpxyrjil4emoql7rfwqv7moevm.py
# Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp]
# Source node to ATen node mapping:
#   decay_chunk => exp_2
#   getitem_5 => unsqueeze_16
#   getitem_6 => unsqueeze_17
#   invert_1 => bitwise_not_1
#   mask_1 => iota_2, iota_3, le_1, logical_and_1, sub_4, unsqueeze_18, unsqueeze_19
#   ones_1 => full_default_3
#   x_segsum_2 => sub_3
#   x_segsum_3 => full_default_4, where_1
# Graph fragment:
#   %cumsum_2 : Tensor "f32[2048, 8, 3][24, 3, 1]cuda:0" = PlaceHolder[target=cumsum_2]
#   %iota_2 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_18 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_2, -2), kwargs = {})
#   %iota_3 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_19 : Tensor "i64[3, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_3, -1), kwargs = {})
#   %sub_4 : Tensor "i64[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_18, %unsqueeze_19), kwargs = {})
#   %le_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_4, 0), kwargs = {})
#   %full_default_3 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([3, 3], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_1, %full_default_3), kwargs = {})
#   %bitwise_not_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and_1,), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %unsqueeze_16 : Tensor "f32[2048, 8, 3, 1][24, 3, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_2, 3), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[2048, 8, 1, 3][24, 3, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_2, 2), kwargs = {})
#   %sub_3 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_16, %unsqueeze_17), kwargs = {})
#   %where_1 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_1, %full_default_4, %sub_3), kwargs = {})
#   %exp_2 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where_1,), kwargs = {})
#   return %exp_2
triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2 = async_compile.triton('triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1376256}, 'kernel_num_gb': 0.000786432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 3)
    x1 = ((xindex // 3) % 3)
    x3 = xindex // 3
    x2 = xindex // 9
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), None, eviction_policy='evict_last')
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp2 & tmp3
    tmp5 = tmp4 == 0
    tmp8 = tmp6 - tmp7
    tmp9 = tl.full([1], float("-inf"), tl.float32)
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = libdevice.exp(tmp10)
    tl.store(out_ptr0 + (x4), tmp11, None)


def get_args():
    arg_0 = rand_strided((2048, 8, 3), (24, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 147456,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000786432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bb/cbb3dza5qnyxqk5nf7lzaf4cj7wcmvul5waalpdkndx6cfehghxm.py
# Topologically Sorted Source Nodes: [A_blocks, A_blocks_1, x_cumsum], Original ATen: [aten.view, aten.permute, aten.cumsum]
# Source node to ATen node mapping:
#   A_blocks => view_1
#   A_blocks_1 => permute
#   x_cumsum => cumsum_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[2048, 128, 8][1024, 8, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view_1 : Tensor "f32[2048, 2, 64, 8][1024, 512, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2048, 2, 64, 8]), kwargs = {})
#   %permute : Tensor "f32[2048, 8, 2, 64][1024, 1, 512, 8]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 3, 1, 2]), kwargs = {})
#   %cumsum_1 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%permute, -1), kwargs = {})
#   return %cumsum_1
triton_per_fused_cumsum_permute_view_3 = async_compile.triton('triton_per_fused_cumsum_permute_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'y': 16384, 'x': 2, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_permute_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8388608, 'x': 0, 'r0_': 16777216}, 'kernel_num_gb': 0.016777216, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_cumsum_permute_view_3(in_ptr0, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 2
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x2 = xindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8*r0_3 + 512*x2 + 1024*y1), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp3, = tl.associative_scan((tmp2,), 2, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r0_3 + 64*x2 + 128*y4), tmp3, xmask)


def get_args():
    arg_0 = rand_strided((2048, 128, 8), (1024, 8, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 128, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 16384, 2, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_permute_view_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_cumsum_permute_view_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016777216
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/b4/cb4jaxvg5nq2tvacvxgw4mlwa7npdox3o6hkw5vjfxawuf2qx2yw.py
# Topologically Sorted Source Nodes: [C_blocks, Y_diag, Y_off], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   C_blocks => view_3
#   Y_diag => clone, permute_1, permute_5, unsqueeze_4, unsqueeze_5
#   Y_off => clone_6, permute_22, permute_25, unsqueeze_23
# Graph fragment:
#   %arg2_1 : Tensor "f32[2048, 128, 8, 16][16384, 128, 16, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view_3 : Tensor "f32[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 5), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[2048, 2, 64, 8, 16, 1, 1][16384, 8192, 128, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 6), kwargs = {})
#   %permute_1 : Tensor "f32[2048, 2, 64, 8, 1, 16, 1][16384, 8192, 128, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_5, [0, 1, 2, 3, 5, 4, 6]), kwargs = {})
#   %permute_5 : Tensor "f32[2048, 2, 8, 64, 16, 1, 1][16384, 8192, 16, 128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2, 5, 4, 6]), kwargs = {})
#   %clone : Tensor "f32[2048, 2, 8, 64, 16, 1, 1][16384, 8192, 1024, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze_23 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 5), kwargs = {})
#   %permute_22 : Tensor "f32[2048, 2, 64, 8, 1, 16][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_23, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %permute_25 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 16, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_22, [0, 1, 3, 2, 5, 4]), kwargs = {})
#   %clone_6 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_25,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone,%clone_6
triton_poi_fused_clone_permute_unsqueeze_view_4 = async_compile.triton('triton_poi_fused_clone_permute_unsqueeze_view_4', '''

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 671088640}, 'kernel_num_gb': 0.402653184, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 64)
    x2 = ((xindex // 1024) % 8)
    x3 = xindex // 8192
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 8192*x3), None, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x4), tmp0, None, eviction_policy='evict_last')
    tl.store(out_ptr1 + (x4), tmp0, None, eviction_policy='evict_last')

def get_args():
    arg_0 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 64, 16, 1, 1), (16384, 8192, 1024, 16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 33554432,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_unsqueeze_view_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_permute_unsqueeze_view_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.402653184
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/gs/cgswslf7phxjshmon3uwqpz5nth2wzrusfk5oq5exqnduswkltsb.py
# Topologically Sorted Source Nodes: [B_blocks, Y_diag, states, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten.slice, aten.sub, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   B_blocks => view_2
#   Y_diag => clone_1, permute_2, permute_6, unsqueeze_6, unsqueeze_7
#   decay_states => exp_1
#   getitem_2 => slice_1
#   states => clone_4, mul_1, permute_11, permute_12, permute_15, unsqueeze_12, unsqueeze_13, unsqueeze_14
#   sub_1 => sub_2
# Graph fragment:
#   %arg1_1 : Tensor "f32[2048, 128, 8, 16][16384, 128, 16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %cumsum : Tensor "f32[2048, 8, 2, 64][1024, 64, 512, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %view_2 : Tensor "f32[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 5), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[2048, 2, 64, 8, 16, 1, 1][16384, 8192, 128, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 6), kwargs = {})
#   %permute_2 : Tensor "f32[2048, 2, 1, 8, 1, 16, 64][16384, 8192, 1, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_7, [0, 1, 5, 3, 6, 4, 2]), kwargs = {})
#   %permute_6 : Tensor "f32[2048, 2, 8, 16, 1, 64, 1][16384, 8192, 16, 1, 1, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_2, [0, 1, 3, 5, 4, 6, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[2048, 2, 8, 16, 1, 64, 1][16384, 8192, 1024, 64, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze_12 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 5), kwargs = {})
#   %permute_11 : Tensor "f32[2048, 2, 8, 1, 16, 64][16384, 8192, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_12, [0, 1, 3, 5, 4, 2]), kwargs = {})
#   %slice_1 : Tensor "f32[2048, 8, 2, 1][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cumsum, 3, -1, 9223372036854775807), kwargs = {})
#   %sub_2 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %cumsum), kwargs = {})
#   %exp_1 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_1, 4), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[2048, 8, 2, 64, 1, 1][1024, 128, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, 5), kwargs = {})
#   %permute_12 : Tensor "f32[2048, 2, 8, 1, 1, 64][1024, 64, 128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_14, [0, 2, 1, 4, 5, 3]), kwargs = {})
#   %mul_1 : Tensor "f32[2048, 2, 8, 1, 16, 64][16384, 8192, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_11, %permute_12), kwargs = {})
#   %permute_15 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 16, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_1, [0, 1, 2, 5, 4, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1,%clone_4
triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5 = async_compile.triton('triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5', '''

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
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 276824064, 'x': 402653184}, 'kernel_num_gb': 0.411041792, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 8)
    y2 = yindex // 512
    y4 = yindex // 64
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y1 + 128*y0 + 8192*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (63 + 64*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y5), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (y0 + 64*x3 + 1024*y4), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 16*y5), tmp5, xmask & ymask)

def get_args():
    arg_0 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 16, 1, 64, 1), (16384, 8192, 1024, 64, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 2097152, 16,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.411041792
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ka/ckaaqdi2fvjrktyrwngltjfsofkwt3luqufjtwgkqcnpzfer5lok.py
# Topologically Sorted Source Nodes: [C_blocks, Y_diag, B_blocks], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.bmm]
# Source node to ATen node mapping:
#   B_blocks => view_2
#   C_blocks => view_3
#   Y_diag => bmm, clone, clone_1, permute_1, permute_2, permute_5, permute_6, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, view_4, view_5
# Graph fragment:
#   %clone : Tensor "f32[2048, 2, 8, 64, 16, 1, 1][16384, 8192, 1024, 16, 1, 1, 1]cuda:0" = PlaceHolder[target=clone]
#   %clone_1 : Tensor "f32[2048, 2, 8, 16, 1, 64, 1][16384, 8192, 1024, 64, 64, 1, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_3 : Tensor "f32[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 5), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[2048, 2, 64, 8, 16, 1, 1][16384, 8192, 128, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 6), kwargs = {})
#   %permute_1 : Tensor "f32[2048, 2, 64, 8, 1, 16, 1][16384, 8192, 128, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_5, [0, 1, 2, 3, 5, 4, 6]), kwargs = {})
#   %permute_5 : Tensor "f32[2048, 2, 8, 64, 16, 1, 1][16384, 8192, 16, 128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2, 5, 4, 6]), kwargs = {})
#   %clone : Tensor "f32[2048, 2, 8, 64, 16, 1, 1][16384, 8192, 1024, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_4 : Tensor "f32[32768, 64, 16][1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [32768, 64, 16]), kwargs = {})
#   %view_2 : Tensor "f32[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 5), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[2048, 2, 64, 8, 16, 1, 1][16384, 8192, 128, 16, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 6), kwargs = {})
#   %permute_2 : Tensor "f32[2048, 2, 1, 8, 1, 16, 64][16384, 8192, 1, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_7, [0, 1, 5, 3, 6, 4, 2]), kwargs = {})
#   %permute_6 : Tensor "f32[2048, 2, 8, 16, 1, 64, 1][16384, 8192, 16, 1, 1, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_2, [0, 1, 3, 5, 4, 6, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[2048, 2, 8, 16, 1, 64, 1][16384, 8192, 1024, 64, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_5 : Tensor "f32[32768, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [32768, 16, 64]), kwargs = {})
#   %bmm : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_4, %view_5), kwargs = {})
#   return %bmm
triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6 = async_compile.triton('triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6', '''

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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.805306368, 'kernel_flop': 4294967296, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 64
    N = 64
    K = 16

    stride_aq = 1024
    stride_am = 16
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 64
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 4096*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def get_args():
    arg_0 = rand_strided((2048, 2, 8, 64, 16, 1, 1), (16384, 8192, 1024, 16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 16, 1, 64, 1), (16384, 8192, 1024, 64, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1, 32768, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.805306368
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/jz/cjzdtsiv6rr6wqviiaohrfarbdwsuif2eid3kp4drpw37fww5hvt.py
# Topologically Sorted Source Nodes: [Y_diag, mask, ones, invert, x_segsum_1, getitem, getitem_1, x_segsum, L], Original ATen: [aten.view, aten.permute, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   L => exp
#   Y_diag => mul, permute_3, permute_7, unsqueeze_8, unsqueeze_9, view_6
#   getitem => unsqueeze
#   getitem_1 => unsqueeze_1
#   invert => bitwise_not
#   mask => iota, iota_1, le, logical_and, sub_1, unsqueeze_2, unsqueeze_3
#   ones => full_default
#   x_segsum => sub
#   x_segsum_1 => full_default_1, where
# Graph fragment:
#   %bmm : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm]
#   %cumsum_1 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0" = PlaceHolder[target=cumsum_1]
#   %view_6 : Tensor "f32[2048, 2, 8, 64, 1, 1, 64][65536, 32768, 4096, 64, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2048, 2, 8, 64, 1, 1, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 64, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 1, 3, 2, 5, 4, 6]), kwargs = {})
#   %iota : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %iota_1 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_3 : Tensor "i64[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %sub_1 : Tensor "i64[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %le : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 64], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %bitwise_not : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %unsqueeze : Tensor "f32[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_1, 4), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2048, 8, 2, 1, 64][1024, 128, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_1, 3), kwargs = {})
#   %sub : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %where : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default_1, %sub), kwargs = {})
#   %exp : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where,), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[2048, 8, 2, 64, 64, 1][65536, 8192, 4096, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 5), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[2048, 8, 2, 64, 64, 1, 1][65536, 8192, 4096, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 6), kwargs = {})
#   %permute_3 : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 4096, 64, 8192, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_9, [0, 2, 3, 1, 5, 6, 4]), kwargs = {})
#   %mul : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 64, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_7, %permute_3), kwargs = {})
#   return %mul
triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7 = async_compile.triton('triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7', '''

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
    size_hints={'y': 2097152, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 8388608, 'x': 1619001344}, 'kernel_num_gb': 1.082130432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y5 = yindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 8)
    y2 = ((yindex // 512) % 2)
    y3 = yindex // 1024
    y7 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x4 + 64*y5), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (y0 + 64*y2 + 128*y1 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x4 + 64*y2 + 128*y1 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = x4 + ((-1)*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1, 1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5 == 0
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp11 = tl.where(tmp6, tmp10, tmp9)
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp0 * tmp12
    tl.store(out_ptr0 + (x4 + 64*y1 + 512*y0 + 32768*y7), tmp13, xmask & ymask)

def get_args():
    arg_0 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 2, 64), (1024, 128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 64, 8, 1, 1, 64), (65536, 32768, 512, 64, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 2097152, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.082130432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/na/cna7hi57gz6chju6runteuibau5d6phjryqjbws2f22uo5i7lwmt.py
# Topologically Sorted Source Nodes: [Y_diag, mask, ones, invert, x_segsum_1, getitem, getitem_1, x_segsum, L, X_blocks], Original ATen: [aten.view, aten.permute, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.mul, aten.clone, aten._unsafe_view, aten.bmm]
# Source node to ATen node mapping:
#   L => exp
#   X_blocks => view
#   Y_diag => bmm_1, clone_2, mul, permute_3, permute_4, permute_7, permute_8, permute_9, unsqueeze_10, unsqueeze_11, unsqueeze_8, unsqueeze_9, view_6, view_7, view_8
#   getitem => unsqueeze
#   getitem_1 => unsqueeze_1
#   invert => bitwise_not
#   mask => iota, iota_1, le, logical_and, sub_1, unsqueeze_2, unsqueeze_3
#   ones => full_default
#   x_segsum => sub
#   x_segsum_1 => full_default_1, where
# Graph fragment:
#   %mul : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 512, 64, 64, 64, 1]cuda:0" = PlaceHolder[target=mul]
#   %view_6 : Tensor "f32[2048, 2, 8, 64, 1, 1, 64][65536, 32768, 4096, 64, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2048, 2, 8, 64, 1, 1, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 64, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 1, 3, 2, 5, 4, 6]), kwargs = {})
#   %iota : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %iota_1 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_3 : Tensor "i64[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %sub_1 : Tensor "i64[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %le : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 64], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %bitwise_not : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %unsqueeze : Tensor "f32[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_1, 4), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2048, 8, 2, 1, 64][1024, 128, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_1, 3), kwargs = {})
#   %sub : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %where : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default_1, %sub), kwargs = {})
#   %exp : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where,), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[2048, 8, 2, 64, 64, 1][65536, 8192, 4096, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 5), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[2048, 8, 2, 64, 64, 1, 1][65536, 8192, 4096, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 6), kwargs = {})
#   %permute_3 : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 4096, 64, 8192, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_9, [0, 2, 3, 1, 5, 6, 4]), kwargs = {})
#   %mul : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 64, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_7, %permute_3), kwargs = {})
#   %permute_8 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 4096, 64, 1, 64, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 1, 3, 2, 6, 4, 5]), kwargs = {})
#   %view_7 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_8, [32768, 64, 64]), kwargs = {})
#   %view : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [2048, 2, 64, 8, 64]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 5), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[2048, 2, 64, 8, 64, 1, 1][65536, 32768, 512, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 6), kwargs = {})
#   %permute_4 : Tensor "f32[2048, 2, 1, 8, 64, 1, 64][65536, 32768, 1, 64, 1, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_11, [0, 1, 5, 3, 4, 6, 2]), kwargs = {})
#   %permute_9 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 64, 512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [0, 1, 3, 6, 4, 5, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 4096, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [32768, 64, 64]), kwargs = {})
#   %bmm_1 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_7, %view_8), kwargs = {})
#   return %buf5
triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8', '''

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1610612736}, 'kernel_num_gb': 1.073741824, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((x2 % 8)) + 512*x1 + 32768*(x2 // 8)), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)

def get_args():
    arg_0 = rand_strided((2048, 2, 64, 8, 1, 1, 64), (65536, 32768, 512, 64, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 134217728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.073741824
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4u/c4uruva5ttjomb2gci5uuvdjwmccdeheq2brzn52ywb34q5qi4cr.py
# Topologically Sorted Source Nodes: [X_blocks, Y_diag, states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   X_blocks => view
#   Y_diag => clone_2, permute_4, permute_9, unsqueeze_10, unsqueeze_11
#   states => clone_3, permute_13, permute_14, unsqueeze_15
# Graph fragment:
#   %arg3_1 : Tensor "f32[2048, 128, 8, 64][65536, 512, 64, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %view : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [2048, 2, 64, 8, 64]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 5), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[2048, 2, 64, 8, 64, 1, 1][65536, 32768, 512, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 6), kwargs = {})
#   %permute_4 : Tensor "f32[2048, 2, 1, 8, 64, 1, 64][65536, 32768, 1, 64, 1, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_11, [0, 1, 5, 3, 4, 6, 2]), kwargs = {})
#   %permute_9 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 64, 512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [0, 1, 3, 6, 4, 5, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 4096, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze_15 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 5), kwargs = {})
#   %permute_13 : Tensor "f32[2048, 2, 8, 64, 1, 64][65536, 32768, 64, 1, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_15, [0, 1, 3, 4, 5, 2]), kwargs = {})
#   %permute_14 : Tensor "f32[2048, 2, 8, 64, 64, 1][65536, 32768, 64, 1, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_13, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %clone_3 : Tensor "f32[2048, 2, 8, 64, 64, 1][65536, 32768, 4096, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2,%clone_3
triton_poi_fused_clone_permute_unsqueeze_view_9 = async_compile.triton('triton_poi_fused_clone_permute_unsqueeze_view_9', '''

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
    size_hints={'y': 2097152, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_unsqueeze_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1610612736, 'x': 1073741824}, 'kernel_num_gb': 1.610612736, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_9(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 512
    y4 = (yindex % 512)
    y0 = (yindex % 64)
    y5 = yindex // 64
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 512*x3 + 32768*y2), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x3 + 4096*y5), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 64*y6), tmp0, xmask & ymask)

def get_args():
    arg_0 = rand_strided((2048, 128, 8, 64), (65536, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 64, 64, 1, 1), (65536, 32768, 4096, 64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 2, 8, 64, 64, 1), (65536, 32768, 4096, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 2097152, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_unsqueeze_view_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_permute_unsqueeze_view_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.610612736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pj/cpj4srsubtzqu6q33fs77eul64f6hwvudsdnsf6ylgtv77kmilff.py
# Topologically Sorted Source Nodes: [B_blocks, X_blocks, states, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.slice, aten.sub, aten.exp, aten.mul, aten.bmm]
# Source node to ATen node mapping:
#   B_blocks => view_2
#   X_blocks => view
#   decay_states => exp_1
#   getitem_2 => slice_1
#   states => bmm_2, clone_3, clone_4, mul_1, permute_11, permute_12, permute_13, permute_14, permute_15, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, view_11, view_12
#   sub_1 => sub_2
# Graph fragment:
#   %clone_3 : Tensor "f32[2048, 2, 8, 64, 64, 1][65536, 32768, 4096, 64, 1, 1]cuda:0" = PlaceHolder[target=clone_3]
#   %clone_4 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0" = PlaceHolder[target=clone_4]
#   %view_2 : Tensor "f32[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %view : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [2048, 2, 64, 8, 64]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 5), kwargs = {})
#   %permute_13 : Tensor "f32[2048, 2, 8, 64, 1, 64][65536, 32768, 64, 1, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_15, [0, 1, 3, 4, 5, 2]), kwargs = {})
#   %permute_14 : Tensor "f32[2048, 2, 8, 64, 64, 1][65536, 32768, 64, 1, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_13, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %clone_3 : Tensor "f32[2048, 2, 8, 64, 64, 1][65536, 32768, 4096, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   %view_11 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [32768, 64, 64]), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 5), kwargs = {})
#   %permute_11 : Tensor "f32[2048, 2, 8, 1, 16, 64][16384, 8192, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_12, [0, 1, 3, 5, 4, 2]), kwargs = {})
#   %slice_1 : Tensor "f32[2048, 8, 2, 1][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cumsum, 3, -1, 9223372036854775807), kwargs = {})
#   %sub_2 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %cumsum), kwargs = {})
#   %exp_1 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_1, 4), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[2048, 8, 2, 64, 1, 1][1024, 128, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, 5), kwargs = {})
#   %permute_12 : Tensor "f32[2048, 2, 8, 1, 1, 64][1024, 64, 128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_14, [0, 2, 1, 4, 5, 3]), kwargs = {})
#   %mul_1 : Tensor "f32[2048, 2, 8, 1, 16, 64][16384, 8192, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_11, %permute_12), kwargs = {})
#   %permute_15 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 16, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_1, [0, 1, 2, 5, 4, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %view_12 : Tensor "f32[32768, 64, 16][1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [32768, 64, 16]), kwargs = {})
#   %bmm_2 : Tensor "f32[32768, 64, 16][1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_11, %view_12), kwargs = {})
#   return %bmm_2
triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10 = async_compile.triton('triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10', '''

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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.805306368, 'kernel_flop': 4294967296, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
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

    M = 64
    N = 16
    K = 64

    stride_aq = 4096
    stride_am = 64
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 16
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16*idx_m + 1024*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def get_args():
    arg_0 = rand_strided((2048, 2, 8, 64, 64, 1), (65536, 32768, 4096, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 64, 16), (1024, 16, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1, 32768, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.805306368
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/z5/cz53frfhe7rwmh3qfn72t4mpjttmrlubsecalffb4xp5vjrhyjak.py
# Topologically Sorted Source Nodes: [Y_diag, mask, ones, invert, x_segsum_1, getitem, getitem_1, x_segsum, L, X_blocks], Original ATen: [aten.view, aten.permute, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.mul, aten.clone, aten._unsafe_view, aten.bmm]
# Source node to ATen node mapping:
#   L => exp
#   X_blocks => view
#   Y_diag => bmm_1, clone_2, mul, permute_3, permute_4, permute_7, permute_8, permute_9, unsqueeze_10, unsqueeze_11, unsqueeze_8, unsqueeze_9, view_6, view_7, view_8
#   getitem => unsqueeze
#   getitem_1 => unsqueeze_1
#   invert => bitwise_not
#   mask => iota, iota_1, le, logical_and, sub_1, unsqueeze_2, unsqueeze_3
#   ones => full_default
#   x_segsum => sub
#   x_segsum_1 => full_default_1, where
# Graph fragment:
#   %buf5 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=buf5]
#   %clone_2 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 4096, 64, 1, 1, 1]cuda:0" = PlaceHolder[target=clone_2]
#   %view_6 : Tensor "f32[2048, 2, 8, 64, 1, 1, 64][65536, 32768, 4096, 64, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2048, 2, 8, 64, 1, 1, 64]), kwargs = {})
#   %permute_7 : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 64, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 1, 3, 2, 5, 4, 6]), kwargs = {})
#   %iota : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %iota_1 : Tensor "i64[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_3 : Tensor "i64[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %sub_1 : Tensor "i64[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %le : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 64], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %bitwise_not : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %unsqueeze : Tensor "f32[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_1, 4), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2048, 8, 2, 1, 64][1024, 128, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_1, 3), kwargs = {})
#   %sub : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %where : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default_1, %sub), kwargs = {})
#   %exp : Tensor "f32[2048, 8, 2, 64, 64][65536, 8192, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where,), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[2048, 8, 2, 64, 64, 1][65536, 8192, 4096, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 5), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[2048, 8, 2, 64, 64, 1, 1][65536, 8192, 4096, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 6), kwargs = {})
#   %permute_3 : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 4096, 64, 8192, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_9, [0, 2, 3, 1, 5, 6, 4]), kwargs = {})
#   %mul : Tensor "f32[2048, 2, 64, 8, 1, 1, 64][65536, 32768, 64, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_7, %permute_3), kwargs = {})
#   %permute_8 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 4096, 64, 1, 64, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 1, 3, 2, 6, 4, 5]), kwargs = {})
#   %view_7 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_8, [32768, 64, 64]), kwargs = {})
#   %view : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 512, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [2048, 2, 64, 8, 64]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 5), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[2048, 2, 64, 8, 64, 1, 1][65536, 32768, 512, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 6), kwargs = {})
#   %permute_4 : Tensor "f32[2048, 2, 1, 8, 64, 1, 64][65536, 32768, 1, 64, 1, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_11, [0, 1, 5, 3, 4, 6, 2]), kwargs = {})
#   %permute_9 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 64, 512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [0, 1, 3, 6, 4, 5, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[2048, 2, 8, 64, 64, 1, 1][65536, 32768, 4096, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [32768, 64, 64]), kwargs = {})
#   %bmm_1 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_7, %view_8), kwargs = {})
#   return %bmm_1
triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11 = async_compile.triton('triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11', '''

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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 1.610612736, 'kernel_flop': 17179869184, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11(arg_A, arg_B, out_ptr0):
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

    M = 64
    N = 64
    K = 64

    stride_aq = 4096
    stride_am = 64
    stride_ak = 1

    stride_bq = 4096
    stride_bk = 64
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 4096*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def get_args():
    arg_0 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 64, 64, 1, 1), (65536, 32768, 4096, 64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1, 32768, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.610612736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ti/ctioi2rykr63dfflqicj36di5gsny23wf43wm36pzxd5bhmbb64d.py
# Topologically Sorted Source Nodes: [initial_states, states, states_1, new_states], Original ATen: [aten.zeros_like, aten.view, aten.permute, aten.cat, aten.unsqueeze, aten.clone]
# Source node to ATen node mapping:
#   initial_states => full_default_2
#   new_states => clone_5, permute_18, permute_20, unsqueeze_22
#   states => permute_16, view_13, view_14
#   states_1 => cat
# Graph fragment:
#   %bmm_2 : Tensor "f32[32768, 64, 16][1024, 16, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %full_default_2 : Tensor "f32[2048, 1, 8, 64, 16][8192, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2048, 1, 8, 64, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_13 : Tensor "f32[2048, 2, 8, 64, 1, 16][16384, 8192, 1024, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [2048, 2, 8, 64, 1, 16]), kwargs = {})
#   %permute_16 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 16]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %view_14 : Tensor "f32[2048, 2, 8, 64, 16][16384, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_16, [2048, 2, 8, 64, 16]), kwargs = {})
#   %cat : Tensor "f32[2048, 3, 8, 64, 16][24576, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default_2, %view_14], 1), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[2048, 3, 8, 64, 16, 1][24576, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 5), kwargs = {})
#   %permute_18 : Tensor "f32[2048, 1, 8, 64, 16, 3][24576, 1, 1024, 16, 1, 8192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_22, [0, 5, 2, 3, 4, 1]), kwargs = {})
#   %permute_20 : Tensor "f32[2048, 8, 3, 64, 16, 1][24576, 1024, 8192, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_18, [0, 2, 5, 3, 4, 1]), kwargs = {})
#   %clone_5 : Tensor "f32[2048, 8, 3, 64, 16, 1][24576, 3072, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12 = async_compile.triton('triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12', '''

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
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 536870912}, 'kernel_num_gb': 0.33554432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 1024) % 3)
    x0 = (xindex % 1024)
    x2 = ((xindex // 3072) % 8)
    x3 = xindex // 24576
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), mask=tmp8 & xmask, other=0.0)
    tmp12 = tl.where(tmp4, tmp7, tmp11)
    tl.store(out_ptr0 + (x4), tmp12, mask=xmask)

def get_args():
    arg_0 = rand_strided((32768, 64, 16), (1024, 16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 50331648,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.33554432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/kx/ckxdnzimzbeosy674xlbsx2p7unebo4kv2dzfmyax3wzr4zjgi7s.py
# Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk, new_states, initial_states, states, states_1], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.view, aten.zeros_like, aten.permute, aten.cat, aten.clone, aten._unsafe_view, aten.bmm]
# Source node to ATen node mapping:
#   decay_chunk => exp_2
#   getitem_5 => unsqueeze_16
#   getitem_6 => unsqueeze_17
#   initial_states => full_default_2
#   invert_1 => bitwise_not_1
#   mask_1 => iota_2, iota_3, le_1, logical_and_1, sub_4, unsqueeze_18, unsqueeze_19
#   new_states => bmm_3, clone_5, permute_18, permute_20, unsqueeze_20, unsqueeze_21, unsqueeze_22, view_15, view_16
#   ones_1 => full_default_3
#   states => permute_16, view_13, view_14
#   states_1 => cat
#   x_segsum_2 => sub_3
#   x_segsum_3 => full_default_4, where_1
# Graph fragment:
#   %exp_2 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0" = PlaceHolder[target=exp_2]
#   %clone_5 : Tensor "f32[2048, 8, 3, 64, 16, 1][24576, 3072, 1024, 16, 1, 1]cuda:0" = PlaceHolder[target=clone_5]
#   %iota_2 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_18 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_2, -2), kwargs = {})
#   %iota_3 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_19 : Tensor "i64[3, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_3, -1), kwargs = {})
#   %sub_4 : Tensor "i64[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_18, %unsqueeze_19), kwargs = {})
#   %le_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_4, 0), kwargs = {})
#   %full_default_3 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([3, 3], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_1, %full_default_3), kwargs = {})
#   %bitwise_not_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and_1,), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %unsqueeze_16 : Tensor "f32[2048, 8, 3, 1][24, 3, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_2, 3), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[2048, 8, 1, 3][24, 3, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_2, 2), kwargs = {})
#   %sub_3 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_16, %unsqueeze_17), kwargs = {})
#   %where_1 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_1, %full_default_4, %sub_3), kwargs = {})
#   %exp_2 : Tensor "f32[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where_1,), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[2048, 8, 3, 3, 1][72, 9, 3, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_2, 4), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[2048, 8, 3, 3, 1, 1][72, 9, 3, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, 5), kwargs = {})
#   %view_15 : Tensor "f32[16384, 3, 3][9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%unsqueeze_21, [16384, 3, 3]), kwargs = {})
#   %full_default_2 : Tensor "f32[2048, 1, 8, 64, 16][8192, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2048, 1, 8, 64, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_13 : Tensor "f32[2048, 2, 8, 64, 1, 16][16384, 8192, 1024, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [2048, 2, 8, 64, 1, 16]), kwargs = {})
#   %permute_16 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 16]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %view_14 : Tensor "f32[2048, 2, 8, 64, 16][16384, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_16, [2048, 2, 8, 64, 16]), kwargs = {})
#   %cat : Tensor "f32[2048, 3, 8, 64, 16][24576, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default_2, %view_14], 1), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[2048, 3, 8, 64, 16, 1][24576, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 5), kwargs = {})
#   %permute_18 : Tensor "f32[2048, 1, 8, 64, 16, 3][24576, 1, 1024, 16, 1, 8192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_22, [0, 5, 2, 3, 4, 1]), kwargs = {})
#   %permute_20 : Tensor "f32[2048, 8, 3, 64, 16, 1][24576, 1024, 8192, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_18, [0, 2, 5, 3, 4, 1]), kwargs = {})
#   %clone_5 : Tensor "f32[2048, 8, 3, 64, 16, 1][24576, 3072, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_16 : Tensor "f32[16384, 3, 1024][3072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [16384, 3, 1024]), kwargs = {})
#   %bmm_3 : Tensor "f32[16384, 3, 1024][3072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_15, %view_16), kwargs = {})
#   return %bmm_3
triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13 = async_compile.triton('triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13', '''

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=1,
num_warps=2,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.403243008, 'kernel_flop': 301989888, 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 3
    N = 1024
    K = 3

    stride_aq = 9
    stride_am = 3
    stride_ak = 1

    stride_bq = 3072
    stride_bk = 1024
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m + 3072*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def get_args():
    arg_0 = rand_strided((2048, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16384, 3, 1024), (3072, 1024, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 32, 16384, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.403243008
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/wv/cwvamvy7mdt3irbwkat55dvyivcrqxep2e4sujnwxfst75qrz7ff.py
# Topologically Sorted Source Nodes: [new_states, states_2, Y_off], Original ATen: [aten.view, aten.permute, aten.slice, aten.unsqueeze, aten.clone]
# Source node to ATen node mapping:
#   Y_off => clone_7, permute_23, permute_26, unsqueeze_24
#   new_states => permute_21, view_17, view_18
#   states_2 => slice_3
# Graph fragment:
#   %bmm_3 : Tensor "f32[16384, 3, 1024][3072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %view_17 : Tensor "f32[2048, 8, 3, 1, 64, 16][24576, 3072, 1024, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_3, [2048, 8, 3, 1, 64, 16]), kwargs = {})
#   %permute_21 : Tensor "f32[2048, 3, 8, 64, 16, 1][24576, 1024, 3072, 16, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 4, 5, 3]), kwargs = {})
#   %view_18 : Tensor "f32[2048, 3, 8, 64, 16][24576, 1024, 3072, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [2048, 3, 8, 64, 16]), kwargs = {})
#   %slice_3 : Tensor "f32[2048, 2, 8, 64, 16][24576, 1024, 3072, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_18, 1, 0, -1), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[2048, 2, 8, 64, 16, 1][24576, 1024, 3072, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_3, 5), kwargs = {})
#   %permute_23 : Tensor "f32[2048, 2, 1, 8, 64, 16][24576, 1024, 1, 3072, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_24, [0, 1, 5, 2, 3, 4]), kwargs = {})
#   %permute_26 : Tensor "f32[2048, 2, 8, 16, 64, 1][24576, 1024, 3072, 1, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_23, [0, 1, 3, 5, 4, 2]), kwargs = {})
#   %clone_7 : Tensor "f32[2048, 2, 8, 16, 64, 1][16384, 8192, 1024, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_26,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_7
triton_poi_fused_clone_permute_slice_unsqueeze_view_14 = async_compile.triton('triton_poi_fused_clone_permute_slice_unsqueeze_view_14', '''

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
    size_hints={'y': 524288, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_slice_unsqueeze_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 134217728, 'x': 268435456}, 'kernel_num_gb': 0.268435456, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_slice_unsqueeze_view_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = (yindex % 16)
    y1 = ((yindex // 16) % 8)
    y2 = ((yindex // 128) % 2)
    y3 = yindex // 256
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x4 + 1024*y2 + 3072*y1 + 24576*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4 + 64*y5), tmp0, xmask & ymask)

def get_args():
    arg_0 = rand_strided((16384, 3, 1024), (3072, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 16, 64, 1), (16384, 8192, 1024, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 524288, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_slice_unsqueeze_view_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_permute_slice_unsqueeze_view_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.268435456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/wf/cwfu22aejx5b2rx5uo5gkdryztnxrxdq7urwbawgodrua2p34dj6.py
# Topologically Sorted Source Nodes: [C_blocks, Y_off, new_states, states_2], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.slice, aten.bmm]
# Source node to ATen node mapping:
#   C_blocks => view_3
#   Y_off => bmm_4, clone_6, clone_7, permute_22, permute_23, permute_25, permute_26, unsqueeze_23, unsqueeze_24, view_19, view_20
#   new_states => permute_21, view_17, view_18
#   states_2 => slice_3
# Graph fragment:
#   %clone_6 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0" = PlaceHolder[target=clone_6]
#   %clone_7 : Tensor "f32[2048, 2, 8, 16, 64, 1][16384, 8192, 1024, 64, 1, 1]cuda:0" = PlaceHolder[target=clone_7]
#   %view_3 : Tensor "f32[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 5), kwargs = {})
#   %permute_22 : Tensor "f32[2048, 2, 64, 8, 1, 16][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_23, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %permute_25 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 16, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_22, [0, 1, 3, 2, 5, 4]), kwargs = {})
#   %clone_6 : Tensor "f32[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_25,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[32768, 64, 16][1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [32768, 64, 16]), kwargs = {})
#   %view_17 : Tensor "f32[2048, 8, 3, 1, 64, 16][24576, 3072, 1024, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_3, [2048, 8, 3, 1, 64, 16]), kwargs = {})
#   %permute_21 : Tensor "f32[2048, 3, 8, 64, 16, 1][24576, 1024, 3072, 16, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 4, 5, 3]), kwargs = {})
#   %view_18 : Tensor "f32[2048, 3, 8, 64, 16][24576, 1024, 3072, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [2048, 3, 8, 64, 16]), kwargs = {})
#   %slice_3 : Tensor "f32[2048, 2, 8, 64, 16][24576, 1024, 3072, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_18, 1, 0, -1), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[2048, 2, 8, 64, 16, 1][24576, 1024, 3072, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_3, 5), kwargs = {})
#   %permute_23 : Tensor "f32[2048, 2, 1, 8, 64, 16][24576, 1024, 1, 3072, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_24, [0, 1, 5, 2, 3, 4]), kwargs = {})
#   %permute_26 : Tensor "f32[2048, 2, 8, 16, 64, 1][24576, 1024, 3072, 1, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_23, [0, 1, 3, 5, 4, 2]), kwargs = {})
#   %clone_7 : Tensor "f32[2048, 2, 8, 16, 64, 1][16384, 8192, 1024, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_26,), kwargs = {memory_format: torch.contiguous_format})
#   %view_20 : Tensor "f32[32768, 16, 64][1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [32768, 16, 64]), kwargs = {})
#   %bmm_4 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_19, %view_20), kwargs = {})
#   return %bmm_4
triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15 = async_compile.triton('triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15', '''

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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.805306368, 'kernel_flop': 4294967296, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 64
    N = 64
    K = 16

    stride_aq = 1024
    stride_am = 16
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 64
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 4096*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def get_args():
    arg_0 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 16, 64, 1), (16384, 8192, 1024, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1, 32768, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.805306368
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/xa/cxazxubrdr5tl6vajx7uagqmwm5phquczkv6z5knb6iawsm7nake.py
# Topologically Sorted Source Nodes: [Y_diag, Y_off, state_decay_out, add, Y], Original ATen: [aten.view, aten.permute, aten.exp, aten.unsqueeze, aten.mul, aten.add, aten.clone]
# Source node to ATen node mapping:
#   Y => clone_8
#   Y_diag => permute_10, view_10, view_9
#   Y_off => mul_2, permute_24, permute_27, unsqueeze_25, unsqueeze_26, view_21, view_22
#   add => add
#   state_decay_out => exp_3
# Graph fragment:
#   %bmm_1 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_4 : Tensor "f32[32768, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %cumsum : Tensor "f32[2048, 8, 2, 64][1024, 64, 512, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %view_9 : Tensor "f32[2048, 2, 8, 64, 1, 64, 1][65536, 32768, 4096, 64, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [2048, 2, 8, 64, 1, 64, 1]), kwargs = {})
#   %permute_10 : Tensor "f32[2048, 2, 64, 8, 64, 1, 1][65536, 32768, 64, 4096, 1, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 1, 3, 2, 5, 6, 4]), kwargs = {})
#   %view_10 : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_10, [2048, 2, 64, 8, 64]), kwargs = {})
#   %view_21 : Tensor "f32[2048, 2, 8, 64, 1, 64][65536, 32768, 4096, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [2048, 2, 8, 64, 1, 64]), kwargs = {})
#   %permute_27 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 64, 4096, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 1, 3, 2, 5, 4]), kwargs = {})
#   %exp_3 : Tensor "f32[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%cumsum,), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_3, 4), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[2048, 8, 2, 64, 1, 1][1024, 128, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_25, 5), kwargs = {})
#   %permute_24 : Tensor "f32[2048, 2, 64, 8, 1, 1][1024, 64, 1, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_26, [0, 2, 3, 1, 4, 5]), kwargs = {})
#   %mul_2 : Tensor "f32[2048, 2, 64, 8, 64, 1][65536, 32768, 64, 4096, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_27, %permute_24), kwargs = {})
#   %view_22 : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_2, [2048, 2, 64, 8, 64]), kwargs = {})
#   %add : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %view_22), kwargs = {})
#   %clone_8 : Tensor "f32[2048, 2, 64, 8, 64][65536, 32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_8
triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16 = async_compile.triton('triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16', '''

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2147483648}, 'kernel_num_gb': 1.619001344, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = tl.max_contiguous(tl.multiple_of(xindex, XBLOCK), XBLOCK)
    x5 = xindex // 64
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp2 = tl.load(in_ptr2 + (x5), None, eviction_policy='evict_last')
    tmp3 = tl.exp(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tl.store(out_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), tmp5, None)

def get_args():
    arg_0 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 2, 64, 8, 64), (65536, 32768, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 134217728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.619001344
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2048, 128, 8), (1024, 8, 1))
        assert_size_stride(arg1_1, (2048, 128, 8, 16), (16384, 128, 16, 1))
        assert_size_stride(arg2_1, (2048, 128, 8, 16), (16384, 128, 16, 1))
        assert_size_stride(arg3_1, (2048, 128, 8, 64), (65536, 512, 64, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf8 = empty_strided_cuda((2048, 8, 2, 64), (1024, 64, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [A_blocks, A_blocks_1, A_cumsum], Original ATen: [aten.view, aten.permute, aten.cumsum]
            stream0 = get_raw_stream(0)
            triton_per_fused_cumsum_permute_view_0.run(arg0_1, buf8, 32768, 64, stream=stream0)
            buf9 = empty_strided_cuda((2048, 8, 3), (24, 3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_4, pad, x_cumsum_1], Original ATen: [aten.select, aten.constant_pad_nd, aten.cumsum]
            stream0 = get_raw_stream(0)
            triton_per_fused_constant_pad_nd_cumsum_select_1.run(buf8, buf9, 16384, 3, stream=stream0)
            buf13 = empty_strided_cuda((2048, 8, 3, 3), (72, 9, 3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp]
            stream0 = get_raw_stream(0)
            triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2.run(buf9, buf13, 147456, stream=stream0)
            del buf9
            buf3 = empty_strided_cuda((2048, 8, 2, 64), (1024, 128, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [A_blocks, A_blocks_1, x_cumsum], Original ATen: [aten.view, aten.permute, aten.cumsum]
            stream0 = get_raw_stream(0)
            triton_per_fused_cumsum_permute_view_3.run(arg0_1, buf3, 16384, 2, 64, stream=stream0)
            del arg0_1
            buf0 = empty_strided_cuda((2048, 2, 8, 64, 16, 1, 1), (16384, 8192, 1024, 16, 1, 1, 1), torch.float32)
            buf16 = empty_strided_cuda((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [C_blocks, Y_diag, Y_off], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_unsqueeze_view_4.run(arg2_1, buf0, buf16, 33554432, stream=stream0)
            del arg2_1
            buf1 = empty_strided_cuda((2048, 2, 8, 16, 1, 64, 1), (16384, 8192, 1024, 64, 64, 1, 1), torch.float32)
            buf11 = empty_strided_cuda((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [B_blocks, Y_diag, states, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten.slice, aten.sub, aten.exp, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5.run(arg1_1, buf8, buf1, buf11, 2097152, 16, stream=stream0)
            del arg1_1
            buf2 = empty_strided_cuda((32768, 64, 64), (4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [C_blocks, Y_diag, B_blocks], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6.run(buf0, buf1, buf2, 1, 32768, 1, stream=stream0)
            del buf0
            buf4 = empty_strided_cuda((2048, 2, 64, 8, 1, 1, 64), (65536, 32768, 512, 64, 64, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [Y_diag, mask, ones, invert, x_segsum_1, getitem, getitem_1, x_segsum, L], Original ATen: [aten.view, aten.permute, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7.run(buf2, buf3, buf4, 2097152, 64, stream=stream0)
            del buf3
            buf5 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [Y_diag, mask, ones, invert, x_segsum_1, getitem, getitem_1, x_segsum, L, X_blocks], Original ATen: [aten.view, aten.permute, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.mul, aten.clone, aten._unsafe_view, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8.run(buf4, buf5, 134217728, stream=stream0)
            buf6 = reinterpret_tensor(buf4, (2048, 2, 8, 64, 64, 1, 1), (65536, 32768, 4096, 64, 1, 1, 1), 0); del buf4  # reuse
            buf10 = empty_strided_cuda((2048, 2, 8, 64, 64, 1), (65536, 32768, 4096, 64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [X_blocks, Y_diag, states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_unsqueeze_view_9.run(arg3_1, buf6, buf10, 2097152, 64, stream=stream0)
            del arg3_1
            buf12 = reinterpret_tensor(buf1, (32768, 64, 16), (1024, 16, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [B_blocks, X_blocks, states, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.slice, aten.sub, aten.exp, aten.mul, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10.run(buf10, buf11, buf12, 1, 32768, 1, stream=stream0)
            del buf11
            buf7 = reinterpret_tensor(buf10, (32768, 64, 64), (4096, 64, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [Y_diag, mask, ones, invert, x_segsum_1, getitem, getitem_1, x_segsum, L, X_blocks], Original ATen: [aten.view, aten.permute, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.mul, aten.clone, aten._unsafe_view, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11.run(buf5, buf6, buf7, 1, 32768, 1, stream=stream0)
            del buf5
            buf14 = empty_strided_cuda((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [initial_states, states, states_1, new_states], Original ATen: [aten.zeros_like, aten.view, aten.permute, aten.cat, aten.unsqueeze, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12.run(buf12, buf14, 50331648, stream=stream0)
            buf15 = empty_strided_cuda((16384, 3, 1024), (3072, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk, new_states, initial_states, states, states_1], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.view, aten.zeros_like, aten.permute, aten.cat, aten.clone, aten._unsafe_view, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13.run(buf13, buf14, buf15, 32, 16384, 1, stream=stream0)
            del buf13
            del buf14
            buf17 = reinterpret_tensor(buf12, (2048, 2, 8, 16, 64, 1), (16384, 8192, 1024, 64, 1, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [new_states, states_2, Y_off], Original ATen: [aten.view, aten.permute, aten.slice, aten.unsqueeze, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_slice_unsqueeze_view_14.run(buf15, buf17, 524288, 64, stream=stream0)
            del buf15
            buf18 = reinterpret_tensor(buf6, (32768, 64, 64), (4096, 64, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [C_blocks, Y_off, new_states, states_2], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.slice, aten.bmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15.run(buf16, buf17, buf18, 1, 32768, 1, stream=stream0)
            del buf16
            del buf17
            buf19 = empty_strided_cuda((2048, 2, 64, 8, 64), (65536, 32768, 512, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [Y_diag, Y_off, state_decay_out, add, Y], Original ATen: [aten.view, aten.permute, aten.exp, aten.unsqueeze, aten.mul, aten.add, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16.run(buf7, buf18, buf8, buf19, 134217728, stream=stream0)
            del buf18
            del buf7
            del buf8
        return (reinterpret_tensor(buf19, (2048, 128, 8, 64), (65536, 512, 64, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 128, 8), (1024, 8, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, 128, 8, 64), (65536, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def segsum(self, x):
        """Naive segment sum calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs
        L = torch.exp(self.segsum(A_blocks))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        
        return Y

# Test parameters
batch_size = 2048
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.rand(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]
'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0
''',
    '''@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0
''',
    '''@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 3)
    x1 = ((xindex // 3) % 3)
    x3 = xindex // 3
    x2 = xindex // 9
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (x0 + 3*x2), None, eviction_policy='evict_last')
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp2 & tmp3
    tmp5 = tmp4 == 0
    tmp8 = tmp6 - tmp7
    tmp9 = tl.full([1], float("-inf"), tl.float32)
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x4), tmp11, None)

''',
    '''@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0
''',
    '''@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 64)
    x2 = ((xindex // 1024) % 8)
    x3 = xindex // 8192
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 8192*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
    tl.store(out_ptr1 + (x4), tmp0, None)

''',
    '''@triton.jit
def triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 8)
    y2 = yindex // 512
    y4 = yindex // 64
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y1 + 128*y0 + 8192*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (63 + 64*y4), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y5), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (y0 + 64*x3 + 1024*y4), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 16*y5), tmp5, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_permute_unsqueeze_view_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 64
    N = 64
    K = 16

    stride_aq = 1024
    stride_am = 16
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 64
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 4096*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_7(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y5 = yindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 8)
    y2 = ((yindex // 512) % 2)
    y3 = yindex // 1024
    y7 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x4 + 64*y5), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (y0 + 64*y2 + 128*y1 + 1024*y3), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x4 + 64*y2 + 128*y1 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = x4 + ((-1)*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1, 1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5 == 0
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp11 = tl.where(tmp6, tmp10, tmp9)
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp0 * tmp12
    tl.store(out_ptr0 + (x4 + 64*y1 + 512*y0 + 32768*y7), tmp13, xmask & ymask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((x2 % 8)) + 512*x1 + 32768*(x2 // 8)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)

''',
    '''@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_9(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 512
    y4 = (yindex % 512)
    y0 = (yindex % 64)
    y5 = yindex // 64
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 512*x3 + 32768*y2), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x3 + 4096*y5), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 64*y6), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_exp_mul_permute_slice_sub_unsqueeze_view_10(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
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

    M = 64
    N = 16
    K = 64

    stride_aq = 4096
    stride_am = 64
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 16
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16*idx_m + 1024*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_bitwise_not_bmm_clone_exp_masked_fill_mul_ones_permute_sub_tril_unsqueeze_view_11(arg_A, arg_B, out_ptr0):
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

    M = 64
    N = 64
    K = 64

    stride_aq = 4096
    stride_am = 64
    stride_ak = 1

    stride_bq = 4096
    stride_bk = 64
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 4096*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 1024) % 3)
    x0 = (xindex % 1024)
    x2 = ((xindex // 3072) % 8)
    x3 = xindex // 24576
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), tmp8, other=0.0)
    tmp12 = tl.where(tmp4, tmp7, tmp11)
    tl.store(out_ptr0 + (x4), tmp12, None)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 3
    N = 1024
    K = 3

    stride_aq = 9
    stride_am = 3
    stride_ak = 1

    stride_bq = 3072
    stride_bk = 1024
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m + 3072*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_clone_permute_slice_unsqueeze_view_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = (yindex % 16)
    y1 = ((yindex // 16) % 8)
    y2 = ((yindex // 128) % 2)
    y3 = yindex // 256
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x4 + 1024*y2 + 3072*y1 + 24576*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4 + 64*y5), tmp0, xmask & ymask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 64
    N = 64
    K = 16

    stride_aq = 1024
    stride_am = 16
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 64
    stride_bn = 1

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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 4096*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    x5 = xindex // 64
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp2 = tl.load(in_ptr2 + (x5), None, eviction_policy='evict_last')
    tmp3 = tl.exp(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tl.store(out_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), tmp5, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((2048, 128, 8), (1024, 8, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, 128, 8, 64), (65536, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]

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
