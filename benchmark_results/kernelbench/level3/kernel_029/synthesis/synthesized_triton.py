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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/nk/cnkiyngu2bhcipeierr5sxfwsmwnmndczmxq7hgmtl6mts6cllwr.py
# Topologically Sorted Source Nodes: [conv2d, flatten, x, x_1, x_3], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   conv2d => convolution
#   flatten => view
#   x => permute
#   x_1 => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
#   x_3 => var_mean_1
# Graph fragment:
#   %buf0 : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg2_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=buf2]
#   %arg3_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_1 : Tensor "f32[10, 3136, 96][301056, 1, 3136]cuda:0" = PlaceHolder[target=add_1]
#   %convolution : Tensor "f32[10, 96, 56, 56][301056, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[10, 96, 3136][301056, 3136, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [10, 96, 3136]), kwargs = {})
#   %permute : Tensor "f32[10, 3136, 96][301056, 1, 3136]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_1), kwargs = {})
#   %add : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_1 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf2,%add_1,%getitem_3,%buf6
triton_red_fused_convolution_native_layer_norm_transpose_view_0 = async_compile.triton('triton_red_fused_convolution_native_layer_norm_transpose_view_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_layer_norm_transpose_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 36628480, 'r0_': 1152}, 'kernel_num_gb': 0.024336512, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp25_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12 - tmp4
        tmp14 = tl.full([1, 1], 96.0, tl.float32)
        tmp15 = (tmp8 / tmp14)
        tmp16 = tl.full([1, 1], 1e-05, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight, roffset == 0
        )
        tmp25_mean = tl.where(r0_mask & xmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(r0_mask & xmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(r0_mask & xmask, tmp25_weight_next, tmp25_weight)
        tl.store(in_out_ptr0 + (x0 + 3136*r0_2 + 301056*x1), tmp23, r0_mask & xmask)
    tmp26, tmp27, tmp28 = triton_helpers.welford(tmp25_mean, tmp25_m2, tmp25_weight, 1)
    tmp25 = tmp26[:, None]
    tmp29 = tmp27[:, None]
    tmp30 = tmp28[:, None]
    tl.store(out_ptr2 + (x3), tmp25, xmask)
    tl.store(out_ptr3 + (x3), tmp29, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 1, 3136), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_transpose_view_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_convolution_native_layer_norm_transpose_view_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024336512
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ut/cutqacpo44fnae7zyg4rte4w3clu64edfmt722wgwcfe2i4jtc6q.py
# Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
# Source node to ATen node mapping:
#   contiguous => clone_2
#   permute => permute_1
#   spatial_mlp_windows => convolution_1
#   windows => view_3
#   x_3 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_4 => view_1
#   x_5 => view_2
#   x_windows => view_4
#   x_windows_heads => view_5
#   x_windows_heads_1 => permute_2
#   x_windows_heads_2 => clone_3, view_6
# Graph fragment:
#   %add_1 : Tensor "f32[10, 3136, 96][301056, 1, 3136]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_3 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf6 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=buf6]
#   %arg5_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_3), kwargs = {})
#   %add_2 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
#   %add_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg6_1), kwargs = {})
#   %view_1 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [10, 56, 56, 96]), kwargs = {})
#   %view_2 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [10, 8, 7, 8, 7, 96]), kwargs = {})
#   %permute_1 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 672, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_2 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [-1, 7, 7, 96]), kwargs = {})
#   %view_4 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [-1, 49, 96]), kwargs = {})
#   %view_5 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [-1, 49, 3, 32]), kwargs = {})
#   %permute_2 : Tensor "f32[640, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [640, 147, 32]), kwargs = {})
#   %convolution_1 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %arg7_1, %arg8_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   return %buf8
triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1 = async_compile.triton('triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1', '''
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
    size_hints={'y': 131072, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 12293120, 'x': 24085248}, 'kernel_num_gb': 0.024336128, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 94080
    xnumel = 32
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 147)
    y1 = yindex // 147
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(((y1 // 8) % 8)) + 3136*x2 + 100352*(y0 // 49) + 301056*(y1 // 64) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 96.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + 32*y3), tmp13, xmask & ymask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 1, 3136), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((640, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 94080, 32,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024336128
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/hm/chmyzxai6b72sst55363zju6vhcovrxpehhidmbgr2yqzeuft64k.py
# Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows, view_5, spatial_mlp_windows_1, spatial_mlp_windows_2, spatial_mlp_windows_3, x_6, permute_1, contiguous_1, x_7, x_8, x_9, layer_norm_2], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   contiguous => clone_2
#   contiguous_1 => clone_5
#   layer_norm_2 => add_5, add_6, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
#   permute => permute_1
#   permute_1 => permute_4
#   spatial_mlp_windows => convolution_1
#   spatial_mlp_windows_1 => permute_3
#   spatial_mlp_windows_2 => clone_4, view_8
#   spatial_mlp_windows_3 => view_9
#   view_5 => view_7
#   windows => view_3
#   x_3 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_4 => view_1
#   x_5 => view_2
#   x_6 => view_10
#   x_7 => view_11
#   x_8 => view_12
#   x_9 => add_4
#   x_windows => view_4
#   x_windows_heads => view_5
#   x_windows_heads_1 => permute_2
#   x_windows_heads_2 => clone_3, view_6
# Graph fragment:
#   %add_1 : Tensor "f32[10, 3136, 96][301056, 1, 3136]cuda:0" = PlaceHolder[target=add_1]
#   %buf9 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0" = PlaceHolder[target=buf9]
#   %arg8_1 : Tensor "f32[147][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %getitem_5 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf11 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=buf11]
#   %arg9_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_3), kwargs = {})
#   %add_2 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
#   %add_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg6_1), kwargs = {})
#   %view_1 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [10, 56, 56, 96]), kwargs = {})
#   %view_2 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [10, 8, 7, 8, 7, 96]), kwargs = {})
#   %permute_1 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 672, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_2 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [-1, 7, 7, 96]), kwargs = {})
#   %view_4 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [-1, 49, 96]), kwargs = {})
#   %view_5 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [-1, 49, 3, 32]), kwargs = {})
#   %permute_2 : Tensor "f32[640, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [640, 147, 32]), kwargs = {})
#   %convolution_1 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %arg7_1, %arg8_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_7 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [-1, 3, 49, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[640, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [640, 49, 96]), kwargs = {})
#   %view_9 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_10 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_9, [10, 8, 8, 7, 7, -1]), kwargs = {})
#   %permute_4 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_5 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_11 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [10, 56, 56, -1]), kwargs = {})
#   %view_12 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [10, 3136, 96]), kwargs = {})
#   %add_4 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_12), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_5), kwargs = {})
#   %add_5 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_4 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_5 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %add_6 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg10_1), kwargs = {})
#   return %getitem_5,%buf11,%add_6
triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12042828, 'r0_': 36127488}, 'kernel_num_gb': 0.036128076, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp12 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp16 - tmp6
        tmp18 = tl.full([1, 1], 96.0, tl.float32)
        tmp19 = (tmp10 / tmp18)
        tmp20 = tl.full([1, 1], 1e-05, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp27, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 1, 3136), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((640, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.036128076
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/iu/ciu3tudttkddfufdeetlp6wknimdefnfdt4ar277amnp2dvdog2v.py
# Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows, view_5, spatial_mlp_windows_1, spatial_mlp_windows_2, spatial_mlp_windows_3, x_6, permute_1, contiguous_1, x_7, x_8, x_9, layer_norm_2, x_10], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   contiguous => clone_2
#   contiguous_1 => clone_5
#   layer_norm_2 => add_5, add_6, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
#   permute => permute_1
#   permute_1 => permute_4
#   spatial_mlp_windows => convolution_1
#   spatial_mlp_windows_1 => permute_3
#   spatial_mlp_windows_2 => clone_4, view_8
#   spatial_mlp_windows_3 => view_9
#   view_5 => view_7
#   windows => view_3
#   x_10 => mm_default_23, permute_5, view_13
#   x_3 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_4 => view_1
#   x_5 => view_2
#   x_6 => view_10
#   x_7 => view_11
#   x_8 => view_12
#   x_9 => add_4
#   x_windows => view_4
#   x_windows_heads => view_5
#   x_windows_heads_1 => permute_2
#   x_windows_heads_2 => clone_3, view_6
# Graph fragment:
#   %add_6 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_6]
#   %arg11_1 : Tensor "f32[384, 96][96, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_3), kwargs = {})
#   %add_2 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
#   %add_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg6_1), kwargs = {})
#   %view_1 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [10, 56, 56, 96]), kwargs = {})
#   %view_2 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [10, 8, 7, 8, 7, 96]), kwargs = {})
#   %permute_1 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 672, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_2 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [-1, 7, 7, 96]), kwargs = {})
#   %view_4 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [-1, 49, 96]), kwargs = {})
#   %view_5 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [-1, 49, 3, 32]), kwargs = {})
#   %permute_2 : Tensor "f32[640, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [640, 147, 32]), kwargs = {})
#   %convolution_1 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %arg7_1, %arg8_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_7 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [-1, 3, 49, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[640, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [640, 49, 96]), kwargs = {})
#   %view_9 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_10 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_9, [10, 8, 8, 7, 7, -1]), kwargs = {})
#   %permute_4 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_5 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_11 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [10, 56, 56, -1]), kwargs = {})
#   %view_12 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [10, 3136, 96]), kwargs = {})
#   %add_4 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_12), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_5), kwargs = {})
#   %add_5 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_4 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_5 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %add_6 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg10_1), kwargs = {})
#   %view_13 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [31360, 96]), kwargs = {})
#   %permute_5 : Tensor "f32[96, 384][1, 96]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg11_1, [1, 0]), kwargs = {})
#   %mm_default_23 : Tensor "f32[31360, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_13, %permute_5), kwargs = {})
#   return %mm_default_23
triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3 = async_compile.triton('triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.060358656, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3(arg_A, arg_B, out_ptr0):
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
    N = 384
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
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1470, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.060358656
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/tu/ctupoit5mzfba5bwv4i2lgqojeh3khocv727vpjqoejbaieqwwtz.py
# Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   x_10 => add_tensor_23, view_14
#   x_11 => add_7, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %arg12_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %mm_default_23 : Tensor "f32[31360, 384][384, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %add_tensor_23 : Tensor "f32[31360, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default_23), kwargs = {})
#   %view_14 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [10, 3136, 384]), kwargs = {})
#   %mul_6 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_7 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_7), kwargs = {})
#   return %mul_8
triton_poi_fused_addmm_gelu_view_4 = async_compile.triton('triton_poi_fused_addmm_gelu_view_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 144508416}, 'kernel_num_gb': 0.096339456, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12042240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)


def get_args():
    arg_0 = rand_strided((10, 3136, 384), (1204224, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 12042240,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_gelu_view_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_gelu_view_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.096339456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pb/cpb2qmklkz3gs6enf7vtjjfv6slvclzwm7iel65iw7tuutfj7by7.py
# Topologically Sorted Source Nodes: [x_10, x_11, x_13], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   x_10 => add_tensor_23, view_14
#   x_11 => add_7, erf, mul_6, mul_7, mul_8
#   x_13 => mm_default_22, permute_6, view_15
# Graph fragment:
#   %mul_8 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0" = PlaceHolder[target=mul_8]
#   %arg13_1 : Tensor "f32[96, 384][384, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %add_tensor_23 : Tensor "f32[31360, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg12_1, %mm_default_23), kwargs = {})
#   %view_14 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [10, 3136, 384]), kwargs = {})
#   %mul_6 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_7 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : Tensor "f32[10, 3136, 384][1204224, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_7), kwargs = {})
#   %view_15 : Tensor "f32[31360, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_8, [31360, 384]), kwargs = {})
#   %permute_6 : Tensor "f32[384, 96][1, 384]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg13_1, [1, 0]), kwargs = {})
#   %mm_default_22 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_15, %permute_6), kwargs = {})
#   return %mm_default_22
triton_tem_fused_addmm_gelu_t_view_5 = async_compile.triton('triton_tem_fused_addmm_gelu_t_view_5', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_t_view_5', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.060358656, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_5(arg_A, arg_B, out_ptr0):
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
    N = 96
    K = 384
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 384
    stride_ak = 1
    stride_bk = 1
    stride_bn = 384

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
        xindex = idx_n + 384*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 96*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 3136, 384), (1204224, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 245, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_gelu_t_view_5.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_gelu_t_view_5.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.060358656
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ys/cyss67fgrpi4zc5me7e4mirf5zirsvw6vtjcul4ai3yboy64tr3x.py
# Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows, view_5, spatial_mlp_windows_1, spatial_mlp_windows_2, spatial_mlp_windows_3, x_6, permute_1, contiguous_1, x_7, x_8, x_9, x_13, x_15, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous => clone_2
#   contiguous_1 => clone_5
#   permute => permute_1
#   permute_1 => permute_4
#   spatial_mlp_windows => convolution_1
#   spatial_mlp_windows_1 => permute_3
#   spatial_mlp_windows_2 => clone_4, view_8
#   spatial_mlp_windows_3 => view_9
#   view_5 => view_7
#   windows => view_3
#   x_13 => add_tensor_22, view_16
#   x_15 => add_8
#   x_16 => var_mean_3
#   x_3 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_4 => view_1
#   x_5 => view_2
#   x_6 => view_10
#   x_7 => view_11
#   x_8 => view_12
#   x_9 => add_4
#   x_windows => view_4
#   x_windows_heads => view_5
#   x_windows_heads_1 => permute_2
#   x_windows_heads_2 => clone_3, view_6
# Graph fragment:
#   %add_1 : Tensor "f32[10, 3136, 96][301056, 1, 3136]cuda:0" = PlaceHolder[target=add_1]
#   %buf9 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0" = PlaceHolder[target=buf9]
#   %arg8_1 : Tensor "f32[147][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg14_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %mm_default_22 : Tensor "f32[31360, 96][96, 1]cuda:0" = PlaceHolder[target=mm_default_22]
#   %add_8 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_8]
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_3), kwargs = {})
#   %add_2 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
#   %add_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg6_1), kwargs = {})
#   %view_1 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [10, 56, 56, 96]), kwargs = {})
#   %view_2 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [10, 8, 7, 8, 7, 96]), kwargs = {})
#   %permute_1 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 672, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_2 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [-1, 7, 7, 96]), kwargs = {})
#   %view_4 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [-1, 49, 96]), kwargs = {})
#   %view_5 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [-1, 49, 3, 32]), kwargs = {})
#   %permute_2 : Tensor "f32[640, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [640, 147, 32]), kwargs = {})
#   %convolution_1 : Tensor "f32[640, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %arg7_1, %arg8_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_7 : Tensor "f32[640, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [-1, 3, 49, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[640, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[640, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[640, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [640, 49, 96]), kwargs = {})
#   %view_9 : Tensor "f32[640, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_10 : Tensor "f32[10, 8, 8, 7, 7, 96][301056, 37632, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_9, [10, 8, 8, 7, 7, -1]), kwargs = {})
#   %permute_4 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_5 : Tensor "f32[10, 8, 7, 8, 7, 96][301056, 37632, 5376, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_11 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [10, 56, 56, -1]), kwargs = {})
#   %view_12 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [10, 3136, 96]), kwargs = {})
#   %add_4 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_12), kwargs = {})
#   %add_tensor_22 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg14_1, %mm_default_22), kwargs = {})
#   %view_16 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [10, 3136, 96]), kwargs = {})
#   %add_8 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_16), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   return %add_8,%getitem_7,%buf19
triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6 = async_compile.triton('triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12544588, 'r0_': 48169344}, 'kernel_num_gb': 0.048420812, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    x3 = xindex
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_out_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r0_2 + 96*x3), tmp8, r0_mask & xmask)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3136, 96), (301056, 1, 3136), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((640, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 31360, 96,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.048420812
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/td/ctdx7xoeiwxuw2vfszojbezclei7bzj2bukhm6fcfzdo6ndvky4o.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   permute_2 => permute_7
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   windows_1 => view_19
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_8 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_8]
#   %getitem_7 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf19 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=buf19]
#   %arg15_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   return %buf21
triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7 = async_compile.triton('triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 42524928}, 'kernel_num_gb': 0.027534848, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3810240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 147)
    x2 = xindex // 4704
    x0 = (xindex % 32)
    x3 = xindex
    tmp0 = (-4) + 7*(((x2 // 9) % 9)) + (((x1 % 49)) // 7)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*((x2 % 9)) + ((((x1 % 49)) % 7))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-21888) + x0 + 32*(x1 // 49) + 96*((((x1 % 49)) % 7)) + 672*((x2 % 9)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 9) % 9)) + 301056*(x2 // 81)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 96.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((810, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3810240,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.027534848
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/br/cbrk6mng26eogrzsjat4jcvbdcn6ht2kkwozfyylsuve76mo6anc.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, layer_norm_4], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   contiguous_3 => clone_11
#   getitem => slice_1, slice_2
#   layer_norm_4 => add_12, add_13, mul_11, mul_12, rsqrt_4, sub_4, var_mean_4
#   permute_2 => permute_7
#   permute_3 => permute_10
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   spatial_mlp_windows_5 => permute_9
#   spatial_mlp_windows_6 => clone_10, view_24
#   spatial_mlp_windows_7 => view_25
#   view_14 => view_23
#   windows_1 => view_19
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_19 => view_26
#   x_20 => view_27
#   x_21 => clone_12
#   x_22 => view_28
#   x_23 => add_11
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_8 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_8]
#   %buf22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0" = PlaceHolder[target=buf22]
#   %arg18_1 : Tensor "f32[147][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %getitem_9 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf24 : Tensor "f32[10, 3136, 1][3136, 1, 31360]cuda:0" = PlaceHolder[target=buf24]
#   %arg19_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_23 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [-1, 3, 49, 32]), kwargs = {})
#   %permute_9 : Tensor "f32[810, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_10 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [810, 49, 96]), kwargs = {})
#   %view_25 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [-1, 7, 7, 96]), kwargs = {})
#   %view_26 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [10, 9, 9, 7, 7, -1]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_11 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_27 : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [10, 63, 63, -1]), kwargs = {})
#   %slice_1 : Tensor "f32[10, 56, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_27, 1, 4, -3), kwargs = {})
#   %slice_2 : Tensor "f32[10, 56, 56, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 2, 4, -3), kwargs = {})
#   %clone_12 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [10, 3136, 96]), kwargs = {})
#   %add_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_9), kwargs = {})
#   %add_12 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_12 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %arg19_1), kwargs = {})
#   %add_13 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %arg20_1), kwargs = {})
#   return %getitem_9,%buf24,%add_13
triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8', '''
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
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 588, 'r0_': 48169728}, 'kernel_num_gb': 0.036128076, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp16 - tmp6
        tmp18 = tl.full([1, 1], 96.0, tl.float32)
        tmp19 = (tmp10 / tmp18)
        tmp20 = tl.full([1, 1], 1e-05, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp27, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((810, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 31360, 96,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.036128076
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bs/cbsgnh4hz6ldkg4c4bneruuzqh3qcwpcimdhw2swvk2jqvtt7vnh.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   contiguous_3 => clone_11
#   getitem => slice_1, slice_2
#   permute_2 => permute_7
#   permute_3 => permute_10
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   spatial_mlp_windows_5 => permute_9
#   spatial_mlp_windows_6 => clone_10, view_24
#   spatial_mlp_windows_7 => view_25
#   view_14 => view_23
#   windows_1 => view_19
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_19 => view_26
#   x_20 => view_27
#   x_21 => clone_12
#   x_22 => view_28
#   x_23 => add_11
#   x_27 => add_tensor_20, view_32
#   x_29 => add_15
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_8 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_8]
#   %buf22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0" = PlaceHolder[target=buf22]
#   %arg18_1 : Tensor "f32[147][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg24_1 : Tensor "f32[96][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %mm_default_20 : Tensor "f32[31360, 96][96, 1]cuda:0" = PlaceHolder[target=mm_default_20]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_23 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [-1, 3, 49, 32]), kwargs = {})
#   %permute_9 : Tensor "f32[810, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_10 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [810, 49, 96]), kwargs = {})
#   %view_25 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [-1, 7, 7, 96]), kwargs = {})
#   %view_26 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [10, 9, 9, 7, 7, -1]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_11 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_27 : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [10, 63, 63, -1]), kwargs = {})
#   %slice_1 : Tensor "f32[10, 56, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_27, 1, 4, -3), kwargs = {})
#   %slice_2 : Tensor "f32[10, 56, 56, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 2, 4, -3), kwargs = {})
#   %clone_12 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [10, 3136, 96]), kwargs = {})
#   %add_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %add_tensor_20 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, %mm_default_20), kwargs = {})
#   %view_32 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [10, 3136, 96]), kwargs = {})
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   return %add_15
triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 60211584}, 'kernel_num_gb': 0.048169932, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 3136)
    x2 = xindex // 301056
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 56))) % 7)) + 224*(((4 + (x1 // 56)) % 7)) + 1568*(x0 // 32) + 4704*((4 + ((x1 % 56))) // 7) + 42336*((4 + (x1 // 56)) // 7) + 381024*x2 + ((x0 % 32))), None)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 56)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 56))) % 7))), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((810, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((31360, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 3010560,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.048169932
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bg/cbg4i27lcf2ad76udqrz37cg6tkpsyjdx7bohlvasgztn2m7zrsb.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   contiguous_3 => clone_11
#   getitem => slice_1, slice_2
#   permute_2 => permute_7
#   permute_3 => permute_10
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   spatial_mlp_windows_5 => permute_9
#   spatial_mlp_windows_6 => clone_10, view_24
#   spatial_mlp_windows_7 => view_25
#   view_14 => view_23
#   windows_1 => view_19
#   x0 => slice_3, slice_4
#   x1 => slice_5, slice_6
#   x2 => slice_7, slice_8
#   x3 => slice_10, slice_9
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_19 => view_26
#   x_20 => view_27
#   x_21 => clone_12
#   x_22 => view_28
#   x_23 => add_11
#   x_27 => add_tensor_20, view_32
#   x_29 => add_15
#   x_30 => view_33
#   x_31 => cat
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_15]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_23 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [-1, 3, 49, 32]), kwargs = {})
#   %permute_9 : Tensor "f32[810, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_10 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [810, 49, 96]), kwargs = {})
#   %view_25 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [-1, 7, 7, 96]), kwargs = {})
#   %view_26 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [10, 9, 9, 7, 7, -1]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_11 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_27 : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [10, 63, 63, -1]), kwargs = {})
#   %slice_1 : Tensor "f32[10, 56, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_27, 1, 4, -3), kwargs = {})
#   %slice_2 : Tensor "f32[10, 56, 56, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 2, 4, -3), kwargs = {})
#   %clone_12 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [10, 3136, 96]), kwargs = {})
#   %add_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %add_tensor_20 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, %mm_default_20), kwargs = {})
#   %view_32 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [10, 3136, 96]), kwargs = {})
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   %view_33 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [10, 56, 56, 96]), kwargs = {})
#   %slice_3 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_4 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_3, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_5 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_6 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_5, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_7 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_8 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_7, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_9 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_10 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_9, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat : Tensor "f32[10, 28, 28, 384][301056, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6, %slice_8, %slice_10], -1), kwargs = {})
#   return %buf31
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9031680}, 'kernel_num_gb': 0.00602112, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 28, 28, 96), (301056, 10752, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00602112
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/va/cva67moea2whldil3ejc4udkec24462sl2s3hwmnispingt4kf45.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   contiguous_3 => clone_11
#   getitem => slice_1, slice_2
#   permute_2 => permute_7
#   permute_3 => permute_10
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   spatial_mlp_windows_5 => permute_9
#   spatial_mlp_windows_6 => clone_10, view_24
#   spatial_mlp_windows_7 => view_25
#   view_14 => view_23
#   windows_1 => view_19
#   x0 => slice_3, slice_4
#   x1 => slice_5, slice_6
#   x2 => slice_7, slice_8
#   x3 => slice_10, slice_9
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_19 => view_26
#   x_20 => view_27
#   x_21 => clone_12
#   x_22 => view_28
#   x_23 => add_11
#   x_27 => add_tensor_20, view_32
#   x_29 => add_15
#   x_30 => view_33
#   x_31 => cat
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_15]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_23 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [-1, 3, 49, 32]), kwargs = {})
#   %permute_9 : Tensor "f32[810, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_10 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [810, 49, 96]), kwargs = {})
#   %view_25 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [-1, 7, 7, 96]), kwargs = {})
#   %view_26 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [10, 9, 9, 7, 7, -1]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_11 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_27 : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [10, 63, 63, -1]), kwargs = {})
#   %slice_1 : Tensor "f32[10, 56, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_27, 1, 4, -3), kwargs = {})
#   %slice_2 : Tensor "f32[10, 56, 56, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 2, 4, -3), kwargs = {})
#   %clone_12 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [10, 3136, 96]), kwargs = {})
#   %add_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %add_tensor_20 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, %mm_default_20), kwargs = {})
#   %view_32 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [10, 3136, 96]), kwargs = {})
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   %view_33 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [10, 56, 56, 96]), kwargs = {})
#   %slice_3 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_4 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_3, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_5 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_6 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_5, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_7 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_8 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_7, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_9 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_10 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_9, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat : Tensor "f32[10, 28, 28, 384][301056, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6, %slice_8, %slice_10], -1), kwargs = {})
#   return %buf32
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9031680}, 'kernel_num_gb': 0.00602112, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 28, 28, 96), (301056, 10752, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00602112
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/f7/cf7tv363rnx3c3o2ncgtzdzpumv6nktnqgypcgedn34rdqsaij7y.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   contiguous_3 => clone_11
#   getitem => slice_1, slice_2
#   permute_2 => permute_7
#   permute_3 => permute_10
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   spatial_mlp_windows_5 => permute_9
#   spatial_mlp_windows_6 => clone_10, view_24
#   spatial_mlp_windows_7 => view_25
#   view_14 => view_23
#   windows_1 => view_19
#   x0 => slice_3, slice_4
#   x1 => slice_5, slice_6
#   x2 => slice_7, slice_8
#   x3 => slice_10, slice_9
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_19 => view_26
#   x_20 => view_27
#   x_21 => clone_12
#   x_22 => view_28
#   x_23 => add_11
#   x_27 => add_tensor_20, view_32
#   x_29 => add_15
#   x_30 => view_33
#   x_31 => cat
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_15]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_23 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [-1, 3, 49, 32]), kwargs = {})
#   %permute_9 : Tensor "f32[810, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_10 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [810, 49, 96]), kwargs = {})
#   %view_25 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [-1, 7, 7, 96]), kwargs = {})
#   %view_26 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [10, 9, 9, 7, 7, -1]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_11 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_27 : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [10, 63, 63, -1]), kwargs = {})
#   %slice_1 : Tensor "f32[10, 56, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_27, 1, 4, -3), kwargs = {})
#   %slice_2 : Tensor "f32[10, 56, 56, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 2, 4, -3), kwargs = {})
#   %clone_12 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [10, 3136, 96]), kwargs = {})
#   %add_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %add_tensor_20 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, %mm_default_20), kwargs = {})
#   %view_32 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [10, 3136, 96]), kwargs = {})
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   %view_33 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [10, 56, 56, 96]), kwargs = {})
#   %slice_3 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_4 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_3, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_5 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_6 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_5, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_7 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_8 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_7, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_9 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_10 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_9, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat : Tensor "f32[10, 28, 28, 384][301056, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6, %slice_8, %slice_10], -1), kwargs = {})
#   return %buf33
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9031680}, 'kernel_num_gb': 0.00602112, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (96 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 28, 28, 96), (301056, 10752, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00602112
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/6a/c6a2gux3j7ocloatdrly3okmoskbaieofxz5qqfw6ezssqrvyeyb.py
# Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_2 => clone_8
#   contiguous_3 => clone_11
#   getitem => slice_1, slice_2
#   permute_2 => permute_7
#   permute_3 => permute_10
#   shifted_x => constant_pad_nd
#   spatial_mlp_windows_4 => convolution_2
#   spatial_mlp_windows_5 => permute_9
#   spatial_mlp_windows_6 => clone_10, view_24
#   spatial_mlp_windows_7 => view_25
#   view_14 => view_23
#   windows_1 => view_19
#   x0 => slice_3, slice_4
#   x1 => slice_5, slice_6
#   x2 => slice_7, slice_8
#   x3 => slice_10, slice_9
#   x_16 => add_10, add_9, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
#   x_17 => view_17
#   x_18 => view_18
#   x_19 => view_26
#   x_20 => view_27
#   x_21 => clone_12
#   x_22 => view_28
#   x_23 => add_11
#   x_27 => add_tensor_20, view_32
#   x_29 => add_15
#   x_30 => view_33
#   x_31 => cat
#   x_windows_1 => view_20
#   x_windows_heads_3 => view_21
#   x_windows_heads_4 => permute_8
#   x_windows_heads_5 => clone_9, view_22
# Graph fragment:
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0" = PlaceHolder[target=add_15]
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %add_9 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[10, 3136, 1][3136, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_9 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg15_1), kwargs = {})
#   %add_10 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg16_1), kwargs = {})
#   %view_17 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [10, 56, 56, 96]), kwargs = {})
#   %constant_pad_nd : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_17, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_18 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [10, 9, 7, 9, 7, 96]), kwargs = {})
#   %permute_7 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 672, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_8 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [-1, 7, 7, 96]), kwargs = {})
#   %view_20 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [-1, 49, 96]), kwargs = {})
#   %view_21 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [-1, 49, 3, 32]), kwargs = {})
#   %permute_8 : Tensor "f32[810, 3, 49, 32][4704, 32, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_9 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_22 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [810, 147, 32]), kwargs = {})
#   %convolution_2 : Tensor "f32[810, 147, 32][4704, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %arg17_1, %arg18_1, [1], [0], [1], False, [0], 3), kwargs = {})
#   %view_23 : Tensor "f32[810, 3, 49, 32][4704, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [-1, 3, 49, 32]), kwargs = {})
#   %permute_9 : Tensor "f32[810, 49, 3, 32][4704, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_10 : Tensor "f32[810, 49, 3, 32][4704, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[810, 49, 96][4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [810, 49, 96]), kwargs = {})
#   %view_25 : Tensor "f32[810, 7, 7, 96][4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [-1, 7, 7, 96]), kwargs = {})
#   %view_26 : Tensor "f32[10, 9, 9, 7, 7, 96][381024, 42336, 4704, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [10, 9, 9, 7, 7, -1]), kwargs = {})
#   %permute_10 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 672, 4704, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_11 : Tensor "f32[10, 9, 7, 9, 7, 96][381024, 42336, 6048, 672, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %view_27 : Tensor "f32[10, 63, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [10, 63, 63, -1]), kwargs = {})
#   %slice_1 : Tensor "f32[10, 56, 63, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_27, 1, 4, -3), kwargs = {})
#   %slice_2 : Tensor "f32[10, 56, 56, 96][381024, 6048, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 2, 4, -3), kwargs = {})
#   %clone_12 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [10, 3136, 96]), kwargs = {})
#   %add_11 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %add_tensor_20 : Tensor "f32[31360, 96][96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, %mm_default_20), kwargs = {})
#   %view_32 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [10, 3136, 96]), kwargs = {})
#   %add_15 : Tensor "f32[10, 3136, 96][301056, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   %view_33 : Tensor "f32[10, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [10, 56, 56, 96]), kwargs = {})
#   %slice_3 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_4 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_3, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_5 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_6 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_5, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_7 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_8 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_7, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_9 : Tensor "f32[10, 28, 56, 96][301056, 10752, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_33, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_10 : Tensor "f32[10, 28, 28, 96][301056, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_9, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat : Tensor "f32[10, 28, 28, 384][301056, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6, %slice_8, %slice_10], -1), kwargs = {})
#   return %buf34
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9031680}, 'kernel_num_gb': 0.00602112, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (5472 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 28, 28, 96), (301056, 10752, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00602112
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/o4/co4zaccbjnobxh4rwgpw3wcjycjyov3cqmukqkywgbnydepe3h4h.py
# Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_32 => view_34
#   x_33 => add_16, add_17, mul_16, mul_17, rsqrt_5, sub_5, var_mean_5
# Graph fragment:
#   %cat : Tensor "f32[10, 28, 28, 384][301056, 10752, 384, 1]cuda:0" = PlaceHolder[target=cat]
#   %getitem_11 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=getitem_11]
#   %buf37 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=buf37]
#   %arg25_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %arg26_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %view_34 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [10, -1, 384]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_34, %getitem_11), kwargs = {})
#   %add_16 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_16 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_17 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg25_1), kwargs = {})
#   %add_17 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg26_1), kwargs = {})
#   return %getitem_11,%buf37,%add_17
triton_per_fused_native_layer_norm_view_14 = async_compile.triton('triton_per_fused_native_layer_norm_view_14', '''
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
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 36129792}, 'kernel_num_gb': 0.024087552, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_native_layer_norm_view_14(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp0 - tmp10
    tmp18 = tl.full([1, 1], 384.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 384*x0), tmp27, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 28, 28, 384), (301056, 10752, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 784, 384), (301056, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 7840, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_view_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_native_layer_norm_view_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024087552
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/zb/czbfi5wub3ntwboilo6astd54xjtmi37wnbzpjtihimkpze4mfz4.py
# Topologically Sorted Source Nodes: [x_32, x_33, x_34], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.mm]
# Source node to ATen node mapping:
#   x_32 => view_34
#   x_33 => add_16, add_17, mul_16, mul_17, rsqrt_5, sub_5, var_mean_5
#   x_34 => mm, permute_13, view_35
# Graph fragment:
#   %add_17 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0" = PlaceHolder[target=add_17]
#   %arg27_1 : Tensor "f32[192, 384][384, 1]cuda:0" = PlaceHolder[target=arg27_1]
#   %view_34 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [10, -1, 384]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_34, %getitem_11), kwargs = {})
#   %add_16 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_16 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_17 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg25_1), kwargs = {})
#   %add_17 : Tensor "f32[10, 784, 384][301056, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg26_1), kwargs = {})
#   %view_35 : Tensor "f32[7840, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_17, [7840, 384]), kwargs = {})
#   %permute_13 : Tensor "f32[384, 192][1, 384]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg27_1, [1, 0]), kwargs = {})
#   %mm : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_35, %permute_13), kwargs = {})
#   return %mm
triton_tem_fused_mm_native_layer_norm_t_view_15 = async_compile.triton('triton_tem_fused_mm_native_layer_norm_t_view_15', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_mm_native_layer_norm_t_view_15', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.018358272, 'kernel_flop': 1156055040, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_15(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 7840
    N = 192
    K = 384
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 384
    stride_ak = 1
    stride_bk = 1
    stride_bn = 384

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
        xindex = idx_n + 384*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 784, 384), (301056, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 246, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_mm_native_layer_norm_t_view_15.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_mm_native_layer_norm_t_view_15.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.018358272
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/du/cduwpmzwtef3oepdfty7p3bg2ahkdsl4p5jv46bkozzoatouqpgi.py
# Topologically Sorted Source Nodes: [x_34, x_35], Original ATen: [aten._unsafe_view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_34 => view_36
#   x_35 => var_mean_6
# Graph fragment:
#   %mm : Tensor "f32[7840, 192][192, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_36 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [10, 784, 192]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_13,%buf42
triton_per_fused__unsafe_view_native_layer_norm_16 = async_compile.triton('triton_per_fused__unsafe_view_native_layer_norm_16', '''
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
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 125440, 'r0_': 6021120}, 'kernel_num_gb': 0.00608384, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_16(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 192, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 7840, 192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_native_layer_norm_16.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_native_layer_norm_16.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00608384
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bb/cbbg3rirqm22woijzyt5ivn7qrhhgnr52t5zymbcxam6ofva4k6j.py
# Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution]
# Source node to ATen node mapping:
#   contiguous_5 => clone_15
#   permute_4 => permute_14
#   spatial_mlp_windows_8 => convolution_3
#   windows_2 => view_39
#   x_34 => view_36
#   x_35 => add_18, add_19, mul_18, mul_19, rsqrt_6, sub_6, var_mean_6
#   x_36 => view_37
#   x_37 => view_38
#   x_windows_2 => view_40
#   x_windows_heads_6 => view_41
#   x_windows_heads_7 => permute_15
#   x_windows_heads_8 => clone_16, view_42
# Graph fragment:
#   %mm : Tensor "f32[7840, 192][192, 1]cuda:0" = PlaceHolder[target=mm]
#   %getitem_13 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf42 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=buf42]
#   %arg28_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %arg29_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %view_36 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [10, 784, 192]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_36, %getitem_13), kwargs = {})
#   %add_18 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_18 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg28_1), kwargs = {})
#   %add_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg29_1), kwargs = {})
#   %view_37 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_19, [10, 28, 28, 192]), kwargs = {})
#   %view_38 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_37, [10, 4, 7, 4, 7, 192]), kwargs = {})
#   %permute_14 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 1344, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_38, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_15 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   %view_39 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_15, [-1, 7, 7, 192]), kwargs = {})
#   %view_40 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [-1, 49, 192]), kwargs = {})
#   %view_41 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_40, [-1, 49, 6, 32]), kwargs = {})
#   %permute_15 : Tensor "f32[160, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_41, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %view_42 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [160, 294, 32]), kwargs = {})
#   %convolution_3 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_42, %arg30_1, %arg31_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   return %buf44
triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17 = async_compile.triton('triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 18064896}, 'kernel_num_gb': 0.012106496, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 294)
    x2 = xindex // 9408
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 4)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 4)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 4)) + 28*(((x1 % 49)) // 7) + 196*(x2 // 4) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 4)) + 28*(((x1 % 49)) // 7) + 196*(x2 // 4) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 192.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((160, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1505280,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012106496
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ry/cryimemqygaovmyntzrgcnf3wc6gapjtk3iejc7onnsbfwtr6dbu.py
# Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8, view_25, spatial_mlp_windows_9, spatial_mlp_windows_10, spatial_mlp_windows_11, x_38, permute_5, contiguous_6, x_39, x_40, x_41, layer_norm_7], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   contiguous_5 => clone_15
#   contiguous_6 => clone_18
#   layer_norm_7 => add_21, add_22, mul_20, mul_21, rsqrt_7, sub_7, var_mean_7
#   permute_4 => permute_14
#   permute_5 => permute_17
#   spatial_mlp_windows_10 => clone_17, view_44
#   spatial_mlp_windows_11 => view_45
#   spatial_mlp_windows_8 => convolution_3
#   spatial_mlp_windows_9 => permute_16
#   view_25 => view_43
#   windows_2 => view_39
#   x_34 => view_36
#   x_35 => add_18, add_19, mul_18, mul_19, rsqrt_6, sub_6, var_mean_6
#   x_36 => view_37
#   x_37 => view_38
#   x_38 => view_46
#   x_39 => view_47
#   x_40 => view_48
#   x_41 => add_20
#   x_windows_2 => view_40
#   x_windows_heads_6 => view_41
#   x_windows_heads_7 => permute_15
#   x_windows_heads_8 => clone_16, view_42
# Graph fragment:
#   %mm : Tensor "f32[7840, 192][192, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf45 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0" = PlaceHolder[target=buf45]
#   %arg31_1 : Tensor "f32[294][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %getitem_15 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf47 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=buf47]
#   %arg32_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %arg33_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg33_1]
#   %view_36 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [10, 784, 192]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_36, %getitem_13), kwargs = {})
#   %add_18 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_18 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg28_1), kwargs = {})
#   %add_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg29_1), kwargs = {})
#   %view_37 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_19, [10, 28, 28, 192]), kwargs = {})
#   %view_38 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_37, [10, 4, 7, 4, 7, 192]), kwargs = {})
#   %permute_14 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 1344, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_38, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_15 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   %view_39 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_15, [-1, 7, 7, 192]), kwargs = {})
#   %view_40 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [-1, 49, 192]), kwargs = {})
#   %view_41 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_40, [-1, 49, 6, 32]), kwargs = {})
#   %permute_15 : Tensor "f32[160, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_41, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %view_42 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [160, 294, 32]), kwargs = {})
#   %convolution_3 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_42, %arg30_1, %arg31_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_43 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_3, [-1, 6, 49, 32]), kwargs = {})
#   %permute_16 : Tensor "f32[160, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_43, [0, 2, 1, 3]), kwargs = {})
#   %clone_17 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_17, [160, 49, 192]), kwargs = {})
#   %view_45 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_44, [-1, 7, 7, 192]), kwargs = {})
#   %view_46 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_45, [10, 4, 4, 7, 7, -1]), kwargs = {})
#   %permute_17 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_46, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_18 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %view_47 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [10, 28, 28, -1]), kwargs = {})
#   %view_48 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_47, [10, 784, 192]), kwargs = {})
#   %add_20 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_48), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_20, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %getitem_15), kwargs = {})
#   %add_21 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_20 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_21 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg32_1), kwargs = {})
#   %add_22 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg33_1), kwargs = {})
#   return %getitem_15,%buf47,%add_22
triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18', '''
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
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1176, 'r0_': 24086016}, 'kernel_num_gb': 0.018066072, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 28)) % 7)) + 224*(((x0 // 28) % 7)) + 1568*(r0_2 // 32) + 9408*(((x0 % 28)) // 7) + 37632*(x0 // 196) + 150528*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 28) % 7)) + 49*(r0_2 // 32) + ((((x0 % 28)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 192.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 192*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((160, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 7840, 192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.018066072
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/5a/c5ayhguswwysv6mbhwvsnefclmnyhh7vasvxu6lor5wyrs7baent.py
# Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8, view_25, spatial_mlp_windows_9, spatial_mlp_windows_10, spatial_mlp_windows_11, x_38, permute_5, contiguous_6, x_39, x_40, x_41, layer_norm_7, x_42], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_5 => clone_15
#   contiguous_6 => clone_18
#   layer_norm_7 => add_21, add_22, mul_20, mul_21, rsqrt_7, sub_7, var_mean_7
#   permute_4 => permute_14
#   permute_5 => permute_17
#   spatial_mlp_windows_10 => clone_17, view_44
#   spatial_mlp_windows_11 => view_45
#   spatial_mlp_windows_8 => convolution_3
#   spatial_mlp_windows_9 => permute_16
#   view_25 => view_43
#   windows_2 => view_39
#   x_34 => view_36
#   x_35 => add_18, add_19, mul_18, mul_19, rsqrt_6, sub_6, var_mean_6
#   x_36 => view_37
#   x_37 => view_38
#   x_38 => view_46
#   x_39 => view_47
#   x_40 => view_48
#   x_41 => add_20
#   x_42 => mm_default_19, permute_18, view_49
#   x_windows_2 => view_40
#   x_windows_heads_6 => view_41
#   x_windows_heads_7 => permute_15
#   x_windows_heads_8 => clone_16, view_42
# Graph fragment:
#   %add_22 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_22]
#   %arg34_1 : Tensor "f32[768, 192][192, 1]cuda:0" = PlaceHolder[target=arg34_1]
#   %view_36 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [10, 784, 192]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_36, %getitem_13), kwargs = {})
#   %add_18 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_18 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg28_1), kwargs = {})
#   %add_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg29_1), kwargs = {})
#   %view_37 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_19, [10, 28, 28, 192]), kwargs = {})
#   %view_38 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_37, [10, 4, 7, 4, 7, 192]), kwargs = {})
#   %permute_14 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 1344, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_38, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_15 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   %view_39 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_15, [-1, 7, 7, 192]), kwargs = {})
#   %view_40 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [-1, 49, 192]), kwargs = {})
#   %view_41 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_40, [-1, 49, 6, 32]), kwargs = {})
#   %permute_15 : Tensor "f32[160, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_41, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %view_42 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [160, 294, 32]), kwargs = {})
#   %convolution_3 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_42, %arg30_1, %arg31_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_43 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_3, [-1, 6, 49, 32]), kwargs = {})
#   %permute_16 : Tensor "f32[160, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_43, [0, 2, 1, 3]), kwargs = {})
#   %clone_17 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_17, [160, 49, 192]), kwargs = {})
#   %view_45 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_44, [-1, 7, 7, 192]), kwargs = {})
#   %view_46 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_45, [10, 4, 4, 7, 7, -1]), kwargs = {})
#   %permute_17 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_46, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_18 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %view_47 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [10, 28, 28, -1]), kwargs = {})
#   %view_48 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_47, [10, 784, 192]), kwargs = {})
#   %add_20 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_48), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_20, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %getitem_15), kwargs = {})
#   %add_21 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_20 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_21 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg32_1), kwargs = {})
#   %add_22 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg33_1), kwargs = {})
#   %view_49 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_22, [7840, 192]), kwargs = {})
#   %permute_18 : Tensor "f32[192, 768][1, 192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg34_1, [1, 0]), kwargs = {})
#   %mm_default_19 : Tensor "f32[7840, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_49, %permute_18), kwargs = {})
#   return %mm_default_19
triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19 = async_compile.triton('triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.030695424, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19(arg_A, arg_B, out_ptr0):
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
    N = 768
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
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 738, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.030695424
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4d/c4dyjpmlycstkbckvgdgbml4dhh5vxhlpvjrgbwenboizsq3gwl5.py
# Topologically Sorted Source Nodes: [x_42, x_43], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   x_42 => add_tensor_19, view_50
#   x_43 => add_23, erf_2, mul_22, mul_23, mul_24
# Graph fragment:
#   %arg35_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_19 : Tensor "f32[7840, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_19]
#   %add_tensor_19 : Tensor "f32[7840, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg35_1, %mm_default_19), kwargs = {})
#   %view_50 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_19, [10, 784, 768]), kwargs = {})
#   %mul_22 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, 0.5), kwargs = {})
#   %mul_23 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_23,), kwargs = {})
#   %add_23 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_24 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %add_23), kwargs = {})
#   return %mul_24
triton_poi_fused_addmm_gelu_view_20 = async_compile.triton('triton_poi_fused_addmm_gelu_view_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 72256512}, 'kernel_num_gb': 0.048172032, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)


def get_args():
    arg_0 = rand_strided((10, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 6021120,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_gelu_view_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_gelu_view_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.048172032
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/j5/cj5t52mgiwvkbnflm2ozjp66qygydtazk5cohwy64uxcmmgqx67w.py
# Topologically Sorted Source Nodes: [x_42, x_43, x_45], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   x_42 => add_tensor_19, view_50
#   x_43 => add_23, erf_2, mul_22, mul_23, mul_24
#   x_45 => mm_default_18, permute_19, view_51
# Graph fragment:
#   %mul_24 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0" = PlaceHolder[target=mul_24]
#   %arg36_1 : Tensor "f32[192, 768][768, 1]cuda:0" = PlaceHolder[target=arg36_1]
#   %add_tensor_19 : Tensor "f32[7840, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg35_1, %mm_default_19), kwargs = {})
#   %view_50 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_19, [10, 784, 768]), kwargs = {})
#   %mul_22 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, 0.5), kwargs = {})
#   %mul_23 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_23,), kwargs = {})
#   %add_23 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_24 : Tensor "f32[10, 784, 768][602112, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %add_23), kwargs = {})
#   %view_51 : Tensor "f32[7840, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_24, [7840, 768]), kwargs = {})
#   %permute_19 : Tensor "f32[768, 192][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg36_1, [1, 0]), kwargs = {})
#   %mm_default_18 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_51, %permute_19), kwargs = {})
#   return %mm_default_18
triton_tem_fused_addmm_gelu_t_view_21 = async_compile.triton('triton_tem_fused_addmm_gelu_t_view_21', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_t_view_21', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.030695424, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_21(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 7840
    N = 192
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_0 = rand_strided((10, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 246, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_gelu_t_view_21.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_gelu_t_view_21.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.030695424
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pw/cpwydxj7pbsajy6gwlqnmzgiqftdua5chcolg6lyodnbubrcvppx.py
# Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8, view_25, spatial_mlp_windows_9, spatial_mlp_windows_10, spatial_mlp_windows_11, x_38, permute_5, contiguous_6, x_39, x_40, x_41, x_45, x_47, x_48], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_5 => clone_15
#   contiguous_6 => clone_18
#   permute_4 => permute_14
#   permute_5 => permute_17
#   spatial_mlp_windows_10 => clone_17, view_44
#   spatial_mlp_windows_11 => view_45
#   spatial_mlp_windows_8 => convolution_3
#   spatial_mlp_windows_9 => permute_16
#   view_25 => view_43
#   windows_2 => view_39
#   x_34 => view_36
#   x_35 => add_18, add_19, mul_18, mul_19, rsqrt_6, sub_6, var_mean_6
#   x_36 => view_37
#   x_37 => view_38
#   x_38 => view_46
#   x_39 => view_47
#   x_40 => view_48
#   x_41 => add_20
#   x_45 => add_tensor_18, view_52
#   x_47 => add_24
#   x_48 => var_mean_8
#   x_windows_2 => view_40
#   x_windows_heads_6 => view_41
#   x_windows_heads_7 => permute_15
#   x_windows_heads_8 => clone_16, view_42
# Graph fragment:
#   %mm : Tensor "f32[7840, 192][192, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf45 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0" = PlaceHolder[target=buf45]
#   %arg31_1 : Tensor "f32[294][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %arg37_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %mm_default_18 : Tensor "f32[7840, 192][192, 1]cuda:0" = PlaceHolder[target=mm_default_18]
#   %add_24 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_24]
#   %view_36 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [10, 784, 192]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_36, %getitem_13), kwargs = {})
#   %add_18 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_18 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg28_1), kwargs = {})
#   %add_19 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg29_1), kwargs = {})
#   %view_37 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_19, [10, 28, 28, 192]), kwargs = {})
#   %view_38 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_37, [10, 4, 7, 4, 7, 192]), kwargs = {})
#   %permute_14 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 1344, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_38, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_15 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   %view_39 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_15, [-1, 7, 7, 192]), kwargs = {})
#   %view_40 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [-1, 49, 192]), kwargs = {})
#   %view_41 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_40, [-1, 49, 6, 32]), kwargs = {})
#   %permute_15 : Tensor "f32[160, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_41, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   %view_42 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [160, 294, 32]), kwargs = {})
#   %convolution_3 : Tensor "f32[160, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_42, %arg30_1, %arg31_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_43 : Tensor "f32[160, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_3, [-1, 6, 49, 32]), kwargs = {})
#   %permute_16 : Tensor "f32[160, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_43, [0, 2, 1, 3]), kwargs = {})
#   %clone_17 : Tensor "f32[160, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[160, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_17, [160, 49, 192]), kwargs = {})
#   %view_45 : Tensor "f32[160, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_44, [-1, 7, 7, 192]), kwargs = {})
#   %view_46 : Tensor "f32[10, 4, 4, 7, 7, 192][150528, 37632, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_45, [10, 4, 4, 7, 7, -1]), kwargs = {})
#   %permute_17 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_46, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_18 : Tensor "f32[10, 4, 7, 4, 7, 192][150528, 37632, 5376, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %view_47 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [10, 28, 28, -1]), kwargs = {})
#   %view_48 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_47, [10, 784, 192]), kwargs = {})
#   %add_20 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_48), kwargs = {})
#   %add_tensor_18 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg37_1, %mm_default_18), kwargs = {})
#   %view_52 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_18, [10, 784, 192]), kwargs = {})
#   %add_24 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_52), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   return %add_24,%getitem_17,%buf55
triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22 = async_compile.triton('triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22', '''
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
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 126616, 'r0_': 30106368}, 'kernel_num_gb': 0.024149144, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*((((x0 % 28)) % 7)) + 224*(((x0 // 28) % 7)) + 1568*(r0_2 // 32) + 9408*(((x0 % 28)) // 7) + 37632*(x0 // 196) + 150528*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((x0 // 28) % 7)) + 49*(r0_2 // 32) + ((((x0 % 28)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 192*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((160, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 7840, 192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024149144
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/26/c26664pxtmmqwepgeludv2aqstr7fyovr65uicjg3lvgsfznnkz6.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   permute_6 => permute_20
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   windows_3 => view_55
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_24 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_24]
#   %getitem_17 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=getitem_17]
#   %buf55 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=buf55]
#   %arg38_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   return %buf57
triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23 = async_compile.triton('triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 24838656}, 'kernel_num_gb': 0.015493376, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2352000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 294)
    x2 = xindex // 9408
    x0 = (xindex % 32)
    x3 = xindex
    tmp0 = (-4) + 7*(((x2 // 5) % 5)) + (((x1 % 49)) // 7)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*((x2 % 5)) + ((((x1 % 49)) % 7))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-22272) + x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 5)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 5) % 5)) + 150528*(x2 // 25)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 192.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((250, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2352000,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.015493376
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/r7/cr7kqefzzes4wtks7kxzb267qq23a3zcyo5nydlhoknnmakvarle.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, layer_norm_9], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   contiguous_8 => clone_24
#   getitem_5 => slice_11, slice_12
#   layer_norm_9 => add_28, add_29, mul_27, mul_28, rsqrt_9, sub_9, var_mean_9
#   permute_6 => permute_20
#   permute_7 => permute_23
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   spatial_mlp_windows_13 => permute_22
#   spatial_mlp_windows_14 => clone_23, view_60
#   spatial_mlp_windows_15 => view_61
#   view_34 => view_59
#   windows_3 => view_55
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_51 => view_62
#   x_52 => view_63
#   x_53 => clone_25
#   x_54 => view_64
#   x_55 => add_27
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_24 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_24]
#   %buf58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0" = PlaceHolder[target=buf58]
#   %arg41_1 : Tensor "f32[294][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %getitem_19 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=getitem_19]
#   %buf60 : Tensor "f32[10, 784, 1][784, 1, 7840]cuda:0" = PlaceHolder[target=buf60]
#   %arg42_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %arg43_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_59 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_4, [-1, 6, 49, 32]), kwargs = {})
#   %permute_22 : Tensor "f32[250, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %clone_23 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_60 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_23, [250, 49, 192]), kwargs = {})
#   %view_61 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [-1, 7, 7, 192]), kwargs = {})
#   %view_62 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_61, [10, 5, 5, 7, 7, -1]), kwargs = {})
#   %permute_23 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_62, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_24 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %view_63 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_24, [10, 35, 35, -1]), kwargs = {})
#   %slice_11 : Tensor "f32[10, 28, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_63, 1, 4, -3), kwargs = {})
#   %slice_12 : Tensor "f32[10, 28, 28, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_11, 2, 4, -3), kwargs = {})
#   %clone_25 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_64 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_25, [10, 784, 192]), kwargs = {})
#   %add_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_64), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_27, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_9 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_27, %getitem_19), kwargs = {})
#   %add_28 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_9), kwargs = {})
#   %mul_28 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %arg42_1), kwargs = {})
#   %add_29 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %arg43_1), kwargs = {})
#   return %getitem_19,%buf60,%add_29
triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24', '''
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
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1176, 'r0_': 24086016}, 'kernel_num_gb': 0.018066072, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 28))) % 7)) + 224*(((4 + (x0 // 28)) % 7)) + 1568*(r0_2 // 32) + 9408*((4 + ((x0 % 28))) // 7) + 47040*((4 + (x0 // 28)) // 7) + 235200*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 28)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 28))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 192.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 192*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((250, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 7840, 192,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.018066072
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/4t/c4tqzpyny7z2ld5ztescdsj62qmcpobuwzotl6duvnbqykipnyni.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   contiguous_8 => clone_24
#   getitem_5 => slice_11, slice_12
#   permute_6 => permute_20
#   permute_7 => permute_23
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   spatial_mlp_windows_13 => permute_22
#   spatial_mlp_windows_14 => clone_23, view_60
#   spatial_mlp_windows_15 => view_61
#   view_34 => view_59
#   windows_3 => view_55
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_51 => view_62
#   x_52 => view_63
#   x_53 => clone_25
#   x_54 => view_64
#   x_55 => add_27
#   x_59 => add_tensor_16, view_68
#   x_61 => add_31
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_24 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_24]
#   %buf58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0" = PlaceHolder[target=buf58]
#   %arg41_1 : Tensor "f32[294][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %arg47_1 : Tensor "f32[192][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %mm_default_16 : Tensor "f32[7840, 192][192, 1]cuda:0" = PlaceHolder[target=mm_default_16]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_59 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_4, [-1, 6, 49, 32]), kwargs = {})
#   %permute_22 : Tensor "f32[250, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %clone_23 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_60 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_23, [250, 49, 192]), kwargs = {})
#   %view_61 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [-1, 7, 7, 192]), kwargs = {})
#   %view_62 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_61, [10, 5, 5, 7, 7, -1]), kwargs = {})
#   %permute_23 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_62, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_24 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %view_63 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_24, [10, 35, 35, -1]), kwargs = {})
#   %slice_11 : Tensor "f32[10, 28, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_63, 1, 4, -3), kwargs = {})
#   %slice_12 : Tensor "f32[10, 28, 28, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_11, 2, 4, -3), kwargs = {})
#   %clone_25 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_64 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_25, [10, 784, 192]), kwargs = {})
#   %add_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_64), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg47_1, %mm_default_16), kwargs = {})
#   %view_68 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [10, 784, 192]), kwargs = {})
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_68), kwargs = {})
#   return %add_31
triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 30106368}, 'kernel_num_gb': 0.024086424, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 784)
    x2 = xindex // 150528
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 28))) % 7)) + 224*(((4 + (x1 // 28)) % 7)) + 1568*(x0 // 32) + 9408*((4 + ((x1 % 28))) // 7) + 47040*((4 + (x1 // 28)) // 7) + 235200*x2 + ((x0 % 32))), xmask)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 28)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 28))) % 7))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((250, 294, 32), (9408, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1505280,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024086424
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/sz/cszx6qriya5yl5wlqwwghd7jwr5o4g6libpdzuv3ts4ktfejbsup.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   contiguous_8 => clone_24
#   getitem_5 => slice_11, slice_12
#   permute_6 => permute_20
#   permute_7 => permute_23
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   spatial_mlp_windows_13 => permute_22
#   spatial_mlp_windows_14 => clone_23, view_60
#   spatial_mlp_windows_15 => view_61
#   view_34 => view_59
#   windows_3 => view_55
#   x0_1 => slice_13, slice_14
#   x1_1 => slice_15, slice_16
#   x2_1 => slice_17, slice_18
#   x3_1 => slice_19, slice_20
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_51 => view_62
#   x_52 => view_63
#   x_53 => clone_25
#   x_54 => view_64
#   x_55 => add_27
#   x_59 => add_tensor_16, view_68
#   x_61 => add_31
#   x_62 => view_69
#   x_63 => cat_1
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_31]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_59 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_4, [-1, 6, 49, 32]), kwargs = {})
#   %permute_22 : Tensor "f32[250, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %clone_23 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_60 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_23, [250, 49, 192]), kwargs = {})
#   %view_61 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [-1, 7, 7, 192]), kwargs = {})
#   %view_62 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_61, [10, 5, 5, 7, 7, -1]), kwargs = {})
#   %permute_23 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_62, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_24 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %view_63 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_24, [10, 35, 35, -1]), kwargs = {})
#   %slice_11 : Tensor "f32[10, 28, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_63, 1, 4, -3), kwargs = {})
#   %slice_12 : Tensor "f32[10, 28, 28, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_11, 2, 4, -3), kwargs = {})
#   %clone_25 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_64 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_25, [10, 784, 192]), kwargs = {})
#   %add_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_64), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg47_1, %mm_default_16), kwargs = {})
#   %view_68 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [10, 784, 192]), kwargs = {})
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_68), kwargs = {})
#   %view_69 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_31, [10, 28, 28, 192]), kwargs = {})
#   %slice_13 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_14 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_13, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_15 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_16 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_15, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_17 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_18 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_17, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_19 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_20 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_19, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_1 : Tensor "f32[10, 14, 14, 768][150528, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_14, %slice_16, %slice_18, %slice_20], -1), kwargs = {})
#   return %buf67
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4515840}, 'kernel_num_gb': 0.00301056, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 14, 14, 192), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 376320,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00301056
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/sv/csvyw5sgiuaeyh5ahbjb7zwlk6vtwpij7nre636flemxmx3qum3n.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   contiguous_8 => clone_24
#   getitem_5 => slice_11, slice_12
#   permute_6 => permute_20
#   permute_7 => permute_23
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   spatial_mlp_windows_13 => permute_22
#   spatial_mlp_windows_14 => clone_23, view_60
#   spatial_mlp_windows_15 => view_61
#   view_34 => view_59
#   windows_3 => view_55
#   x0_1 => slice_13, slice_14
#   x1_1 => slice_15, slice_16
#   x2_1 => slice_17, slice_18
#   x3_1 => slice_19, slice_20
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_51 => view_62
#   x_52 => view_63
#   x_53 => clone_25
#   x_54 => view_64
#   x_55 => add_27
#   x_59 => add_tensor_16, view_68
#   x_61 => add_31
#   x_62 => view_69
#   x_63 => cat_1
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_31]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_59 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_4, [-1, 6, 49, 32]), kwargs = {})
#   %permute_22 : Tensor "f32[250, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %clone_23 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_60 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_23, [250, 49, 192]), kwargs = {})
#   %view_61 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [-1, 7, 7, 192]), kwargs = {})
#   %view_62 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_61, [10, 5, 5, 7, 7, -1]), kwargs = {})
#   %permute_23 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_62, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_24 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %view_63 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_24, [10, 35, 35, -1]), kwargs = {})
#   %slice_11 : Tensor "f32[10, 28, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_63, 1, 4, -3), kwargs = {})
#   %slice_12 : Tensor "f32[10, 28, 28, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_11, 2, 4, -3), kwargs = {})
#   %clone_25 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_64 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_25, [10, 784, 192]), kwargs = {})
#   %add_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_64), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg47_1, %mm_default_16), kwargs = {})
#   %view_68 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [10, 784, 192]), kwargs = {})
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_68), kwargs = {})
#   %view_69 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_31, [10, 28, 28, 192]), kwargs = {})
#   %slice_13 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_14 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_13, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_15 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_16 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_15, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_17 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_18 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_17, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_19 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_20 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_19, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_1 : Tensor "f32[10, 14, 14, 768][150528, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_14, %slice_16, %slice_18, %slice_20], -1), kwargs = {})
#   return %buf68
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4515840}, 'kernel_num_gb': 0.00301056, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 14, 14, 192), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 376320,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00301056
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ja/cjasyhf4knjy4mrlevpqwc27dzp22f3jqr32mip2ot2jxsbrrkie.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   contiguous_8 => clone_24
#   getitem_5 => slice_11, slice_12
#   permute_6 => permute_20
#   permute_7 => permute_23
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   spatial_mlp_windows_13 => permute_22
#   spatial_mlp_windows_14 => clone_23, view_60
#   spatial_mlp_windows_15 => view_61
#   view_34 => view_59
#   windows_3 => view_55
#   x0_1 => slice_13, slice_14
#   x1_1 => slice_15, slice_16
#   x2_1 => slice_17, slice_18
#   x3_1 => slice_19, slice_20
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_51 => view_62
#   x_52 => view_63
#   x_53 => clone_25
#   x_54 => view_64
#   x_55 => add_27
#   x_59 => add_tensor_16, view_68
#   x_61 => add_31
#   x_62 => view_69
#   x_63 => cat_1
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_31]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_59 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_4, [-1, 6, 49, 32]), kwargs = {})
#   %permute_22 : Tensor "f32[250, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %clone_23 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_60 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_23, [250, 49, 192]), kwargs = {})
#   %view_61 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [-1, 7, 7, 192]), kwargs = {})
#   %view_62 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_61, [10, 5, 5, 7, 7, -1]), kwargs = {})
#   %permute_23 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_62, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_24 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %view_63 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_24, [10, 35, 35, -1]), kwargs = {})
#   %slice_11 : Tensor "f32[10, 28, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_63, 1, 4, -3), kwargs = {})
#   %slice_12 : Tensor "f32[10, 28, 28, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_11, 2, 4, -3), kwargs = {})
#   %clone_25 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_64 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_25, [10, 784, 192]), kwargs = {})
#   %add_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_64), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg47_1, %mm_default_16), kwargs = {})
#   %view_68 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [10, 784, 192]), kwargs = {})
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_68), kwargs = {})
#   %view_69 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_31, [10, 28, 28, 192]), kwargs = {})
#   %slice_13 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_14 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_13, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_15 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_16 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_15, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_17 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_18 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_17, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_19 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_20 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_19, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_1 : Tensor "f32[10, 14, 14, 768][150528, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_14, %slice_16, %slice_18, %slice_20], -1), kwargs = {})
#   return %buf69
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4515840}, 'kernel_num_gb': 0.00301056, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 14, 14, 192), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 376320,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00301056
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/e7/ce75azyojlnk5otdti4vofjhb7kgh4loolkg4rjvtyacrsfgnoqf.py
# Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_7 => clone_21
#   contiguous_8 => clone_24
#   getitem_5 => slice_11, slice_12
#   permute_6 => permute_20
#   permute_7 => permute_23
#   shifted_x_1 => constant_pad_nd_1
#   spatial_mlp_windows_12 => convolution_4
#   spatial_mlp_windows_13 => permute_22
#   spatial_mlp_windows_14 => clone_23, view_60
#   spatial_mlp_windows_15 => view_61
#   view_34 => view_59
#   windows_3 => view_55
#   x0_1 => slice_13, slice_14
#   x1_1 => slice_15, slice_16
#   x2_1 => slice_17, slice_18
#   x3_1 => slice_19, slice_20
#   x_48 => add_25, add_26, mul_25, mul_26, rsqrt_8, sub_8, var_mean_8
#   x_49 => view_53
#   x_50 => view_54
#   x_51 => view_62
#   x_52 => view_63
#   x_53 => clone_25
#   x_54 => view_64
#   x_55 => add_27
#   x_59 => add_tensor_16, view_68
#   x_61 => add_31
#   x_62 => view_69
#   x_63 => cat_1
#   x_windows_3 => view_56
#   x_windows_heads_10 => permute_21
#   x_windows_heads_11 => clone_22, view_58
#   x_windows_heads_9 => view_57
# Graph fragment:
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0" = PlaceHolder[target=add_31]
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[10, 784, 1][784, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_25 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %arg38_1), kwargs = {})
#   %add_26 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg39_1), kwargs = {})
#   %view_53 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_26, [10, 28, 28, 192]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_53, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_54 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_1, [10, 5, 7, 5, 7, 192]), kwargs = {})
#   %permute_20 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 1344, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_55 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [-1, 7, 7, 192]), kwargs = {})
#   %view_56 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [-1, 49, 192]), kwargs = {})
#   %view_57 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_56, [-1, 49, 6, 32]), kwargs = {})
#   %permute_21 : Tensor "f32[250, 6, 49, 32][9408, 32, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_57, [0, 2, 1, 3]), kwargs = {})
#   %clone_22 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_22, [250, 294, 32]), kwargs = {})
#   %convolution_4 : Tensor "f32[250, 294, 32][9408, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_58, %arg40_1, %arg41_1, [1], [0], [1], False, [0], 6), kwargs = {})
#   %view_59 : Tensor "f32[250, 6, 49, 32][9408, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_4, [-1, 6, 49, 32]), kwargs = {})
#   %permute_22 : Tensor "f32[250, 49, 6, 32][9408, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %clone_23 : Tensor "f32[250, 49, 6, 32][9408, 192, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_60 : Tensor "f32[250, 49, 192][9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_23, [250, 49, 192]), kwargs = {})
#   %view_61 : Tensor "f32[250, 7, 7, 192][9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [-1, 7, 7, 192]), kwargs = {})
#   %view_62 : Tensor "f32[10, 5, 5, 7, 7, 192][235200, 47040, 9408, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_61, [10, 5, 5, 7, 7, -1]), kwargs = {})
#   %permute_23 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 1344, 9408, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_62, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_24 : Tensor "f32[10, 5, 7, 5, 7, 192][235200, 47040, 6720, 1344, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %view_63 : Tensor "f32[10, 35, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_24, [10, 35, 35, -1]), kwargs = {})
#   %slice_11 : Tensor "f32[10, 28, 35, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_63, 1, 4, -3), kwargs = {})
#   %slice_12 : Tensor "f32[10, 28, 28, 192][235200, 6720, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_11, 2, 4, -3), kwargs = {})
#   %clone_25 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_64 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_25, [10, 784, 192]), kwargs = {})
#   %add_27 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_64), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7840, 192][192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg47_1, %mm_default_16), kwargs = {})
#   %view_68 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [10, 784, 192]), kwargs = {})
#   %add_31 : Tensor "f32[10, 784, 192][150528, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_68), kwargs = {})
#   %view_69 : Tensor "f32[10, 28, 28, 192][150528, 5376, 192, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_31, [10, 28, 28, 192]), kwargs = {})
#   %slice_13 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_14 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_13, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_15 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_16 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_15, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_17 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_18 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_17, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_19 : Tensor "f32[10, 14, 28, 192][150528, 10752, 192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_69, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_20 : Tensor "f32[10, 14, 14, 192][150528, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_19, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_1 : Tensor "f32[10, 14, 14, 768][150528, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_14, %slice_16, %slice_18, %slice_20], -1), kwargs = {})
#   return %buf70
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4515840}, 'kernel_num_gb': 0.00301056, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (5568 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 14, 14, 192), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 376320,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00301056
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/6c/c6clgoq6dnca2la3cly46cucqhgy2tav5amfhjdhywvkew2d2rid.py
# Topologically Sorted Source Nodes: [x_64, x_65], Original ATen: [aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_64 => view_70
#   x_65 => add_32, add_33, mul_32, mul_33, rsqrt_10, sub_10, var_mean_10
# Graph fragment:
#   %cat_1 : Tensor "f32[10, 14, 14, 768][150528, 10752, 768, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %getitem_21 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_21]
#   %buf73 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf73]
#   %arg48_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg48_1]
#   %arg49_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg49_1]
#   %view_70 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_1, [10, -1, 768]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_70, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_70, %getitem_21), kwargs = {})
#   %add_32 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_10 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_32 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_10), kwargs = {})
#   %mul_33 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg48_1), kwargs = {})
#   %add_33 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg49_1), kwargs = {})
#   return %getitem_21,%buf73,%add_33
triton_per_fused_native_layer_norm_view_30 = async_compile.triton('triton_per_fused_native_layer_norm_view_30', '''
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
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 18069504}, 'kernel_num_gb': 0.012048384, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_native_layer_norm_view_30(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp0 - tmp10
    tmp18 = tl.full([1, 1], 768.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp27, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 14, 14, 768), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1960, 768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_view_30.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_native_layer_norm_view_30.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012048384
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/u4/cu4idhfhevnhywp5uvtpjx27zk574vww7xmj6uybyabktw3vtarn.py
# Topologically Sorted Source Nodes: [x_64, x_65, x_66], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.mm]
# Source node to ATen node mapping:
#   x_64 => view_70
#   x_65 => add_32, add_33, mul_32, mul_33, rsqrt_10, sub_10, var_mean_10
#   x_66 => mm_1, permute_26, view_71
# Graph fragment:
#   %add_33 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0" = PlaceHolder[target=add_33]
#   %arg50_1 : Tensor "f32[384, 768][768, 1]cuda:0" = PlaceHolder[target=arg50_1]
#   %view_70 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_1, [10, -1, 768]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_70, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_70, %getitem_21), kwargs = {})
#   %add_32 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_10 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_32 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_10), kwargs = {})
#   %mul_33 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg48_1), kwargs = {})
#   %add_33 : Tensor "f32[10, 196, 768][150528, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg49_1), kwargs = {})
#   %view_71 : Tensor "f32[1960, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_33, [1960, 768]), kwargs = {})
#   %permute_26 : Tensor "f32[768, 384][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg50_1, [1, 0]), kwargs = {})
#   %mm_1 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_71, %permute_26), kwargs = {})
#   return %mm_1
triton_tem_fused_mm_native_layer_norm_t_view_31 = async_compile.triton('triton_tem_fused_mm_native_layer_norm_t_view_31', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_mm_native_layer_norm_t_view_31', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.010211328, 'kernel_flop': 1156055040, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_31(arg_A, arg_B, out_ptr0):
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
    N = 384
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 93, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_mm_native_layer_norm_t_view_31.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_mm_native_layer_norm_t_view_31.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.010211328
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/f2/cf2go67tebruvibhjuewl3nxzompcdf57e64h2gaph2s4miq4ljz.py
# Topologically Sorted Source Nodes: [x_66, x_67], Original ATen: [aten._unsafe_view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_66 => view_72
#   x_67 => var_mean_11
# Graph fragment:
#   %mm_1 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %view_72 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [10, 196, 384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_23,%buf78
triton_per_fused__unsafe_view_native_layer_norm_32 = async_compile.triton('triton_per_fused__unsafe_view_native_layer_norm_32', '''
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_native_layer_norm_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 31360, 'r0_': 3010560}, 'kernel_num_gb': 0.00302624, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_32(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 1960, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_native_layer_norm_32.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_native_layer_norm_32.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00302624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bt/cbt3pppzgxzepswqm24nwn2j6vs2t3pjuykk5lvtsu2ybsjffxzi.py
# Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution]
# Source node to ATen node mapping:
#   contiguous_10 => clone_28
#   permute_8 => permute_27
#   spatial_mlp_windows_16 => convolution_5
#   windows_4 => view_75
#   x_66 => view_72
#   x_67 => add_34, add_35, mul_34, mul_35, rsqrt_11, sub_11, var_mean_11
#   x_68 => view_73
#   x_69 => view_74
#   x_windows_4 => view_76
#   x_windows_heads_12 => view_77
#   x_windows_heads_13 => permute_28
#   x_windows_heads_14 => clone_29, view_78
# Graph fragment:
#   %mm_1 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %getitem_23 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_23]
#   %buf78 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf78]
#   %arg51_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %arg52_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %view_72 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [10, 196, 384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_72, %getitem_23), kwargs = {})
#   %add_34 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_11 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
#   %mul_34 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %arg51_1), kwargs = {})
#   %add_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %arg52_1), kwargs = {})
#   %view_73 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_35, [10, 14, 14, 384]), kwargs = {})
#   %view_74 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_73, [10, 2, 7, 2, 7, 384]), kwargs = {})
#   %permute_27 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 2688, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_28 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_27,), kwargs = {memory_format: torch.contiguous_format})
#   %view_75 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_28, [-1, 7, 7, 384]), kwargs = {})
#   %view_76 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_75, [-1, 49, 384]), kwargs = {})
#   %view_77 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_76, [-1, 49, 12, 32]), kwargs = {})
#   %permute_28 : Tensor "f32[40, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_77, [0, 2, 1, 3]), kwargs = {})
#   %clone_29 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_78 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_29, [40, 588, 32]), kwargs = {})
#   %convolution_5 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_78, %arg53_1, %arg54_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   return %buf80
triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33 = async_compile.triton('triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9034752}, 'kernel_num_gb': 0.006039872, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 588)
    x2 = xindex // 18816
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 2)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 2)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 384.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006039872
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/2s/c2s32hc3w5oy2winyl44yvhom3vhajd4h5h56bggqcjn6g3df5f6.py
# Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16, view_45, spatial_mlp_windows_17, spatial_mlp_windows_18, spatial_mlp_windows_19, x_70, permute_9, contiguous_11, x_71, x_72, x_73, layer_norm_12], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   contiguous_10 => clone_28
#   contiguous_11 => clone_31
#   layer_norm_12 => add_37, add_38, mul_36, mul_37, rsqrt_12, sub_12, var_mean_12
#   permute_8 => permute_27
#   permute_9 => permute_30
#   spatial_mlp_windows_16 => convolution_5
#   spatial_mlp_windows_17 => permute_29
#   spatial_mlp_windows_18 => clone_30, view_80
#   spatial_mlp_windows_19 => view_81
#   view_45 => view_79
#   windows_4 => view_75
#   x_66 => view_72
#   x_67 => add_34, add_35, mul_34, mul_35, rsqrt_11, sub_11, var_mean_11
#   x_68 => view_73
#   x_69 => view_74
#   x_70 => view_82
#   x_71 => view_83
#   x_72 => view_84
#   x_73 => add_36
#   x_windows_4 => view_76
#   x_windows_heads_12 => view_77
#   x_windows_heads_13 => permute_28
#   x_windows_heads_14 => clone_29, view_78
# Graph fragment:
#   %mm_1 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %buf81 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0" = PlaceHolder[target=buf81]
#   %arg54_1 : Tensor "f32[588][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %getitem_25 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_25]
#   %buf83 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf83]
#   %arg55_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg55_1]
#   %arg56_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %view_72 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [10, 196, 384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_72, %getitem_23), kwargs = {})
#   %add_34 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_11 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
#   %mul_34 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %arg51_1), kwargs = {})
#   %add_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %arg52_1), kwargs = {})
#   %view_73 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_35, [10, 14, 14, 384]), kwargs = {})
#   %view_74 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_73, [10, 2, 7, 2, 7, 384]), kwargs = {})
#   %permute_27 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 2688, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_28 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_27,), kwargs = {memory_format: torch.contiguous_format})
#   %view_75 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_28, [-1, 7, 7, 384]), kwargs = {})
#   %view_76 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_75, [-1, 49, 384]), kwargs = {})
#   %view_77 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_76, [-1, 49, 12, 32]), kwargs = {})
#   %permute_28 : Tensor "f32[40, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_77, [0, 2, 1, 3]), kwargs = {})
#   %clone_29 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_78 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_29, [40, 588, 32]), kwargs = {})
#   %convolution_5 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_78, %arg53_1, %arg54_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_79 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_5, [-1, 12, 49, 32]), kwargs = {})
#   %permute_29 : Tensor "f32[40, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_79, [0, 2, 1, 3]), kwargs = {})
#   %clone_30 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_29,), kwargs = {memory_format: torch.contiguous_format})
#   %view_80 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_30, [40, 49, 384]), kwargs = {})
#   %view_81 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_80, [-1, 7, 7, 384]), kwargs = {})
#   %view_82 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_81, [10, 2, 2, 7, 7, -1]), kwargs = {})
#   %permute_30 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_82, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_31 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_30,), kwargs = {memory_format: torch.contiguous_format})
#   %view_83 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_31, [10, 14, 14, -1]), kwargs = {})
#   %view_84 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_83, [10, 196, 384]), kwargs = {})
#   %add_36 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_72, %view_84), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_36, %getitem_25), kwargs = {})
#   %add_37 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_12 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_37,), kwargs = {})
#   %mul_36 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_12), kwargs = {})
#   %mul_37 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %arg55_1), kwargs = {})
#   %add_38 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %arg56_1), kwargs = {})
#   return %getitem_25,%buf83,%add_38
triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34', '''
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2352, 'r0_': 12045312}, 'kernel_num_gb': 0.009037104, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009037104
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/pk/cpkwq2ldjy2t3ac3isztzfie4wyxzs5o4d5h7tfow4ayc6khtura.py
# Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16, view_45, spatial_mlp_windows_17, spatial_mlp_windows_18, spatial_mlp_windows_19, x_70, permute_9, contiguous_11, x_71, x_72, x_73, layer_norm_12, x_74], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_10 => clone_28
#   contiguous_11 => clone_31
#   layer_norm_12 => add_37, add_38, mul_36, mul_37, rsqrt_12, sub_12, var_mean_12
#   permute_8 => permute_27
#   permute_9 => permute_30
#   spatial_mlp_windows_16 => convolution_5
#   spatial_mlp_windows_17 => permute_29
#   spatial_mlp_windows_18 => clone_30, view_80
#   spatial_mlp_windows_19 => view_81
#   view_45 => view_79
#   windows_4 => view_75
#   x_66 => view_72
#   x_67 => add_34, add_35, mul_34, mul_35, rsqrt_11, sub_11, var_mean_11
#   x_68 => view_73
#   x_69 => view_74
#   x_70 => view_82
#   x_71 => view_83
#   x_72 => view_84
#   x_73 => add_36
#   x_74 => mm_default_15, permute_31, view_85
#   x_windows_4 => view_76
#   x_windows_heads_12 => view_77
#   x_windows_heads_13 => permute_28
#   x_windows_heads_14 => clone_29, view_78
# Graph fragment:
#   %add_38 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_38]
#   %arg57_1 : Tensor "f32[1536, 384][384, 1]cuda:0" = PlaceHolder[target=arg57_1]
#   %view_72 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [10, 196, 384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_72, %getitem_23), kwargs = {})
#   %add_34 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_11 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
#   %mul_34 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %arg51_1), kwargs = {})
#   %add_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %arg52_1), kwargs = {})
#   %view_73 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_35, [10, 14, 14, 384]), kwargs = {})
#   %view_74 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_73, [10, 2, 7, 2, 7, 384]), kwargs = {})
#   %permute_27 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 2688, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_28 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_27,), kwargs = {memory_format: torch.contiguous_format})
#   %view_75 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_28, [-1, 7, 7, 384]), kwargs = {})
#   %view_76 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_75, [-1, 49, 384]), kwargs = {})
#   %view_77 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_76, [-1, 49, 12, 32]), kwargs = {})
#   %permute_28 : Tensor "f32[40, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_77, [0, 2, 1, 3]), kwargs = {})
#   %clone_29 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_78 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_29, [40, 588, 32]), kwargs = {})
#   %convolution_5 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_78, %arg53_1, %arg54_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_79 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_5, [-1, 12, 49, 32]), kwargs = {})
#   %permute_29 : Tensor "f32[40, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_79, [0, 2, 1, 3]), kwargs = {})
#   %clone_30 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_29,), kwargs = {memory_format: torch.contiguous_format})
#   %view_80 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_30, [40, 49, 384]), kwargs = {})
#   %view_81 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_80, [-1, 7, 7, 384]), kwargs = {})
#   %view_82 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_81, [10, 2, 2, 7, 7, -1]), kwargs = {})
#   %permute_30 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_82, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_31 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_30,), kwargs = {memory_format: torch.contiguous_format})
#   %view_83 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_31, [10, 14, 14, -1]), kwargs = {})
#   %view_84 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_83, [10, 196, 384]), kwargs = {})
#   %add_36 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_72, %view_84), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_36, %getitem_25), kwargs = {})
#   %add_37 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_12 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_37,), kwargs = {})
#   %mul_36 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_12), kwargs = {})
#   %mul_37 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %arg55_1), kwargs = {})
#   %add_38 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %arg56_1), kwargs = {})
#   %view_85 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_38, [1960, 384]), kwargs = {})
#   %permute_31 : Tensor "f32[384, 1536][1, 384]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg57_1, [1, 0]), kwargs = {})
#   %mm_default_15 : Tensor "f32[1960, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_85, %permute_31), kwargs = {})
#   return %mm_default_15
triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35 = async_compile.triton('triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.017412096, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35(arg_A, arg_B, out_ptr0):
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
    N = 1536
    K = 384
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 384
    stride_ak = 1
    stride_bk = 1
    stride_bn = 384

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
        xindex = idx_n + 384*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1536*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1536*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 372, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.017412096
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/vm/cvmebx7gtu6vywgj4w6uggcw6klmxflm2qfxmsukzws2gvyjgtgv.py
# Topologically Sorted Source Nodes: [x_74, x_75], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   x_74 => add_tensor_15, view_86
#   x_75 => add_39, erf_4, mul_38, mul_39, mul_40
# Graph fragment:
#   %arg58_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %mm_default_15 : Tensor "f32[1960, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_15]
#   %add_tensor_15 : Tensor "f32[1960, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, %mm_default_15), kwargs = {})
#   %view_86 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [10, 196, 1536]), kwargs = {})
#   %mul_38 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_86, 0.5), kwargs = {})
#   %mul_39 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_86, 0.7071067811865476), kwargs = {})
#   %erf_4 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_39,), kwargs = {})
#   %add_39 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_40 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %add_39), kwargs = {})
#   return %mul_40
triton_poi_fused_addmm_gelu_view_36 = async_compile.triton('triton_poi_fused_addmm_gelu_view_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 36132864}, 'kernel_num_gb': 0.024090624, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 1536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)


def get_args():
    arg_0 = rand_strided((10, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 3010560,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_gelu_view_36.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_gelu_view_36.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.024090624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/5b/c5bjgw4hl5edtq6vdub4bml5jx7muhlz3itswspn3x7v2dkfbjlx.py
# Topologically Sorted Source Nodes: [x_74, x_75, x_77], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   x_74 => add_tensor_15, view_86
#   x_75 => add_39, erf_4, mul_38, mul_39, mul_40
#   x_77 => mm_default_14, permute_32, view_87
# Graph fragment:
#   %mul_40 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0" = PlaceHolder[target=mul_40]
#   %arg59_1 : Tensor "f32[384, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg59_1]
#   %add_tensor_15 : Tensor "f32[1960, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, %mm_default_15), kwargs = {})
#   %view_86 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [10, 196, 1536]), kwargs = {})
#   %mul_38 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_86, 0.5), kwargs = {})
#   %mul_39 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_86, 0.7071067811865476), kwargs = {})
#   %erf_4 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_39,), kwargs = {})
#   %add_39 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_40 : Tensor "f32[10, 196, 1536][301056, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %add_39), kwargs = {})
#   %view_87 : Tensor "f32[1960, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_40, [1960, 1536]), kwargs = {})
#   %permute_32 : Tensor "f32[1536, 384][1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg59_1, [1, 0]), kwargs = {})
#   %mm_default_14 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_87, %permute_32), kwargs = {})
#   return %mm_default_14
triton_tem_fused_addmm_gelu_t_view_37 = async_compile.triton('triton_tem_fused_addmm_gelu_t_view_37', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_t_view_37', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.017412096, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_37(arg_A, arg_B, out_ptr0):
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
    N = 384
    K = 1536
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1536
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1536

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
        xindex = idx_n + 1536*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 93, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_gelu_t_view_37.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_gelu_t_view_37.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.017412096
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/o7/co764xjeiw7jqiasthg5ys3yzhggohtffdpmmxbrpmzvzowgicog.py
# Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16, view_45, spatial_mlp_windows_17, spatial_mlp_windows_18, spatial_mlp_windows_19, x_70, permute_9, contiguous_11, x_71, x_72, x_73, x_77, x_79, x_80], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_10 => clone_28
#   contiguous_11 => clone_31
#   permute_8 => permute_27
#   permute_9 => permute_30
#   spatial_mlp_windows_16 => convolution_5
#   spatial_mlp_windows_17 => permute_29
#   spatial_mlp_windows_18 => clone_30, view_80
#   spatial_mlp_windows_19 => view_81
#   view_45 => view_79
#   windows_4 => view_75
#   x_66 => view_72
#   x_67 => add_34, add_35, mul_34, mul_35, rsqrt_11, sub_11, var_mean_11
#   x_68 => view_73
#   x_69 => view_74
#   x_70 => view_82
#   x_71 => view_83
#   x_72 => view_84
#   x_73 => add_36
#   x_77 => add_tensor_14, view_88
#   x_79 => add_40
#   x_80 => var_mean_13
#   x_windows_4 => view_76
#   x_windows_heads_12 => view_77
#   x_windows_heads_13 => permute_28
#   x_windows_heads_14 => clone_29, view_78
# Graph fragment:
#   %mm_1 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %buf81 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0" = PlaceHolder[target=buf81]
#   %arg54_1 : Tensor "f32[588][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %arg60_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %mm_default_14 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_default_14]
#   %add_40 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_40]
#   %view_72 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [10, 196, 384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_72, %getitem_23), kwargs = {})
#   %add_34 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_11 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
#   %mul_34 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %arg51_1), kwargs = {})
#   %add_35 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %arg52_1), kwargs = {})
#   %view_73 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_35, [10, 14, 14, 384]), kwargs = {})
#   %view_74 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_73, [10, 2, 7, 2, 7, 384]), kwargs = {})
#   %permute_27 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 2688, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_28 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_27,), kwargs = {memory_format: torch.contiguous_format})
#   %view_75 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_28, [-1, 7, 7, 384]), kwargs = {})
#   %view_76 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_75, [-1, 49, 384]), kwargs = {})
#   %view_77 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_76, [-1, 49, 12, 32]), kwargs = {})
#   %permute_28 : Tensor "f32[40, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_77, [0, 2, 1, 3]), kwargs = {})
#   %clone_29 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_78 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_29, [40, 588, 32]), kwargs = {})
#   %convolution_5 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_78, %arg53_1, %arg54_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_79 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_5, [-1, 12, 49, 32]), kwargs = {})
#   %permute_29 : Tensor "f32[40, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_79, [0, 2, 1, 3]), kwargs = {})
#   %clone_30 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_29,), kwargs = {memory_format: torch.contiguous_format})
#   %view_80 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_30, [40, 49, 384]), kwargs = {})
#   %view_81 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_80, [-1, 7, 7, 384]), kwargs = {})
#   %view_82 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_81, [10, 2, 2, 7, 7, -1]), kwargs = {})
#   %permute_30 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_82, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_31 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_30,), kwargs = {memory_format: torch.contiguous_format})
#   %view_83 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_31, [10, 14, 14, -1]), kwargs = {})
#   %view_84 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_83, [10, 196, 384]), kwargs = {})
#   %add_36 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_72, %view_84), kwargs = {})
#   %add_tensor_14 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg60_1, %mm_default_14), kwargs = {})
#   %view_88 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_14, [10, 196, 384]), kwargs = {})
#   %add_40 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %view_88), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_40, [2]), kwargs = {correction: 0, keepdim: True})
#   return %add_40,%getitem_27,%buf91
triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38 = async_compile.triton('triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38', '''
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33712, 'r0_': 15054336}, 'kernel_num_gb': 0.012061808, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1960, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012061808
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ow/cowafd4hqbva3qxdvbgnslkrwxgwlmk7bj6nqtwn2zk6jossz3xp.py
# Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
# Source node to ATen node mapping:
#   contiguous_12 => clone_34
#   permute_10 => permute_33
#   shifted_x_2 => constant_pad_nd_2
#   spatial_mlp_windows_20 => convolution_6
#   windows_5 => view_91
#   x_80 => add_41, add_42, mul_41, mul_42, rsqrt_13, sub_13, var_mean_13
#   x_81 => view_89
#   x_82 => view_90
#   x_windows_5 => view_92
#   x_windows_heads_15 => view_93
#   x_windows_heads_16 => permute_34
#   x_windows_heads_17 => clone_35, view_94
# Graph fragment:
#   %add_40 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_40]
#   %getitem_27 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf91 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf91]
#   %arg61_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %arg62_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_40, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %getitem_27), kwargs = {})
#   %add_41 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_13 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
#   %mul_41 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %mul_42 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %arg61_1), kwargs = {})
#   %add_42 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %arg62_1), kwargs = {})
#   %view_89 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_42, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_2 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_89, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_90 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_2, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_33 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_90, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_34 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_33,), kwargs = {memory_format: torch.contiguous_format})
#   %view_91 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_34, [-1, 7, 7, 384]), kwargs = {})
#   %view_92 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_91, [-1, 49, 384]), kwargs = {})
#   %view_93 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_92, [-1, 49, 12, 32]), kwargs = {})
#   %permute_34 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_93, [0, 2, 1, 3]), kwargs = {})
#   %clone_35 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   %view_94 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [90, 588, 32]), kwargs = {})
#   %convolution_6 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_94, %arg63_1, %arg64_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   return %buf93
triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39 = async_compile.triton('triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16561152}, 'kernel_num_gb': 0.009803072, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1693440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 588)
    x2 = xindex // 18816
    x0 = (xindex % 32)
    x3 = xindex
    tmp0 = (-4) + 7*(((x2 // 3) % 3)) + (((x1 % 49)) // 7)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*((x2 % 3)) + ((((x1 % 49)) % 7))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-23040) + x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 3)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 3) % 3)) + 75264*(x2 // 9)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-60) + 7*((x2 % 3)) + 14*(((x1 % 49)) // 7) + 98*(((x2 // 3) % 3)) + 196*(x2 // 9) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-60) + 7*((x2 % 3)) + 14*(((x1 % 49)) // 7) + 98*(((x2 // 3) % 3)) + 196*(x2 // 9) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 384.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((90, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1693440,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009803072
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/xx/cxxqqtmg5rhetmh6ttv72qccgei67zkco4do7inbiiydpkg4zkft.py
# Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20, view_54, spatial_mlp_windows_21, spatial_mlp_windows_22, spatial_mlp_windows_23, x_83, permute_11, contiguous_13, x_84, getitem_10, x_85, x_86, x_87, layer_norm_14], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
# Source node to ATen node mapping:
#   contiguous_12 => clone_34
#   contiguous_13 => clone_37
#   getitem_10 => slice_21, slice_22
#   layer_norm_14 => add_44, add_45, mul_43, mul_44, rsqrt_14, sub_14, var_mean_14
#   permute_10 => permute_33
#   permute_11 => permute_36
#   shifted_x_2 => constant_pad_nd_2
#   spatial_mlp_windows_20 => convolution_6
#   spatial_mlp_windows_21 => permute_35
#   spatial_mlp_windows_22 => clone_36, view_96
#   spatial_mlp_windows_23 => view_97
#   view_54 => view_95
#   windows_5 => view_91
#   x_80 => add_41, add_42, mul_41, mul_42, rsqrt_13, sub_13, var_mean_13
#   x_81 => view_89
#   x_82 => view_90
#   x_83 => view_98
#   x_84 => view_99
#   x_85 => clone_38
#   x_86 => view_100
#   x_87 => add_43
#   x_windows_5 => view_92
#   x_windows_heads_15 => view_93
#   x_windows_heads_16 => permute_34
#   x_windows_heads_17 => clone_35, view_94
# Graph fragment:
#   %add_40 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_40]
#   %buf94 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0" = PlaceHolder[target=buf94]
#   %arg64_1 : Tensor "f32[588][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %getitem_29 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_29]
#   %buf96 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf96]
#   %arg65_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %arg66_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg66_1]
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_40, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %getitem_27), kwargs = {})
#   %add_41 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_13 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
#   %mul_41 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %mul_42 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %arg61_1), kwargs = {})
#   %add_42 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %arg62_1), kwargs = {})
#   %view_89 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_42, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_2 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_89, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_90 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_2, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_33 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_90, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_34 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_33,), kwargs = {memory_format: torch.contiguous_format})
#   %view_91 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_34, [-1, 7, 7, 384]), kwargs = {})
#   %view_92 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_91, [-1, 49, 384]), kwargs = {})
#   %view_93 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_92, [-1, 49, 12, 32]), kwargs = {})
#   %permute_34 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_93, [0, 2, 1, 3]), kwargs = {})
#   %clone_35 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   %view_94 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [90, 588, 32]), kwargs = {})
#   %convolution_6 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_94, %arg63_1, %arg64_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_95 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [-1, 12, 49, 32]), kwargs = {})
#   %permute_35 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_95, [0, 2, 1, 3]), kwargs = {})
#   %clone_36 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_35,), kwargs = {memory_format: torch.contiguous_format})
#   %view_96 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_36, [90, 49, 384]), kwargs = {})
#   %view_97 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_96, [-1, 7, 7, 384]), kwargs = {})
#   %view_98 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_97, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_36 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_98, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_37 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_36,), kwargs = {memory_format: torch.contiguous_format})
#   %view_99 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_37, [10, 21, 21, -1]), kwargs = {})
#   %slice_21 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_99, 1, 4, -3), kwargs = {})
#   %slice_22 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_21, 2, 4, -3), kwargs = {})
#   %clone_38 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_100 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_38, [10, 196, 384]), kwargs = {})
#   %add_43 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %view_100), kwargs = {})
#   %var_mean_14 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_43, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_14 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_43, %getitem_29), kwargs = {})
#   %add_44 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_14 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_44,), kwargs = {})
#   %mul_43 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_14), kwargs = {})
#   %mul_44 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %arg65_1), kwargs = {})
#   %add_45 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %arg66_1), kwargs = {})
#   return %getitem_29,%buf96,%add_45
triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40', '''
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2352, 'r0_': 12045312}, 'kernel_num_gb': 0.009037104, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 14))) % 7)) + 224*(((4 + (x0 // 14)) % 7)) + 1568*(r0_2 // 32) + 18816*((4 + ((x0 % 14))) // 7) + 56448*((4 + (x0 // 14)) // 7) + 169344*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 14)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 14))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((90, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009037104
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ys/cyswjkzrqwbonndmlwcpxhi5bxo5e3jekvz3cq35muqv7pvojihs.py
# Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20, view_54, spatial_mlp_windows_21, spatial_mlp_windows_22, spatial_mlp_windows_23, x_83, permute_11, contiguous_13, x_84, getitem_10, x_85, x_86, x_87, x_91, x_93, x_94], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_12 => clone_34
#   contiguous_13 => clone_37
#   getitem_10 => slice_21, slice_22
#   permute_10 => permute_33
#   permute_11 => permute_36
#   shifted_x_2 => constant_pad_nd_2
#   spatial_mlp_windows_20 => convolution_6
#   spatial_mlp_windows_21 => permute_35
#   spatial_mlp_windows_22 => clone_36, view_96
#   spatial_mlp_windows_23 => view_97
#   view_54 => view_95
#   windows_5 => view_91
#   x_80 => add_41, add_42, mul_41, mul_42, rsqrt_13, sub_13, var_mean_13
#   x_81 => view_89
#   x_82 => view_90
#   x_83 => view_98
#   x_84 => view_99
#   x_85 => clone_38
#   x_86 => view_100
#   x_87 => add_43
#   x_91 => add_tensor_12, view_104
#   x_93 => add_47
#   x_94 => var_mean_15
#   x_windows_5 => view_92
#   x_windows_heads_15 => view_93
#   x_windows_heads_16 => permute_34
#   x_windows_heads_17 => clone_35, view_94
# Graph fragment:
#   %add_40 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_40]
#   %buf94 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0" = PlaceHolder[target=buf94]
#   %arg64_1 : Tensor "f32[588][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %arg70_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %mm_default_12 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_default_12]
#   %add_47 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_47]
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_40, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %getitem_27), kwargs = {})
#   %add_41 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_13 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
#   %mul_41 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %mul_42 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %arg61_1), kwargs = {})
#   %add_42 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %arg62_1), kwargs = {})
#   %view_89 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_42, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_2 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_89, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_90 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_2, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_33 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_90, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_34 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_33,), kwargs = {memory_format: torch.contiguous_format})
#   %view_91 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_34, [-1, 7, 7, 384]), kwargs = {})
#   %view_92 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_91, [-1, 49, 384]), kwargs = {})
#   %view_93 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_92, [-1, 49, 12, 32]), kwargs = {})
#   %permute_34 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_93, [0, 2, 1, 3]), kwargs = {})
#   %clone_35 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   %view_94 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [90, 588, 32]), kwargs = {})
#   %convolution_6 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_94, %arg63_1, %arg64_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_95 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [-1, 12, 49, 32]), kwargs = {})
#   %permute_35 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_95, [0, 2, 1, 3]), kwargs = {})
#   %clone_36 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_35,), kwargs = {memory_format: torch.contiguous_format})
#   %view_96 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_36, [90, 49, 384]), kwargs = {})
#   %view_97 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_96, [-1, 7, 7, 384]), kwargs = {})
#   %view_98 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_97, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_36 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_98, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_37 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_36,), kwargs = {memory_format: torch.contiguous_format})
#   %view_99 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_37, [10, 21, 21, -1]), kwargs = {})
#   %slice_21 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_99, 1, 4, -3), kwargs = {})
#   %slice_22 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_21, 2, 4, -3), kwargs = {})
#   %clone_38 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_22,), kwargs = {memory_format: torch.contiguous_format})
#   %view_100 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_38, [10, 196, 384]), kwargs = {})
#   %add_43 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %view_100), kwargs = {})
#   %add_tensor_12 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg70_1, %mm_default_12), kwargs = {})
#   %view_104 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_12, [10, 196, 384]), kwargs = {})
#   %add_47 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %view_104), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_47, [2]), kwargs = {correction: 0, keepdim: True})
#   return %add_47,%getitem_31,%buf104
triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41 = async_compile.triton('triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41', '''
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33712, 'r0_': 15054336}, 'kernel_num_gb': 0.012061808, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x0 % 14))) % 7)) + 224*(((4 + (x0 // 14)) % 7)) + 1568*(r0_2 // 32) + 18816*((4 + ((x0 % 14))) // 7) + 56448*((4 + (x0 // 14)) // 7) + 169344*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x0 // 14)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 14))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((90, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1960, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012061808
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ol/col6fkdzp6pdsirfdv42eyu6mqhhaixe7362y5cdjo6ludyka6mu.py
# Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
# Source node to ATen node mapping:
#   contiguous_15 => clone_41
#   permute_12 => permute_39
#   spatial_mlp_windows_24 => convolution_7
#   windows_6 => view_107
#   x_94 => add_48, add_49, mul_48, mul_49, rsqrt_15, sub_15, var_mean_15
#   x_95 => view_105
#   x_96 => view_106
#   x_windows_6 => view_108
#   x_windows_heads_18 => view_109
#   x_windows_heads_19 => permute_40
#   x_windows_heads_20 => clone_42, view_110
# Graph fragment:
#   %add_47 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_47]
#   %getitem_31 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_31]
#   %buf104 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf104]
#   %arg71_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg71_1]
#   %arg72_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_47, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_15 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_47, %getitem_31), kwargs = {})
#   %add_48 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_15 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %mul_48 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_15), kwargs = {})
#   %mul_49 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %arg71_1), kwargs = {})
#   %add_49 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %arg72_1), kwargs = {})
#   %view_105 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_49, [10, 14, 14, 384]), kwargs = {})
#   %view_106 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_105, [10, 2, 7, 2, 7, 384]), kwargs = {})
#   %permute_39 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 2688, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_41 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %view_107 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_41, [-1, 7, 7, 384]), kwargs = {})
#   %view_108 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_107, [-1, 49, 384]), kwargs = {})
#   %view_109 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [-1, 49, 12, 32]), kwargs = {})
#   %permute_40 : Tensor "f32[40, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [0, 2, 1, 3]), kwargs = {})
#   %clone_42 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_40,), kwargs = {memory_format: torch.contiguous_format})
#   %view_110 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_42, [40, 588, 32]), kwargs = {})
#   %convolution_7 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_110, %arg73_1, %arg74_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   return %buf106
triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42 = async_compile.triton('triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9034752}, 'kernel_num_gb': 0.006039872, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 588)
    x2 = xindex // 18816
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 2)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 2)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 384.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006039872
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/2c/c2c72nh5nop42vlwcxmuueaagsnxjoc5wybvoxvtm4enumww6ehe.py
# Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24, view_63, spatial_mlp_windows_25, spatial_mlp_windows_26, spatial_mlp_windows_27, x_97, permute_13, contiguous_16, x_98, x_99, x_100, layer_norm_16], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   contiguous_15 => clone_41
#   contiguous_16 => clone_44
#   layer_norm_16 => add_51, add_52, mul_50, mul_51, rsqrt_16, sub_16, var_mean_16
#   permute_12 => permute_39
#   permute_13 => permute_42
#   spatial_mlp_windows_24 => convolution_7
#   spatial_mlp_windows_25 => permute_41
#   spatial_mlp_windows_26 => clone_43, view_112
#   spatial_mlp_windows_27 => view_113
#   view_63 => view_111
#   windows_6 => view_107
#   x_100 => add_50
#   x_94 => add_48, add_49, mul_48, mul_49, rsqrt_15, sub_15, var_mean_15
#   x_95 => view_105
#   x_96 => view_106
#   x_97 => view_114
#   x_98 => view_115
#   x_99 => view_116
#   x_windows_6 => view_108
#   x_windows_heads_18 => view_109
#   x_windows_heads_19 => permute_40
#   x_windows_heads_20 => clone_42, view_110
# Graph fragment:
#   %add_47 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_47]
#   %buf107 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0" = PlaceHolder[target=buf107]
#   %arg74_1 : Tensor "f32[588][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %getitem_33 : Tensor "f32[10, 196, 1][196, 1, 1960]cuda:0" = PlaceHolder[target=getitem_33]
#   %buf109 : Tensor "f32[10, 196, 1][196, 1, 1984]cuda:0" = PlaceHolder[target=buf109]
#   %arg75_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %arg76_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg76_1]
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_47, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_15 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_47, %getitem_31), kwargs = {})
#   %add_48 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_15 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %mul_48 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_15), kwargs = {})
#   %mul_49 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %arg71_1), kwargs = {})
#   %add_49 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %arg72_1), kwargs = {})
#   %view_105 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_49, [10, 14, 14, 384]), kwargs = {})
#   %view_106 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_105, [10, 2, 7, 2, 7, 384]), kwargs = {})
#   %permute_39 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 2688, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_41 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %view_107 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_41, [-1, 7, 7, 384]), kwargs = {})
#   %view_108 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_107, [-1, 49, 384]), kwargs = {})
#   %view_109 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [-1, 49, 12, 32]), kwargs = {})
#   %permute_40 : Tensor "f32[40, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [0, 2, 1, 3]), kwargs = {})
#   %clone_42 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_40,), kwargs = {memory_format: torch.contiguous_format})
#   %view_110 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_42, [40, 588, 32]), kwargs = {})
#   %convolution_7 : Tensor "f32[40, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_110, %arg73_1, %arg74_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_111 : Tensor "f32[40, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_7, [-1, 12, 49, 32]), kwargs = {})
#   %permute_41 : Tensor "f32[40, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_111, [0, 2, 1, 3]), kwargs = {})
#   %clone_43 : Tensor "f32[40, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   %view_112 : Tensor "f32[40, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_43, [40, 49, 384]), kwargs = {})
#   %view_113 : Tensor "f32[40, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_112, [-1, 7, 7, 384]), kwargs = {})
#   %view_114 : Tensor "f32[10, 2, 2, 7, 7, 384][75264, 37632, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_113, [10, 2, 2, 7, 7, -1]), kwargs = {})
#   %permute_42 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_114, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_44 : Tensor "f32[10, 2, 7, 2, 7, 384][75264, 37632, 5376, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_42,), kwargs = {memory_format: torch.contiguous_format})
#   %view_115 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_44, [10, 14, 14, -1]), kwargs = {})
#   %view_116 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_115, [10, 196, 384]), kwargs = {})
#   %add_50 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, %view_116), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_50, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_16 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_50, %getitem_33), kwargs = {})
#   %add_51 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_16 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_51,), kwargs = {})
#   %mul_50 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %rsqrt_16), kwargs = {})
#   %mul_51 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %arg75_1), kwargs = {})
#   %add_52 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %arg76_1), kwargs = {})
#   return %getitem_33,%buf109,%add_52
triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43', '''
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2352, 'r0_': 12045312}, 'kernel_num_gb': 0.009037104, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009037104
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/q2/cq2eiw2vbzs6ojpgsxmrjzndjlqayxn22g3bddcoqtm6rij2mvqg.py
# Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   contiguous_22 => clone_60
#   contiguous_23 => clone_63
#   getitem_12 => slice_25, slice_26
#   permute_18 => permute_57
#   permute_19 => permute_60
#   shifted_x_4 => constant_pad_nd_4
#   spatial_mlp_windows_36 => convolution_10
#   spatial_mlp_windows_37 => permute_59
#   spatial_mlp_windows_38 => clone_62, view_160
#   spatial_mlp_windows_39 => view_161
#   view_90 => view_159
#   windows_9 => view_155
#   x_134 => add_69, add_70, mul_69, mul_70, rsqrt_21, sub_21, var_mean_21
#   x_135 => view_153
#   x_136 => view_154
#   x_137 => view_162
#   x_138 => view_163
#   x_139 => clone_64
#   x_140 => view_164
#   x_141 => add_71
#   x_145 => add_tensor_4, view_168
#   x_147 => add_75
#   x_windows_9 => view_156
#   x_windows_heads_27 => view_157
#   x_windows_heads_28 => permute_58
#   x_windows_heads_29 => clone_61, view_158
# Graph fragment:
#   %add_68 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_68]
#   %buf146 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0" = PlaceHolder[target=buf146]
#   %arg104_1 : Tensor "f32[588][1]cuda:0" = PlaceHolder[target=arg104_1]
#   %arg110_1 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=arg110_1]
#   %mm_default_4 : Tensor "f32[1960, 384][384, 1]cuda:0" = PlaceHolder[target=mm_default_4]
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_68, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_68, %getitem_43), kwargs = {})
#   %add_69 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_21 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_69 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %mul_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %arg101_1), kwargs = {})
#   %add_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg102_1), kwargs = {})
#   %view_153 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_4 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_153, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_154 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_4, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_57 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_154, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_60 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_57,), kwargs = {memory_format: torch.contiguous_format})
#   %view_155 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [-1, 7, 7, 384]), kwargs = {})
#   %view_156 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_155, [-1, 49, 384]), kwargs = {})
#   %view_157 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_156, [-1, 49, 12, 32]), kwargs = {})
#   %permute_58 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1, 3]), kwargs = {})
#   %clone_61 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_58,), kwargs = {memory_format: torch.contiguous_format})
#   %view_158 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_61, [90, 588, 32]), kwargs = {})
#   %convolution_10 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_158, %arg103_1, %arg104_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_159 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [-1, 12, 49, 32]), kwargs = {})
#   %permute_59 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_159, [0, 2, 1, 3]), kwargs = {})
#   %clone_62 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
#   %view_160 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_62, [90, 49, 384]), kwargs = {})
#   %view_161 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [-1, 7, 7, 384]), kwargs = {})
#   %view_162 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_60 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_63 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
#   %view_163 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_63, [10, 21, 21, -1]), kwargs = {})
#   %slice_25 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_163, 1, 4, -3), kwargs = {})
#   %slice_26 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_25, 2, 4, -3), kwargs = {})
#   %clone_64 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
#   %view_164 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_64, [10, 196, 384]), kwargs = {})
#   %add_71 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %view_164), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg110_1, %mm_default_4), kwargs = {})
#   %view_168 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [10, 196, 384]), kwargs = {})
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %view_168), kwargs = {})
#   return %add_75
triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15054336}, 'kernel_num_gb': 0.012046128, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 196)
    x2 = xindex // 75264
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 14))) % 7)) + 224*(((4 + (x1 // 14)) % 7)) + 1568*(x0 // 32) + 18816*((4 + ((x1 % 14))) // 7) + 56448*((4 + (x1 // 14)) // 7) + 169344*x2 + ((x0 % 32))), xmask)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 14)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 14))) % 7))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((90, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 752640,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012046128
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/bt/cbtk4jd4ub6g6uytwh5ar4a2xwigezp7rgv2c7z77vxxrmbhhjbo.py
# Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_22 => clone_60
#   contiguous_23 => clone_63
#   getitem_12 => slice_25, slice_26
#   permute_18 => permute_57
#   permute_19 => permute_60
#   shifted_x_4 => constant_pad_nd_4
#   spatial_mlp_windows_36 => convolution_10
#   spatial_mlp_windows_37 => permute_59
#   spatial_mlp_windows_38 => clone_62, view_160
#   spatial_mlp_windows_39 => view_161
#   view_90 => view_159
#   windows_9 => view_155
#   x0_2 => slice_27, slice_28
#   x1_2 => slice_29, slice_30
#   x2_2 => slice_31, slice_32
#   x3_2 => slice_33, slice_34
#   x_134 => add_69, add_70, mul_69, mul_70, rsqrt_21, sub_21, var_mean_21
#   x_135 => view_153
#   x_136 => view_154
#   x_137 => view_162
#   x_138 => view_163
#   x_139 => clone_64
#   x_140 => view_164
#   x_141 => add_71
#   x_145 => add_tensor_4, view_168
#   x_147 => add_75
#   x_148 => view_169
#   x_149 => cat_2
#   x_windows_9 => view_156
#   x_windows_heads_27 => view_157
#   x_windows_heads_28 => permute_58
#   x_windows_heads_29 => clone_61, view_158
# Graph fragment:
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_75]
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_68, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_68, %getitem_43), kwargs = {})
#   %add_69 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_21 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_69 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %mul_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %arg101_1), kwargs = {})
#   %add_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg102_1), kwargs = {})
#   %view_153 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_4 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_153, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_154 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_4, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_57 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_154, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_60 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_57,), kwargs = {memory_format: torch.contiguous_format})
#   %view_155 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [-1, 7, 7, 384]), kwargs = {})
#   %view_156 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_155, [-1, 49, 384]), kwargs = {})
#   %view_157 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_156, [-1, 49, 12, 32]), kwargs = {})
#   %permute_58 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1, 3]), kwargs = {})
#   %clone_61 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_58,), kwargs = {memory_format: torch.contiguous_format})
#   %view_158 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_61, [90, 588, 32]), kwargs = {})
#   %convolution_10 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_158, %arg103_1, %arg104_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_159 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [-1, 12, 49, 32]), kwargs = {})
#   %permute_59 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_159, [0, 2, 1, 3]), kwargs = {})
#   %clone_62 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
#   %view_160 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_62, [90, 49, 384]), kwargs = {})
#   %view_161 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [-1, 7, 7, 384]), kwargs = {})
#   %view_162 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_60 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_63 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
#   %view_163 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_63, [10, 21, 21, -1]), kwargs = {})
#   %slice_25 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_163, 1, 4, -3), kwargs = {})
#   %slice_26 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_25, 2, 4, -3), kwargs = {})
#   %clone_64 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
#   %view_164 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_64, [10, 196, 384]), kwargs = {})
#   %add_71 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %view_164), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg110_1, %mm_default_4), kwargs = {})
#   %view_168 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [10, 196, 384]), kwargs = {})
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %view_168), kwargs = {})
#   %view_169 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_75, [10, 14, 14, 384]), kwargs = {})
#   %slice_27 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_28 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_27, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_29 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_30 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_29, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_31 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_32 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_31, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_33 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_34 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_33, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_2 : Tensor "f32[10, 7, 7, 1536][75264, 10752, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_28, %slice_30, %slice_32, %slice_34], -1), kwargs = {})
#   return %buf155
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2257920}, 'kernel_num_gb': 0.00150528, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 7, 7, 384), (75264, 10752, 1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 188160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00150528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/lx/clxubvwhr2n6iffvdum2djk4m2ubt42vcqaemw6glbwcmcpddrhr.py
# Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_22 => clone_60
#   contiguous_23 => clone_63
#   getitem_12 => slice_25, slice_26
#   permute_18 => permute_57
#   permute_19 => permute_60
#   shifted_x_4 => constant_pad_nd_4
#   spatial_mlp_windows_36 => convolution_10
#   spatial_mlp_windows_37 => permute_59
#   spatial_mlp_windows_38 => clone_62, view_160
#   spatial_mlp_windows_39 => view_161
#   view_90 => view_159
#   windows_9 => view_155
#   x0_2 => slice_27, slice_28
#   x1_2 => slice_29, slice_30
#   x2_2 => slice_31, slice_32
#   x3_2 => slice_33, slice_34
#   x_134 => add_69, add_70, mul_69, mul_70, rsqrt_21, sub_21, var_mean_21
#   x_135 => view_153
#   x_136 => view_154
#   x_137 => view_162
#   x_138 => view_163
#   x_139 => clone_64
#   x_140 => view_164
#   x_141 => add_71
#   x_145 => add_tensor_4, view_168
#   x_147 => add_75
#   x_148 => view_169
#   x_149 => cat_2
#   x_windows_9 => view_156
#   x_windows_heads_27 => view_157
#   x_windows_heads_28 => permute_58
#   x_windows_heads_29 => clone_61, view_158
# Graph fragment:
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_75]
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_68, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_68, %getitem_43), kwargs = {})
#   %add_69 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_21 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_69 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %mul_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %arg101_1), kwargs = {})
#   %add_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg102_1), kwargs = {})
#   %view_153 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_4 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_153, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_154 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_4, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_57 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_154, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_60 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_57,), kwargs = {memory_format: torch.contiguous_format})
#   %view_155 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [-1, 7, 7, 384]), kwargs = {})
#   %view_156 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_155, [-1, 49, 384]), kwargs = {})
#   %view_157 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_156, [-1, 49, 12, 32]), kwargs = {})
#   %permute_58 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1, 3]), kwargs = {})
#   %clone_61 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_58,), kwargs = {memory_format: torch.contiguous_format})
#   %view_158 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_61, [90, 588, 32]), kwargs = {})
#   %convolution_10 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_158, %arg103_1, %arg104_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_159 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [-1, 12, 49, 32]), kwargs = {})
#   %permute_59 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_159, [0, 2, 1, 3]), kwargs = {})
#   %clone_62 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
#   %view_160 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_62, [90, 49, 384]), kwargs = {})
#   %view_161 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [-1, 7, 7, 384]), kwargs = {})
#   %view_162 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_60 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_63 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
#   %view_163 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_63, [10, 21, 21, -1]), kwargs = {})
#   %slice_25 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_163, 1, 4, -3), kwargs = {})
#   %slice_26 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_25, 2, 4, -3), kwargs = {})
#   %clone_64 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
#   %view_164 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_64, [10, 196, 384]), kwargs = {})
#   %add_71 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %view_164), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg110_1, %mm_default_4), kwargs = {})
#   %view_168 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [10, 196, 384]), kwargs = {})
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %view_168), kwargs = {})
#   %view_169 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_75, [10, 14, 14, 384]), kwargs = {})
#   %slice_27 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_28 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_27, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_29 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_30 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_29, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_31 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_32 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_31, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_33 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_34 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_33, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_2 : Tensor "f32[10, 7, 7, 1536][75264, 10752, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_28, %slice_30, %slice_32, %slice_34], -1), kwargs = {})
#   return %buf156
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2257920}, 'kernel_num_gb': 0.00150528, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 7, 7, 384), (75264, 10752, 1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 188160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00150528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/wc/cwc6jfnqoqcamy4ktj6jrawf3ulmtv7ru4lz6ytyytil7wvfrsiq.py
# Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_22 => clone_60
#   contiguous_23 => clone_63
#   getitem_12 => slice_25, slice_26
#   permute_18 => permute_57
#   permute_19 => permute_60
#   shifted_x_4 => constant_pad_nd_4
#   spatial_mlp_windows_36 => convolution_10
#   spatial_mlp_windows_37 => permute_59
#   spatial_mlp_windows_38 => clone_62, view_160
#   spatial_mlp_windows_39 => view_161
#   view_90 => view_159
#   windows_9 => view_155
#   x0_2 => slice_27, slice_28
#   x1_2 => slice_29, slice_30
#   x2_2 => slice_31, slice_32
#   x3_2 => slice_33, slice_34
#   x_134 => add_69, add_70, mul_69, mul_70, rsqrt_21, sub_21, var_mean_21
#   x_135 => view_153
#   x_136 => view_154
#   x_137 => view_162
#   x_138 => view_163
#   x_139 => clone_64
#   x_140 => view_164
#   x_141 => add_71
#   x_145 => add_tensor_4, view_168
#   x_147 => add_75
#   x_148 => view_169
#   x_149 => cat_2
#   x_windows_9 => view_156
#   x_windows_heads_27 => view_157
#   x_windows_heads_28 => permute_58
#   x_windows_heads_29 => clone_61, view_158
# Graph fragment:
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_75]
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_68, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_68, %getitem_43), kwargs = {})
#   %add_69 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_21 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_69 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %mul_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %arg101_1), kwargs = {})
#   %add_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg102_1), kwargs = {})
#   %view_153 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_4 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_153, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_154 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_4, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_57 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_154, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_60 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_57,), kwargs = {memory_format: torch.contiguous_format})
#   %view_155 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [-1, 7, 7, 384]), kwargs = {})
#   %view_156 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_155, [-1, 49, 384]), kwargs = {})
#   %view_157 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_156, [-1, 49, 12, 32]), kwargs = {})
#   %permute_58 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1, 3]), kwargs = {})
#   %clone_61 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_58,), kwargs = {memory_format: torch.contiguous_format})
#   %view_158 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_61, [90, 588, 32]), kwargs = {})
#   %convolution_10 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_158, %arg103_1, %arg104_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_159 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [-1, 12, 49, 32]), kwargs = {})
#   %permute_59 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_159, [0, 2, 1, 3]), kwargs = {})
#   %clone_62 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
#   %view_160 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_62, [90, 49, 384]), kwargs = {})
#   %view_161 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [-1, 7, 7, 384]), kwargs = {})
#   %view_162 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_60 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_63 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
#   %view_163 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_63, [10, 21, 21, -1]), kwargs = {})
#   %slice_25 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_163, 1, 4, -3), kwargs = {})
#   %slice_26 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_25, 2, 4, -3), kwargs = {})
#   %clone_64 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
#   %view_164 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_64, [10, 196, 384]), kwargs = {})
#   %add_71 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %view_164), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg110_1, %mm_default_4), kwargs = {})
#   %view_168 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [10, 196, 384]), kwargs = {})
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %view_168), kwargs = {})
#   %view_169 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_75, [10, 14, 14, 384]), kwargs = {})
#   %slice_27 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_28 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_27, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_29 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_30 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_29, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_31 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_32 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_31, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_33 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_34 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_33, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_2 : Tensor "f32[10, 7, 7, 1536][75264, 10752, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_28, %slice_30, %slice_32, %slice_34], -1), kwargs = {})
#   return %buf157
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2257920}, 'kernel_num_gb': 0.00150528, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (384 + x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 7, 7, 384), (75264, 10752, 1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 188160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00150528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/qy/cqy5otos337itedz57gkidaxpsurl6zif4noy4y54sskdiw5tktj.py
# Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
# Source node to ATen node mapping:
#   contiguous_22 => clone_60
#   contiguous_23 => clone_63
#   getitem_12 => slice_25, slice_26
#   permute_18 => permute_57
#   permute_19 => permute_60
#   shifted_x_4 => constant_pad_nd_4
#   spatial_mlp_windows_36 => convolution_10
#   spatial_mlp_windows_37 => permute_59
#   spatial_mlp_windows_38 => clone_62, view_160
#   spatial_mlp_windows_39 => view_161
#   view_90 => view_159
#   windows_9 => view_155
#   x0_2 => slice_27, slice_28
#   x1_2 => slice_29, slice_30
#   x2_2 => slice_31, slice_32
#   x3_2 => slice_33, slice_34
#   x_134 => add_69, add_70, mul_69, mul_70, rsqrt_21, sub_21, var_mean_21
#   x_135 => view_153
#   x_136 => view_154
#   x_137 => view_162
#   x_138 => view_163
#   x_139 => clone_64
#   x_140 => view_164
#   x_141 => add_71
#   x_145 => add_tensor_4, view_168
#   x_147 => add_75
#   x_148 => view_169
#   x_149 => cat_2
#   x_windows_9 => view_156
#   x_windows_heads_27 => view_157
#   x_windows_heads_28 => permute_58
#   x_windows_heads_29 => clone_61, view_158
# Graph fragment:
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0" = PlaceHolder[target=add_75]
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_68, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_68, %getitem_43), kwargs = {})
#   %add_69 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_21 : Tensor "f32[10, 196, 1][196, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_69 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %mul_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %arg101_1), kwargs = {})
#   %add_70 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg102_1), kwargs = {})
#   %view_153 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [10, 14, 14, 384]), kwargs = {})
#   %constant_pad_nd_4 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_153, [0, 0, 4, 3, 4, 3], 0.0), kwargs = {})
#   %view_154 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd_4, [10, 3, 7, 3, 7, 384]), kwargs = {})
#   %permute_57 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 2688, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_154, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_60 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_57,), kwargs = {memory_format: torch.contiguous_format})
#   %view_155 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [-1, 7, 7, 384]), kwargs = {})
#   %view_156 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_155, [-1, 49, 384]), kwargs = {})
#   %view_157 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_156, [-1, 49, 12, 32]), kwargs = {})
#   %permute_58 : Tensor "f32[90, 12, 49, 32][18816, 32, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1, 3]), kwargs = {})
#   %clone_61 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_58,), kwargs = {memory_format: torch.contiguous_format})
#   %view_158 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_61, [90, 588, 32]), kwargs = {})
#   %convolution_10 : Tensor "f32[90, 588, 32][18816, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_158, %arg103_1, %arg104_1, [1], [0], [1], False, [0], 12), kwargs = {})
#   %view_159 : Tensor "f32[90, 12, 49, 32][18816, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [-1, 12, 49, 32]), kwargs = {})
#   %permute_59 : Tensor "f32[90, 49, 12, 32][18816, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_159, [0, 2, 1, 3]), kwargs = {})
#   %clone_62 : Tensor "f32[90, 49, 12, 32][18816, 384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
#   %view_160 : Tensor "f32[90, 49, 384][18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_62, [90, 49, 384]), kwargs = {})
#   %view_161 : Tensor "f32[90, 7, 7, 384][18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [-1, 7, 7, 384]), kwargs = {})
#   %view_162 : Tensor "f32[10, 3, 3, 7, 7, 384][169344, 56448, 18816, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [10, 3, 3, 7, 7, -1]), kwargs = {})
#   %permute_60 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 2688, 18816, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %clone_63 : Tensor "f32[10, 3, 7, 3, 7, 384][169344, 56448, 8064, 2688, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
#   %view_163 : Tensor "f32[10, 21, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_63, [10, 21, 21, -1]), kwargs = {})
#   %slice_25 : Tensor "f32[10, 14, 21, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_163, 1, 4, -3), kwargs = {})
#   %slice_26 : Tensor "f32[10, 14, 14, 384][169344, 8064, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_25, 2, 4, -3), kwargs = {})
#   %clone_64 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
#   %view_164 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_64, [10, 196, 384]), kwargs = {})
#   %add_71 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %view_164), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1960, 384][384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg110_1, %mm_default_4), kwargs = {})
#   %view_168 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [10, 196, 384]), kwargs = {})
#   %add_75 : Tensor "f32[10, 196, 384][75264, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %view_168), kwargs = {})
#   %view_169 : Tensor "f32[10, 14, 14, 384][75264, 5376, 384, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%add_75, [10, 14, 14, 384]), kwargs = {})
#   %slice_27 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_28 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_27, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_29 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_30 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_29, 2, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_31 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 0, 9223372036854775807, 2), kwargs = {})
#   %slice_32 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_31, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_33 : Tensor "f32[10, 7, 14, 384][75264, 10752, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_169, 1, 1, 9223372036854775807, 2), kwargs = {})
#   %slice_34 : Tensor "f32[10, 7, 7, 384][75264, 10752, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_33, 2, 1, 9223372036854775807, 2), kwargs = {})
#   %cat_2 : Tensor "f32[10, 7, 7, 1536][75264, 10752, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_28, %slice_30, %slice_32, %slice_34], -1), kwargs = {})
#   return %buf158
triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2257920}, 'kernel_num_gb': 0.00150528, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (5760 + x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)


def get_args():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 7, 7, 384), (75264, 10752, 1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 188160,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.00150528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/2e/c2efjwgjrjdiv2zdq7o24af6ozb5mzkbv3ckxoalbqkb3l3gpkuz.py
# Topologically Sorted Source Nodes: [x_150, x_151], Original ATen: [aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_150 => view_170
#   x_151 => add_76, add_77, mul_76, mul_77, rsqrt_23, sub_23, var_mean_23
# Graph fragment:
#   %cat_2 : Tensor "f32[10, 7, 7, 1536][75264, 10752, 1536, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %getitem_47 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=getitem_47]
#   %buf161 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=buf161]
#   %arg111_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg111_1]
#   %arg112_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg112_1]
#   %view_170 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_2, [10, -1, 1536]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_170, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_23 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_170, %getitem_47), kwargs = {})
#   %add_76 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_23 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_76 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_23), kwargs = {})
#   %mul_77 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %arg111_1), kwargs = {})
#   %add_77 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %arg112_1), kwargs = {})
#   return %getitem_47,%buf161,%add_77
triton_red_fused_native_layer_norm_view_49 = async_compile.triton('triton_red_fused_native_layer_norm_view_49', '''
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
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_view_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 9043968}, 'kernel_num_gb': 0.006033408, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_native_layer_norm_view_49(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8 - tmp2
        tmp10 = tl.full([1, 1], 1536.0, tl.float32)
        tmp11 = (tmp6 / tmp10)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp19, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 7, 7, 1536), (75264, 10752, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 49, 1536), (75264, 1536, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 490, 1536,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_view_49.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_native_layer_norm_view_49.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006033408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/zo/czolw7znerezjmkofy5xz74mn435pzfgupli37kwpx5s4atz4mce.py
# Topologically Sorted Source Nodes: [x_150, x_151, x_152], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.mm]
# Source node to ATen node mapping:
#   x_150 => view_170
#   x_151 => add_76, add_77, mul_76, mul_77, rsqrt_23, sub_23, var_mean_23
#   x_152 => mm_2, permute_63, view_171
# Graph fragment:
#   %add_77 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0" = PlaceHolder[target=add_77]
#   %arg113_1 : Tensor "f32[768, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg113_1]
#   %view_170 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_2, [10, -1, 1536]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_170, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_23 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_170, %getitem_47), kwargs = {})
#   %add_76 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_23 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_76 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_23), kwargs = {})
#   %mul_77 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %arg111_1), kwargs = {})
#   %add_77 : Tensor "f32[10, 49, 1536][75264, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %arg112_1), kwargs = {})
#   %view_171 : Tensor "f32[490, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [490, 1536]), kwargs = {})
#   %permute_63 : Tensor "f32[1536, 768][1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg113_1, [1, 0]), kwargs = {})
#   %mm_2 : Tensor "f32[490, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_171, %permute_63), kwargs = {})
#   return %mm_2
triton_tem_fused_mm_native_layer_norm_t_view_50 = async_compile.triton('triton_tem_fused_mm_native_layer_norm_t_view_50', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_mm_native_layer_norm_t_view_50', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.009234432, 'kernel_flop': 1156055040, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_50(arg_A, arg_B, out_ptr0):
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

    M = 490
    N = 768
    K = 1536
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1536
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1536

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
        xindex = idx_n + 1536*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 49, 1536), (75264, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 96, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_mm_native_layer_norm_t_view_50.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_mm_native_layer_norm_t_view_50.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.009234432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/zk/czkt2hhqvuiz3mfxo5i3mr4kf34orb25zncytehiyoi6ek3titep.py
# Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten._unsafe_view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_152 => view_172
#   x_153 => var_mean_24
# Graph fragment:
#   %mm_2 : Tensor "f32[490, 768][768, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_172 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [10, 49, 768]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_49,%buf166
triton_per_fused__unsafe_view_native_layer_norm_51 = async_compile.triton('triton_per_fused__unsafe_view_native_layer_norm_51', '''
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
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_native_layer_norm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 7840, 'r0_': 1505280}, 'kernel_num_gb': 0.0015092, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_51(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)


def get_args():
    arg_0 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 490, 768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_native_layer_norm_51.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_native_layer_norm_51.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.0015092
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/74/c7466lo7zclmjunua4qc7rzfq53e3n4h6y4dbjukkjb3oj4hv6lo.py
# Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   permute_20 => permute_64
#   spatial_mlp_windows_40 => convolution_11
#   windows_10 => view_175
#   x_152 => view_172
#   x_153 => add_78, add_79, mul_78, mul_79, rsqrt_24, sub_24, var_mean_24
#   x_154 => view_173
#   x_155 => view_174
#   x_windows_10 => view_176
#   x_windows_heads_30 => view_177
#   x_windows_heads_31 => permute_65
#   x_windows_heads_32 => clone_67, view_178
# Graph fragment:
#   %mm_2 : Tensor "f32[490, 768][768, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %getitem_49 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=getitem_49]
#   %buf166 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=buf166]
#   %arg114_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg114_1]
#   %arg115_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg115_1]
#   %view_172 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [10, 49, 768]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_172, %getitem_49), kwargs = {})
#   %add_78 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_78,), kwargs = {})
#   %mul_78 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %arg114_1), kwargs = {})
#   %add_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %arg115_1), kwargs = {})
#   %view_173 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_79, [10, 7, 7, 768]), kwargs = {})
#   %view_174 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_173, [10, 1, 7, 1, 7, 768]), kwargs = {})
#   %permute_64 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_174, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_175 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_64, [-1, 7, 7, 768]), kwargs = {})
#   %view_176 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_175, [-1, 49, 768]), kwargs = {})
#   %view_177 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_176, [-1, 49, 24, 32]), kwargs = {})
#   %permute_65 : Tensor "f32[10, 24, 49, 32][37632, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_177, [0, 2, 1, 3]), kwargs = {})
#   %clone_67 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_65,), kwargs = {memory_format: torch.contiguous_format})
#   %view_178 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_67, [10, 1176, 32]), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_178, %arg116_1, %arg117_1, [1], [0], [1], False, [0], 24), kwargs = {})
#   return %buf168
triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52 = async_compile.triton('triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4521984}, 'kernel_num_gb': 0.003020624, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 1176)
    x2 = xindex // 37632
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1176, 32), (37632, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 376320,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.003020624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/kj/ckj3kzlaqjgfk2v5uhr4qvr6glhi2wdskgnp5mlcp4mlzx6hy737.py
# Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40, view_101, spatial_mlp_windows_41, spatial_mlp_windows_42, spatial_mlp_windows_43, x_156, permute_21, x_157, x_158, x_159, layer_norm_25], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   layer_norm_25 => add_81, add_82, mul_80, mul_81, rsqrt_25, sub_25, var_mean_25
#   permute_20 => permute_64
#   permute_21 => permute_67
#   spatial_mlp_windows_40 => convolution_11
#   spatial_mlp_windows_41 => permute_66
#   spatial_mlp_windows_42 => clone_68, view_180
#   spatial_mlp_windows_43 => view_181
#   view_101 => view_179
#   windows_10 => view_175
#   x_152 => view_172
#   x_153 => add_78, add_79, mul_78, mul_79, rsqrt_24, sub_24, var_mean_24
#   x_154 => view_173
#   x_155 => view_174
#   x_156 => view_182
#   x_157 => view_183
#   x_158 => view_184
#   x_159 => add_80
#   x_windows_10 => view_176
#   x_windows_heads_30 => view_177
#   x_windows_heads_31 => permute_65
#   x_windows_heads_32 => clone_67, view_178
# Graph fragment:
#   %mm_2 : Tensor "f32[490, 768][768, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %buf169 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0" = PlaceHolder[target=buf169]
#   %arg117_1 : Tensor "f32[1176][1]cuda:0" = PlaceHolder[target=arg117_1]
#   %getitem_51 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=getitem_51]
#   %buf171 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=buf171]
#   %arg118_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %arg119_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg119_1]
#   %view_172 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [10, 49, 768]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_172, %getitem_49), kwargs = {})
#   %add_78 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_78,), kwargs = {})
#   %mul_78 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %arg114_1), kwargs = {})
#   %add_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %arg115_1), kwargs = {})
#   %view_173 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_79, [10, 7, 7, 768]), kwargs = {})
#   %view_174 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_173, [10, 1, 7, 1, 7, 768]), kwargs = {})
#   %permute_64 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_174, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_175 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_64, [-1, 7, 7, 768]), kwargs = {})
#   %view_176 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_175, [-1, 49, 768]), kwargs = {})
#   %view_177 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_176, [-1, 49, 24, 32]), kwargs = {})
#   %permute_65 : Tensor "f32[10, 24, 49, 32][37632, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_177, [0, 2, 1, 3]), kwargs = {})
#   %clone_67 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_65,), kwargs = {memory_format: torch.contiguous_format})
#   %view_178 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_67, [10, 1176, 32]), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_178, %arg116_1, %arg117_1, [1], [0], [1], False, [0], 24), kwargs = {})
#   %view_179 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [-1, 24, 49, 32]), kwargs = {})
#   %permute_66 : Tensor "f32[10, 49, 24, 32][37632, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_179, [0, 2, 1, 3]), kwargs = {})
#   %clone_68 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_66,), kwargs = {memory_format: torch.contiguous_format})
#   %view_180 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_68, [10, 49, 768]), kwargs = {})
#   %view_181 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_180, [-1, 7, 7, 768]), kwargs = {})
#   %view_182 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_181, [10, 1, 1, 7, 7, -1]), kwargs = {})
#   %permute_67 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_182, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_183 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_67, [10, 7, 7, -1]), kwargs = {})
#   %view_184 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [10, 49, 768]), kwargs = {})
#   %add_80 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_172, %view_184), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_80, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_80, %getitem_51), kwargs = {})
#   %add_81 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-05), kwargs = {})
#   %rsqrt_25 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_81,), kwargs = {})
#   %mul_80 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_81 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, %arg118_1), kwargs = {})
#   %add_82 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %arg119_1), kwargs = {})
#   return %getitem_51,%buf171,%add_82
triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53', '''
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
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4704, 'r0_': 6027264}, 'kernel_num_gb': 0.004526688, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex
    x0 = (xindex % 49)
    x1 = xindex // 49
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 768.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1176, 32), (37632, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1176,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 490, 768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004526688
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/nk/cnkktxrpp2jxopcbjozpiz5544hpa3tzi66t6uxzr4dnalqbyhsx.py
# Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40, view_101, spatial_mlp_windows_41, spatial_mlp_windows_42, spatial_mlp_windows_43, x_156, permute_21, x_157, x_158, x_159, layer_norm_25, x_160], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution, aten.add, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   layer_norm_25 => add_81, add_82, mul_80, mul_81, rsqrt_25, sub_25, var_mean_25
#   permute_20 => permute_64
#   permute_21 => permute_67
#   spatial_mlp_windows_40 => convolution_11
#   spatial_mlp_windows_41 => permute_66
#   spatial_mlp_windows_42 => clone_68, view_180
#   spatial_mlp_windows_43 => view_181
#   view_101 => view_179
#   windows_10 => view_175
#   x_152 => view_172
#   x_153 => add_78, add_79, mul_78, mul_79, rsqrt_24, sub_24, var_mean_24
#   x_154 => view_173
#   x_155 => view_174
#   x_156 => view_182
#   x_157 => view_183
#   x_158 => view_184
#   x_159 => add_80
#   x_160 => mm_default_3, permute_68, view_185
#   x_windows_10 => view_176
#   x_windows_heads_30 => view_177
#   x_windows_heads_31 => permute_65
#   x_windows_heads_32 => clone_67, view_178
# Graph fragment:
#   %add_82 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0" = PlaceHolder[target=add_82]
#   %arg120_1 : Tensor "f32[3072, 768][768, 1]cuda:0" = PlaceHolder[target=arg120_1]
#   %view_172 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [10, 49, 768]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_172, %getitem_49), kwargs = {})
#   %add_78 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_78,), kwargs = {})
#   %mul_78 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %arg114_1), kwargs = {})
#   %add_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %arg115_1), kwargs = {})
#   %view_173 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_79, [10, 7, 7, 768]), kwargs = {})
#   %view_174 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_173, [10, 1, 7, 1, 7, 768]), kwargs = {})
#   %permute_64 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_174, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_175 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_64, [-1, 7, 7, 768]), kwargs = {})
#   %view_176 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_175, [-1, 49, 768]), kwargs = {})
#   %view_177 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_176, [-1, 49, 24, 32]), kwargs = {})
#   %permute_65 : Tensor "f32[10, 24, 49, 32][37632, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_177, [0, 2, 1, 3]), kwargs = {})
#   %clone_67 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_65,), kwargs = {memory_format: torch.contiguous_format})
#   %view_178 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_67, [10, 1176, 32]), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_178, %arg116_1, %arg117_1, [1], [0], [1], False, [0], 24), kwargs = {})
#   %view_179 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [-1, 24, 49, 32]), kwargs = {})
#   %permute_66 : Tensor "f32[10, 49, 24, 32][37632, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_179, [0, 2, 1, 3]), kwargs = {})
#   %clone_68 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_66,), kwargs = {memory_format: torch.contiguous_format})
#   %view_180 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_68, [10, 49, 768]), kwargs = {})
#   %view_181 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_180, [-1, 7, 7, 768]), kwargs = {})
#   %view_182 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_181, [10, 1, 1, 7, 7, -1]), kwargs = {})
#   %permute_67 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_182, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_183 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_67, [10, 7, 7, -1]), kwargs = {})
#   %view_184 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [10, 49, 768]), kwargs = {})
#   %add_80 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_172, %view_184), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_80, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_80, %getitem_51), kwargs = {})
#   %add_81 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-05), kwargs = {})
#   %rsqrt_25 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_81,), kwargs = {})
#   %mul_80 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_81 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, %arg118_1), kwargs = {})
#   %add_82 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %arg119_1), kwargs = {})
#   %view_185 : Tensor "f32[490, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_82, [490, 768]), kwargs = {})
#   %permute_68 : Tensor "f32[768, 3072][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg120_1, [1, 0]), kwargs = {})
#   %mm_default_3 : Tensor "f32[490, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_185, %permute_68), kwargs = {})
#   return %mm_default_3
triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54 = async_compile.triton('triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.016963584, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 3072
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 3072*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 3072*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 192, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016963584
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/fl/cflph4dxornnn7xgvjrwkt65os6bpkkkvoq5yx4imd3rbmicnhia.py
# Topologically Sorted Source Nodes: [x_160, x_161], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   x_160 => add_tensor_3, view_186
#   x_161 => add_83, erf_10, mul_82, mul_83, mul_84
# Graph fragment:
#   %arg121_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg121_1]
#   %mm_default_3 : Tensor "f32[490, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_3]
#   %add_tensor_3 : Tensor "f32[490, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg121_1, %mm_default_3), kwargs = {})
#   %view_186 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [10, 49, 3072]), kwargs = {})
#   %mul_82 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_186, 0.5), kwargs = {})
#   %mul_83 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_186, 0.7071067811865476), kwargs = {})
#   %erf_10 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_83,), kwargs = {})
#   %add_83 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_10, 1), kwargs = {})
#   %mul_84 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %add_83), kwargs = {})
#   return %mul_84
triton_poi_fused_addmm_gelu_view_55 = async_compile.triton('triton_poi_fused_addmm_gelu_view_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 18075648}, 'kernel_num_gb': 0.012054528, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_55(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)


def get_args():
    arg_0 = rand_strided((10, 49, 3072), (150528, 3072, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1505280,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_gelu_view_55.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_addmm_gelu_view_55.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.012054528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/3s/c3swkh3vkpfrxopvyohxnifg2i7zbe4fm7audn2hlzqjzvjofxqf.py
# Topologically Sorted Source Nodes: [x_160, x_161, x_163], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
# Source node to ATen node mapping:
#   x_160 => add_tensor_3, view_186
#   x_161 => add_83, erf_10, mul_82, mul_83, mul_84
#   x_163 => mm_default_2, permute_69, view_187
# Graph fragment:
#   %mul_84 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0" = PlaceHolder[target=mul_84]
#   %arg122_1 : Tensor "f32[768, 3072][3072, 1]cuda:0" = PlaceHolder[target=arg122_1]
#   %add_tensor_3 : Tensor "f32[490, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg121_1, %mm_default_3), kwargs = {})
#   %view_186 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [10, 49, 3072]), kwargs = {})
#   %mul_82 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_186, 0.5), kwargs = {})
#   %mul_83 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_186, 0.7071067811865476), kwargs = {})
#   %erf_10 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_83,), kwargs = {})
#   %add_83 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_10, 1), kwargs = {})
#   %mul_84 : Tensor "f32[10, 49, 3072][150528, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %add_83), kwargs = {})
#   %view_187 : Tensor "f32[490, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_84, [490, 3072]), kwargs = {})
#   %permute_69 : Tensor "f32[3072, 768][1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg122_1, [1, 0]), kwargs = {})
#   %mm_default_2 : Tensor "f32[490, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_187, %permute_69), kwargs = {})
#   return %mm_default_2
triton_tem_fused_addmm_gelu_t_view_56 = async_compile.triton('triton_tem_fused_addmm_gelu_t_view_56', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_t_view_56', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.016963584, 'kernel_flop': 2312110080, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_56(arg_A, arg_B, out_ptr0):
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

    M = 490
    N = 768
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

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
        xindex = idx_n + 3072*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 3072*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def get_args():
    arg_0 = rand_strided((10, 49, 3072), (150528, 3072, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 96, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_gelu_t_view_56.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_gelu_t_view_56.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.016963584
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/6q/c6qr7qjq26ofavcwybubajaaurjclcq7uyq6a4flklhxx3sr4a4l.py
# Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40, view_101, spatial_mlp_windows_41, spatial_mlp_windows_42, spatial_mlp_windows_43, x_156, permute_21, x_157, x_158, x_159, x_163, x_165, x_166], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution, aten.add, aten.addmm]
# Source node to ATen node mapping:
#   permute_20 => permute_64
#   permute_21 => permute_67
#   spatial_mlp_windows_40 => convolution_11
#   spatial_mlp_windows_41 => permute_66
#   spatial_mlp_windows_42 => clone_68, view_180
#   spatial_mlp_windows_43 => view_181
#   view_101 => view_179
#   windows_10 => view_175
#   x_152 => view_172
#   x_153 => add_78, add_79, mul_78, mul_79, rsqrt_24, sub_24, var_mean_24
#   x_154 => view_173
#   x_155 => view_174
#   x_156 => view_182
#   x_157 => view_183
#   x_158 => view_184
#   x_159 => add_80
#   x_163 => add_tensor_2, view_188
#   x_165 => add_84
#   x_166 => var_mean_26
#   x_windows_10 => view_176
#   x_windows_heads_30 => view_177
#   x_windows_heads_31 => permute_65
#   x_windows_heads_32 => clone_67, view_178
# Graph fragment:
#   %mm_2 : Tensor "f32[490, 768][768, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %buf169 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0" = PlaceHolder[target=buf169]
#   %arg117_1 : Tensor "f32[1176][1]cuda:0" = PlaceHolder[target=arg117_1]
#   %arg123_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %mm_default_2 : Tensor "f32[490, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %add_84 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0" = PlaceHolder[target=add_84]
#   %view_172 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [10, 49, 768]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_172, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_172, %getitem_49), kwargs = {})
#   %add_78 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_78,), kwargs = {})
#   %mul_78 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %arg114_1), kwargs = {})
#   %add_79 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %arg115_1), kwargs = {})
#   %view_173 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_79, [10, 7, 7, 768]), kwargs = {})
#   %view_174 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_173, [10, 1, 7, 1, 7, 768]), kwargs = {})
#   %permute_64 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_174, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_175 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_64, [-1, 7, 7, 768]), kwargs = {})
#   %view_176 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_175, [-1, 49, 768]), kwargs = {})
#   %view_177 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_176, [-1, 49, 24, 32]), kwargs = {})
#   %permute_65 : Tensor "f32[10, 24, 49, 32][37632, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_177, [0, 2, 1, 3]), kwargs = {})
#   %clone_67 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_65,), kwargs = {memory_format: torch.contiguous_format})
#   %view_178 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_67, [10, 1176, 32]), kwargs = {})
#   %convolution_11 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_178, %arg116_1, %arg117_1, [1], [0], [1], False, [0], 24), kwargs = {})
#   %view_179 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [-1, 24, 49, 32]), kwargs = {})
#   %permute_66 : Tensor "f32[10, 49, 24, 32][37632, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_179, [0, 2, 1, 3]), kwargs = {})
#   %clone_68 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_66,), kwargs = {memory_format: torch.contiguous_format})
#   %view_180 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_68, [10, 49, 768]), kwargs = {})
#   %view_181 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_180, [-1, 7, 7, 768]), kwargs = {})
#   %view_182 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_181, [10, 1, 1, 7, 7, -1]), kwargs = {})
#   %permute_67 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_182, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_183 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_67, [10, 7, 7, -1]), kwargs = {})
#   %view_184 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [10, 49, 768]), kwargs = {})
#   %add_80 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_172, %view_184), kwargs = {})
#   %add_tensor_2 : Tensor "f32[490, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, %mm_default_2), kwargs = {})
#   %view_188 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [10, 49, 768]), kwargs = {})
#   %add_84 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %view_188), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_84, [2]), kwargs = {correction: 0, keepdim: True})
#   return %add_84,%getitem_53,%buf179
triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57 = async_compile.triton('triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57', '''
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
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12544, 'r0_': 7529472}, 'kernel_num_gb': 0.006032816, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex
    x0 = (xindex % 49)
    x1 = xindex // 49
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)


def get_args():
    arg_0 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1176, 32), (37632, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1176,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 490, 768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.006032816
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/nj/cnjcz2nu7j2yp2qgydkql4uvi3acwd7gopweieiueins7cr7u4na.py
# Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution]
# Source node to ATen node mapping:
#   permute_22 => permute_70
#   spatial_mlp_windows_44 => convolution_12
#   windows_11 => view_191
#   x_166 => add_85, add_86, mul_85, mul_86, rsqrt_26, sub_26, var_mean_26
#   x_167 => view_189
#   x_168 => view_190
#   x_windows_11 => view_192
#   x_windows_heads_33 => view_193
#   x_windows_heads_34 => permute_71
#   x_windows_heads_35 => clone_71, view_194
# Graph fragment:
#   %add_84 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0" = PlaceHolder[target=add_84]
#   %getitem_53 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=getitem_53]
#   %buf179 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=buf179]
#   %arg124_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg124_1]
#   %arg125_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_84, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_84, %getitem_53), kwargs = {})
#   %add_85 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-05), kwargs = {})
#   %rsqrt_26 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_85,), kwargs = {})
#   %mul_85 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_26), kwargs = {})
#   %mul_86 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %arg124_1), kwargs = {})
#   %add_86 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %arg125_1), kwargs = {})
#   %view_189 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_86, [10, 7, 7, 768]), kwargs = {})
#   %view_190 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_189, [10, 1, 7, 1, 7, 768]), kwargs = {})
#   %permute_70 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_190, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_191 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_70, [-1, 7, 7, 768]), kwargs = {})
#   %view_192 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_191, [-1, 49, 768]), kwargs = {})
#   %view_193 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_192, [-1, 49, 24, 32]), kwargs = {})
#   %permute_71 : Tensor "f32[10, 24, 49, 32][37632, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_193, [0, 2, 1, 3]), kwargs = {})
#   %clone_71 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_71,), kwargs = {memory_format: torch.contiguous_format})
#   %view_194 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_71, [10, 1176, 32]), kwargs = {})
#   %convolution_12 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_194, %arg126_1, %arg127_1, [1], [0], [1], False, [0], 24), kwargs = {})
#   return %buf181
triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58 = async_compile.triton('triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4521984}, 'kernel_num_gb': 0.003020624, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 1176)
    x2 = xindex // 37632
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)


def get_args():
    arg_0 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1176, 32), (37632, 32, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 376320,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.003020624
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/z3/cz35fcojs7duuj7cj7insuigk32z7hzcgloei4by5lkaumv3f2n7.py
# Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44, view_110, spatial_mlp_windows_45, spatial_mlp_windows_46, spatial_mlp_windows_47, x_169, permute_23, x_170, x_171, x_172, layer_norm_27], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   layer_norm_27 => add_88, add_89, mul_87, mul_88, rsqrt_27, sub_27, var_mean_27
#   permute_22 => permute_70
#   permute_23 => permute_73
#   spatial_mlp_windows_44 => convolution_12
#   spatial_mlp_windows_45 => permute_72
#   spatial_mlp_windows_46 => clone_72, view_196
#   spatial_mlp_windows_47 => view_197
#   view_110 => view_195
#   windows_11 => view_191
#   x_166 => add_85, add_86, mul_85, mul_86, rsqrt_26, sub_26, var_mean_26
#   x_167 => view_189
#   x_168 => view_190
#   x_169 => view_198
#   x_170 => view_199
#   x_171 => view_200
#   x_172 => add_87
#   x_windows_11 => view_192
#   x_windows_heads_33 => view_193
#   x_windows_heads_34 => permute_71
#   x_windows_heads_35 => clone_71, view_194
# Graph fragment:
#   %add_84 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0" = PlaceHolder[target=add_84]
#   %buf182 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0" = PlaceHolder[target=buf182]
#   %arg127_1 : Tensor "f32[1176][1]cuda:0" = PlaceHolder[target=arg127_1]
#   %getitem_55 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=getitem_55]
#   %buf184 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=buf184]
#   %arg128_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg128_1]
#   %arg129_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_84, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_84, %getitem_53), kwargs = {})
#   %add_85 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-05), kwargs = {})
#   %rsqrt_26 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_85,), kwargs = {})
#   %mul_85 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_26), kwargs = {})
#   %mul_86 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %arg124_1), kwargs = {})
#   %add_86 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %arg125_1), kwargs = {})
#   %view_189 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_86, [10, 7, 7, 768]), kwargs = {})
#   %view_190 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_189, [10, 1, 7, 1, 7, 768]), kwargs = {})
#   %permute_70 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 5376, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_190, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_191 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_70, [-1, 7, 7, 768]), kwargs = {})
#   %view_192 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_191, [-1, 49, 768]), kwargs = {})
#   %view_193 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_192, [-1, 49, 24, 32]), kwargs = {})
#   %permute_71 : Tensor "f32[10, 24, 49, 32][37632, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_193, [0, 2, 1, 3]), kwargs = {})
#   %clone_71 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_71,), kwargs = {memory_format: torch.contiguous_format})
#   %view_194 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_71, [10, 1176, 32]), kwargs = {})
#   %convolution_12 : Tensor "f32[10, 1176, 32][37632, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_194, %arg126_1, %arg127_1, [1], [0], [1], False, [0], 24), kwargs = {})
#   %view_195 : Tensor "f32[10, 24, 49, 32][37632, 1568, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_12, [-1, 24, 49, 32]), kwargs = {})
#   %permute_72 : Tensor "f32[10, 49, 24, 32][37632, 32, 1568, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_195, [0, 2, 1, 3]), kwargs = {})
#   %clone_72 : Tensor "f32[10, 49, 24, 32][37632, 768, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_196 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_72, [10, 49, 768]), kwargs = {})
#   %view_197 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_196, [-1, 7, 7, 768]), kwargs = {})
#   %view_198 : Tensor "f32[10, 1, 1, 7, 7, 768][37632, 37632, 37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_197, [10, 1, 1, 7, 7, -1]), kwargs = {})
#   %permute_73 : Tensor "f32[10, 1, 7, 1, 7, 768][37632, 37632, 5376, 37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_198, [0, 1, 3, 2, 4, 5]), kwargs = {})
#   %view_199 : Tensor "f32[10, 7, 7, 768][37632, 5376, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_73, [10, 7, 7, -1]), kwargs = {})
#   %view_200 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_199, [10, 49, 768]), kwargs = {})
#   %add_87 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %view_200), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_87, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_87, %getitem_55), kwargs = {})
#   %add_88 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-05), kwargs = {})
#   %rsqrt_27 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_88,), kwargs = {})
#   %mul_87 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %mul_88 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %arg128_1), kwargs = {})
#   %add_89 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %arg129_1), kwargs = {})
#   return %getitem_55,%buf184,%add_89
triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59', '''
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
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4704, 'r0_': 6027264}, 'kernel_num_gb': 0.004526688, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex
    x0 = (xindex % 49)
    x1 = xindex // 49
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 768.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1176, 32), (37632, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1176,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 490, 768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.004526688
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/aa/caalca23ky3ideitjgfezcok34b624gqb7mpn2ada63h6sawr7vu.py
# Topologically Sorted Source Nodes: [x_179, transpose_25, x_180], Original ATen: [aten.native_layer_norm, aten.transpose, aten.unsqueeze, aten.mean]
# Source node to ATen node mapping:
#   transpose_25 => permute_76
#   x_179 => add_92, add_93, mul_92, mul_93, rsqrt_28, sub_28, var_mean_28
#   x_180 => mean, unsqueeze
# Graph fragment:
#   %add_91 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0" = PlaceHolder[target=add_91]
#   %getitem_57 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=getitem_57]
#   %buf192 : Tensor "f32[10, 49, 1][49, 1, 490]cuda:0" = PlaceHolder[target=buf192]
#   %arg134_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg134_1]
#   %arg135_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg135_1]
#   %buf194 : Tensor "f32[10, 768, 1, 1][768, 1, 7680, 7680]cuda:0" = PlaceHolder[target=buf194]
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_91, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_91, %getitem_57), kwargs = {})
#   %add_92 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_56, 1e-05), kwargs = {})
#   %rsqrt_28 : Tensor "f32[10, 49, 1][49, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_92,), kwargs = {})
#   %mul_92 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_28), kwargs = {})
#   %mul_93 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_92, %arg134_1), kwargs = {})
#   %add_93 : Tensor "f32[10, 49, 768][37632, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %arg135_1), kwargs = {})
#   %permute_76 : Tensor "f32[10, 768, 49][37632, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_93, [0, 2, 1]), kwargs = {})
#   %unsqueeze : Tensor "f32[10, 768, 1, 49][37632, 1, 37632, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_76, -2), kwargs = {})
#   %mean : Tensor "f32[10, 768, 1, 1][768, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%unsqueeze, [-1, -2], True), kwargs = {})
#   return %buf194,%mean
triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60 = async_compile.triton('triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60', '''
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
    size_hints={'x': 8192, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1572864, 'r0_': 3920}, 'kernel_num_gb': 0.001576784, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7680
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
    x0 = (xindex % 768)
    x1 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 37632*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 49*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 49*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.full([1, 1], 49.0, tl.float32)
    tmp19 = (tmp17 / tmp18)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((10, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 7680, 49,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.001576784
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ih/cihvm4ppuvmkbxyy6ifbccq565lhogp2c4g7xdaw5vhk7batntd4.py
# Topologically Sorted Source Nodes: [x_180, x_181, x_182], Original ATen: [aten.as_strided, aten.squeeze, aten.view, aten.addmm, aten.t]
# Source node to ATen node mapping:
#   x_180 => as_strided, squeeze
#   x_181 => view_205
#   x_182 => addmm_default, constant_pad_nd_default, permute_77
# Graph fragment:
#   %mean : Tensor "f32[10, 768, 1, 1][768, 1, 1, 1]cuda:0" = PlaceHolder[target=mean]
#   %arg137_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg137_1]
#   %constant_pad_nd_default : Tensor "f32[12, 768][768, 1]cuda:0" = PlaceHolder[target=constant_pad_nd_default]
#   %arg136_1 : Tensor "f32[1000, 768][768, 1]cuda:0" = PlaceHolder[target=arg136_1]
#   %as_strided : Tensor "f32[10, 768, 1, 1][768, 1, 768, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.as_strided.default](args = (%mean, [10, 768, 1, 1], [768, 1, 768, 768]), kwargs = {})
#   %squeeze : Tensor "f32[10, 768, 1][768, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%as_strided, -2), kwargs = {})
#   %view_205 : Tensor "f32[10, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%squeeze, [10, 768]), kwargs = {})
#   %constant_pad_nd_default : Tensor "f32[12, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_205, [0, 0, 0, 2]), kwargs = {})
#   %permute_77 : Tensor "f32[768, 1000][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg136_1, [1, 0]), kwargs = {})
#   %addmm_default : Tensor "f32[12, 1000][1000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg137_1, %constant_pad_nd_default, %permute_77), kwargs = {})
#   return %constant_pad_nd_default,%addmm_default
triton_tem_fused_addmm_as_strided_squeeze_t_view_61 = async_compile.triton('triton_tem_fused_addmm_as_strided_squeeze_t_view_61', '''
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
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_as_strided_squeeze_t_view_61', 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.003204864, 'kernel_flop': 18432000, 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused_addmm_as_strided_squeeze_t_view_61(in_ptr0, arg_B, in_ptr2, out_ptr0):
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
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        tmp2 = tl.broadcast_to(idx_m, [BLOCK_M, BLOCK_K])
        tmp3 = tl.full([1], 10, tl.int64)
        tmp4 = tmp2 < tmp3
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 768*idx_m, [BLOCK_M, BLOCK_K])), tmp4, eviction_policy='evict_last', other=0.0)
        a = tmp5.broadcast_to(xindex.shape)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((12, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 32, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_as_strided_squeeze_t_view_61.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_addmm_as_strided_squeeze_t_view_61.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.003204864
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1 = args
        args.clear()
        assert_size_stride(arg0_1, (10, 3, 224, 224), (150528, 50176, 224, 1))
        assert_size_stride(arg1_1, (96, 3, 4, 4), (48, 16, 4, 1))
        assert_size_stride(arg2_1, (96, ), (1, ))
        assert_size_stride(arg3_1, (96, ), (1, ))
        assert_size_stride(arg4_1, (96, ), (1, ))
        assert_size_stride(arg5_1, (96, ), (1, ))
        assert_size_stride(arg6_1, (96, ), (1, ))
        assert_size_stride(arg7_1, (147, 49, 1), (49, 1, 1))
        assert_size_stride(arg8_1, (147, ), (1, ))
        assert_size_stride(arg9_1, (96, ), (1, ))
        assert_size_stride(arg10_1, (96, ), (1, ))
        assert_size_stride(arg11_1, (384, 96), (96, 1))
        assert_size_stride(arg12_1, (384, ), (1, ))
        assert_size_stride(arg13_1, (96, 384), (384, 1))
        assert_size_stride(arg14_1, (96, ), (1, ))
        assert_size_stride(arg15_1, (96, ), (1, ))
        assert_size_stride(arg16_1, (96, ), (1, ))
        assert_size_stride(arg17_1, (147, 49, 1), (49, 1, 1))
        assert_size_stride(arg18_1, (147, ), (1, ))
        assert_size_stride(arg19_1, (96, ), (1, ))
        assert_size_stride(arg20_1, (96, ), (1, ))
        assert_size_stride(arg21_1, (384, 96), (96, 1))
        assert_size_stride(arg22_1, (384, ), (1, ))
        assert_size_stride(arg23_1, (96, 384), (384, 1))
        assert_size_stride(arg24_1, (96, ), (1, ))
        assert_size_stride(arg25_1, (384, ), (1, ))
        assert_size_stride(arg26_1, (384, ), (1, ))
        assert_size_stride(arg27_1, (192, 384), (384, 1))
        assert_size_stride(arg28_1, (192, ), (1, ))
        assert_size_stride(arg29_1, (192, ), (1, ))
        assert_size_stride(arg30_1, (294, 49, 1), (49, 1, 1))
        assert_size_stride(arg31_1, (294, ), (1, ))
        assert_size_stride(arg32_1, (192, ), (1, ))
        assert_size_stride(arg33_1, (192, ), (1, ))
        assert_size_stride(arg34_1, (768, 192), (192, 1))
        assert_size_stride(arg35_1, (768, ), (1, ))
        assert_size_stride(arg36_1, (192, 768), (768, 1))
        assert_size_stride(arg37_1, (192, ), (1, ))
        assert_size_stride(arg38_1, (192, ), (1, ))
        assert_size_stride(arg39_1, (192, ), (1, ))
        assert_size_stride(arg40_1, (294, 49, 1), (49, 1, 1))
        assert_size_stride(arg41_1, (294, ), (1, ))
        assert_size_stride(arg42_1, (192, ), (1, ))
        assert_size_stride(arg43_1, (192, ), (1, ))
        assert_size_stride(arg44_1, (768, 192), (192, 1))
        assert_size_stride(arg45_1, (768, ), (1, ))
        assert_size_stride(arg46_1, (192, 768), (768, 1))
        assert_size_stride(arg47_1, (192, ), (1, ))
        assert_size_stride(arg48_1, (768, ), (1, ))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (384, 768), (768, 1))
        assert_size_stride(arg51_1, (384, ), (1, ))
        assert_size_stride(arg52_1, (384, ), (1, ))
        assert_size_stride(arg53_1, (588, 49, 1), (49, 1, 1))
        assert_size_stride(arg54_1, (588, ), (1, ))
        assert_size_stride(arg55_1, (384, ), (1, ))
        assert_size_stride(arg56_1, (384, ), (1, ))
        assert_size_stride(arg57_1, (1536, 384), (384, 1))
        assert_size_stride(arg58_1, (1536, ), (1, ))
        assert_size_stride(arg59_1, (384, 1536), (1536, 1))
        assert_size_stride(arg60_1, (384, ), (1, ))
        assert_size_stride(arg61_1, (384, ), (1, ))
        assert_size_stride(arg62_1, (384, ), (1, ))
        assert_size_stride(arg63_1, (588, 49, 1), (49, 1, 1))
        assert_size_stride(arg64_1, (588, ), (1, ))
        assert_size_stride(arg65_1, (384, ), (1, ))
        assert_size_stride(arg66_1, (384, ), (1, ))
        assert_size_stride(arg67_1, (1536, 384), (384, 1))
        assert_size_stride(arg68_1, (1536, ), (1, ))
        assert_size_stride(arg69_1, (384, 1536), (1536, 1))
        assert_size_stride(arg70_1, (384, ), (1, ))
        assert_size_stride(arg71_1, (384, ), (1, ))
        assert_size_stride(arg72_1, (384, ), (1, ))
        assert_size_stride(arg73_1, (588, 49, 1), (49, 1, 1))
        assert_size_stride(arg74_1, (588, ), (1, ))
        assert_size_stride(arg75_1, (384, ), (1, ))
        assert_size_stride(arg76_1, (384, ), (1, ))
        assert_size_stride(arg77_1, (1536, 384), (384, 1))
        assert_size_stride(arg78_1, (1536, ), (1, ))
        assert_size_stride(arg79_1, (384, 1536), (1536, 1))
        assert_size_stride(arg80_1, (384, ), (1, ))
        assert_size_stride(arg81_1, (384, ), (1, ))
        assert_size_stride(arg82_1, (384, ), (1, ))
        assert_size_stride(arg83_1, (588, 49, 1), (49, 1, 1))
        assert_size_stride(arg84_1, (588, ), (1, ))
        assert_size_stride(arg85_1, (384, ), (1, ))
        assert_size_stride(arg86_1, (384, ), (1, ))
        assert_size_stride(arg87_1, (1536, 384), (384, 1))
        assert_size_stride(arg88_1, (1536, ), (1, ))
        assert_size_stride(arg89_1, (384, 1536), (1536, 1))
        assert_size_stride(arg90_1, (384, ), (1, ))
        assert_size_stride(arg91_1, (384, ), (1, ))
        assert_size_stride(arg92_1, (384, ), (1, ))
        assert_size_stride(arg93_1, (588, 49, 1), (49, 1, 1))
        assert_size_stride(arg94_1, (588, ), (1, ))
        assert_size_stride(arg95_1, (384, ), (1, ))
        assert_size_stride(arg96_1, (384, ), (1, ))
        assert_size_stride(arg97_1, (1536, 384), (384, 1))
        assert_size_stride(arg98_1, (1536, ), (1, ))
        assert_size_stride(arg99_1, (384, 1536), (1536, 1))
        assert_size_stride(arg100_1, (384, ), (1, ))
        assert_size_stride(arg101_1, (384, ), (1, ))
        assert_size_stride(arg102_1, (384, ), (1, ))
        assert_size_stride(arg103_1, (588, 49, 1), (49, 1, 1))
        assert_size_stride(arg104_1, (588, ), (1, ))
        assert_size_stride(arg105_1, (384, ), (1, ))
        assert_size_stride(arg106_1, (384, ), (1, ))
        assert_size_stride(arg107_1, (1536, 384), (384, 1))
        assert_size_stride(arg108_1, (1536, ), (1, ))
        assert_size_stride(arg109_1, (384, 1536), (1536, 1))
        assert_size_stride(arg110_1, (384, ), (1, ))
        assert_size_stride(arg111_1, (1536, ), (1, ))
        assert_size_stride(arg112_1, (1536, ), (1, ))
        assert_size_stride(arg113_1, (768, 1536), (1536, 1))
        assert_size_stride(arg114_1, (768, ), (1, ))
        assert_size_stride(arg115_1, (768, ), (1, ))
        assert_size_stride(arg116_1, (1176, 49, 1), (49, 1, 1))
        assert_size_stride(arg117_1, (1176, ), (1, ))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (3072, 768), (768, 1))
        assert_size_stride(arg121_1, (3072, ), (1, ))
        assert_size_stride(arg122_1, (768, 3072), (3072, 1))
        assert_size_stride(arg123_1, (768, ), (1, ))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (768, ), (1, ))
        assert_size_stride(arg126_1, (1176, 49, 1), (49, 1, 1))
        assert_size_stride(arg127_1, (1176, ), (1, ))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (3072, 768), (768, 1))
        assert_size_stride(arg131_1, (3072, ), (1, ))
        assert_size_stride(arg132_1, (768, 3072), (3072, 1))
        assert_size_stride(arg133_1, (768, ), (1, ))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (1000, 768), (768, 1))
        assert_size_stride(arg137_1, (1000, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (10, 96, 56, 56), (301056, 3136, 56, 1), 'unknown_op')
            del arg0_1
            del arg1_1
            buf4 = reinterpret_tensor(buf0, (10, 3136, 96), (301056, 1, 3136), 0); del buf0  # reuse
            buf5 = empty_strided_cuda((10, 3136, 1), (3136, 1, 31360), torch.float32)
            buf6 = empty_strided_cuda((10, 3136, 1), (3136, 1, 31360), torch.float32)
            # Topologically Sorted Source Nodes: [conv2d, flatten, x, x_1, x_3], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_layer_norm_transpose_view_0.run(buf4, arg2_1, arg3_1, arg4_1, buf5, buf6, 31360, 96, stream=stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            buf8 = empty_strided_cuda((640, 147, 32), (4704, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1.run(buf4, buf5, buf6, arg5_1, arg6_1, buf8, 94080, 32, stream=stream0)
            del arg5_1
            del arg6_1
            # Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf9 = extern_kernels.convolution(buf8, arg7_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=3, bias=None)
            assert_size_stride(buf9, (640, 147, 32), (4704, 32, 1), 'torch.ops.aten.convolution.default')
            del arg7_1
            buf13 = reinterpret_tensor(buf8, (10, 3136, 96), (301056, 96, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows, view_5, spatial_mlp_windows_1, spatial_mlp_windows_2, spatial_mlp_windows_3, x_6, permute_1, contiguous_1, x_7, x_8, x_9, layer_norm_2], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2.run(buf4, buf9, arg8_1, arg9_1, arg10_1, buf13, 31360, 96, stream=stream0)
            del arg10_1
            del arg9_1
            buf14 = empty_strided_cuda((31360, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows, view_5, spatial_mlp_windows_1, spatial_mlp_windows_2, spatial_mlp_windows_3, x_6, permute_1, contiguous_1, x_7, x_8, x_9, layer_norm_2, x_10], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3.run(buf13, arg11_1, buf14, 1470, 1, 1, stream=stream0)
            del arg11_1
            del buf13
            buf15 = reinterpret_tensor(buf14, (10, 3136, 384), (1204224, 384, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_4.run(buf15, arg12_1, 12042240, stream=stream0)
            del arg12_1
            buf16 = empty_strided_cuda((31360, 96), (96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_10, x_11, x_13], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_5.run(buf15, arg13_1, buf16, 245, 1, 1, stream=stream0)
            del arg13_1
            buf17 = reinterpret_tensor(buf16, (10, 3136, 96), (301056, 96, 1), 0); del buf16  # reuse
            buf18 = buf6; del buf6  # reuse
            buf19 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_4, x_5, permute, contiguous, windows, x_windows, x_windows_heads, x_windows_heads_1, x_windows_heads_2, spatial_mlp_windows, view_5, spatial_mlp_windows_1, spatial_mlp_windows_2, spatial_mlp_windows_3, x_6, permute_1, contiguous_1, x_7, x_8, x_9, x_13, x_15, x_16], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6.run(buf17, buf4, buf9, arg8_1, arg14_1, buf18, buf19, 31360, 96, stream=stream0)
            del arg14_1
            del arg8_1
            del buf4
            buf21 = empty_strided_cuda((810, 147, 32), (4704, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7.run(buf17, buf18, buf19, arg15_1, arg16_1, buf21, 3810240, stream=stream0)
            del arg15_1
            del arg16_1
            del buf18
            del buf19
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf22 = extern_kernels.convolution(buf21, arg17_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=3, bias=None)
            assert_size_stride(buf22, (810, 147, 32), (4704, 32, 1), 'torch.ops.aten.convolution.default')
            del arg17_1
            del buf21
            buf26 = reinterpret_tensor(buf9, (10, 3136, 96), (301056, 96, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, layer_norm_4], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8.run(buf17, buf22, arg18_1, arg19_1, arg20_1, buf26, 31360, 96, stream=stream0)
            del arg19_1
            del arg20_1
            buf27 = reinterpret_tensor(buf15, (31360, 384), (384, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, layer_norm_4, x_24], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3.run(buf26, arg21_1, buf27, 1470, 1, 1, stream=stream0)
            del arg21_1
            del buf26
            buf28 = reinterpret_tensor(buf27, (10, 3136, 384), (1204224, 384, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_4.run(buf28, arg22_1, 12042240, stream=stream0)
            del arg22_1
            buf29 = empty_strided_cuda((31360, 96), (96, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_24, x_25, x_27], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_5.run(buf28, arg23_1, buf29, 245, 1, 1, stream=stream0)
            del arg23_1
            del buf28
            buf30 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9.run(buf30, buf22, arg18_1, arg24_1, buf29, 3010560, stream=stream0)
            del arg18_1
            del arg24_1
            del buf22
            buf35 = reinterpret_tensor(buf29, (10, 28, 28, 384), (301056, 10752, 384, 1), 0); del buf29  # reuse
            buf31 = reinterpret_tensor(buf35, (10, 28, 28, 96), (301056, 10752, 384, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10.run(buf30, buf31, 752640, stream=stream0)
            buf32 = reinterpret_tensor(buf35, (10, 28, 28, 96), (301056, 10752, 384, 1), 96)  # alias
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11.run(buf30, buf32, 752640, stream=stream0)
            buf33 = reinterpret_tensor(buf35, (10, 28, 28, 96), (301056, 10752, 384, 1), 192)  # alias
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12.run(buf30, buf33, 752640, stream=stream0)
            buf34 = reinterpret_tensor(buf35, (10, 28, 28, 96), (301056, 10752, 384, 1), 288)  # alias
            # Topologically Sorted Source Nodes: [x_16, x_17, shifted_x, x_18, permute_2, contiguous_2, windows_1, x_windows_1, x_windows_heads_3, x_windows_heads_4, x_windows_heads_5, spatial_mlp_windows_4, view_14, spatial_mlp_windows_5, spatial_mlp_windows_6, spatial_mlp_windows_7, x_19, permute_3, contiguous_3, x_20, getitem, x_21, x_22, x_23, x_27, x_29, x_30, x0, x1, x2, x3, x_31], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13.run(buf30, buf34, 752640, stream=stream0)
            buf39 = reinterpret_tensor(buf30, (10, 784, 384), (301056, 384, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_view_14.run(buf35, arg25_1, arg26_1, buf39, 7840, 384, stream=stream0)
            del arg25_1
            del arg26_1
            del buf31
            del buf32
            del buf33
            del buf34
            del buf35
            buf40 = empty_strided_cuda((7840, 192), (192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_32, x_33, x_34], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_native_layer_norm_t_view_15.run(buf39, arg27_1, buf40, 246, 1, 1, stream=stream0)
            del arg27_1
            buf41 = empty_strided_cuda((10, 784, 1), (784, 1, 7840), torch.float32)
            buf42 = empty_strided_cuda((10, 784, 1), (784, 1, 7840), torch.float32)
            # Topologically Sorted Source Nodes: [x_34, x_35], Original ATen: [aten._unsafe_view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_native_layer_norm_16.run(buf40, buf41, buf42, 7840, 192, stream=stream0)
            buf44 = empty_strided_cuda((160, 294, 32), (9408, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17.run(buf40, buf41, buf42, arg28_1, arg29_1, buf44, 1505280, stream=stream0)
            del arg28_1
            del arg29_1
            # Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution]
            buf45 = extern_kernels.convolution(buf44, arg30_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=6, bias=None)
            assert_size_stride(buf45, (160, 294, 32), (9408, 32, 1), 'torch.ops.aten.convolution.default')
            del arg30_1
            buf49 = reinterpret_tensor(buf44, (10, 784, 192), (150528, 192, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8, view_25, spatial_mlp_windows_9, spatial_mlp_windows_10, spatial_mlp_windows_11, x_38, permute_5, contiguous_6, x_39, x_40, x_41, layer_norm_7], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18.run(buf40, buf45, arg31_1, arg32_1, arg33_1, buf49, 7840, 192, stream=stream0)
            del arg32_1
            del arg33_1
            buf50 = empty_strided_cuda((7840, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8, view_25, spatial_mlp_windows_9, spatial_mlp_windows_10, spatial_mlp_windows_11, x_38, permute_5, contiguous_6, x_39, x_40, x_41, layer_norm_7, x_42], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19.run(buf49, arg34_1, buf50, 738, 1, 1, stream=stream0)
            del arg34_1
            buf51 = reinterpret_tensor(buf50, (10, 784, 768), (602112, 768, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_43], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_20.run(buf51, arg35_1, 6021120, stream=stream0)
            del arg35_1
            buf52 = reinterpret_tensor(buf49, (7840, 192), (192, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_43, x_45], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_21.run(buf51, arg36_1, buf52, 246, 1, 1, stream=stream0)
            del arg36_1
            buf53 = reinterpret_tensor(buf40, (10, 784, 192), (150528, 192, 1), 0); del buf40  # reuse
            buf54 = buf42; del buf42  # reuse
            buf55 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [x_34, x_35, x_36, x_37, permute_4, contiguous_5, windows_2, x_windows_2, x_windows_heads_6, x_windows_heads_7, x_windows_heads_8, spatial_mlp_windows_8, view_25, spatial_mlp_windows_9, spatial_mlp_windows_10, spatial_mlp_windows_11, x_38, permute_5, contiguous_6, x_39, x_40, x_41, x_45, x_47, x_48], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22.run(buf53, buf45, arg31_1, arg37_1, buf52, buf54, buf55, 7840, 192, stream=stream0)
            del arg31_1
            del arg37_1
            del buf45
            buf57 = empty_strided_cuda((250, 294, 32), (9408, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23.run(buf53, buf54, buf55, arg38_1, arg39_1, buf57, 2352000, stream=stream0)
            del arg38_1
            del arg39_1
            del buf54
            del buf55
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf58 = extern_kernels.convolution(buf57, arg40_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=6, bias=None)
            assert_size_stride(buf58, (250, 294, 32), (9408, 32, 1), 'torch.ops.aten.convolution.default')
            del arg40_1
            del buf57
            buf62 = reinterpret_tensor(buf52, (10, 784, 192), (150528, 192, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, layer_norm_9], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24.run(buf53, buf58, arg41_1, arg42_1, arg43_1, buf62, 7840, 192, stream=stream0)
            del arg42_1
            del arg43_1
            buf63 = reinterpret_tensor(buf51, (7840, 768), (768, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, layer_norm_9, x_56], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19.run(buf62, arg44_1, buf63, 738, 1, 1, stream=stream0)
            del arg44_1
            buf64 = reinterpret_tensor(buf63, (10, 784, 768), (602112, 768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [x_56, x_57], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_20.run(buf64, arg45_1, 6021120, stream=stream0)
            del arg45_1
            buf65 = reinterpret_tensor(buf62, (7840, 192), (192, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [x_56, x_57, x_59], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_21.run(buf64, arg46_1, buf65, 246, 1, 1, stream=stream0)
            del arg46_1
            del buf64
            buf66 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25.run(buf66, buf58, arg41_1, arg47_1, buf65, 1505280, stream=stream0)
            del arg41_1
            del arg47_1
            del buf58
            buf71 = reinterpret_tensor(buf65, (10, 14, 14, 768), (150528, 10752, 768, 1), 0); del buf65  # reuse
            buf67 = reinterpret_tensor(buf71, (10, 14, 14, 192), (150528, 10752, 768, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26.run(buf66, buf67, 376320, stream=stream0)
            buf68 = reinterpret_tensor(buf71, (10, 14, 14, 192), (150528, 10752, 768, 1), 192)  # alias
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27.run(buf66, buf68, 376320, stream=stream0)
            buf69 = reinterpret_tensor(buf71, (10, 14, 14, 192), (150528, 10752, 768, 1), 384)  # alias
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28.run(buf66, buf69, 376320, stream=stream0)
            buf70 = reinterpret_tensor(buf71, (10, 14, 14, 192), (150528, 10752, 768, 1), 576)  # alias
            # Topologically Sorted Source Nodes: [x_48, x_49, shifted_x_1, x_50, permute_6, contiguous_7, windows_3, x_windows_3, x_windows_heads_9, x_windows_heads_10, x_windows_heads_11, spatial_mlp_windows_12, view_34, spatial_mlp_windows_13, spatial_mlp_windows_14, spatial_mlp_windows_15, x_51, permute_7, contiguous_8, x_52, getitem_5, x_53, x_54, x_55, x_59, x_61, x_62, x0_1, x1_1, x2_1, x3_1, x_63], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29.run(buf66, buf70, 376320, stream=stream0)
            buf75 = reinterpret_tensor(buf66, (10, 196, 768), (150528, 768, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [x_64, x_65], Original ATen: [aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_view_30.run(buf71, arg48_1, arg49_1, buf75, 1960, 768, stream=stream0)
            del arg48_1
            del arg49_1
            del buf67
            del buf68
            del buf69
            del buf70
            del buf71
            buf76 = empty_strided_cuda((1960, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_64, x_65, x_66], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_native_layer_norm_t_view_31.run(buf75, arg50_1, buf76, 93, 1, 1, stream=stream0)
            del arg50_1
            buf77 = empty_strided_cuda((10, 196, 1), (196, 1, 1960), torch.float32)
            buf78 = empty_strided_cuda((10, 196, 1), (196, 1, 1984), torch.float32)
            # Topologically Sorted Source Nodes: [x_66, x_67], Original ATen: [aten._unsafe_view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_native_layer_norm_32.run(buf76, buf77, buf78, 1960, 384, stream=stream0)
            buf80 = empty_strided_cuda((40, 588, 32), (18816, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33.run(buf76, buf77, buf78, arg51_1, arg52_1, buf80, 752640, stream=stream0)
            del arg51_1
            del arg52_1
            # Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution]
            buf81 = extern_kernels.convolution(buf80, arg53_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=12, bias=None)
            assert_size_stride(buf81, (40, 588, 32), (18816, 32, 1), 'torch.ops.aten.convolution.default')
            del arg53_1
            buf85 = reinterpret_tensor(buf80, (10, 196, 384), (75264, 384, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16, view_45, spatial_mlp_windows_17, spatial_mlp_windows_18, spatial_mlp_windows_19, x_70, permute_9, contiguous_11, x_71, x_72, x_73, layer_norm_12], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34.run(buf76, buf81, arg54_1, arg55_1, arg56_1, buf85, 1960, 384, stream=stream0)
            del arg55_1
            del arg56_1
            buf86 = reinterpret_tensor(buf39, (1960, 1536), (1536, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16, view_45, spatial_mlp_windows_17, spatial_mlp_windows_18, spatial_mlp_windows_19, x_70, permute_9, contiguous_11, x_71, x_72, x_73, layer_norm_12, x_74], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(buf85, arg57_1, buf86, 372, 1, 1, stream=stream0)
            del arg57_1
            buf87 = reinterpret_tensor(buf86, (10, 196, 1536), (301056, 1536, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [x_74, x_75], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_36.run(buf87, arg58_1, 3010560, stream=stream0)
            del arg58_1
            buf88 = reinterpret_tensor(buf85, (1960, 384), (384, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [x_74, x_75, x_77], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_37.run(buf87, arg59_1, buf88, 93, 1, 1, stream=stream0)
            del arg59_1
            buf89 = reinterpret_tensor(buf76, (10, 196, 384), (75264, 384, 1), 0); del buf76  # reuse
            buf90 = reinterpret_tensor(buf78, (10, 196, 1), (196, 1, 1960), 0); del buf78  # reuse
            buf91 = reinterpret_tensor(buf77, (10, 196, 1), (196, 1, 1984), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [x_66, x_67, x_68, x_69, permute_8, contiguous_10, windows_4, x_windows_4, x_windows_heads_12, x_windows_heads_13, x_windows_heads_14, spatial_mlp_windows_16, view_45, spatial_mlp_windows_17, spatial_mlp_windows_18, spatial_mlp_windows_19, x_70, permute_9, contiguous_11, x_71, x_72, x_73, x_77, x_79, x_80], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38.run(buf89, buf81, arg54_1, arg60_1, buf88, buf90, buf91, 1960, 384, stream=stream0)
            del arg54_1
            del arg60_1
            del buf81
            buf93 = empty_strided_cuda((90, 588, 32), (18816, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39.run(buf89, buf90, buf91, arg61_1, arg62_1, buf93, 1693440, stream=stream0)
            del arg61_1
            del arg62_1
            # Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf94 = extern_kernels.convolution(buf93, arg63_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=12, bias=None)
            assert_size_stride(buf94, (90, 588, 32), (18816, 32, 1), 'torch.ops.aten.convolution.default')
            del arg63_1
            del buf93
            buf98 = reinterpret_tensor(buf88, (10, 196, 384), (75264, 384, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20, view_54, spatial_mlp_windows_21, spatial_mlp_windows_22, spatial_mlp_windows_23, x_83, permute_11, contiguous_13, x_84, getitem_10, x_85, x_86, x_87, layer_norm_14], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40.run(buf89, buf94, arg64_1, arg65_1, arg66_1, buf98, 1960, 384, stream=stream0)
            del arg65_1
            del arg66_1
            buf99 = reinterpret_tensor(buf87, (1960, 1536), (1536, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20, view_54, spatial_mlp_windows_21, spatial_mlp_windows_22, spatial_mlp_windows_23, x_83, permute_11, contiguous_13, x_84, getitem_10, x_85, x_86, x_87, layer_norm_14, x_88], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(buf98, arg67_1, buf99, 372, 1, 1, stream=stream0)
            del arg67_1
            buf100 = reinterpret_tensor(buf99, (10, 196, 1536), (301056, 1536, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [x_88, x_89], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_36.run(buf100, arg68_1, 3010560, stream=stream0)
            del arg68_1
            buf101 = reinterpret_tensor(buf98, (1960, 384), (384, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [x_88, x_89, x_91], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_37.run(buf100, arg69_1, buf101, 93, 1, 1, stream=stream0)
            del arg69_1
            buf102 = buf89; del buf89  # reuse
            buf103 = reinterpret_tensor(buf91, (10, 196, 1), (196, 1, 1960), 0); del buf91  # reuse
            buf104 = reinterpret_tensor(buf90, (10, 196, 1), (196, 1, 1984), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [x_80, x_81, shifted_x_2, x_82, permute_10, contiguous_12, windows_5, x_windows_5, x_windows_heads_15, x_windows_heads_16, x_windows_heads_17, spatial_mlp_windows_20, view_54, spatial_mlp_windows_21, spatial_mlp_windows_22, spatial_mlp_windows_23, x_83, permute_11, contiguous_13, x_84, getitem_10, x_85, x_86, x_87, x_91, x_93, x_94], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41.run(buf102, buf94, arg64_1, arg70_1, buf101, buf103, buf104, 1960, 384, stream=stream0)
            del arg64_1
            del arg70_1
            buf106 = reinterpret_tensor(buf101, (40, 588, 32), (18816, 32, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42.run(buf102, buf103, buf104, arg71_1, arg72_1, buf106, 752640, stream=stream0)
            del arg71_1
            del arg72_1
            # Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf107 = extern_kernels.convolution(buf106, arg73_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=12, bias=None)
            assert_size_stride(buf107, (40, 588, 32), (18816, 32, 1), 'torch.ops.aten.convolution.default')
            del arg73_1
            buf111 = reinterpret_tensor(buf106, (10, 196, 384), (75264, 384, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24, view_63, spatial_mlp_windows_25, spatial_mlp_windows_26, spatial_mlp_windows_27, x_97, permute_13, contiguous_16, x_98, x_99, x_100, layer_norm_16], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43.run(buf102, buf107, arg74_1, arg75_1, arg76_1, buf111, 1960, 384, stream=stream0)
            del arg75_1
            del arg76_1
            buf112 = reinterpret_tensor(buf100, (1960, 1536), (1536, 1), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24, view_63, spatial_mlp_windows_25, spatial_mlp_windows_26, spatial_mlp_windows_27, x_97, permute_13, contiguous_16, x_98, x_99, x_100, layer_norm_16, x_101], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(buf111, arg77_1, buf112, 372, 1, 1, stream=stream0)
            del arg77_1
            buf113 = reinterpret_tensor(buf112, (10, 196, 1536), (301056, 1536, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [x_101, x_102], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_36.run(buf113, arg78_1, 3010560, stream=stream0)
            del arg78_1
            buf114 = reinterpret_tensor(buf111, (1960, 384), (384, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [x_101, x_102, x_104], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_37.run(buf113, arg79_1, buf114, 93, 1, 1, stream=stream0)
            del arg79_1
            buf115 = buf102; del buf102  # reuse
            buf116 = reinterpret_tensor(buf104, (10, 196, 1), (196, 1, 1960), 0); del buf104  # reuse
            buf117 = reinterpret_tensor(buf103, (10, 196, 1), (196, 1, 1984), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, x_96, permute_12, contiguous_15, windows_6, x_windows_6, x_windows_heads_18, x_windows_heads_19, x_windows_heads_20, spatial_mlp_windows_24, view_63, spatial_mlp_windows_25, spatial_mlp_windows_26, spatial_mlp_windows_27, x_97, permute_13, contiguous_16, x_98, x_99, x_100, x_104, x_106, x_107], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38.run(buf115, buf107, arg74_1, arg80_1, buf114, buf116, buf117, 1960, 384, stream=stream0)
            del arg74_1
            del arg80_1
            del buf107
            buf119 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_108, shifted_x_3, x_109, permute_14, contiguous_17, windows_7, x_windows_7, x_windows_heads_21, x_windows_heads_22, x_windows_heads_23, spatial_mlp_windows_28], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39.run(buf115, buf116, buf117, arg81_1, arg82_1, buf119, 1693440, stream=stream0)
            del arg81_1
            del arg82_1
            # Topologically Sorted Source Nodes: [x_107, x_108, shifted_x_3, x_109, permute_14, contiguous_17, windows_7, x_windows_7, x_windows_heads_21, x_windows_heads_22, x_windows_heads_23, spatial_mlp_windows_28], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf120 = extern_kernels.convolution(buf119, arg83_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=12, bias=None)
            assert_size_stride(buf120, (90, 588, 32), (18816, 32, 1), 'torch.ops.aten.convolution.default')
            del arg83_1
            del buf119
            buf124 = reinterpret_tensor(buf114, (10, 196, 384), (75264, 384, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_108, shifted_x_3, x_109, permute_14, contiguous_17, windows_7, x_windows_7, x_windows_heads_21, x_windows_heads_22, x_windows_heads_23, spatial_mlp_windows_28, view_72, spatial_mlp_windows_29, spatial_mlp_windows_30, spatial_mlp_windows_31, x_110, permute_15, contiguous_18, x_111, getitem_11, x_112, x_113, x_114, layer_norm_18], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40.run(buf115, buf120, arg84_1, arg85_1, arg86_1, buf124, 1960, 384, stream=stream0)
            del arg85_1
            del arg86_1
            buf125 = reinterpret_tensor(buf113, (1960, 1536), (1536, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_108, shifted_x_3, x_109, permute_14, contiguous_17, windows_7, x_windows_7, x_windows_heads_21, x_windows_heads_22, x_windows_heads_23, spatial_mlp_windows_28, view_72, spatial_mlp_windows_29, spatial_mlp_windows_30, spatial_mlp_windows_31, x_110, permute_15, contiguous_18, x_111, getitem_11, x_112, x_113, x_114, layer_norm_18, x_115], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(buf124, arg87_1, buf125, 372, 1, 1, stream=stream0)
            del arg87_1
            buf126 = reinterpret_tensor(buf125, (10, 196, 1536), (301056, 1536, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [x_115, x_116], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_36.run(buf126, arg88_1, 3010560, stream=stream0)
            del arg88_1
            buf127 = reinterpret_tensor(buf124, (1960, 384), (384, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [x_115, x_116, x_118], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_37.run(buf126, arg89_1, buf127, 93, 1, 1, stream=stream0)
            del arg89_1
            buf128 = buf115; del buf115  # reuse
            buf129 = reinterpret_tensor(buf117, (10, 196, 1), (196, 1, 1960), 0); del buf117  # reuse
            buf130 = reinterpret_tensor(buf116, (10, 196, 1), (196, 1, 1984), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [x_107, x_108, shifted_x_3, x_109, permute_14, contiguous_17, windows_7, x_windows_7, x_windows_heads_21, x_windows_heads_22, x_windows_heads_23, spatial_mlp_windows_28, view_72, spatial_mlp_windows_29, spatial_mlp_windows_30, spatial_mlp_windows_31, x_110, permute_15, contiguous_18, x_111, getitem_11, x_112, x_113, x_114, x_118, x_120, x_121], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41.run(buf128, buf120, arg84_1, arg90_1, buf127, buf129, buf130, 1960, 384, stream=stream0)
            del arg84_1
            del arg90_1
            buf132 = reinterpret_tensor(buf127, (40, 588, 32), (18816, 32, 1), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [x_121, x_122, x_123, permute_16, contiguous_20, windows_8, x_windows_8, x_windows_heads_24, x_windows_heads_25, x_windows_heads_26, spatial_mlp_windows_32], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42.run(buf128, buf129, buf130, arg91_1, arg92_1, buf132, 752640, stream=stream0)
            del arg91_1
            del arg92_1
            # Topologically Sorted Source Nodes: [x_121, x_122, x_123, permute_16, contiguous_20, windows_8, x_windows_8, x_windows_heads_24, x_windows_heads_25, x_windows_heads_26, spatial_mlp_windows_32], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf133 = extern_kernels.convolution(buf132, arg93_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=12, bias=None)
            assert_size_stride(buf133, (40, 588, 32), (18816, 32, 1), 'torch.ops.aten.convolution.default')
            del arg93_1
            buf137 = reinterpret_tensor(buf132, (10, 196, 384), (75264, 384, 1), 0); del buf132  # reuse
            # Topologically Sorted Source Nodes: [x_121, x_122, x_123, permute_16, contiguous_20, windows_8, x_windows_8, x_windows_heads_24, x_windows_heads_25, x_windows_heads_26, spatial_mlp_windows_32, view_81, spatial_mlp_windows_33, spatial_mlp_windows_34, spatial_mlp_windows_35, x_124, permute_17, contiguous_21, x_125, x_126, x_127, layer_norm_20], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43.run(buf128, buf133, arg94_1, arg95_1, arg96_1, buf137, 1960, 384, stream=stream0)
            del arg95_1
            del arg96_1
            buf138 = reinterpret_tensor(buf126, (1960, 1536), (1536, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [x_121, x_122, x_123, permute_16, contiguous_20, windows_8, x_windows_8, x_windows_heads_24, x_windows_heads_25, x_windows_heads_26, spatial_mlp_windows_32, view_81, spatial_mlp_windows_33, spatial_mlp_windows_34, spatial_mlp_windows_35, x_124, permute_17, contiguous_21, x_125, x_126, x_127, layer_norm_20, x_128], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(buf137, arg97_1, buf138, 372, 1, 1, stream=stream0)
            del arg97_1
            buf139 = reinterpret_tensor(buf138, (10, 196, 1536), (301056, 1536, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [x_128, x_129], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_36.run(buf139, arg98_1, 3010560, stream=stream0)
            del arg98_1
            buf140 = reinterpret_tensor(buf137, (1960, 384), (384, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [x_128, x_129, x_131], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_37.run(buf139, arg99_1, buf140, 93, 1, 1, stream=stream0)
            del arg99_1
            buf141 = buf128; del buf128  # reuse
            buf142 = reinterpret_tensor(buf130, (10, 196, 1), (196, 1, 1960), 0); del buf130  # reuse
            buf143 = reinterpret_tensor(buf129, (10, 196, 1), (196, 1, 1984), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [x_121, x_122, x_123, permute_16, contiguous_20, windows_8, x_windows_8, x_windows_heads_24, x_windows_heads_25, x_windows_heads_26, spatial_mlp_windows_32, view_81, spatial_mlp_windows_33, spatial_mlp_windows_34, spatial_mlp_windows_35, x_124, permute_17, contiguous_21, x_125, x_126, x_127, x_131, x_133, x_134], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38.run(buf141, buf133, arg94_1, arg100_1, buf140, buf142, buf143, 1960, 384, stream=stream0)
            del arg100_1
            del arg94_1
            del buf133
            buf145 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39.run(buf141, buf142, buf143, arg101_1, arg102_1, buf145, 1693440, stream=stream0)
            del arg101_1
            del arg102_1
            del buf142
            del buf143
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution]
            buf146 = extern_kernels.convolution(buf145, arg103_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=12, bias=None)
            assert_size_stride(buf146, (90, 588, 32), (18816, 32, 1), 'torch.ops.aten.convolution.default')
            del arg103_1
            del buf145
            buf150 = reinterpret_tensor(buf140, (10, 196, 384), (75264, 384, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, layer_norm_22], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40.run(buf141, buf146, arg104_1, arg105_1, arg106_1, buf150, 1960, 384, stream=stream0)
            del arg105_1
            del arg106_1
            buf151 = reinterpret_tensor(buf139, (1960, 1536), (1536, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, layer_norm_22, x_142], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35.run(buf150, arg107_1, buf151, 372, 1, 1, stream=stream0)
            del arg107_1
            buf152 = reinterpret_tensor(buf151, (10, 196, 1536), (301056, 1536, 1), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [x_142, x_143], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_36.run(buf152, arg108_1, 3010560, stream=stream0)
            del arg108_1
            buf153 = reinterpret_tensor(buf150, (1960, 384), (384, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [x_142, x_143, x_145], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_37.run(buf152, arg109_1, buf153, 93, 1, 1, stream=stream0)
            del arg109_1
            del buf152
            buf154 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44.run(buf154, buf146, arg104_1, arg110_1, buf153, 752640, stream=stream0)
            del arg104_1
            del arg110_1
            del buf146
            buf159 = reinterpret_tensor(buf153, (10, 7, 7, 1536), (75264, 10752, 1536, 1), 0); del buf153  # reuse
            buf155 = reinterpret_tensor(buf159, (10, 7, 7, 384), (75264, 10752, 1536, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45.run(buf154, buf155, 188160, stream=stream0)
            buf156 = reinterpret_tensor(buf159, (10, 7, 7, 384), (75264, 10752, 1536, 1), 384)  # alias
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46.run(buf154, buf156, 188160, stream=stream0)
            buf157 = reinterpret_tensor(buf159, (10, 7, 7, 384), (75264, 10752, 1536, 1), 768)  # alias
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47.run(buf154, buf157, 188160, stream=stream0)
            buf158 = reinterpret_tensor(buf159, (10, 7, 7, 384), (75264, 10752, 1536, 1), 1152)  # alias
            # Topologically Sorted Source Nodes: [x_134, x_135, shifted_x_4, x_136, permute_18, contiguous_22, windows_9, x_windows_9, x_windows_heads_27, x_windows_heads_28, x_windows_heads_29, spatial_mlp_windows_36, view_90, spatial_mlp_windows_37, spatial_mlp_windows_38, spatial_mlp_windows_39, x_137, permute_19, contiguous_23, x_138, getitem_12, x_139, x_140, x_141, x_145, x_147, x_148, x0_2, x1_2, x2_2, x3_2, x_149], Original ATen: [aten.native_layer_norm, aten.view, aten.constant_pad_nd, aten.permute, aten.clone, aten.transpose, aten._unsafe_view, aten.convolution, aten.slice, aten.add, aten.addmm, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48.run(buf154, buf158, 188160, stream=stream0)
            buf163 = reinterpret_tensor(buf154, (10, 49, 1536), (75264, 1536, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [x_150, x_151], Original ATen: [aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_view_49.run(buf159, arg111_1, arg112_1, buf163, 490, 1536, stream=stream0)
            del arg111_1
            del arg112_1
            del buf155
            del buf156
            del buf157
            del buf158
            del buf159
            buf164 = empty_strided_cuda((490, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_150, x_151, x_152], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_tem_fused_mm_native_layer_norm_t_view_50.run(buf163, arg113_1, buf164, 96, 1, 1, stream=stream0)
            del arg113_1
            del buf163
            buf165 = empty_strided_cuda((10, 49, 1), (49, 1, 490), torch.float32)
            buf166 = empty_strided_cuda((10, 49, 1), (49, 1, 490), torch.float32)
            # Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten._unsafe_view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_native_layer_norm_51.run(buf164, buf165, buf166, 490, 768, stream=stream0)
            buf168 = empty_strided_cuda((10, 1176, 32), (37632, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52.run(buf164, buf165, buf166, arg114_1, arg115_1, buf168, 376320, stream=stream0)
            del arg114_1
            del arg115_1
            # Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution]
            buf169 = extern_kernels.convolution(buf168, arg116_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=24, bias=None)
            assert_size_stride(buf169, (10, 1176, 32), (37632, 32, 1), 'torch.ops.aten.convolution.default')
            del arg116_1
            buf173 = reinterpret_tensor(buf168, (10, 49, 768), (37632, 768, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40, view_101, spatial_mlp_windows_41, spatial_mlp_windows_42, spatial_mlp_windows_43, x_156, permute_21, x_157, x_158, x_159, layer_norm_25], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53.run(buf164, buf169, arg117_1, arg118_1, arg119_1, buf173, 490, 768, stream=stream0)
            del arg118_1
            del arg119_1
            buf174 = reinterpret_tensor(buf75, (490, 3072), (3072, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40, view_101, spatial_mlp_windows_41, spatial_mlp_windows_42, spatial_mlp_windows_43, x_156, permute_21, x_157, x_158, x_159, layer_norm_25, x_160], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54.run(buf173, arg120_1, buf174, 192, 1, 1, stream=stream0)
            del arg120_1
            buf175 = reinterpret_tensor(buf174, (10, 49, 3072), (150528, 3072, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [x_160, x_161], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_55.run(buf175, arg121_1, 1505280, stream=stream0)
            del arg121_1
            buf176 = reinterpret_tensor(buf173, (490, 768), (768, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [x_160, x_161, x_163], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_56.run(buf175, arg122_1, buf176, 96, 1, 1, stream=stream0)
            del arg122_1
            buf177 = reinterpret_tensor(buf164, (10, 49, 768), (37632, 768, 1), 0); del buf164  # reuse
            buf178 = buf166; del buf166  # reuse
            buf179 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [x_152, x_153, x_154, x_155, permute_20, windows_10, x_windows_10, x_windows_heads_30, x_windows_heads_31, x_windows_heads_32, spatial_mlp_windows_40, view_101, spatial_mlp_windows_41, spatial_mlp_windows_42, spatial_mlp_windows_43, x_156, permute_21, x_157, x_158, x_159, x_163, x_165, x_166], Original ATen: [aten._unsafe_view, aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57.run(buf177, buf169, arg117_1, arg123_1, buf176, buf178, buf179, 490, 768, stream=stream0)
            del arg117_1
            del arg123_1
            del buf169
            buf181 = reinterpret_tensor(buf176, (10, 1176, 32), (37632, 32, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58.run(buf177, buf178, buf179, arg124_1, arg125_1, buf181, 376320, stream=stream0)
            del arg124_1
            del arg125_1
            # Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution]
            buf182 = extern_kernels.convolution(buf181, arg126_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=24, bias=None)
            assert_size_stride(buf182, (10, 1176, 32), (37632, 32, 1), 'torch.ops.aten.convolution.default')
            del arg126_1
            buf186 = reinterpret_tensor(buf181, (10, 49, 768), (37632, 768, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44, view_110, spatial_mlp_windows_45, spatial_mlp_windows_46, spatial_mlp_windows_47, x_169, permute_23, x_170, x_171, x_172, layer_norm_27], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59.run(buf177, buf182, arg127_1, arg128_1, arg129_1, buf186, 490, 768, stream=stream0)
            del arg128_1
            del arg129_1
            buf187 = reinterpret_tensor(buf175, (490, 3072), (3072, 1), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44, view_110, spatial_mlp_windows_45, spatial_mlp_windows_46, spatial_mlp_windows_47, x_169, permute_23, x_170, x_171, x_172, layer_norm_27, x_173], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution, aten.add, aten.t, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54.run(buf186, arg130_1, buf187, 192, 1, 1, stream=stream0)
            del arg130_1
            buf188 = reinterpret_tensor(buf187, (10, 49, 3072), (150528, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_55.run(buf188, arg131_1, 1505280, stream=stream0)
            del arg131_1
            buf189 = reinterpret_tensor(buf186, (490, 768), (768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [x_173, x_174, x_176], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_gelu_t_view_56.run(buf188, arg132_1, buf189, 96, 1, 1, stream=stream0)
            del arg132_1
            del buf188
            buf190 = buf177; del buf177  # reuse
            buf191 = buf179; del buf179  # reuse
            buf192 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [x_166, x_167, x_168, permute_22, windows_11, x_windows_11, x_windows_heads_33, x_windows_heads_34, x_windows_heads_35, spatial_mlp_windows_44, view_110, spatial_mlp_windows_45, spatial_mlp_windows_46, spatial_mlp_windows_47, x_169, permute_23, x_170, x_171, x_172, x_176, x_178, x_179], Original ATen: [aten.native_layer_norm, aten.view, aten.permute, aten.transpose, aten.clone, aten._unsafe_view, aten.convolution, aten.add, aten.addmm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57.run(buf190, buf182, arg127_1, arg133_1, buf189, buf191, buf192, 490, 768, stream=stream0)
            del arg127_1
            del arg133_1
            del buf182
            del buf189
            buf194 = empty_strided_cuda((10, 768, 1, 1), (768, 1, 7680, 7680), torch.float32)
            buf195 = reinterpret_tensor(buf194, (10, 768, 1, 1), (768, 1, 1, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [x_179, transpose_25, x_180], Original ATen: [aten.native_layer_norm, aten.transpose, aten.unsqueeze, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60.run(buf195, buf190, buf191, buf192, arg134_1, arg135_1, 7680, 49, stream=stream0)
            del arg134_1
            del arg135_1
            del buf190
            del buf191
            del buf192
            buf197 = empty_strided_cuda((12, 1000), (1000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_180, x_181, x_182], Original ATen: [aten.as_strided, aten.squeeze, aten.view, aten.addmm, aten.t]
            stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_as_strided_squeeze_t_view_61.run(arg137_1, arg136_1, buf195, buf197, 32, 1, 1, stream=stream0)
            del arg136_1
            del arg137_1
            del buf195
        return (reinterpret_tensor(buf197, (10, 1000), (1000, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((96, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((147, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((147, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((147, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((147, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((294, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((294, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((294, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((294, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1176, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1176, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    r""" Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Window/Shifted-Window Spatial MLP
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # merge windows
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \\
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Model(nn.Module):
    r""" Swin MLP

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
batch_size = 10
image_size = 224

def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_red_fused_convolution_native_layer_norm_transpose_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp25_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12 - tmp4
        tmp14 = tl.full([1, 1], 96.0, tl.float32)
        tmp15 = (tmp8 / tmp14)
        tmp16 = tl.full([1, 1], 1e-05, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight, roffset == 0
        )
        tmp25_mean = tl.where(r0_mask & xmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(r0_mask & xmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(r0_mask & xmask, tmp25_weight_next, tmp25_weight)
        tl.store(in_out_ptr0 + (x0 + 3136*r0_2 + 301056*x1), tmp23, r0_mask & xmask)
    tmp26, tmp27, tmp28 = triton_helpers.welford(tmp25_mean, tmp25_m2, tmp25_weight, 1)
    tmp25 = tmp26[:, None]
    tmp29 = tmp27[:, None]
    tmp30 = tmp28[:, None]
    tl.store(out_ptr2 + (x3), tmp25, xmask)
    tl.store(out_ptr3 + (x3), tmp29, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 94080
    xnumel = 32
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 147)
    y1 = yindex // 147
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(((y1 // 8) % 8)) + 3136*x2 + 100352*(y0 // 49) + 301056*(y1 // 64) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 96.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + 32*y3), tmp13, xmask & ymask)

''',
    '''@triton.jit
def triton_red_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp12 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp16 - tmp6
        tmp18 = tl.full([1, 1], 96.0, tl.float32)
        tmp19 = (tmp10 / tmp18)
        tmp20 = tl.full([1, 1], 1e-05, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp27, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_3(arg_A, arg_B, out_ptr0):
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
    N = 384
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
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_gelu_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12042240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_gelu_t_view_5(arg_A, arg_B, out_ptr0):
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
    N = 96
    K = 384
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 384
    stride_ak = 1
    stride_bk = 1
    stride_bn = 384

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
        xindex = idx_n + 384*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 96*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    x3 = xindex
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_out_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r0_2 + 96*x3), tmp8, r0_mask & xmask)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3810240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 147)
    x2 = xindex // 4704
    x0 = (xindex % 32)
    x3 = xindex
    tmp0 = (-4) + 7*(((x2 // 9) % 9)) + (((x1 % 49)) // 7)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*((x2 % 9)) + ((((x1 % 49)) % 7))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-21888) + x0 + 32*(x1 // 49) + 96*((((x1 % 49)) % 7)) + 672*((x2 % 9)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 9) % 9)) + 301056*(x2 // 81)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-228) + 7*((x2 % 9)) + 56*(((x1 % 49)) // 7) + 392*(((x2 // 9) % 9)) + 3136*(x2 // 81) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 96.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)

''',
    '''@triton.jit
def triton_red_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 56))) % 7)) + 224*(((4 + (x0 // 56)) % 7)) + 1568*(r0_2 // 32) + 4704*((4 + ((x0 % 56))) // 7) + 42336*((4 + (x0 // 56)) // 7) + 381024*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (7*(((4 + (x0 // 56)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 56))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp16 - tmp6
        tmp18 = tl.full([1, 1], 96.0, tl.float32)
        tmp19 = (tmp10 / tmp18)
        tmp20 = tl.full([1, 1], 1e-05, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r0_2 + 96*x3), tmp27, r0_mask & xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 3136)
    x2 = xindex // 301056
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 56))) % 7)) + 224*(((4 + (x1 // 56)) % 7)) + 1568*(x0 // 32) + 4704*((4 + ((x1 % 56))) // 7) + 42336*((4 + (x1 // 56)) // 7) + 381024*x2 + ((x0 % 32))), None)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 56)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 56))) % 7))), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (96 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 28)
    x2 = xindex // 2688
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (5472 + x0 + 192*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 384*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_per_fused_native_layer_norm_view_14(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp0 - tmp10
    tmp18 = tl.full([1, 1], 384.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 384*x0), tmp27, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_15(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 7840
    N = 192
    K = 384
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 384
    stride_ak = 1
    stride_bk = 1
    stride_bn = 384

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
        xindex = idx_n + 384*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_per_fused__unsafe_view_native_layer_norm_16(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 192, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 294)
    x2 = xindex // 9408
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 4)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 4)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 4)) + 28*(((x1 % 49)) // 7) + 196*(x2 // 4) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 4)) + 28*(((x1 % 49)) // 7) + 196*(x2 // 4) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 192.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 28)) % 7)) + 224*(((x0 // 28) % 7)) + 1568*(r0_2 // 32) + 9408*(((x0 % 28)) // 7) + 37632*(x0 // 196) + 150528*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 28) % 7)) + 49*(r0_2 // 32) + ((((x0 % 28)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 192.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 192*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_19(arg_A, arg_B, out_ptr0):
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
    N = 768
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
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_gelu_t_view_21(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 7840
    N = 192
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 192*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*((((x0 % 28)) % 7)) + 224*(((x0 // 28) % 7)) + 1568*(r0_2 // 32) + 9408*(((x0 % 28)) // 7) + 37632*(x0 // 196) + 150528*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((x0 // 28) % 7)) + 49*(r0_2 // 32) + ((((x0 % 28)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 192*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2352000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 294)
    x2 = xindex // 9408
    x0 = (xindex % 32)
    x3 = xindex
    tmp0 = (-4) + 7*(((x2 // 5) % 5)) + (((x1 % 49)) // 7)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*((x2 % 5)) + ((((x1 % 49)) % 7))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-22272) + x0 + 32*(x1 // 49) + 192*((((x1 % 49)) % 7)) + 1344*((x2 % 5)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 5) % 5)) + 150528*(x2 // 25)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-116) + 7*((x2 % 5)) + 28*(((x1 % 49)) // 7) + 196*(((x2 // 5) % 5)) + 784*(x2 // 25) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 192.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
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
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 28))) % 7)) + 224*(((4 + (x0 // 28)) % 7)) + 1568*(r0_2 // 32) + 9408*((4 + ((x0 % 28))) // 7) + 47040*((4 + (x0 // 28)) // 7) + 235200*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 28)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 28))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 192.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 192*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 784)
    x2 = xindex // 150528
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 28))) % 7)) + 224*(((4 + (x1 // 28)) % 7)) + 1568*(x0 // 32) + 9408*((4 + ((x1 % 28))) // 7) + 47040*((4 + (x1 // 28)) // 7) + 235200*x2 + ((x0 % 32))), xmask)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 28)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 28))) % 7))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 14)
    x2 = xindex // 2688
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (5568 + x0 + 384*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 768*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_per_fused_native_layer_norm_view_30(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp0 - tmp10
    tmp18 = tl.full([1, 1], 768.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp27, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_31(arg_A, arg_B, out_ptr0):
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
    N = 384
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_32(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 588)
    x2 = xindex // 18816
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 2)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 2)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 384.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_35(arg_A, arg_B, out_ptr0):
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
    N = 1536
    K = 384
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 384
    stride_ak = 1
    stride_bk = 1
    stride_bn = 384

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
        xindex = idx_n + 384*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1536*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1536*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_gelu_view_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 1536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)

''',
    '''@triton.jit
def triton_tem_fused_addmm_gelu_t_view_37(arg_A, arg_B, out_ptr0):
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
    N = 384
    K = 1536
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1536
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1536

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
        xindex = idx_n + 1536*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 384*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 384*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1693440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 588)
    x2 = xindex // 18816
    x0 = (xindex % 32)
    x3 = xindex
    tmp0 = (-4) + 7*(((x2 // 3) % 3)) + (((x1 % 49)) // 7)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*((x2 % 3)) + ((((x1 % 49)) % 7))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-23040) + x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 3)) + 5376*(((x1 % 49)) // 7) + 37632*(((x2 // 3) % 3)) + 75264*(x2 // 9)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-60) + 7*((x2 % 3)) + 14*(((x1 % 49)) // 7) + 98*(((x2 // 3) % 3)) + 196*(x2 // 9) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-60) + 7*((x2 % 3)) + 14*(((x1 % 49)) // 7) + 98*(((x2 // 3) % 3)) + 196*(x2 // 9) + ((((x1 % 49)) % 7))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full([1], 384.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 14))) % 7)) + 224*(((4 + (x0 // 14)) % 7)) + 1568*(r0_2 // 32) + 18816*((4 + ((x0 % 14))) // 7) + 56448*((4 + (x0 // 14)) // 7) + 169344*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 14)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 14))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x0 % 14))) % 7)) + 224*(((4 + (x0 // 14)) % 7)) + 1568*(r0_2 // 32) + 18816*((4 + ((x0 % 14))) // 7) + 56448*((4 + (x0 // 14)) // 7) + 169344*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x0 // 14)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 14))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 384*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 588)
    x2 = xindex // 18816
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 384*((((x1 % 49)) % 7)) + 2688*((x2 % 2)) + 5376*(((x1 % 49)) // 7) + 37632*(x2 // 2)), xmask)
    tmp1 = tl.load(in_ptr1 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (7*((x2 % 2)) + 14*(((x1 % 49)) // 7) + 98*(x2 // 2) + ((((x1 % 49)) % 7))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 384.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
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
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 196)
    x2 = xindex // 75264
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 14))) % 7)) + 224*(((4 + (x1 // 14)) % 7)) + 1568*(x0 // 32) + 18816*((4 + ((x1 % 14))) // 7) + 56448*((4 + (x1 // 14)) // 7) + 169344*x2 + ((x0 % 32))), xmask)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 14)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 14))) % 7))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (5376 + x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (384 + x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_cat_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = xindex // 2688
    x3 = xindex // 384
    tmp0 = tl.load(in_ptr0 + (5760 + x0 + 768*x1 + 10752*x2), xmask)
    tl.store(out_ptr0 + (x0 + 1536*x3), tmp0, xmask)

''',
    '''@triton.jit
def triton_red_fused_native_layer_norm_view_49(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8 - tmp2
        tmp10 = tl.full([1, 1], 1536.0, tl.float32)
        tmp11 = (tmp6 / tmp10)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp19, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_50(arg_A, arg_B, out_ptr0):
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

    M = 490
    N = 768
    K = 1536
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1536
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1536

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
        xindex = idx_n + 1536*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1536*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_51(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 1176)
    x2 = xindex // 37632
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex
    x0 = (xindex % 49)
    x1 = xindex // 49
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 768.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 3072
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 3072*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 3072*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_poi_fused_addmm_gelu_view_55(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_gelu_t_view_56(arg_A, arg_B, out_ptr0):
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

    M = 490
    N = 768
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

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
        xindex = idx_n + 3072*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 3072*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex
    x0 = (xindex % 49)
    x1 = xindex // 49
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp8, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)

''',
    '''@triton.jit
def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 1176)
    x2 = xindex // 37632
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*(x1 // 49) + 768*((x1 % 49)) + 37632*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (49*x2 + ((x1 % 49))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + 32*(x1 // 49)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)

''',
    '''@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex
    x0 = (xindex % 49)
    x1 = xindex // 49
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*x0 + 1568*(r0_2 // 32) + 37632*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 49*(r0_2 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 768.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp31, r0_mask & xmask)

''',
    '''@triton.jit
def triton_per_fused_mean_native_layer_norm_transpose_unsqueeze_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7680
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
    x0 = (xindex % 768)
    x1 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 37632*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 49*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 49*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 768.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.full([1, 1], 49.0, tl.float32)
    tmp19 = (tmp17 / tmp18)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)

''',
    '''@triton.jit
def triton_tem_fused_addmm_as_strided_squeeze_t_view_61(in_ptr0, arg_B, in_ptr2, out_ptr0):
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
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        tmp2 = tl.broadcast_to(idx_m, [BLOCK_M, BLOCK_K])
        tmp3 = tl.full([1], 10, tl.int64)
        tmp4 = tmp2 < tmp3
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 768*idx_m, [BLOCK_M, BLOCK_K])), tmp4, eviction_policy='evict_last', other=0.0)
        a = tmp5.broadcast_to(xindex.shape)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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
    arg0_1 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((96, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((147, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((147, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((294, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((294, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((294,), (1,), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((588, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1176, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1176,), (1,), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1176, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1176,), (1,), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1]

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
