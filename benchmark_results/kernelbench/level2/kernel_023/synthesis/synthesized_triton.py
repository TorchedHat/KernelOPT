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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/c5/cc5y3fbl4bnb5csrv7xtfx77y5vb3tfvzdmrc6qcqge2ae6joezq.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => var_mean, view
# Graph fragment:
#   %buf0 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %view : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [128, 8, 3, 19800]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf2
triton_red_fused_convolution_native_group_norm_0 = async_compile.triton('triton_red_fused_convolution_native_group_norm_0', '''
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
    size_hints={'x': 1024, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16384, 'r0_': 243302400}, 'kernel_num_gb': 0.243310688, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 59400
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 8)
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_3 = r0_index // 19800
        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 3*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp8, xmask)


def get_args():
    arg_0 = rand_strided((128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1024, 59400,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_convolution_native_group_norm_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.243310688
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/7a/c7a7xukmbocvxsbnlrolarxdm2twkr6mk3rfahl6ykkqaexqvxal.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.mean]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, var_mean, view, view_1
#   x_2 => mean
# Graph fragment:
#   %buf0 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_1 : Tensor "f32[128, 8, 1, 1][8, 1, 1024, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[128, 8, 1, 1][8, 1, 1024, 1024]cuda:0" = PlaceHolder[target=buf2]
#   %arg3_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[24][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %view : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [128, 8, 3, 19800]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[128, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[128, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [128, 24, 22, 30, 30]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 24, 1][24, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 24, 1, 1, 1][24, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 4), kwargs = {})
#   %mul_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 24, 1][24, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 24, 1, 1, 1][24, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 4), kwargs = {})
#   %add_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_7), kwargs = {})
#   %mean : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [1, 2, 3, 4]), kwargs = {})
#   return %buf4
triton_per_fused_convolution_mean_native_group_norm_1 = async_compile.triton('triton_per_fused_convolution_mean_native_group_norm_1', '''
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
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_native_group_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1106208, 'r0_': 243302400}, 'kernel_num_gb': 0.24386384, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 138240
    r0_numel = 440
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
    r0_3 = r0_index
    x0 = (xindex % 180)
    x4 = xindex // 180
    x1 = ((xindex // 180) % 6)
    x2 = xindex // 1080
    x5 = (xindex % 1080)
    tmp0 = tl.load(in_ptr0 + (r0_3 + 30*((((r0_3 + 440*x0) % 30)) // 30) + 440*x0 + 900*((30*((((r0_3 + 440*x0) // 30) % 30)) + (((r0_3 + 440*x0) % 30))) // 900) + 79200*x4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (4*x1 + ((r0_3 + 440*x0) // 19800)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (8*x2 + ((4*x1 + ((r0_3 + 440*x0) // 19800)) // 3)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (8*x2 + ((4*x1 + ((r0_3 + 440*x0) // 19800)) // 3)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (4*x1 + ((r0_3 + 440*x0) // 19800)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (4*x1 + ((r0_3 + 440*x0) // 19800)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 59400.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x5 + 1088*x2), tmp19, xmask)


def get_args():
    arg_0 = rand_strided((128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 6, 180), (1088, 180, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 138240, 440,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_mean_native_group_norm_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_convolution_mean_native_group_norm_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.24386384
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/ig/ciglmujgtm3clxouudkxnvaqrzpdsuadesar64suuhiljnwhd3kp.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.mean]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, var_mean, view, view_1
#   x_2 => mean
# Graph fragment:
#   %buf4 : Tensor "f32[128, 6, 180][1088, 180, 1]cuda:0" = PlaceHolder[target=buf4]
#   %convolution : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %view : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [128, 8, 3, 19800]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[128, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[128, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [128, 24, 22, 30, 30]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 24, 1][24, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 24, 1, 1, 1][24, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 4), kwargs = {})
#   %mul_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 24, 1][24, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 24, 1, 1, 1][24, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 4), kwargs = {})
#   %add_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_7), kwargs = {})
#   %mean : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [1, 2, 3, 4]), kwargs = {})
#   return %buf5
triton_per_fused_convolution_mean_native_group_norm_2 = async_compile.triton('triton_per_fused_convolution_mean_native_group_norm_2', '''
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
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6144, 'r0_': 552960}, 'kernel_num_gb': 0.000556032, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 180
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
    x0 = (xindex % 6)
    x1 = xindex // 6
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 180*x0 + 1088*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp4, xmask)


def get_args():
    arg_0 = rand_strided((128, 6, 180), (1088, 180, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 768, 180,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_mean_native_group_norm_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_convolution_mean_native_group_norm_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.000556032
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/kv/ckvagrdk5klp5n672w3xoln6uixaoreroxgzfbcjc2fngyb45udq.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.mean]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, var_mean, view, view_1
#   x_2 => mean
# Graph fragment:
#   %buf5 : Tensor "f32[128, 6][6, 1]cuda:0" = PlaceHolder[target=buf5]
#   %buf6 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=buf6]
#   %convolution : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %view : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [128, 8, 3, 19800]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[128, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[128, 8, 1, 1][8, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 8, 3, 19800][475200, 59400, 19800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [128, 24, 22, 30, 30]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 24, 1][24, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 24, 1, 1, 1][24, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 4), kwargs = {})
#   %mul_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 24, 1][24, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 24, 1, 1, 1][24, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 4), kwargs = {})
#   %add_1 : Tensor "f32[128, 24, 22, 30, 30][475200, 19800, 900, 30, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_7), kwargs = {})
#   %mean : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [1, 2, 3, 4]), kwargs = {})
#   return %buf6,%mean
triton_per_fused_convolution_mean_native_group_norm_3 = async_compile.triton('triton_per_fused_convolution_mean_native_group_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_native_group_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 2304}, 'kernel_num_gb': 4.096e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 6
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 6*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl.full([1, 1], 475200.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 128, 6,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_mean_native_group_norm_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_convolution_mean_native_group_norm_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 4.096e-06
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (24, 3, 3, 3, 3), (81, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (24, ), (1, ))
        assert_size_stride(arg2_1, (128, 3, 24, 32, 32), (73728, 24576, 1024, 32, 1))
        assert_size_stride(arg3_1, (24, ), (1, ))
        assert_size_stride(arg4_1, (24, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), 'unknown_op')
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((128, 8, 1, 1), (8, 1, 1024, 1024), torch.float32)
            buf2 = empty_strided_cuda((128, 8, 1, 1), (8, 1, 1024, 1024), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_0.run(buf0, arg1_1, buf1, buf2, 1024, 59400, stream=stream0)
            buf4 = empty_strided_cuda((128, 6, 180), (1088, 180, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_mean_native_group_norm_1.run(buf0, arg1_1, buf1, buf2, arg3_1, arg4_1, buf4, 138240, 440, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del buf0
            del buf1
            del buf2
            buf5 = empty_strided_cuda((128, 6), (6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_mean_native_group_norm_2.run(buf4, buf5, 768, 180, stream=stream0)
            del buf4
            buf6 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_mean_native_group_norm_3.run(buf7, buf5, 128, 6, stream=stream0)
            del buf5
        return (buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((24, 3, 3, 3, 3), (81, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 24, 32, 32), (73728, 24576, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 59400
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 8)
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_3 = r0_index // 19800
        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 3*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp8, xmask)

''',
    '''@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 138240
    r0_numel = 440
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
    r0_3 = r0_index
    x0 = (xindex % 180)
    x4 = xindex // 180
    x1 = ((xindex // 180) % 6)
    x2 = xindex // 1080
    x5 = (xindex % 1080)
    tmp0 = tl.load(in_ptr0 + (r0_3 + 30*((((r0_3 + 440*x0) % 30)) // 30) + 440*x0 + 900*((30*((((r0_3 + 440*x0) // 30) % 30)) + (((r0_3 + 440*x0) % 30))) // 900) + 79200*x4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (4*x1 + ((r0_3 + 440*x0) // 19800)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (8*x2 + ((4*x1 + ((r0_3 + 440*x0) // 19800)) // 3)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (8*x2 + ((4*x1 + ((r0_3 + 440*x0) // 19800)) // 3)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (4*x1 + ((r0_3 + 440*x0) // 19800)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (4*x1 + ((r0_3 + 440*x0) // 19800)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 59400.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x5 + 1088*x2), tmp19, xmask)

''',
    '''@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 180
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
    x0 = (xindex % 6)
    x1 = xindex // 6
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 180*x0 + 1088*x1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp4, xmask)

''',
    '''@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 6
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 6*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl.full([1, 1], 475200.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((24, 3, 3, 3, 3), (81, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 24, 32, 32), (73728, 24576, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]

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
