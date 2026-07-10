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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/zb/czb5farksvtt2un4rh4664ss4sswptdlyl2xzv6cqyfnkbgbwx47.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add
#   x_2 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %buf0 : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=arg3_1]
#   %getitem_1 : Tensor "f32[32, 64, 32, 64, 1][131072, 2048, 64, 1, 4194304]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[32, 64, 32, 64, 1][131072, 2048, 64, 1, 4194304]cuda:0" = PlaceHolder[target=buf2]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [1, 1, 1], 1), kwargs = {})
#   %add : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %arg3_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[32, 64, 32, 64, 1][131072, 2048, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[32, 64, 32, 64, 1][131072, 2048, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   return %getitem_1,%buf2,%add_2
triton_per_fused_add_convolution_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_convolution_native_layer_norm_0', '''
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
    size_hints={'x': 4194304, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 3221225984}, 'kernel_num_gb': 2.14748442, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_convolution_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 2048) % 64)
    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [1, 1])
    tmp26 = tl.load(in_ptr2 + (r0_3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 64, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp6 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp19 = tmp5 - tmp13
    tmp20 = tl.full([1, 1], 64.0, tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tl.full([1, 1], 1e-05, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp29, None)


def get_args():
    arg_0 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4194304, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_layer_norm_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_convolution_native_layer_norm_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 2.14748442
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/2f/c2fobewihwkfurhmeu6l2rxluzvrq7l5c7qdfnoeunv6egfpoj7o.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten.add, aten.native_layer_norm, aten.avg_pool3d, aten.gelu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add
#   x_2 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
#   x_3 => avg_pool3d
#   x_4 => add_3, erf, mul_2, mul_3, mul_4
# Graph fragment:
#   %add_2 : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0" = PlaceHolder[target=add_2]
#   %avg_pool3d : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0" = PlaceHolder[target=avg_pool3d]
#   %convolution : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [1, 1, 1], 1), kwargs = {})
#   %add : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %arg3_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[32, 64, 32, 64, 1][131072, 2048, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[32, 64, 32, 64, 1][131072, 2048, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : Tensor "f32[32, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %avg_pool3d : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%add_2, [2, 2, 2], [2, 2, 2]), kwargs = {})
#   %mul_2 : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool3d, 0.5), kwargs = {})
#   %mul_3 : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool3d, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_3,), kwargs = {})
#   %add_3 : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_4 : Tensor "f32[32, 64, 16, 32, 32][1048576, 16384, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add_3), kwargs = {})
#   return %avg_pool3d,%mul_4
triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1 = async_compile.triton('triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 268435456}, 'kernel_num_gb': 1.34217728, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 0.5, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = tl.full([1], 1.0, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, None)


def get_args():
    arg_0 = rand_strided((32, 64, 16, 32, 32), (1048576, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 33554432,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.34217728
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        assert_size_stride(arg0_1, (32, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1))
        assert_size_stride(arg3_1, (), ())
        assert_size_stride(arg4_1, (64, ), (1, ))
        assert_size_stride(arg5_1, (64, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
            assert_size_stride(buf0, (32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf4 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_convolution_native_layer_norm_0.run(buf4, arg1_1, arg3_1, arg4_1, arg5_1, 4194304, 64, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf5 = empty_strided_cuda((32, 64, 16, 32, 32), (1048576, 16384, 1024, 32, 1), torch.float32)
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.convolution, aten.add, aten.native_layer_norm, aten.avg_pool3d, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1.run(buf6, buf4, 33554432, stream=stream0)
            del buf4
        return (buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_per_fused_add_convolution_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 2048) % 64)
    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [1, 1])
    tmp26 = tl.load(in_ptr2 + (r0_3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 64, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp6 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp19 = tmp5 - tmp13
    tmp20 = tl.full([1, 1], 64.0, tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tl.full([1, 1], 1e-05, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp29, None)

''',
    '''@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_gelu_native_layer_norm_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1], 0.5, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.erf(tmp20)
    tmp22 = tl.full([1], 1.0, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((32, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]

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
