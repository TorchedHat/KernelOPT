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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/it/citp2eyvfjyusy3dcebyd6ibm5ybcjdohwflyjznybslvwdfhypm.py
# Topologically Sorted Source Nodes: [kl_div, log], Original ATen: [aten.xlogy, aten.log, aten.mul, aten.sub, aten.sum]
# Source node to ATen node mapping:
#   kl_div => eq, full_default, full_default_1, isnan, log_1, mul, mul_1, sub, sum_1, where, where_1
#   log => log
# Graph fragment:
#   %arg1_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %isnan : Tensor "b8[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%arg1_1,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : Tensor "b8[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 0), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%arg1_1,), kwargs = {})
#   %mul_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %log_1), kwargs = {})
#   %where : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul_1), kwargs = {})
#   %where_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %log : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%arg0_1,), kwargs = {})
#   %mul : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %log), kwargs = {})
#   %sub : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub,), kwargs = {})
#   return %buf1
triton_red_fused_log_mul_sub_sum_xlogy_0 = async_compile.triton('triton_red_fused_log_mul_sub_sum_xlogy_0', '''
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
    size_hints={'x': 2048, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_log_mul_sub_sum_xlogy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16384, 'r0_': 2147483648}, 'kernel_num_gb': 2.14749184, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_log_mul_sub_sum_xlogy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.isnan(tmp0).to(tl.int1)
        tmp2 = tl.full([1, 1], 0.0, tl.float32)
        tmp3 = tmp0 == tmp2
        tmp4 = tl_math.log(tmp0)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.where(tmp3, tmp2, tmp5)
        tmp7 = tl.full([1, 1], float("nan"), tl.float32)
        tmp8 = tl.where(tmp1, tmp7, tmp6)
        tmp10 = tl_math.log(tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tmp8 - tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)


def get_args():
    arg_0 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 2048, 131072,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_log_mul_sub_sum_xlogy_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_log_mul_sub_sum_xlogy_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 2.14749184
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/me/cmefdauljwxjt7prmdqrjdwmtst665hf6h72qk7g376cbzhfikgw.py
# Topologically Sorted Source Nodes: [kl_div, log], Original ATen: [aten.xlogy, aten.log, aten.mul, aten.sub, aten.sum, aten.div]
# Source node to ATen node mapping:
#   kl_div => div, eq, full_default, full_default_1, isnan, log_1, mul, mul_1, sub, sum_1, where, where_1
#   log => log
# Graph fragment:
#   %buf1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=buf1]
#   %sum_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_1]
#   %isnan : Tensor "b8[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%arg1_1,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : Tensor "b8[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 0), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%arg1_1,), kwargs = {})
#   %mul_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %log_1), kwargs = {})
#   %where : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul_1), kwargs = {})
#   %where_1 : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %log : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%arg0_1,), kwargs = {})
#   %mul : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %log), kwargs = {})
#   %sub : Tensor "f32[16384, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub,), kwargs = {})
#   %div : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, 16384), kwargs = {})
#   return %sum_1,%div
triton_red_fused_div_log_mul_sub_sum_xlogy_1 = async_compile.triton('triton_red_fused_div_log_mul_sub_sum_xlogy_1', '''
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
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_log_mul_sub_sum_xlogy_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 8192}, 'kernel_num_gb': 8.2e-06, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused_div_log_mul_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.full([1, 1], 6.103515625e-05, tl.float32)
    tmp5 = tmp2 * tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp5, None)


def get_args():
    arg_0 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 1, 2048,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_log_mul_sub_sum_xlogy_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_div_log_mul_sub_sum_xlogy_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 8.2e-06
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
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16384, 16384), (16384, 1))
        assert_size_stride(arg1_1, (16384, 16384), (16384, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((2048, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [kl_div, log], Original ATen: [aten.xlogy, aten.log, aten.mul, aten.sub, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_log_mul_sub_sum_xlogy_0.run(arg1_1, arg0_1, buf1, 2048, 131072, stream=stream0)
            del arg0_1
            del arg1_1
            buf2 = empty_strided_cuda((), (), torch.float32)
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [kl_div, log], Original ATen: [aten.xlogy, aten.log, aten.mul, aten.sub, aten.sum, aten.div]
            stream0 = get_raw_stream(0)
            triton_red_fused_div_log_mul_sub_sum_xlogy_1.run(buf3, buf1, 1, 2048, stream=stream0)
            del buf1
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []
'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_red_fused_log_mul_sub_sum_xlogy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.isnan(tmp0).to(tl.int1)
        tmp2 = tl.full([1, 1], 0.0, tl.float32)
        tmp3 = tmp0 == tmp2
        tmp4 = tl.log(tmp0)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.where(tmp3, tmp2, tmp5)
        tmp7 = tl.full([1, 1], float("nan"), tl.float32)
        tmp8 = tl.where(tmp1, tmp7, tmp6)
        tmp10 = tl.log(tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tmp8 - tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)

''',
    '''@triton.jit
def triton_red_fused_div_log_mul_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.full([1, 1], 6.103515625e-05, tl.float32)
    tmp5 = tmp2 * tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp5, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1]

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
