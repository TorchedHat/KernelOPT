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


# kernel path: /workspace/tmp/torchinductor_user/xg/cxgd6rzheg3uvg4ahmzvezmpz7b2p2iyiepr5nblwdinfvrzncry.py
# Topologically Sorted Source Nodes: [x, x_1, add, sigmoid, mul, x_2, x_3, x_4], Original ATen: [aten.convolution, aten.logsumexp, aten.add, aten.sigmoid, aten.mul, aten.div, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   add => add_1
#   mul => mul
#   sigmoid => sigmoid
#   x => convolution
#   x_1 => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
#   x_2 => div
#   x_3 => sub_1
#   x_4 => clamp_max, clamp_min
# Graph fragment:
#   %buf0 : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %amax : Tensor "f32[128, 1, 31, 63, 63][124000, 15872000, 4000, 63, 1]cuda:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[128, 1, 31, 63, 63][124000, 15872000, 4000, 63, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg3_1 : Tensor "f32[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %amax : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convolution, [1], True), kwargs = {})
#   %abs_1 : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : Tensor "b8[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %where), kwargs = {})
#   %exp : Tensor "f32[128, 16, 31, 63, 63][1968624, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %where), kwargs = {})
#   %add_1 : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 3), kwargs = {})
#   %sigmoid : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %sigmoid), kwargs = {})
#   %div : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 6), kwargs = {})
#   %sub_1 : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %arg3_1), kwargs = {})
#   %clamp_min : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, -1), kwargs = {})
#   %clamp_max : Tensor "f32[128, 1, 31, 63, 63][123039, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   return %amax,%sum_1,%clamp_max
triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0 = async_compile.triton('triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0', '''
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
    size_hints={'x': 16777216, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1133927424, 'r0_': 64}, 'kernel_num_gb': 1.070931524, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15748992
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
    r0_3 = r0_index
    x2 = xindex // 123039
    x5 = (xindex % 123039)
    x0 = (xindex % 3969)
    x4 = xindex // 3969
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + 123039*r0_3 + 1968624*x2), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (0))
    tmp27 = tl.broadcast_to(tmp26, [1, 1])
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tl.full([1, 1], float("inf"), tl.float32)
    tmp9 = tmp7 == tmp8
    tmp10 = tl.full([1, 1], 0.0, tl.float32)
    tmp11 = tl.where(tmp9, tmp10, tmp6)
    tmp12 = tmp2 - tmp11
    tmp13 = libdevice.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl_math.log(tmp17)
    tmp19 = tmp18 + tmp11
    tmp20 = tl.full([1, 1], 3.0, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp19 * tmp22
    tmp24 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp28 = tmp25 - tmp27
    tmp29 = tl.full([1, 1], -1.0, tl.float32)
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = triton_helpers.minimum(tmp30, tmp31)
    tl.store(out_ptr2 + (x6), tmp32, xmask)


def get_args():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 1, 31, 63, 63), (123039, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 15748992, 16,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.070931524
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
        assert_size_stride(arg0_1, (3, 16, 3, 3, 3), (432, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (16, ), (1, ))
        assert_size_stride(arg2_1, (128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1))
        assert_size_stride(arg3_1, (1, 1, 1, 1), (1, 1, 1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf3 = empty_strided_cuda((128, 1, 31, 63, 63), (123039, 123039, 3969, 63, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, add, sigmoid, mul, x_2, x_3, x_4], Original ATen: [aten.convolution, aten.logsumexp, aten.add, aten.sigmoid, aten.mul, aten.div, aten.sub, aten.clamp]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0.run(buf0, arg1_1, arg3_1, buf3, 15748992, 16, stream=stream0)
            del arg1_1
            del arg3_1
            del buf0
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((3, 16, 3, 3, 3), (432, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.sigmoid(x + 3) / 6
        x = x - self.bias
        x = torch.clamp(x, min=-1, max=1)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)  

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15748992
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
    r0_3 = r0_index
    x2 = xindex // 123039
    x5 = (xindex % 123039)
    x0 = (xindex % 3969)
    x4 = xindex // 3969
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + 123039*r0_3 + 1968624*x2), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (0))
    tmp27 = tl.broadcast_to(tmp26, [1, 1])
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tl.full([1, 1], float("inf"), tl.float32)
    tmp9 = tmp7 == tmp8
    tmp10 = tl.full([1, 1], 0.0, tl.float32)
    tmp11 = tl.where(tmp9, tmp10, tmp6)
    tmp12 = tmp2 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.log(tmp17)
    tmp19 = tmp18 + tmp11
    tmp20 = tl.full([1, 1], 3.0, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp19 * tmp22
    tmp24 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp28 = tmp25 - tmp27
    tmp29 = tl.full([1, 1], -1.0, tl.float32)
    tmp30 = tl.maximum(tmp28, tmp29)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = tl.minimum(tmp30, tmp31)
    tl.store(out_ptr2 + (x6), tmp32, xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((3, 16, 3, 3, 3), (432, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
