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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/in/cin6apvbe44wk7dx4ft5r6yjws276gx3tiidymeluztw45jnw4lb.py
# Topologically Sorted Source Nodes: [x, x_1, mean, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.mean, aten.sub]
# Source node to ATen node mapping:
#   mean => mean
#   x => convolution
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_10, unsqueeze_11, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, unsqueeze_8, unsqueeze_9
#   x_2 => sub_1
# Graph fragment:
#   %buf0 : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %add_1 : Tensor "f32[16, 32, 31, 63, 63][3968000, 124000, 4000, 63, 1]cuda:0" = PlaceHolder[target=add_1]
#   %buf2 : Tensor "f32[16, 32, 1, 1, 1][32, 1, 512, 512, 512]cuda:0" = PlaceHolder[target=buf2]
#   %convolution : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %unsqueeze : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, -1), kwargs = {})
#   %sub : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_2), kwargs = {})
#   %add : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_3, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_1 : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, -1), kwargs = {})
#   %mul_2 : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_8), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[32, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %add_1 : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_11), kwargs = {})
#   %mean : Tensor "f32[16, 32, 1, 1, 1][32, 1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_1, [2, 3, 4], True), kwargs = {})
#   %sub_1 : Tensor "f32[16, 32, 31, 63, 63][3937248, 123039, 3969, 63, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %mean), kwargs = {})
#   return %add_1,%buf2,%sub_1
triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0', '''
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
    size_hints={'x': 512, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 640, 'r0_': 755951616}, 'kernel_num_gb': 0.755952256, 'kernel_flop': 0}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 123039
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_2 = (r0_index % 3969)
        r0_3 = r0_index // 3969
        tmp0 = tl.load(in_ptr0 + (r0_5 + 123039*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
        tl.store(out_ptr0 + (r0_2 + 4000*r0_3 + 124000*x4), tmp17, r0_mask & xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 3969)
        r0_3 = r0_index // 3969
        r0_5 = r0_index
        tmp21 = tl.load(out_ptr0 + (r0_2 + 4000*r0_3 + 124000*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.full([1, 1], 123039.0, tl.float32)
        tmp23 = (tmp19 / tmp22)
        tmp24 = tmp21 - tmp23
        tl.store(out_ptr2 + (r0_5 + 123039*x4), tmp24, r0_mask & xmask)


def get_args():
    arg_0 = rand_strided((16, 32, 31, 63, 63), (3937248, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((16, 32, 31, 63, 63), (3968000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((16, 32, 31, 63, 63), (3937248, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 512, 123039,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.755952256
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16, 32, 3, 3, 3), (864, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (32, ), (1, ))
        assert_size_stride(arg2_1, (16, 16, 16, 32, 32), (262144, 16384, 1024, 32, 1))
        assert_size_stride(arg3_1, (32, ), (1, ))
        assert_size_stride(arg4_1, (32, ), (1, ))
        assert_size_stride(arg5_1, (32, ), (1, ))
        assert_size_stride(arg6_1, (32, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf0, (16, 32, 31, 63, 63), (3937248, 123039, 3969, 63, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((16, 32, 31, 63, 63), (3968000, 124000, 4000, 63, 1), torch.float32)
            buf3 = empty_strided_cuda((16, 32, 31, 63, 63), (3937248, 123039, 3969, 63, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, mean, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.mean, aten.sub]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0.run(buf0, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, buf1, buf3, 512, 123039, stream=stream0)
            del arg1_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del buf0
            del buf1
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((16, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 16, 16, 32, 32), (262144, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)  # Subtract mean along spatial dimensions
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 123039
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_2 = (r0_index % 3969)
        r0_3 = r0_index // 3969
        tmp0 = tl.load(in_ptr0 + (r0_5 + 123039*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
        tl.store(out_ptr0 + (r0_2 + 4000*r0_3 + 124000*x4), tmp17, r0_mask & xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 3969)
        r0_3 = r0_index // 3969
        r0_5 = r0_index
        tmp21 = tl.load(out_ptr0 + (r0_2 + 4000*r0_3 + 124000*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.full([1, 1], 123039.0, tl.float32)
        tmp23 = (tmp19 / tmp22)
        tmp24 = tmp21 - tmp23
        tl.store(out_ptr2 + (r0_5 + 123039*x4), tmp24, r0_mask & xmask)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((16, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 16, 16, 32, 32), (262144, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]

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
