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


# kernel path: /mnt/podman_storage/ahpoddar/tmp/torchinductor_ahpoddar/b4/cb4ucwgnrmaxgtv5xun2y67ivbyupu5vpz7rmdd5wst4rah7uxg3.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.sigmoid]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => div
#   x_2 => sigmoid
# Graph fragment:
#   %buf0 : Tensor "f32[16, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem : Tensor "f32[16, 1, 32, 64, 64][131072, 2097152, 4096, 64, 1]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[16, 1, 32, 64, 64][131072, 2097152, 4096, 64, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %convolution : Tensor "f32[16, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2, 2], [1, 1, 1], [1, 1, 1], True, [1, 1, 1], 1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convolution, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[16, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[16, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[16, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %sigmoid : Tensor "f32[16, 64, 32, 64, 64][8388608, 131072, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%div,), kwargs = {})
#   return %getitem,%getitem_1,%sigmoid
triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0 = async_compile.triton('triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0', '''
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
    size_hints={'x': 2097152, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '24393E16EE255EA1526A6E7CA61516A01815456CCC5BB7B8EE244A91B17851CD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1610612736, 'r0_': 256}, 'kernel_num_gb': 1.07374208, 'kernel_flop': 0}
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    x0 = (xindex % 131072)
    x1 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), None)
    tmp1 = tl.load(in_ptr0 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp3 - tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tmp2 - tmp7
    tmp14 = libdevice.exp(tmp13)
    tmp15 = (tmp14 / tmp12)
    tmp16 = tl.sigmoid(tmp15)
    tl.store(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), tmp16, None)


def get_args():
    arg_0 = rand_strided((16, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 2097152, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.07374208
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
        assert_size_stride(arg0_1, (32, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
        assert_size_stride(arg1_1, (64, ), (1, ))
        assert_size_stride(arg2_1, (16, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
            assert_size_stride(buf0, (16, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.sigmoid]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0.run(buf3, arg1_1, 2097152, 64, stream=stream0)
            del arg1_1
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.softmax(x)
        x = self.sigmoid(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]'''

_TRITON_KERNEL_SOURCES = [
    '''@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    x0 = (xindex % 131072)
    x1 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), None)
    tmp1 = tl.load(in_ptr0 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tmp2 - tmp7
    tmp14 = tl.exp(tmp13)
    tmp15 = (tmp14 / tmp12)
    tmp16 = tl.sigmoid(tmp15)
    tl.store(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), tmp16, None)

'''
]

def kernel_function(*inputs):
    """kernelopt entry point -- delegates to Inductor's call()."""
    return call(list(inputs))

from torch._dynamo.testing import rand_strided

def get_inputs():
    arg0_1 = rand_strided((32, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
