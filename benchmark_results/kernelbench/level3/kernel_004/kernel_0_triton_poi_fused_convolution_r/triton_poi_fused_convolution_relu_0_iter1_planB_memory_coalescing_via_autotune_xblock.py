import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 784) % 6)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 19267584
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_relu_0[grid](arg_0, arg_1, 19267584)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 6, 28, 28), (4704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")
