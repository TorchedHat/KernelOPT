import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1[grid](arg_0, arg_1, 8388608)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 32, 32, 64, 64), (4194304, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
