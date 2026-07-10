import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 7)
    x2 = ((xindex // 49) % 3)
    x3 = xindex // 147
    x4 = xindex
    base = 2*x0 + 30*x1 + 450*x2 + 1600*x3
    pool_i = tl.arange(0, 8)
    d_x = pool_i % 2
    d_y = (pool_i // 2) % 2
    d_z = pool_i // 4
    win = d_x + 15*d_y + 225*d_z
    all_offs = base[:, None] + win[None, :]
    vals = tl.load(in_ptr0 + all_offs, xmask[:, None], other=float('-inf'), eviction_policy='evict_first')
    result = tl.max(vals, axis=1)
    tl.store(out_ptr0 + x4, result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 301056
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2[grid](arg_0, arg_1, 301056)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 3, 7, 7), (2352, 147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
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
