import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 2048}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 251983872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = ((xindex // 123039) % 16)
    x0 = (xindex % 3969)
    x5 = xindex // 3969
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (0), mask=True, other=0.0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x0 + 4000*x5), tmp5, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 251983872
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_mul_0[grid](arg_0, arg_1, arg_2, arg_3, 251983872)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 31, 63, 63), (1984000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
