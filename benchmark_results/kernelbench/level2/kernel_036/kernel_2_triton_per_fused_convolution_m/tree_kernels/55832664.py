import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=1),
        triton.Config({'XBLOCK': 1},  num_warps=2),
        triton.Config({'XBLOCK': 2},  num_warps=1),
        triton.Config({'XBLOCK': 2},  num_warps=2),
        triton.Config({'XBLOCK': 4},  num_warps=2),
        triton.Config({'XBLOCK': 4},  num_warps=4),
        triton.Config({'XBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_convolution_min_2(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.max_contiguous(tl.multiple_of(tl.arange(0, R0_BLOCK), R0_BLOCK), R0_BLOCK)[None, :]
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr1 + (r0_1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = triton_helpers.min2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp5, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 1048576
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_min_2[grid](arg_0, arg_1, arg_2, 1048576, 128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 128, 256, 256), (8388608, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 1, 256, 256), (65536, 1048576, 256, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
