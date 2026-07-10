import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_convolution_min_2(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    x0_w = x0 % 256
    x0_h = (x0 // 256) % 256
    x0_n = x0 // 65536
    acc = tl.full([XBLOCK, 1], float('inf'), dtype=tl.float32)
    for r0_start in tl.static_range(0, 128, R0_BLOCK):
        r0_index = r0_start + tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < r0_numel
        combined_mask = xmask & r0_mask
        in_offset = r0_index + 128*x0_w + 32768*x0_h + 8388608*x0_n
        tmp0 = tl.load(in_ptr0 + in_offset, mask=combined_mask, other=float('inf'), eviction_policy='evict_first')
        tmp1 = tl.load(in_ptr1 + r0_index, mask=r0_mask, other=float('inf'), eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tile_min = triton_helpers.min2(tmp2, 1)[:, None]
        acc = tl.minimum(acc, tile_min)
    out_offset = x0_w + 256*x0_h + 65536*x0_n
    tl.store(out_ptr0 + out_offset, acc.to(tl.float32), mask=xmask)

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
