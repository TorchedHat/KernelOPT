import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64},   num_warps=4),
        triton.Config({'XBLOCK': 64},   num_warps=8),
        triton.Config({'XBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 128},  num_warps=8),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2764800
    r0_numel = 22
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = (xindex % 900)
    x4 = xindex // 900
    x1 = ((xindex // 900) % 24)
    x5 = xindex

    # Load bias once, reuse in both passes
    tmp_bias = tl.load(in_ptr1 + (x1), mask=xmask, eviction_policy='evict_last', other=0.0)

    # Pass 1: r0 in [0, 16) — all 16 lanes valid (16 < 22), NO r0_mask needed
    r0_idx_p1 = tl.arange(0, R0_BLOCK)[None, :]   # values 0..15
    tmp0_p1 = tl.load(in_ptr0 + (x0 + 900*r0_idx_p1 + 19800*x4), mask=xmask, other=0.0, eviction_policy='evict_first')
    tmp2_p1 = tmp0_p1 + tmp_bias
    tmp_min_p1 = triton_helpers.min2(tl.broadcast_to(tmp2_p1, [XBLOCK, R0_BLOCK]), 1)[:, None]

    # Pass 2: r0 in [16, 32) — only indices 16..21 are valid (6 of 16 lanes)
    r0_idx_p2 = 16 + tl.arange(0, R0_BLOCK)[None, :]  # values 16..31
    r0_mask_p2 = r0_idx_p2 < r0_numel                  # True for 16..21 only
    tmp0_p2 = tl.load(in_ptr0 + (x0 + 900*r0_idx_p2 + 19800*x4), mask=xmask & r0_mask_p2, other=0.0, eviction_policy='evict_first')
    tmp2_p2 = tmp0_p2 + tmp_bias
    tmp5_p2 = tl.where(r0_mask_p2, tl.broadcast_to(tmp2_p2, [XBLOCK, R0_BLOCK]), float('inf'))
    tmp_min_p2 = triton_helpers.min2(tmp5_p2, 1)[:, None]

    # Combine partial mins and store
    tmp6 = tl.minimum(tmp_min_p1, tmp_min_p2).to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp6, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 2764800
    r0_numel = 22
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_min_0[grid](arg_0, arg_1, arg_2, 2764800, 22)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 24, 30, 30), (21600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
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
