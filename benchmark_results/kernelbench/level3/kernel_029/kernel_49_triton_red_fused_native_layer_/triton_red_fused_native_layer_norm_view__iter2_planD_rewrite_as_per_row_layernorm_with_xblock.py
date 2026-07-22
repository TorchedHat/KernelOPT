import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['r0_numel'],
)
@triton.jit
def triton_red_fused_native_layer_norm_view_49(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 490
    r0_numel = 1536
    x0 = tl.program_id(0)  # scalar row index — one block per row
    xmask = x0 < xnumel    # scalar boolean
    r0_base = tl.arange(0, R0_BLOCK)  # 1D shape [R0_BLOCK]
    tmp2_mean   = tl.zeros([R0_BLOCK], tl.float32)
    tmp2_m2     = tl.zeros([R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([R0_BLOCK], tl.float32)
    # Pass 1: Welford online accumulation over columns
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        tmp0 = tl.load(in_ptr0 + (r0_index + 1536 * x0), mask=r0_mask & xmask, other=0.0, eviction_policy='evict_last')
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp0, tmp2_mean, tmp2_m2, tmp2_weight, r0_offset == 0
        )
        tmp2_mean   = tl.where(r0_mask & xmask, tmp2_mean_next,   tmp2_mean)
        tmp2_m2     = tl.where(r0_mask & xmask, tmp2_m2_next,     tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    # Final reduction across R0_BLOCK tiles -> scalars
    mean, m2, weight = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 0)  # dim=0
    rstd = libdevice.rsqrt(m2 / 1536.0 + 1e-5)  # scalar
    # Pass 2: normalize + scale + shift
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        tmp8  = tl.load(in_ptr0 + (r0_index + 1536 * x0), mask=r0_mask & xmask, other=0.0, eviction_policy='evict_first')
        tmp16 = tl.load(in_ptr1 + r0_index, mask=r0_mask, other=0.0, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr2 + r0_index, mask=r0_mask, other=0.0, eviction_policy='evict_last')
        out = (tmp8 - mean) * rstd * tmp16 + tmp18
        tl.store(out_ptr2 + (r0_index + 1536 * x0), out, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 490
    r0_numel = 1536
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_native_layer_norm_view_49[grid](arg_0, arg_1, arg_2, arg_3, 490, 1536)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10, 7, 7, 1536), (75264, 10752, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 49, 1536), (75264, 1536, 1), device='cuda:0', dtype=torch.float32)
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
