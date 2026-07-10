import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=3),
        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=4),
        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=4),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_add_convolution_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([1, XBLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:, None]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK, 1], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 2048) % 64)
    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), mask=r0_mask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (x1), mask=xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + tl.zeros([1, XBLOCK], tl.int32), mask=xmask)
    tmp4 = tl.broadcast_to(tmp3, [1, XBLOCK])
    tmp26 = tl.load(in_ptr2 + (r0_3), mask=r0_mask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_3), mask=r0_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    # Single-pass Konig-Huygens: compute sum(x) and sum(x^2) independently (no data dependency)
    tmp_sq = tmp5 * tmp5                                          # x^2, [R0_BLOCK, XBLOCK]
    tmp_sum    = tl.sum(tmp5, 0)[None, :].to(tl.float32)         # sum(x),   [1, XBLOCK] -- INDEPENDENT
    tmp_sum_sq = tl.sum(tmp_sq, 0)[None, :].to(tl.float32)       # sum(x^2), [1, XBLOCK] -- INDEPENDENT
    N = tl.full([1, 1], 64.0, tl.float32)
    tmp_mean = tmp_sum / N                                        # E[x], [1, XBLOCK]
    tmp_ex2  = tmp_sum_sq / N                                     # E[x^2], [1, XBLOCK]
    # Var(x) = E[x^2] - E[x]^2 via FMA: -mean*mean + E[x^2]
    tmp_var  = tl.math.fma(-tmp_mean, tmp_mean, tmp_ex2)          # [1, XBLOCK]
    tmp_eps  = tl.full([1, 1], 1e-05, tl.float32)
    tmp_rstd = libdevice.rsqrt(tmp_var + tmp_eps)                 # [1, XBLOCK]
    tmp_xnorm = (tmp5 - tmp_mean) * tmp_rstd                      # [R0_BLOCK, XBLOCK]
    # Affine transform as single FFMA instruction
    tmp29 = tl.math.fma(tmp_xnorm, tmp26, tmp28)                  # [R0_BLOCK, XBLOCK]
    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp29, mask=r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 4194304
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_convolution_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4194304, 64)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
