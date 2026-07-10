import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    # xnumel = 115200 = 128 * 900 (batch x spatial)
    # r0_numel = 24 (heads, reduction axis)
    # Each CTA handles ONE (batch, spatial) position
    # XBLOCK is kept in signature for compatibility but unused in 1D layout
    R0_BLOCK: tl.constexpr = 32

    pid = tl.program_id(0)       # one CTA per (batch, spatial) pair
    spatial_idx = pid % 900      # x0: position within HW (stride=1 in global)
    batch_idx   = pid // 900     # x1: batch index

    # Head indices 0..23, padded to R0_BLOCK=32
    r_idx  = tl.arange(0, R0_BLOCK)        # [R0_BLOCK] contiguous in register tile
    r_mask = r_idx < r0_numel              # mask out indices 24..31

    # Address: matches original formula x0 + 900*r0_2 + 21600*x1 exactly
    addr = spatial_idx + 900 * r_idx + 21600 * batch_idx  # [R0_BLOCK]

    # Load all 24 head values; masked lanes get -inf (safe for max)
    vals = tl.load(in_out_ptr0 + addr, mask=r_mask, other=float('-inf'),
                   eviction_policy='evict_first')  # [R0_BLOCK]

    # Softmax in registers: no shared memory staging, no bank conflicts
    row_max  = tl.max(tl.where(r_mask, vals, float('-inf')), axis=0)  # scalar
    shifted  = vals - row_max                                           # [R0_BLOCK]
    exp_vals = tl.exp(shifted)                                          # [R0_BLOCK]
    exp_sum  = tl.sum(tl.where(r_mask, exp_vals, 0.0), axis=0)        # scalar
    result   = exp_vals / exp_sum                                       # [R0_BLOCK]

    # Store back (only valid head indices)
    tl.store(in_out_ptr0 + addr, result, mask=r_mask,
             eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    xnumel = 115200
    r0_numel = 24
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24, XBLOCK=128)
    return arg_0

def get_inputs():
    arg_0 = rand_strided((128, 24, 30, 30), (21600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0]

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
