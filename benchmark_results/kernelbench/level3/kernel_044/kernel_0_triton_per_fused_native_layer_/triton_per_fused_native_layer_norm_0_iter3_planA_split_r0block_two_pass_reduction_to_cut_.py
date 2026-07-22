import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 16}, num_warps=16, num_stages=2),
    triton.Config({'XBLOCK': 32}, num_warps=16, num_stages=2),
    triton.Config({'XBLOCK': 64}, num_warps=16, num_stages=2),
], key=['xnumel', 'r0_numel'])
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 65536
    r0_numel = 768
    CHUNK: tl.constexpr = 512  # two passes of 512 cover all 768 elements
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    xmask = xindex < xnumel
    x0 = xindex

    # Pass 1: r0 indices [0, 512) -- all valid (512 < 768)
    r0_a = tl.arange(0, CHUNK)[None, :]  # [1, 512]
    mask_a = r0_a < r0_numel  # all True since 512 < 768
    inp_a = tl.load(in_ptr0 + (r0_a + 768 * x0), mask=xmask & mask_a, other=0.0)  # [XBLOCK, 512]
    sum_a = tl.sum(inp_a, 1)[:, None]  # [XBLOCK, 1]

    # Pass 2: r0 indices [512, 1024) -- valid only for [512, 768)
    r0_b = CHUNK + tl.arange(0, CHUNK)[None, :]  # [1, 512], values 512..1023
    mask_b = r0_b < r0_numel  # True for 512..767, False for 768..1023
    inp_b = tl.load(in_ptr0 + (r0_b + 768 * x0), mask=xmask & mask_b, other=0.0)  # [XBLOCK, 512]
    sum_b = tl.sum(inp_b, 1)[:, None]  # [XBLOCK, 1]

    # Compute mean
    N = tl.full([1, 1], 768.0, tl.float32)
    mean = (sum_a + sum_b) / N  # [XBLOCK, 1]

    # Compute variance using (x - mean)^2 form for numerical stability
    diff_a = inp_a - mean
    var_a = tl.sum(diff_a * diff_a, 1)[:, None]  # [XBLOCK, 1]
    diff_b = inp_b - mean
    var_b = tl.sum(diff_b * diff_b, 1)[:, None]  # [XBLOCK, 1]
    variance = (var_a + var_b) / N

    eps = tl.full([1, 1], 1e-05, tl.float32)
    rstd = libdevice.rsqrt(variance + eps)  # [XBLOCK, 1]

    # Output pass 1: write normalized values for indices [0, 512)
    w_a = tl.load(in_ptr1 + r0_a, mask=mask_a, eviction_policy='evict_last', other=0.0)
    b_a = tl.load(in_ptr2 + r0_a, mask=mask_a, eviction_policy='evict_last', other=0.0)
    out_a = diff_a * rstd * w_a + b_a  # FMA-friendly chain
    tl.store(out_ptr2 + (r0_a + 768 * x0), out_a, mask=xmask & mask_a)

    # Output pass 2: write normalized values for indices [512, 768)
    w_b = tl.load(in_ptr1 + r0_b, mask=mask_b, eviction_policy='evict_last', other=0.0)
    b_b = tl.load(in_ptr2 + r0_b, mask=mask_b, eviction_policy='evict_last', other=0.0)
    out_b = diff_b * rstd * w_b + b_b  # FMA-friendly chain
    tl.store(out_ptr2 + (r0_b + 768 * x0), out_b, mask=xmask & mask_b)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 65536
    r0_numel = 768
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 65536, 768)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
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
