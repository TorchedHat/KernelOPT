import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8},   num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 64  # next power-of-2 >= r0_numel=49
    xnumel = 12800
    r0_numel = 49
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    r0index = tl.arange(0, R0_BLOCK)[None, :]           # [1, R0_BLOCK]
    xmask  = (xindex < xnumel)                          # [XBLOCK, 1]
    r0mask = (r0index < 49)                             # [1, R0_BLOCK]
    x0 = xindex % 1280                                  # [XBLOCK, 1]
    x1 = xindex // 1280                                 # [XBLOCK, 1]
    mean   = tl.load(in_ptr1 + x0, mask=xmask, eviction_policy='evict_last')  # [XBLOCK, 1]
    var    = tl.load(in_ptr2 + x0, mask=xmask, eviction_policy='evict_last')  # [XBLOCK, 1]
    weight = tl.load(in_ptr3 + x0, mask=xmask, eviction_policy='evict_last')  # [XBLOCK, 1]
    bias   = tl.load(in_ptr4 + x0, mask=xmask, eviction_policy='evict_last')  # [XBLOCK, 1]
    inv_std = 1.0 / tl.sqrt_rn(var + 1e-05)             # [XBLOCK, 1]
    # Combined mask for 2D load: both x and r0 must be in bounds
    xr_mask = xmask & r0mask                            # [XBLOCK, R0_BLOCK]
    # Single vectorized 2D load of ALL reduction elements at once
    val = tl.load(in_ptr0 + (x0 + 1280 * r0index + 62720 * x1),
                  mask=xr_mask, other=0.0)              # [XBLOCK, R0_BLOCK]
    normed   = (val - mean) * inv_std                   # [XBLOCK, R0_BLOCK]
    scaled   = normed * weight + bias                   # [XBLOCK, R0_BLOCK]
    relu_out = tl.maximum(scaled, 0.0)                  # [XBLOCK, R0_BLOCK]
    # Zero out padding elements (r0index >= 49) before summing
    acc_2d   = tl.where(xr_mask, relu_out, tl.zeros([XBLOCK, R0_BLOCK], tl.float32))
    result   = tl.sum(acc_2d, axis=1) / 49.0            # [XBLOCK]
    out_idx  = xoffset + tl.arange(0, XBLOCK)           # [XBLOCK]
    tl.store(in_out_ptr0 + out_idx, result, mask=(out_idx < xnumel))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 12800
    r0_numel = 49
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12800, 49)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 1280, 1, 1), (1280, 1, 12800, 12800), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((490, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1280,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
