import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12800
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]          # [1, R0_BLOCK]
    xmask = (xoffset + tl.arange(0, XBLOCK)) < xnumel  # [XBLOCK]
    r0_mask = r0_index < r0_numel                        # [1, R0_BLOCK]
    x0 = xindex % 1280                                   # [XBLOCK, 1]
    x1 = xindex // 1280                                  # [XBLOCK, 1]
    x0_1d = (xoffset + tl.arange(0, XBLOCK)) % 1280    # [XBLOCK]
    # Load BN params with 1D xmask
    tmp1  = tl.load(in_ptr1 + x0_1d, mask=xmask, eviction_policy='evict_last')  # [XBLOCK]
    tmp3  = tl.load(in_ptr2 + x0_1d, mask=xmask, eviction_policy='evict_last')  # [XBLOCK]
    tmp12 = tl.load(in_ptr3 + x0_1d, mask=xmask, eviction_policy='evict_last')  # [XBLOCK]
    tmp14 = tl.load(in_ptr4 + x0_1d, mask=xmask, eviction_policy='evict_last')  # [XBLOCK]
    inv_std = 1.0 / tl.sqrt_rn(tmp3 + 1e-05)            # [XBLOCK]
    # Expand BN params to [XBLOCK, 1] for 2D broadcasting
    tmp1_2d  = tmp1[:, None]
    tmp12_2d = tmp12[:, None]
    tmp14_2d = tmp14[:, None]
    inv_std_2d = inv_std[:, None]
    # Load all 49 reduction elements in one 2D tile [XBLOCK, R0_BLOCK]
    val = tl.load(
        in_ptr0 + (x0 + 1280 * r0_index + 62720 * x1),
        mask=xmask & r0_mask, other=0.0
    )
    # Apply BN + ReLU elementwise on 2D tile
    normed = (val - tmp1_2d) * inv_std_2d * tmp12_2d + tmp14_2d  # [XBLOCK, R0_BLOCK]
    relu_out = tl.maximum(tl.zeros([XBLOCK, R0_BLOCK], tl.float32), normed)
    # Zero out padding elements (r0_index >= r0_numel) before summing
    relu_valid = tl.where(r0_mask, relu_out, tl.zeros([XBLOCK, R0_BLOCK], tl.float32))
    result = tl.sum(relu_valid, axis=1) / 49.0           # [XBLOCK]
    # Store 1D result
    out_index = xoffset + tl.arange(0, XBLOCK)
    out_guard = out_index < xnumel
    tl.store(in_out_ptr0 + out_index, result, mask=out_guard)

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
