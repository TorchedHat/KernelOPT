import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # 2D NHWC tiling: XBLOCK spatial positions x 128 channels (masked to C=96)
    # hw_numel = N*H*W = 10*56*56 = 31360, C = 96 (stride-1 in NHWC layout)
    CBLOCK: tl.constexpr = 128   # next power-of-2 >= 96
    hw_numel = 31360
    C = 96
    hw_offset = tl.program_id(0) * XBLOCK
    hw_idx    = hw_offset + tl.arange(0, XBLOCK)   # [XBLOCK]
    hw_mask   = hw_idx < hw_numel                   # [XBLOCK]
    c_idx     = tl.arange(0, CBLOCK)               # [128]
    c_mask    = c_idx < C                           # [128] — mask out channels >= 96

    # Load per-channel stats ONCE per CTA (96 floats, L1-resident)
    mean   = tl.load(in_ptr0 + c_idx, c_mask, other=0.0)   # [128]
    var    = tl.load(in_ptr1 + c_idx, c_mask, other=0.0)
    weight = tl.load(in_ptr2 + c_idx, c_mask, other=1.0)
    bias   = tl.load(in_ptr3 + c_idx, c_mask, other=0.0)

    # Precompute fused scale/offset per channel [128]
    eps     = tl.full([CBLOCK], 1e-05, tl.float32)
    inv_std = tl.math.rsqrt(var + eps)
    scale   = weight * inv_std
    offset  = bias - mean * scale

    # 2D index: NHWC layout, channel is stride-1 (coalesced)
    flat_idx  = hw_idx[:, None] * C + c_idx[None, :]        # [XBLOCK, 128]
    flat_mask = hw_mask[:, None] & c_mask[None, :]           # [XBLOCK, 128]

    x_val  = tl.load(in_out_ptr0 + flat_idx, flat_mask, other=0.0)   # [XBLOCK, 128]
    result = x_val * scale[None, :] + offset[None, :]                  # broadcast [1, 128]
    result = tl.maximum(result, tl.full([1], 0.0, tl.float32))
    result = tl.minimum(result, tl.full([1], 6.0, tl.float32))
    tl.store(in_out_ptr0 + flat_idx, result, flat_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
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
