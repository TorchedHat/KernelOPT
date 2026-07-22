import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 64}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK: tl.constexpr):
    # XBLOCK here covers the spatial/batch dimension (ynumel = xnumel // 160 = 31360)
    # Channel dimension is fixed at 256 (with masking to 160)
    CDIM: tl.constexpr = 256

    ynumel = xnumel // 160  # = 31360

    # channel dimension: fixed 256 with masking to 160
    x0 = tl.arange(0, CDIM)[:, None]      # [256, 1] channel indices
    xmask = x0 < 160                       # [256, 1] valid channel mask

    # spatial/batch dimension: tiled with XBLOCK (autotuned)
    yoffset = tl.program_id(0) * XBLOCK
    y0 = yoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK] spatial indices
    ymask = y0 < ynumel                    # [1, XBLOCK]

    full_mask = xmask & ymask              # [256, XBLOCK]

    # BN params: load once per channel, broadcast across XBLOCK spatial elements (L1 reuse)
    tmp23 = tl.load(in_ptr4 + x0, mask=xmask, eviction_policy='evict_last', other=0.0)  # mean
    tmp25 = tl.load(in_ptr5 + x0, mask=xmask, eviction_policy='evict_last', other=0.0)  # var
    tmp34 = tl.load(in_ptr6 + x0, mask=xmask, eviction_policy='evict_last', other=0.0)  # weight
    tmp36 = tl.load(in_ptr7 + x0, mask=xmask, eviction_policy='evict_last', other=0.0)  # bias

    # 4 unconditional masked loads (no tl.where chain):
    # channels 0-63 from in_ptr0 (stride: 64 per spatial element)
    mask0 = (x0 < 64) & ymask
    val0 = tl.load(in_ptr0 + (64 * y0 + x0), mask=mask0, eviction_policy='evict_last', other=0.0)

    # channels 64-95 from in_ptr1 (stride: 32 per spatial element)
    mask1 = (x0 >= 64) & (x0 < 96) & ymask
    val1 = tl.load(in_ptr1 + (32 * y0 + (x0 - 64)), mask=mask1, eviction_policy='evict_last', other=0.0)

    # channels 96-127 from in_ptr2 (stride: 32 per spatial element)
    mask2 = (x0 >= 96) & (x0 < 128) & ymask
    val2 = tl.load(in_ptr2 + (32 * y0 + (x0 - 96)), mask=mask2, eviction_policy='evict_last', other=0.0)

    # channels 128-159 from in_ptr3 (stride: 32 per spatial element)
    mask3 = (x0 >= 128) & xmask & ymask
    val3 = tl.load(in_ptr3 + (32 * y0 + (x0 - 128)), mask=mask3, eviction_policy='evict_last', other=0.0)

    # Sum: exactly one non-zero per channel row (masks are mutually exclusive)
    tmp22 = val0 + val1 + val2 + val3

    # BN + ReLU (unchanged arithmetic)
    tmp24 = tmp22 - tmp23
    tmp26 = tl.full([1, 1], 1e-05, tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp33 = tmp24 * tmp28
    tmp37 = libdevice.fma(tmp33, tmp34, tmp36)
    tmp38 = tl.zeros([CDIM, XBLOCK], tl.int32)
    tmp39 = tl.maximum(tmp38, tmp37)

    # Store: reconstruct flat index as 160*y0 + x0
    tl.store(in_out_ptr0 + (160 * y0 + x0), tmp39, mask=full_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    arg_8 = inputs[8]
    xnumel = 5017600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 5017600)
    return arg_8

def get_inputs():
    arg_0 = rand_strided((10, 160, 56, 56), (501760, 1, 8960, 160), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8]

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
