import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},   num_warps=1),
        triton.Config({'XBLOCK': 8},   num_warps=2),
        triton.Config({'XBLOCK': 16},  num_warps=2),
        triton.Config({'XBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
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
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]         # [1, 64]
    xmask = xindex < xnumel                             # [XBLOCK, 1]
    r0_mask = r0_index < 49                             # [1, 64]
    x0 = xindex % 1280                                  # [XBLOCK, 1]
    x1 = xindex // 1280                                 # [XBLOCK, 1]
    tmp1  = tl.load(in_ptr1 + x0, mask=xmask, eviction_policy='evict_last')   # mean   [XBLOCK, 1]
    tmp3  = tl.load(in_ptr2 + x0, mask=xmask, eviction_policy='evict_last')   # var    [XBLOCK, 1]
    tmp12 = tl.load(in_ptr3 + x0, mask=xmask, eviction_policy='evict_last')   # weight [XBLOCK, 1]
    tmp14 = tl.load(in_ptr4 + x0, mask=xmask, eviction_policy='evict_last')   # bias   [XBLOCK, 1]
    tmp5 = tmp3 + 1e-05
    tmp6 = tl.sqrt_rn(tmp5)
    inv_std = 1.0 / tmp6                                # [XBLOCK, 1]
    # Load all r0 elements in parallel: [XBLOCK, R0_BLOCK]
    val = tl.load(in_ptr0 + (x0 + 1280 * r0_index + 62720 * x1), mask=xmask & r0_mask, other=0.0)
    # BN + ReLU elementwise on 2D tile
    tmp2  = val - tmp1
    tmp11 = tmp2 * inv_std
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    relu_out = tl.maximum(tl.zeros([XBLOCK, R0_BLOCK], tl.float32), tmp15)
    # Parallel reduction: zero out padded elements (indices 49..63), then sum
    result = tl.sum(tl.where(r0_mask, relu_out, tl.zeros([XBLOCK, R0_BLOCK], tl.float32)), axis=1) / 49.0
    # result shape is [XBLOCK]; xindex/xmask are [XBLOCK,1] — reshape to 1D for store
    xindex_1d = tl.reshape(xindex, [XBLOCK])
    xmask_1d  = tl.reshape(xmask,  [XBLOCK])
    tl.store(in_out_ptr0 + xindex_1d, result, mask=xmask_1d)

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
