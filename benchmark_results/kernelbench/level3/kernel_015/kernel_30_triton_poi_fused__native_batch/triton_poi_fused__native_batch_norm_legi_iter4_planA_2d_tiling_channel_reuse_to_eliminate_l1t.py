import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(
        in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0,
        xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: XBLOCK = number of spatial rows per block (autotuned: 1, 2, 4)
    # Channel dim is fixed at 384 (stride-1). Use CBLOCK=512 (power of 2) with mask.
    CBLOCK: tl.constexpr = 512
    # xnumel = 7840 = 3010560 // 384 (batch * H * W spatial positions)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1] — spatial rows
    cindex = tl.arange(0, CBLOCK)[None, :]              # [1, 512]   — channel
    cmask = cindex < 384                                 # [1, 512]

    # Load channel vectors ONCE per block — broadcast across XBLOCK rows
    tmp1  = tl.load(in_ptr1 + cindex, mask=cmask, other=0.0, eviction_policy='evict_last')  # mean
    tmp3  = tl.load(in_ptr2 + cindex, mask=cmask, other=0.0, eviction_policy='evict_last')  # var
    tmp12 = tl.load(in_ptr3 + cindex, mask=cmask, other=0.0, eviction_policy='evict_last')  # weight
    tmp14 = tl.load(in_ptr4 + cindex, mask=cmask, other=0.0, eviction_policy='evict_last')  # bias

    # Hoist rsqrt: computed once per channel per block
    tmp5 = tmp3 + 1e-05
    tmp6 = tl.math.rsqrt(tmp5)  # [1, 512]

    # 2D flat index: xindex*384 + cindex = original x2
    flat_idx = xindex * 384 + cindex  # [XBLOCK, 512]

    # Load main tensor tile [XBLOCK, 512] — coalesced along channel (stride-1)
    tmp0 = tl.load(in_ptr0 + flat_idx, mask=cmask, other=0.0)  # [XBLOCK, 512]

    # BN + ReLU
    tmp2  = tmp0 - tmp1
    tmp11 = tmp2 * tmp6
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.maximum(tl.zeros([XBLOCK, CBLOCK], dtype=tl.float32), tmp15)
    tl.store(out_ptr0 + flat_idx, tmp17, mask=cmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 3010560)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 384, 28, 28), (301056, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 384, 28, 28), (301056, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
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
