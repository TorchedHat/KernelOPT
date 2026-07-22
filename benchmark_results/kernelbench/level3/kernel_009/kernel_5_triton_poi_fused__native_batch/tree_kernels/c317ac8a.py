import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 4},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 8},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=8,  num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    # CBLOCK=64 channels per block (hardcoded power-of-2 for cheap bitwise ops)
    CBLOCK: tl.constexpr = 64
    spatial_numel = 31360
    C = 60
    pid0 = tl.program_id(0)  # spatial tile
    pid1 = tl.program_id(1)  # channel tile
    # Linearize 2D block: tid = spatial_local * CBLOCK + channel_local
    # => consecutive threads = consecutive channels => stride-1 coalesced!
    tids = tl.arange(0, XBLOCK * CBLOCK)[:]
    spatial_local = tids // CBLOCK   # [0..XBLOCK)
    channel_local = tids % CBLOCK    # [0..CBLOCK)
    spatial_idx = pid0 * XBLOCK + spatial_local
    channel_idx = pid1 * CBLOCK + channel_local
    smask = spatial_idx < spatial_numel
    cmask = channel_idx < C
    mask = smask & cmask
    flat_idx = spatial_idx * C + channel_idx  # stride-1 within warp!
    tmp0  = tl.load(in_out_ptr0 + flat_idx, mask)
    tmp1  = tl.load(in_ptr0  + channel_idx, mask, eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr1  + channel_idx, mask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2  + channel_idx, mask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3  + channel_idx, mask, eviction_policy='evict_last')
    tmp2  = tmp0 - tmp1
    tmp4  = tl.full([XBLOCK * CBLOCK], 1e-05, tl.float32)
    tmp5  = tmp3 + tmp4
    tmp6  = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp6
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.maximum(0.0, tmp15)
    tl.store(in_out_ptr0 + flat_idx, tmp17, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1881600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 60, 56, 56), (188160, 1, 3360, 60), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((60,), (1,), device='cuda:0', dtype=torch.float32)
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
