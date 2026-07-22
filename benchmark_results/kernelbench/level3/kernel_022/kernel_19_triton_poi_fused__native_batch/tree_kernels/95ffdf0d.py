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
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    CHANNELS: tl.constexpr = 240
    SPATIAL: tl.constexpr = 7840
    pid_x = tl.program_id(0)
    s_offset = pid_x * XBLOCK
    s_idx = s_offset + tl.arange(0, XBLOCK)   # [XBLOCK]
    s_mask = s_idx < SPATIAL
    c_idx = tl.arange(0, 256)                  # [256], next power-of-2 >= 240
    c_mask = c_idx < CHANNELS
    flat_idx = s_idx[:, None] * CHANNELS + c_idx[None, :]  # [XBLOCK, 256]
    full_mask = s_mask[:, None] & c_mask[None, :]
    tmp0  = tl.load(in_out_ptr0 + flat_idx, full_mask)
    tmp1  = tl.load(in_ptr0 + c_idx[None, :], c_mask[None, :], eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr1 + c_idx[None, :], c_mask[None, :], eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + c_idx[None, :], c_mask[None, :], eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + c_idx[None, :], c_mask[None, :], eviction_policy='evict_last')
    tmp2  = tmp0 - tmp1
    tmp5  = tmp3 + tl.full([1, 1], 1e-05, tl.float32)
    tmp10 = tl.math.rsqrt(tmp5)
    tmp11 = tmp2 * tmp10
    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
    tmp17 = tl.maximum(tmp15, tl.full([1, 1], 0.0, tl.float32))
    tmp19 = tl.minimum(tmp17, tl.full([1, 1], 6.0, tl.float32))
    tl.store(in_out_ptr0 + flat_idx, tmp19, full_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1881600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
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
