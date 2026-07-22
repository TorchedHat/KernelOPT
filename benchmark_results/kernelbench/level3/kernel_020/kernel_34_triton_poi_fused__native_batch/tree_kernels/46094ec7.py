import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=8,  num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: program_id(0) tiles over 960 channels, program_id(1) over 490 outer positions
    # XBLOCK tiles the channel dimension (must be power of 2, <= 960)
    # Each program_id(1) handles exactly 1 outer position
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)

    c_offset = pid_c * XBLOCK + tl.arange(0, XBLOCK)
    c_mask = c_offset < 960

    # flat index into in_out_ptr0: outer_pos * 960 + channel
    flat_idx = pid_r * 960 + c_offset
    full_mask = (pid_r < (xnumel // 960)) & c_mask

    # Channel-broadcast loads (no modulo needed)
    tmp1  = tl.load(in_ptr0 + c_offset, c_mask, eviction_policy='evict_last', other=0.0)
    tmp3  = tl.load(in_ptr1 + c_offset, c_mask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + c_offset, c_mask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + c_offset, c_mask, eviction_policy='evict_last', other=0.0)

    # Main tensor load
    tmp0 = tl.load(in_out_ptr0 + flat_idx, full_mask, eviction_policy='evict_first', other=0.0)

    # Batch norm + hardtanh computation
    tmp2  = tmp0 - tmp1
    tmp5  = tmp3 + 1e-05
    tmp6  = tl.math.rsqrt(tmp5)
    tmp11 = tmp2 * tmp6
    tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
    tmp17 = tl.maximum(tmp15, 0.0)
    tmp19 = tl.minimum(tmp17, 6.0)
    tl.store(in_out_ptr0 + flat_idx, tmp19, full_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 470400
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 470400)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
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
