import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: axis-0 = spatial tiles (SBLOCK=1), axis-1 = channel tiles (CBLOCK=XBLOCK=128)
    # spatial_numel = 1960, channel_numel = 480
    # grid = (1960, triton.cdiv(480, XBLOCK)) = (1960, 4)
    s_pid = tl.program_id(0)
    c_pid = tl.program_id(1)
    # SBLOCK=1: each spatial block covers exactly 1 spatial position
    s_idx = s_pid  # scalar spatial index
    c_idx = c_pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    c_mask = c_idx < 480
    x2 = s_idx * 480 + c_idx
    x0 = c_idx
    tmp0 = tl.load(in_out_ptr0 + x2, c_mask)
    tmp1 = tl.load(in_ptr0 + x0, c_mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, c_mask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + x0, c_mask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + x0, c_mask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp10 = tl.math.rsqrt(tmp5)
    tmp11 = tmp2 * tmp10
    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + x2, tmp19, c_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 940800
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_26[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 940800, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((480,), (1,), device='cuda:0', dtype=torch.float32)
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
