import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    # No xmask: 134217728 % 128 == 0, all tiles are full
    x4 = xindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    # SCALAR HOIST: 2 scalar loads instead of 128-element vector gather
    # Within each XBLOCK=128 tile, xindex // 64 produces exactly 2 values:
    #   threads 0-63  -> x5_lo = program_id * 2
    #   threads 64-127 -> x5_hi = program_id * 2 + 1
    x5_lo = tl.program_id(0) * 2       # scalar int: lower-half index into in_ptr2
    x5_hi = tl.program_id(0) * 2 + 1   # scalar int: upper-half index into in_ptr2
    tmp2_lo = tl.load(in_ptr2 + x5_lo)  # scalar load
    tmp2_hi = tl.load(in_ptr2 + x5_hi)  # scalar load
    lane = tl.arange(0, XBLOCK)[:]
    tmp2 = tl.where(lane < 64, tmp2_lo, tmp2_hi)  # register-only broadcast, zero memory cost
    tmp3 = tl.exp(tmp2)
    tmp5 = libdevice.fma(tmp1, tmp3, tmp0)  # fused: tmp1 * tmp3 + tmp0
    tl.store(out_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), tmp5, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 134217728
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_clone_exp_mul_permute_unsqueeze_view_16[grid](arg_0, arg_1, arg_2, arg_3, 134217728, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 2, 64, 8, 64), (65536, 32768, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
