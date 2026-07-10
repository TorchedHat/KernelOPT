import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused_add_addmm_mul_native_group_norm_sigmoid_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    x0 = (xindex % 64)   # [XBLOCK, 1]
    x3 = xindex           # [XBLOCK, 1]

    # Single-pass Welford: only [XBLOCK,1] accumulators live at any time
    welford_mean = tl.zeros([XBLOCK, 1], dtype=tl.float32)
    welford_m2   = tl.zeros([XBLOCK, 1], dtype=tl.float32)

    for r in tl.static_range(R0_BLOCK):  # fully unrolled (R0_BLOCK=64 constexpr)
        r_idx = tl.full([XBLOCK, 1], r, dtype=tl.int32)
        val0 = tl.load(in_ptr0 + (r_idx + 64 * x0), None, eviction_policy='evict_last')
        val1 = tl.load(in_ptr1 + (r_idx + 64 * x3), None)
        val2 = tl.load(in_ptr2 + (r_idx + 64 * x0), None, eviction_policy='evict_last')
        # SiLU(val0+val1) + val2  (same formula as original)
        act = val0 + val1
        act = tl.sigmoid(act) * act + val2
        # Welford online update for iteration r (count = r+1)
        count = tl.full([XBLOCK, 1], r + 1, dtype=tl.float32)
        delta = act - welford_mean
        welford_mean = welford_mean + delta / count
        delta2 = act - welford_mean
        welford_m2 = welford_m2 + delta * delta2

    tl.store(out_ptr0 + x3, welford_mean, None)  # mean  (matches tmp14)
    tl.store(out_ptr1 + x3, welford_m2,   None)  # M2 = sum((x_i-mean)^2) (matches tmp19)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2097152
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_addmm_mul_native_group_norm_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2097152, 64, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32768, 64, 1, 1), (64, 1, 2097152, 2097152), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32768, 64, 1, 1), (64, 1, 2097152, 2097152), device='cuda:0', dtype=torch.float32)
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
