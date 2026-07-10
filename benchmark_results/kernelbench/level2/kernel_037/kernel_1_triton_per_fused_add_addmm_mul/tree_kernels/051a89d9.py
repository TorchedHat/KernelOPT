import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_add_addmm_mul_native_group_norm_sigmoid_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    xmask = xindex < xnumel
    x0 = (xindex % 64)   # [XBLOCK, 1]
    x3 = xindex           # [XBLOCK, 1]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]  # [1, R0_BLOCK]

    # Vectorized loads: full [XBLOCK, R0_BLOCK] tiles
    tmp0 = tl.load(in_ptr0 + (r0_index + 64 * x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_index + 64 * x3), xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_index + 64 * x0), xmask, eviction_policy='evict_last', other=0.0)

    # SiLU(tmp0 + tmp1) + tmp5
    act = tmp0 + tmp1
    act = tl.sigmoid(act) * act + tmp5

    # Two-pass reduction: mean and M2 = sum((x_i - mean)^2)
    tmp_sum = tl.sum(act, 1)[:, None]          # [XBLOCK, 1]
    tmp_mean = tmp_sum / R0_BLOCK               # [XBLOCK, 1]
    tmp_diff = act - tmp_mean                   # [XBLOCK, R0_BLOCK]
    tmp_m2 = tl.sum(tmp_diff * tmp_diff, 1)[:, None]  # [XBLOCK, 1]

    tl.store(out_ptr0 + x3, tmp_mean, xmask)   # mean
    tl.store(out_ptr1 + x3, tmp_m2,   xmask)   # M2 = sum((x_i - mean)^2)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2097152
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_addmm_mul_native_group_norm_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2097152, 64)
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
