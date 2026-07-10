import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=1),
        triton.Config({'XBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 1}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_min_native_group_norm_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_BLOCK: tl.constexpr = 64
    row_idx = tl.program_id(0)           # one block = one row
    r_idx = tl.arange(0, R0_BLOCK)      # flat [64], no 2D broadcasting
    # Load 64 contiguous floats for this row (stride-1 dim, no mask needed)
    tmp0 = tl.load(in_ptr0 + (r_idx + R0_BLOCK * row_idx))
    # Row-wise min reduction over the 64 elements
    tmp1 = tl.min(tmp0, axis=0)          # scalar result
    # Store scalar to output
    tl.store(out_ptr0 + row_idx, tmp1.to(tl.float32))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1024
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_min_native_group_norm_3[grid](arg_0, arg_1, 1024, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 1, 64), (64, 65536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 1), (1, 1024), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
