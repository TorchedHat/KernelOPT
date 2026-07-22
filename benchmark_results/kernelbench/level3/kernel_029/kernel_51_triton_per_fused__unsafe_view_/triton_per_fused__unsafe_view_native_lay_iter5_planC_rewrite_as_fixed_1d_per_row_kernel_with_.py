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
        triton.Config({'XBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 1}, num_warps=16),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_51(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    # With XBLOCK=1, each program handles exactly one row
    x0 = tl.program_id(0)           # scalar row index
    xmask = x0 < xnumel             # scalar bool guard
    # 1D reduction index over the row
    r0_index = tl.arange(0, R0_BLOCK)   # shape [R0_BLOCK]
    r0_mask = r0_index < r0_numel        # shape [R0_BLOCK]
    # Load one full row: shape [R0_BLOCK]
    tmp0 = tl.load(in_ptr0 + (r0_index + 768 * x0), mask=r0_mask, other=0.0)
    # Mean: masked sum then divide
    tmp_sum = tl.sum(tl.where(r0_mask, tmp0, 0.0), axis=0)  # scalar
    mean = tmp_sum / 768.0                                    # scalar
    # Variance numerator: sum of squared deviations
    diff = tmp0 - mean                                        # [R0_BLOCK]
    sq = diff * diff                                          # [R0_BLOCK]
    var_sum = tl.sum(tl.where(r0_mask, sq, 0.0), axis=0)    # scalar
    # Store mean and variance sum for this row
    tl.store(out_ptr0 + x0, mean, mask=xmask)
    tl.store(out_ptr1 + x0, var_sum, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 490
    r0_numel = 768
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__unsafe_view_native_layer_norm_51[grid](arg_0, arg_1, arg_2, 490, 768)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 49, 1), (49, 1, 490), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
