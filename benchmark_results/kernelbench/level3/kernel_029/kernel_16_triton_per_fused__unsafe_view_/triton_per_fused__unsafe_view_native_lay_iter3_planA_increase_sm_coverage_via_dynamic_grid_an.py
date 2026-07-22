import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 4}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_16(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 7840
    r0_numel = 192
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    combined_mask = r0_mask & xmask
    tmp0 = tl.load(in_ptr0 + (r0_1 + 192*x0), mask=combined_mask, other=0.0)
    # Zero out-of-bounds lanes once; reuse for both sums
    tmp_vals = tl.where(combined_mask, tmp0, 0.0)
    # Compute Σx and Σx² in one pass over the same in-register tile
    tmp_sum  = tl.sum(tmp_vals, 1)[:, None].to(tl.float32)            # Σx
    tmp_sum2 = tl.sum(tmp_vals * tmp_vals, 1)[:, None].to(tl.float32) # Σx²
    tmp_n    = tl.full([XBLOCK, 1], 192.0, tl.float32)
    tmp_mean = tmp_sum / tmp_n                          # mean = Σx/N
    # König formula: Σ(x-mean)² = Σx² - (Σx)²/N  (no second global load)
    tmp_var_sum = tmp_sum2 - (tmp_sum * tmp_sum) / tmp_n
    tl.store(out_ptr0 + (x0), tmp_mean, mask=xmask)
    tl.store(out_ptr1 + (x0), tmp_var_sum, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 7840
    r0_numel = 192
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__unsafe_view_native_layer_norm_16[grid](arg_0, arg_1, arg_2, 7840, 192)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 784, 1), (784, 1, 7840), device='cuda:0', dtype=torch.float32)
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
