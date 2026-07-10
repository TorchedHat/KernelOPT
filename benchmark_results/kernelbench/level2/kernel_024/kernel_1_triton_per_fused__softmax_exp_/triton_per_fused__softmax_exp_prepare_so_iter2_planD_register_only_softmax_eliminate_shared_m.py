import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_2 = r0_index
    x0 = (xindex % 900)
    x1 = xindex // 900
    # Load with other=float('-inf') so masked lanes don't inflate the max
    tmp0 = tl.load(in_out_ptr0 + (x0 + 900 * r0_2 + 21600 * x1), mask=xmask & r0_mask, other=float('-inf'))
    # Register-only row-wise max (no shared memory, uses warp shuffles)
    tmp6 = tl.max(tmp0, axis=1, keep_dims=True)
    # Subtract max and exponentiate
    tmp7 = tmp0 - tmp6
    tmp8 = tl.exp(tmp7)
    # Zero out masked lanes for the sum
    tmp11 = tl.where(xmask & r0_mask, tmp8, 0.0)
    # Register-only row-wise sum
    tmp12 = tl.sum(tmp11, axis=1, keep_dims=True)
    # Normalize
    tmp15 = tmp11 / tmp12
    tl.store(in_out_ptr0 + (x0 + 900 * r0_2 + 21600 * x1), tmp15, mask=xmask & r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    xnumel = 115200
    r0_numel = 24
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24)
    return arg_0

def get_inputs():
    arg_0 = rand_strided((128, 24, 30, 30), (21600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0]

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
