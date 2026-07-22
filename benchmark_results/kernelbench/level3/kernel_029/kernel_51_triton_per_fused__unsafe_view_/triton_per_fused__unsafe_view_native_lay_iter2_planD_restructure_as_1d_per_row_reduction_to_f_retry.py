import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__unsafe_view_native_layer_norm_51(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 490
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    x0 = tl.program_id(0)  # scalar row index, one block per row
    xmask = x0 < xnumel
    r0_index = tl.arange(0, R0_BLOCK)  # 1D shape [1024]
    r0_mask = r0_index < r0_numel      # 1D bool [1024]
    tmp0 = tl.load(in_ptr0 + (r0_index + 768 * x0), mask=r0_mask & xmask, other=0.0)  # [1024]
    tmp_sum = tl.sum(tl.where(r0_mask, tmp0, 0.0), axis=0).to(tl.float32)  # scalar
    mean = tmp_sum / 768.0
    diff = tmp0 - mean
    diff2 = tl.where(r0_mask, diff * diff, 0.0)
    var_sum = tl.sum(diff2, axis=0).to(tl.float32)  # scalar
    tl.store(out_ptr0 + x0, mean, mask=xmask)    # scalar store
    tl.store(out_ptr1 + x0, var_sum, mask=xmask) # scalar store

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 490
    r0_numel = 768
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__unsafe_view_native_layer_norm_51[grid](arg_0, arg_1, arg_2, 490, 768, XBLOCK=128)
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
