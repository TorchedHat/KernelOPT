import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: program_id(0) over channel dim, program_id(1) over spatial dim
    # XBLOCK covers channel dim (C=208), YBLOCK=4 hardcoded for spatial dim
    YBLOCK: tl.constexpr = 4
    C: tl.constexpr = 208
    pid_x = tl.program_id(0)  # over channel dim
    pid_y = tl.program_id(1)  # over spatial/batch dim
    x0 = pid_x * XBLOCK + tl.arange(0, XBLOCK)[None, :]  # shape [YBLOCK, XBLOCK]
    x1 = pid_y * YBLOCK + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, XBLOCK]
    x0_mask = x0 < C
    x1_mask = x1 < 501760
    mask = x0_mask & x1_mask
    flat_idx = x1 * C + x0  # contiguous read from NHWC input
    tmp0 = tl.load(in_ptr0 + flat_idx, mask=mask)
    tmp1 = tl.load(in_ptr1 + x0, mask=x0_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + 512 * x1), tmp2, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 104366080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_10[grid](arg_0, arg_1, arg_2, 104366080, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 208, 224, 224), (10436608, 1, 46592, 208), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((208,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 208, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
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
