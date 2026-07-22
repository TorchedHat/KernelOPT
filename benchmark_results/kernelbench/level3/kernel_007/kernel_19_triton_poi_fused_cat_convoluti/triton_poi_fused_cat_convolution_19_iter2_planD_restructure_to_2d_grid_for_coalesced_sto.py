import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_convolution_19(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: pid_x covers x0 (column/fast dim, width=32), pid_y covers x1 (row/slow dim)
    pid_x = tl.program_id(0)  # covers x0 dimension
    pid_y = tl.program_id(1)  # covers x1 dimension
    YBLOCK: tl.constexpr = 8
    x0 = pid_x * XBLOCK + tl.arange(0, XBLOCK)   # shape [XBLOCK]
    x1 = pid_y * YBLOCK + tl.arange(0, YBLOCK)   # shape [YBLOCK]
    x0_2d = x0[None, :]   # [1, XBLOCK]
    x1_2d = x1[:, None]   # [YBLOCK, 1]
    xmask = x0_2d < 32    # guard x0 < 32
    ymask = x1_2d < 7840  # guard x1 < 7840
    mask = xmask & ymask
    tmp0 = tl.load(in_ptr0 + (x1_2d * 32 + x0_2d), mask)
    tl.store(out_ptr0 + (x0_2d + 256 * x1_2d), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 250880
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_19[grid](arg_0, arg_1, 250880, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((7840, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
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
