import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid_n = tl.program_id(0)   # flat (x2, x0) axis, range [0, 4096)
    pid_m = tl.program_id(1)   # head (x1) axis, range [0, 8)
    n_offs = pid_n * XBLOCK + tl.arange(0, XBLOCK)  # shape [XBLOCK]
    n_mask = n_offs < 524288
    x0 = n_offs % 32           # k dimension, stride-1 in source
    x2 = n_offs // 32          # sequence dimension
    x1 = pid_m                 # head FIXED per block, no stride issue
    load_offset = n_offs + 524288 * x1  # = x0 + 32*x2 + 524288*x1, fully contiguous
    tmp0 = tl.load(in_ptr0 + load_offset, mask=n_mask)
    store_offset = x0 + 32 * x1 + 256 * x2  # matches output strides (256, 32, 1)
    tl.store(out_ptr0 + store_offset, tmp0, mask=n_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 8, 32), (256, 32, 1), device='cuda:0', dtype=torch.float32)
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
