import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: pid_n tiles over flat (x2, x0) space; pid_m = x1 head index
    pid_n = tl.program_id(0)   # tile over (x2, x0) flat space [0, 524288)
    pid_m = tl.program_id(1)   # x1 head index in [0, 8)

    flat_offset = pid_n * XBLOCK
    flat_index = flat_offset + tl.arange(0, XBLOCK)  # [XBLOCK]
    flat_mask = flat_index < 524288  # 16384*32=524288, always true for XBLOCK=256

    x0 = flat_index % 32    # k dim, stride-1 in source tensor
    x2 = flat_index // 32   # n dim

    # Load: in_ptr0 shape (8,16384,32) strides (524288,32,1)
    # All threads share pid_m (x1), x0 varies stride-1 -> coalesced
    load_ptr = in_ptr0 + (pid_m * 524288 + x2 * 32 + x0)
    tmp0 = tl.load(load_ptr, mask=flat_mask)

    # Store: out_ptr0 shape (16384,8,32) strides (256,32,1)
    # xindex = x0 + x1*32 + x2*256 = x0 + pid_m*32 + x2*256
    store_ptr = out_ptr0 + (x0 + pid_m * 32 + x2 * 256)
    tl.store(store_ptr, tmp0, mask=flat_mask)

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
