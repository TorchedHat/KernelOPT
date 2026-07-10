import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid_x = tl.program_id(0)  # column tile index
    pid_y = tl.program_id(1)  # row index (one row per CTA in Y)
    col_offsets = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK] column indices
    # in_ptr0: shape (1024,1), strides (1,1024) -> element i at offset i*1 = i
    tmp0 = tl.load(in_ptr0 + col_offsets)  # coalesced [XBLOCK] floats
    # in_ptr1: shape (1,8192,1,1), strides (8192,1,1,1) -> element [0,j,0,0] at offset j*1 = j
    tmp1 = tl.load(in_ptr1 + pid_y)  # scalar broadcast, one load per CTA
    # out_ptr0: shape (1,8192,1024,1), strides (8388608,1024,1,1) -> [0,j,i,0] at j*1024 + i*1
    tl.store(out_ptr0 + pid_y * 1024 + col_offsets, tmp0 + tmp1)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_4[grid](arg_0, arg_1, arg_2, 8388608, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1024, 1), (1, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 8192, 1024, 1), (8388608, 1024, 1, 1), device='cuda:0', dtype=torch.float32)
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
