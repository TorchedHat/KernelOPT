import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # grid = (8192, 1): axis-0 is row (0..8191), axis-1 unused
    # Eliminates xindex % 1024 and xindex // 1024 entirely
    row_id = tl.program_id(0)   # directly the x1 index (0..8191), no division needed
    col_idx = tl.arange(0, XBLOCK)  # 0..1023, compile-time constant, no modulo needed

    # in_ptr0: rand_strided((1024,1), (1,1024)) -> stride[0]=1
    # col_idx=0..1023 with stride 1 -> coalesced load of 1024 consecutive floats
    tmp0 = tl.load(in_ptr0 + col_idx)

    # in_ptr1: rand_strided((1,8192,1,1), (8192,1,1,1)) -> stride[1]=1
    # row_id indexes the size-8192 dim which has stride 1 -> scalar load
    tmp1 = tl.load(in_ptr1 + row_id)

    # out_ptr0: rand_strided((1,8192,1024,1), (8388608,1024,1,1)) -> stride[1]=1024, stride[2]=1
    # row row_id starts at row_id*1024, col col_idx has stride 1
    out_offset = row_id * XBLOCK + col_idx
    tl.store(out_ptr0 + out_offset, tmp0 + tmp1)

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
