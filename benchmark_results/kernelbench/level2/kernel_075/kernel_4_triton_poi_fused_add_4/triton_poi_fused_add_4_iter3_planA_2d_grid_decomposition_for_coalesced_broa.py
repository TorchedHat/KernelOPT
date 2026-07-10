import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid decomposition: program_id(0) = column tile, program_id(1) = row tile
    # Grid is launched as (triton.cdiv(1024, XBLOCK), triton.cdiv(8192, YBLOCK))
    # With XBLOCK=1024, YBLOCK=8: grid = (1, 1024)
    YBLOCK: tl.constexpr = 8
    col_offset = tl.program_id(0) * XBLOCK
    row_offset = tl.program_id(1) * YBLOCK
    col_idx = col_offset + tl.arange(0, XBLOCK)   # [XBLOCK]
    row_idx = row_offset + tl.arange(0, YBLOCK)   # [YBLOCK]
    # Coalesced load: consecutive floats from in_ptr0 (stride-1 access)
    row_vals = tl.load(in_ptr0 + col_idx)          # [XBLOCK] loaded ONCE, reused YBLOCK times
    # Scalar broadcast: one float per row from in_ptr1
    col_vals = tl.load(in_ptr1 + row_idx)          # [YBLOCK]
    # 2D broadcast add in registers: [YBLOCK, XBLOCK]
    result = row_vals[None, :] + col_vals[:, None]
    # Output offsets: out shape (1,8192,1024,1) stride (8388608,1024,1,1)
    out_offsets = row_idx[:, None] * 1024 + col_idx[None, :]  # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + out_offsets, result)

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
