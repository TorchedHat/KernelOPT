import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Persistent kernel: 132 CTAs (one per SM), each processes multiple rows
    # TILE_SIZE = 1024 (full broadcast row), NUM_SMS = 132
    TILE_SIZE: tl.constexpr = 1024
    NUM_SMS: tl.constexpr = 132
    total_tiles = 8192  # xnumel // TILE_SIZE = 8388608 // 1024

    sm_id = tl.program_id(0)  # 0..131
    col_idx = tl.arange(0, TILE_SIZE)  # [0..1023]

    # Load in_ptr0 ONCE per CTA: 1024 floats, reused across all assigned rows
    row_vals = tl.load(in_ptr0 + col_idx)

    tile_id = sm_id
    while tile_id < total_tiles:
        # Load scalar for this row from in_ptr1
        col_val = tl.load(in_ptr1 + tile_id)
        result = row_vals + col_val
        tl.store(out_ptr0 + tile_id * TILE_SIZE + col_idx, result)
        tile_id += NUM_SMS

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
