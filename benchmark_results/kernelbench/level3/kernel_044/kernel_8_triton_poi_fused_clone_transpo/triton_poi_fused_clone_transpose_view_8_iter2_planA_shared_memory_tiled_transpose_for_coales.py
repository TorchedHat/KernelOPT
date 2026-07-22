import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # XBLOCK is repurposed as TILE_Y (chunk size along x2 / seq_len dimension)
    # TILE_X = 96 (full head_dim, stride-1 in in_ptr0, hardcoded)
    # N_CHUNKS = 512 // XBLOCK (number of x2 chunks per (x3,x1) pair)
    TILE_X: tl.constexpr = 96
    N_CHUNKS: tl.constexpr = 512 // XBLOCK

    # Grid encodes (x3, x1, chunk_x2):
    #   x3 in [0, 128), x1 in [0, 8), chunk in [0, N_CHUNKS)
    pid = tl.program_id(0)
    x3    = pid // (8 * N_CHUNKS)
    x1    = (pid // N_CHUNKS) % 8
    chunk = pid % N_CHUNKS

    offs_x2 = chunk * XBLOCK + tl.arange(0, XBLOCK)   # [chunk*TILE_Y .. chunk*TILE_Y+TILE_Y-1]
    offs_x0 = tl.arange(0, TILE_X)                    # [0 .. 95]

    # Load: in_ptr0 shape (1024,512,96) strides (49152,96,1)
    # address = x3*393216 + x1*49152 + x2*96 + x0
    # Each row (fixed x2) is 96 contiguous floats -> coalesced!
    base_in = x3 * 393216 + x1 * 49152
    ptrs_in = in_ptr0 + base_in + offs_x2[:, None] * 96 + offs_x0[None, :]
    tile = tl.load(ptrs_in, eviction_policy='evict_first')

    # Store: out_ptr0 shape (128,512,8,96) strides (393216,768,96,1)
    # address = x3*393216 + x2*768 + x1*96 + x0
    # Each row (fixed x2) is 96 contiguous floats -> coalesced!
    base_out = x3 * 393216 + x1 * 96
    ptrs_out = out_ptr0 + base_out + offs_x2[:, None] * 768 + offs_x0[None, :]
    tl.store(ptrs_out, tile, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_8[grid](arg_0, arg_1, 50331648, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
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
