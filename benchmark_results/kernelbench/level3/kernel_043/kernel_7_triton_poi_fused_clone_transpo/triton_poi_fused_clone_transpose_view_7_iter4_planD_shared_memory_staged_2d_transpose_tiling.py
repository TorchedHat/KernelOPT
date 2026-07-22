import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiled kernel: each CTA handles one (x2, x3) pair, all 8 x1 values, all 96 x0 values
    # Total elements per CTA: 8 * 96 = 768; 256 threads * 3 elems each
    # grid=(512, 128) = 65536 CTAs total
    x2 = tl.program_id(0)  # in [0, 512)
    x3 = tl.program_id(1)  # in [0, 128)

    TILE_X0 = 96
    TILE_X1 = 8
    BLOCK = XBLOCK  # XBLOCK=256 at launch

    tid = tl.arange(0, XBLOCK)  # [0..255]
    # Each thread handles 3 consecutive flat elements in [0, 768)
    e0 = tid * 3 + 0
    e1 = tid * 3 + 1
    e2 = tid * 3 + 2

    # Decompose flat element index into (x1_local, x0_local)
    x1_0 = e0 // TILE_X0
    x0_0 = e0 % TILE_X0
    x1_1 = e1 // TILE_X0
    x0_1 = e1 % TILE_X0
    x1_2 = e2 // TILE_X0
    x0_2 = e2 % TILE_X0

    # Source: in_ptr0[x0 + 96*x2 + 49152*x1 + 393216*x3]
    base = 96 * x2 + 393216 * x3
    v0 = tl.load(in_ptr0 + base + x0_0 + 49152 * x1_0)
    v1 = tl.load(in_ptr0 + base + x0_1 + 49152 * x1_1)
    v2 = tl.load(in_ptr0 + base + x0_2 + 49152 * x1_2)

    # Destination: out_ptr0[x3*393216 + x2*768 + x1*96 + x0]
    dst_base = x3 * 393216 + x2 * 768
    tl.store(out_ptr0 + dst_base + x1_0 * TILE_X0 + x0_0, v0)
    tl.store(out_ptr0 + dst_base + x1_1 * TILE_X0 + x0_1, v1)
    tl.store(out_ptr0 + dst_base + x1_2 * TILE_X0 + x0_2, v2)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_7[grid](arg_0, arg_1, 50331648, XBLOCK=128)
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
