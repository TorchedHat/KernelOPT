import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Use a square BLOCK tile for coalesced transpose
    # BLOCK = 32, grid = (xnumel//BLOCK, ynumel//BLOCK) = (1024, 32)
    BLOCK = 32
    # pid(0) covers x dimension (0..1023), pid(1) covers y dimension (0..31)
    xoffset = tl.program_id(0) * BLOCK
    yoffset = tl.program_id(1) * BLOCK

    # rows = y indices [BLOCK, 1], cols = x indices [1, BLOCK]
    rows = yoffset + tl.arange(0, BLOCK)[:, None]   # shape [BLOCK, 1]
    cols = xoffset + tl.arange(0, BLOCK)[None, :]   # shape [1, BLOCK]

    # Load: coalesced along cols (x is stride-1 in input)
    # load_idx shape: [BLOCK, BLOCK]
    load_idx = cols + 32768 * rows
    tile = tl.load(in_ptr0 + load_idx)  # shape [BLOCK, BLOCK], no eviction hint

    # Transpose in SRAM: tile_T shape [BLOCK, BLOCK]
    # After trans: axis0 corresponds to original cols (x), axis1 to original rows (y)
    tile_T = tl.trans(tile)  # shape [BLOCK, BLOCK]

    # Store: out_rows = x values (axis0 after trans), out_cols = y values (axis1 after trans)
    out_rows = xoffset + tl.arange(0, BLOCK)[:, None]   # shape [BLOCK, 1]  (x values)
    out_cols = yoffset + tl.arange(0, BLOCK)[None, :]   # shape [1, BLOCK]  (y values)

    # Output store index: y0 + 128*x2 + 4194304*y1
    # where y0 = out_cols % 128, y1 = out_cols // 128, x2 = out_rows
    # Consecutive threads vary out_cols (innermost) -> out_cols%128 varies by 1 -> coalesced
    y0 = out_cols % 128
    y1 = out_cols // 128
    store_idx = y0 + 128 * out_rows + 4194304 * y1
    tl.store(out_ptr0 + store_idx, tile_T)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1024
    xnumel = 32768
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_38[grid](arg_0, arg_1, 1024, 32768, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 128, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 128, 64, 512), (4194304, 1, 65536, 128), device='cuda:0', dtype=torch.float32)
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
