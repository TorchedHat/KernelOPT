import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # ynumel = 4096 = N*C = 32*128  (flat channel index)
    # xnumel = 31500 = H*W = 126*250 (flat spatial index)
    # YBLOCK = TILE_C (channel tile size), XBLOCK = TILE_X (spatial tile size)
    ynumel = 4096
    xnumel = 31500

    # CTA tile boundaries
    c_tile_start = tl.program_id(1) * YBLOCK   # flat channel offset [0, 4096)
    x_tile_start = tl.program_id(0) * XBLOCK   # spatial offset [0, 31500)

    # 1D index vectors
    c_idx = c_tile_start + tl.arange(0, YBLOCK)   # [YBLOCK]
    x_idx = x_tile_start + tl.arange(0, XBLOCK)   # [XBLOCK]

    # Broadcast to 2D tile [YBLOCK, XBLOCK]
    c_2d = c_idx[:, None]   # [YBLOCK, 1]
    x_2d = x_idx[None, :]   # [1, XBLOCK]

    # Decompose flat channel index into batch n and channel c_local
    # y0 = c_local (stride-1 in input), y1 = n (batch)
    y0 = c_2d % 128          # channel within batch: stride-1 in input
    y1 = c_2d // 128         # batch index

    # Boundary masks
    c_mask = c_2d < ynumel   # [YBLOCK, 1]
    x_mask = x_2d < xnumel   # [1, XBLOCK]
    mask_2d = c_mask & x_mask # [YBLOCK, XBLOCK]

    # COALESCED LOAD: consecutive threads (along YBLOCK/channel) step along y0 (stride-1 in input)
    # Input address: y0 + 128*x2 + 4032000*y1
    in_addr = y0 + 128 * x_2d + 4032000 * y1   # [YBLOCK, XBLOCK]
    tile = tl.load(in_ptr0 + in_addr, mask=mask_2d, other=0.0)
    # No evict_last: data is not reused after this transpose

    # COALESCED STORE: consecutive threads (along XBLOCK/spatial) step along x2 (stride-1 in output)
    # Output address: x2 + 31500*y3  where y3 = c_2d (flat channel index)
    out_addr = x_2d + 31500 * c_2d   # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + out_addr, tile, mask=mask_2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
