import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK = TILE_hw (spatial dimension tiles)
    # XBLOCK = TILE_c  (channel dimension tiles, stride-1 in input)
    # Grid: (cdiv(C, XBLOCK), cdiv(HW, YBLOCK), N)
    # pid0 -> channel tile, pid1 -> spatial tile, pid2 -> batch
    C = 128
    H = 126
    W = 250
    HW = H * W  # 31500
    N = 32

    pid_c  = tl.program_id(0)  # tile along C
    pid_hw = tl.program_id(1)  # tile along HW
    pid_n  = tl.program_id(2)  # batch index

    c_base  = pid_c  * XBLOCK
    hw_base = pid_hw * YBLOCK

    tc  = tl.arange(0, XBLOCK)   # [XBLOCK] channel offsets within tile
    thw = tl.arange(0, YBLOCK)   # [YBLOCK] spatial offsets within tile

    c_idx  = c_base  + tc      # [XBLOCK]
    hw_idx = hw_base + thw     # [YBLOCK]
    h_idx  = hw_idx // W       # [YBLOCK]
    w_idx  = hw_idx  % W       # [YBLOCK]

    # Coalesced LOAD: c is stride-1 in input (in_sc=1), make it the fast/inner dim
    # Shape: [YBLOCK, XBLOCK] = [TILE_hw, TILE_c]
    c_mask  = c_idx[None, :]  < C    # [1, XBLOCK]
    hw_mask = hw_idx[:, None] < HW   # [YBLOCK, 1]
    load_mask = hw_mask & c_mask     # [YBLOCK, XBLOCK]

    in_off = (
        pid_n * 4032000
        + h_idx[:, None] * 32000
        + w_idx[:, None] * 128
        + c_idx[None, :] * 1          # stride-1, COALESCED
    )  # shape [YBLOCK, XBLOCK]
    tile_data = tl.load(in_ptr0 + in_off, mask=load_mask, other=0.0)

    # Transpose the register tile: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    tile_t     = tl.trans(tile_data)   # [XBLOCK, YBLOCK] = [TILE_c, TILE_hw]
    store_mask = tl.trans(load_mask)   # [XBLOCK, YBLOCK]

    # Coalesced STORE: hw is stride-1 in output (out_sw=1), make it the fast/inner dim
    # Shape: [XBLOCK, YBLOCK] = [TILE_c, TILE_hw]
    out_off = (
        pid_n * 4032000
        + c_idx[:, None]  * 31500
        + h_idx[None, :]  * 250
        + w_idx[None, :]  * 1         # stride-1, COALESCED
    )  # shape [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + out_off, tile_t, mask=store_mask)

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
