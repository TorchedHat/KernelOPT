import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # YBLOCK = TILE_C (channel tile, stride=1 in in_ptr0 -> coalesced loads)
    # XBLOCK = TILE_X (spatial tile)
    ynumel = 4096
    xnumel = 31500

    pid_x = tl.program_id(0)   # spatial tile index
    pid_y = tl.program_id(1)   # channel-group / batch tile index

    # Channel indices [0..YBLOCK-1], stride=1 in in_ptr0 -> COALESCED
    c_idx = tl.arange(0, YBLOCK)           # shape [YBLOCK]

    # Spatial indices for this tile
    x_base = pid_x * XBLOCK
    x_idx = x_base + tl.arange(0, XBLOCK)  # shape [XBLOCK]
    xmask = x_idx < xnumel

    # y1: batch index (which group of 128 channels)
    y1 = pid_y  # each CTA covers exactly one batch element (YBLOCK=128 == channel dim)

    # Load [YBLOCK, XBLOCK]: address = c_idx[:,None] + 128*x_idx[None,:] + 4032000*y1
    # Consecutive elements along c_idx (dim 0) are stride-1 in in_ptr0 -> fully coalesced
    addr_load = c_idx[:, None] + 128 * x_idx[None, :] + 4032000 * y1
    tile = tl.load(in_ptr0 + addr_load, mask=xmask[None, :], other=0.0)

    # Store [YBLOCK, XBLOCK]: address = x_idx[None,:] + 31500*y3[:,None]
    # where y3 = c_idx + 128*y1 (absolute y index)
    y3 = c_idx + 128 * y1                  # shape [YBLOCK]
    ymask = y3 < ynumel
    addr_store = x_idx[None, :] + 31500 * y3[:, None]
    tl.store(out_ptr0 + addr_store, tile, mask=(xmask[None, :] & ymask[:, None]))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
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
