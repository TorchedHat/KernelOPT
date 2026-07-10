import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Input:  (N=32, C=128, H=126, W=250) strides (4032000, 1, 32000, 128) NHWC
    # Output: (N=32, C=128, H=126, W=250) strides (4032000, 31500, 250, 1) NCHW
    # xnumel=31500 (H*W), ynumel=4096 (N*C)
    # XBLOCK = TILE_HW (spatial tile), YBLOCK = TILE_C = 128 (full channel dim)
    # pid axis 0 = HW tile, pid axis 1 = batch (N=32)
    pid_hw = tl.program_id(0)   # HW tile index
    pid_n  = tl.program_id(1)   # batch index 0..31

    hw_start = pid_hw * XBLOCK
    hw_offs  = hw_start + tl.arange(0, XBLOCK)   # [XBLOCK] = hw indices
    c_offs   = tl.arange(0, YBLOCK)              # [YBLOCK=128] = channel indices 0..127
    hw_mask  = hw_offs < xnumel                  # guard xnumel=31500

    # Tile shape [XBLOCK, YBLOCK]: inner dim = C (stride-1 in input) -> COALESCED LOADS
    # load_idx[hw, c] = pid_n*4032000 + c*1 + hw*128
    load_idx = (pid_n * 4032000 +
                c_offs[None, :] * 1 +
                hw_offs[:, None] * 128)  # [XBLOCK, YBLOCK]
    load_mask = hw_mask[:, None]         # [XBLOCK, 1] broadcast over YBLOCK
    tile = tl.load(in_ptr0 + load_idx, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK]

    # Store: store_idx[hw, c] = pid_n*4032000 + c*31500 + hw*1
    # Correctness: n*4032000 + c*31500 + hw = n*31500*128 + c*31500 + hw
    #            = 31500*(n*128+c) + hw = x2 + 31500*y3  (original formula verified)
    store_idx = (pid_n * 4032000 +
                 c_offs[None, :] * 31500 +
                 hw_offs[:, None] * 1)   # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + store_idx, tile, mask=load_mask)

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
