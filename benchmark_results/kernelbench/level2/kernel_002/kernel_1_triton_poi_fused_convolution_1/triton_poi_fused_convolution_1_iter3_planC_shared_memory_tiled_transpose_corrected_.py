import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Input strides:  (N=32,C=128,H=126,W=250) -> (4032000, 1, 32000, 128)
    # Output strides: (N=32,C=128,H=126,W=250) -> (4032000, 31500, 250, 1)
    # ynumel=4096 (N*C=32*128), xnumel=31500 (H*W=126*250)
    # Load:  in_ptr0[y0 + 128*x2 + 4032000*y1]  y0=y3%128 has stride=1 in input
    # Store: out_ptr0[x2 + 31500*y3]             x2 has stride=1 in output
    #
    # STRATEGY: Use XBLOCK as TILE_X (x2 dimension), YBLOCK as TILE_Y (y3 dimension)
    #   tx = arange(XBLOCK)[:, None] -> outer dim for loads (x2), inner dim for stores
    #   ty = arange(YBLOCK)[None, :] -> inner dim for loads (y0, stride=1 -> COALESCED LOAD)
    #   Load shape: [XBLOCK, YBLOCK] — coalesced along ty (y0 stride=1)
    #   tl.trans(vals) -> [YBLOCK, XBLOCK]
    #   Store shape: [YBLOCK, XBLOCK] — coalesced along tx2 (x2 stride=1 -> COALESCED STORE)
    xnumel = 31500
    ynumel = 4096

    x_tile_start = tl.program_id(0) * XBLOCK
    y_tile_start = tl.program_id(1) * YBLOCK

    # tx: outer dim for loads (x2), inner dim for stores
    tx = tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    # ty: inner dim for loads (coalesced along y0), outer dim for stores
    ty = tl.arange(0, YBLOCK)[None, :]   # [1, YBLOCK]

    x2 = x_tile_start + tx   # [XBLOCK, 1] — H*W index
    y3 = y_tile_start + ty   # [1, YBLOCK] — N*C index

    y0 = y3 % 128    # channel: stride=1 in input, varies along ty (innermost) -> COALESCED LOAD
    y1 = y3 // 128   # batch index

    xmask = x2 < xnumel   # [XBLOCK, 1]
    ymask = y3 < ynumel   # [1, YBLOCK]
    mask_load = xmask & ymask  # [XBLOCK, YBLOCK]

    # COALESCED LOAD: ty is innermost, consecutive warp threads differ by 1 in y0
    vals = tl.load(
        in_ptr0 + (y0 + 128 * x2 + 4032000 * y1),
        mask=mask_load,
        other=0.0
    )  # shape [XBLOCK, YBLOCK]

    # Transpose in registers: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    vals_t = tl.trans(vals)

    # After tl.trans: shape [YBLOCK, XBLOCK]
    # Recompute indices for store shape [YBLOCK, XBLOCK]
    tx2 = tl.arange(0, XBLOCK)[None, :]   # [1, XBLOCK] innermost for store
    ty2 = tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1] outer for store
    x2_store = x_tile_start + tx2         # [1, XBLOCK]
    y3_store = y_tile_start + ty2         # [YBLOCK, 1]
    xmask_store = x2_store < xnumel
    ymask_store = y3_store < ynumel
    mask_store = xmask_store & ymask_store

    # COALESCED STORE: tx2 is innermost, consecutive warp threads differ by 1 in x2
    tl.store(
        out_ptr0 + (x2_store + 31500 * y3_store),
        vals_t,
        mask=mask_store
    )

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
