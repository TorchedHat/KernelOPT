import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Use XBLOCK as the tile size for the register-transpose approach.
    # XBLOCK == YBLOCK is enforced by configs (square tiles required for tl.trans).
    # program_id(0) = tile index along x2 (spatial, 31500 values)
    # program_id(1) = tile index along y0 (channel, 128 values)
    # program_id(2) = batch index (y1 in [0,32))
    pid_x = tl.program_id(0)   # tile index along x2 (spatial)
    pid_y = tl.program_id(1)   # tile index along y0 (channel, stride-1 in input)
    pid_b = tl.program_id(2)   # batch index
    x_base = pid_x * XBLOCK    # base x2 index
    y_base = pid_y * XBLOCK    # base y0 index (XBLOCK == YBLOCK for square tiles)
    tx = tl.arange(0, XBLOCK)  # inner dim (fast-varying, maps to stride-1 dim)
    ty = tl.arange(0, XBLOCK)  # outer dim (slow-varying)
    # Phase 1: coalesced load — tx varies y0 (stride-1 in input), ty varies x2
    x2_idx = x_base + ty[:, None]   # [XBLOCK, 1] — x2 (spatial)
    y0_idx = y_base + tx[None, :]   # [1, XBLOCK] — y0 (channel, stride-1 in input)
    xmask = x2_idx < xnumel         # x2 < 31500
    ymask = y0_idx < 128            # y0 < 128
    load_idx = y0_idx + 128 * x2_idx + 4032000 * pid_b  # [XBLOCK, XBLOCK]
    tile_data = tl.load(in_ptr0 + load_idx, mask=xmask & ymask, other=0.0)
    # tile_data shape: [XBLOCK_x2, XBLOCK_y0] — loaded coalesced along y0 (stride-1)
    # Phase 2: register transpose
    tile_T = tl.trans(tile_data)  # [XBLOCK_y0, XBLOCK_x2]
    # Phase 3: coalesced store — tx varies x2 (stride-1 in output), ty varies y0
    y0_out = y_base + ty[:, None]   # [XBLOCK, 1] — y0 (channel)
    x2_out = x_base + tx[None, :]   # [1, XBLOCK] — x2 (stride-1 in output)
    y3_out = pid_b * 128 + y0_out   # [XBLOCK, 1] — flattened (batch, channel) index
    xmask_out = x2_out < xnumel     # x2 < 31500
    ymask_out = y0_out < 128        # y0 < 128
    store_idx = x2_out + 31500 * y3_out  # [XBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_idx, tile_T, mask=xmask_out & ymask_out)

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
