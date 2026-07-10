import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4, num_stages=2),
], key=['xnumel', 'ynumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    pid_x = tl.program_id(0)  # tile along x2 (output fast dim)
    pid_y = tl.program_id(1)  # tile along y3
    x_offs = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    y_offs = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    x_mask = x_offs < xnumel
    y_mask = y_offs < ynumel
    y0 = y_offs % 128   # fast dim in input (stride=1)
    y1 = y_offs // 128  # batch dim in input (stride=4032000)
    # Load [YBLOCK, XBLOCK]: coalesced along y0 (stride=1), strided along x2 (stride=128)
    load_idx = y0[:, None] + 128 * x_offs[None, :] + 4032000 * y1[:, None]
    mask2d = y_mask[:, None] & x_mask[None, :]
    tile = tl.load(in_ptr0 + load_idx, mask=mask2d, other=0.0)
    # Transpose in SMEM: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    tile_T = tl.trans(tile)
    # Store [XBLOCK, YBLOCK]: coalesced along x2 (stride=1 in output)
    store_idx = x_offs[:, None] + 31500 * y_offs[None, :]
    store_mask = x_mask[:, None] & y_mask[None, :]
    tl.store(out_ptr0 + store_idx, tile_T, mask=store_mask)

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
