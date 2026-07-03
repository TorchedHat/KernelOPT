import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Use XBLOCK as the square tile size (YBLOCK == XBLOCK enforced by configs)
    # pid_x covers the x dimension (xnumel=31500), pid_y covers the y dimension (ynumel=4096)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # Offsets for this tile
    x_off = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # shape [XBLOCK]
    y_off = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # shape [YBLOCK]

    # Boundary masks
    x_mask = x_off < xnumel  # shape [XBLOCK]
    y_mask = y_off < ynumel  # shape [YBLOCK]

    # Decompose y into channel components
    # y0 = y % 128 (stride=1 in source -> coalesced when inner dim)
    # y1 = y // 128
    y0 = y_off % 128  # shape [YBLOCK]
    y1 = y_off // 128  # shape [YBLOCK]

    # Load: make y (channel, stride=1) the inner (contiguous) dimension
    # src_addr shape: [XBLOCK, YBLOCK] where inner dim is y0 (stride=1 in source)
    # x_off[:,None] has shape [XBLOCK,1], y0[None,:] has shape [1,YBLOCK]
    src_addr = y0[None, :] + 128 * x_off[:, None] + 4032000 * y1[None, :]
    # Combined mask: x valid along rows, y valid along cols
    load_mask = x_mask[:, None] & y_mask[None, :]
    tile_data = tl.load(in_ptr0 + src_addr, mask=load_mask, other=0.0)
    # tile_data shape: [XBLOCK, YBLOCK], inner dim = y0 (COALESCED read)

    # Store: make x the inner (contiguous) dimension
    # dst_addr shape: [YBLOCK, XBLOCK] where inner dim is x_off (stride=1 in dst)
    # y_off[:,None] has shape [YBLOCK,1], x_off[None,:] has shape [1,XBLOCK]
    dst_addr = x_off[None, :] + 31500 * y_off[:, None]
    store_mask = y_mask[:, None] & x_mask[None, :]
    # tl.trans flips [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK], making inner dim = x_off (COALESCED write)
    tl.store(out_ptr0 + dst_addr, tl.trans(tile_data), mask=store_mask)

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
