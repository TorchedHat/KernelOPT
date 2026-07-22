import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=1),
        triton.Config({'XBLOCK': 4},  num_warps=2),
        triton.Config({'XBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_cat_211(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    # Restructured to 2D grid: axis 0 = spatial (ynumel=490), axis 1 = channel (32)
    # XBLOCK tiles the spatial dimension (ynumel=490)
    # Channel dimension is always 32 (handled by axis 1 with 1 block of 32)
    pid_y = tl.program_id(0)  # spatial tile (batch*H*W)
    pid_x = tl.program_id(1)  # channel tile (always 0, since we use 1 block of 32)
    # Channel indices: always 0..31
    x0 = pid_x * 32 + tl.arange(0, 32)[None, :]   # shape [1, 32]
    # Spatial indices
    y1 = pid_y * XBLOCK + tl.arange(0, XBLOCK)[:, None]   # shape [XBLOCK, 1]
    xmask = x0 < 32
    ymask = y1 < 490
    mask = xmask & ymask
    # Flat index into in_ptr0 (stride 1568,1,224,32): x2 = y1*32 + x0
    x2 = y1 * 32 + x0
    tmp0 = tl.load(in_ptr0 + x2, mask)
    # out_ptr0 stride (78400,1,11200,1600): offset = x0 + 1600*y1
    tl.store(out_ptr0 + (x0 + 1600 * y1), tmp0, mask)
    # out_ptr1 stride (79968,1,11424,1632): offset = x0 + 1632*y1
    tl.store(out_ptr1 + (x0 + 1632 * y1), tmp0, mask)
    # out_ptr2 stride (81536,1,11648,1664): offset = x0 + 1664*y1
    tl.store(out_ptr2 + (x0 + 1664 * y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 15680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_211[grid](arg_0, arg_1, arg_2, arg_3, 15680)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 7, 7), (78400, 1, 11200, 1600), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (79968, 1, 11424, 1632), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 7, 7), (81536, 1, 11648, 1664), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
