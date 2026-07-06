import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=8, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # pid_x tiles over x (spatial) dimension, pid_y tiles over y (channel/batch) dimension
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x_start = pid_x * XBLOCK
    y_start = pid_y * YBLOCK
    # tx indexes spatial (x), ty indexes channel/batch (y)
    tx = tl.arange(0, XBLOCK)
    ty = tl.arange(0, YBLOCK)
    # Load: rows=x (spatial), cols=y (channel/batch)
    # Shape: [XBLOCK, YBLOCK]
    xidx = x_start + tx[:, None]   # [XBLOCK, 1] -> spatial index
    yidx = y_start + ty[None, :]   # [1, YBLOCK] -> channel/batch index
    y0 = yidx % 128
    y1 = yidx // 128
    # Input address: y0 + 128*xidx + 4032000*y1
    # (stride-1 in y0/channel direction -> coalesced along YBLOCK dimension)
    load_addr = y0 + 128 * xidx + 4032000 * y1
    load_mask = (xidx < xnumel) & (yidx < ynumel)
    tile_data = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK]
    # Transpose in-register: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    tile_t = tl.trans(tile_data)
    # Store: rows=y (channel/batch), cols=x (spatial)
    # Shape: [YBLOCK, XBLOCK]
    yidx2 = y_start + ty[:, None]  # [YBLOCK, 1]
    xidx2 = x_start + tx[None, :]  # [1, XBLOCK]
    # Output address: xidx2 + 31500*yidx2
    # (stride-1 in xidx2/spatial direction -> coalesced along XBLOCK dimension)
    store_addr = xidx2 + 31500 * yidx2
    store_mask = (yidx2 < ynumel) & (xidx2 < xnumel)
    tl.store(out_ptr0 + store_addr, tile_t, mask=store_mask)

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
