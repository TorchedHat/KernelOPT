import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)

    ybase = pid_y * YBLOCK
    yoffs = tl.arange(0, YBLOCK)       # [YBLOCK]
    yindex = ybase + yoffs             # [YBLOCK]
    ymask = yindex < ynumel

    xbase = pid_x * XBLOCK
    xoffs = tl.arange(0, XBLOCK)       # [XBLOCK]
    xindex = xbase + xoffs             # [XBLOCK]
    xmask = xindex < xnumel

    y0 = yindex % 128    # channel index, stride=1 in input
    y1 = yindex // 128   # batch index,   stride=4032000 in input

    # COALESCED LOAD: y0 (stride-1) varies along YBLOCK dim -> shape [YBLOCK, XBLOCK]
    # Within a warp, threads differ along YBLOCK axis (y0 stride-1 in input)
    load_addr = y0[:, None] + 128 * xindex[None, :] + 4032000 * y1[:, None]
    load_mask = ymask[:, None] & xmask[None, :]
    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)

    # COALESCED STORE: xindex (stride-1) varies along XBLOCK dim -> shape [YBLOCK, XBLOCK]
    store_addr = xindex[None, :] + 31500 * yindex[:, None]
    tl.store(out_ptr0 + store_addr, tile, mask=load_mask)

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
