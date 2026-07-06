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
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)  # tiles along xnumel (HW=31500)
    pid_y = tl.program_id(1)  # tiles along ynumel (NC=4096)
    tx = tl.arange(0, XBLOCK)   # x (HW) offsets within tile
    ty = tl.arange(0, YBLOCK)   # y (NC) offsets within tile
    x_idx = pid_x * XBLOCK + tx  # global HW indices [XBLOCK]
    y_idx = pid_y * YBLOCK + ty  # global NC indices [YBLOCK]
    c_idx = y_idx % 128          # channel (stride-1 in input)
    n_idx = y_idx // 128         # batch
    h_idx = x_idx // 250         # height
    w_idx = x_idx % 250          # width
    x_mask = x_idx < xnumel
    y_mask = y_idx < ynumel
    mask2d = y_mask[:, None] & x_mask[None, :]
    # COALESCED LOAD: c_idx is stride-1 in input, varies along y-dim
    load_addr = (n_idx[:, None] * 4032000 + c_idx[:, None] * 1
                 + h_idx[None, :] * 32000 + w_idx[None, :] * 128)
    tile_data = tl.load(in_ptr0 + load_addr, mask=mask2d, other=0.0)
    # COALESCED STORE: w_idx is stride-1 in output, varies along x-dim
    store_addr = (n_idx[:, None] * 4032000 + c_idx[:, None] * 31500
                  + h_idx[None, :] * 250 + w_idx[None, :] * 1)
    tl.store(out_ptr0 + store_addr, tile_data, mask=mask2d)

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
