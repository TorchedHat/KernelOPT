import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
], key=['xnumel', 'ynumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # x_offs: spatial indices [XBLOCK], stride-1 in dst
    x_offs = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    # y_offs: n*c indices [YBLOCK], stride-1 in src (channel dim)
    y_offs = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    xmask = x_offs < xnumel   # [XBLOCK]
    ymask = y_offs < ynumel   # [YBLOCK]
    mask2d = xmask[:, None] & ymask[None, :]  # [XBLOCK, YBLOCK]
    # Decompose y into channel (c) and batch (n)
    y0_2d = (y_offs % 128)[None, :]   # [1, YBLOCK] — channel index, stride-1 in src
    y1_2d = (y_offs // 128)[None, :]  # [1, YBLOCK] — batch index
    x2_2d = x_offs[:, None]           # [XBLOCK, 1] — spatial index
    # COALESCED LOAD: consecutive threads along YBLOCK (last dim) -> stride-1 in src
    # src layout: (n, c, h, w) with strides (4032000, 1, 32000, 128)
    # src_idx = c*1 + n*4032000 + spatial*128 = y0 + 4032000*y1 + 128*x2
    src_idx = y0_2d + 128 * x2_2d + 4032000 * y1_2d  # [XBLOCK, YBLOCK]
    tile = tl.load(in_ptr0 + src_idx, mask=mask2d, other=0.0)
    # COALESCED STORE: consecutive threads along XBLOCK (first dim) -> stride-1 in dst
    # dst layout: (n, c, h, w) with strides (4032000, 31500, 250, 1)
    # dst_idx = x2*1 + y*31500 = x2 + 31500*y
    dst_idx = x2_2d + 31500 * y_offs[None, :]  # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + dst_idx, tile, mask=mask2d)

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
