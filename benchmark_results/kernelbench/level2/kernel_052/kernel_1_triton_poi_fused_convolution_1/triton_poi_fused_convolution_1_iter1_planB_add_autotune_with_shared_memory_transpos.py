import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    row = tl.arange(0, YBLOCK)
    col = tl.arange(0, XBLOCK)
    x_idx = pid_x * XBLOCK + col[None, :]   # [1, XBLOCK]
    y_idx = pid_y * YBLOCK + row[:, None]   # [YBLOCK, 1]
    y0 = y_idx % 128
    y1 = y_idx // 128
    x_mask = x_idx < xnumel
    y_mask = y_idx < ynumel
    mask = x_mask & y_mask
    src_offset = y0 + 128 * x_idx + 4032000 * y1
    tile_data = tl.load(in_ptr0 + src_offset, mask=mask, other=0.0)
    tile_T = tl.trans(tile_data)
    x_idx_T = pid_x * XBLOCK + row[:, None]
    y_idx_T = pid_y * YBLOCK + col[None, :]
    mask_T = (x_idx_T < xnumel) & (y_idx_T < ynumel)
    dst_offset = x_idx_T + 31500 * y_idx_T
    tl.store(out_ptr0 + dst_offset, tile_T, mask=mask_T)

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
