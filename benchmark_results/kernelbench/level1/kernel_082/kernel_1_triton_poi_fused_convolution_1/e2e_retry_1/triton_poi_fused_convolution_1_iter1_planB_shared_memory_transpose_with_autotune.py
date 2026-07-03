import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=8, num_stages=4),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # Row indices (ynumel dimension = 4096 = 32 batches * 128 channels)
    rows = pid_m * YBLOCK + tl.arange(0, YBLOCK)
    # Col indices (xnumel dimension = 31500 = 126 * 250 spatial)
    cols = pid_n * XBLOCK + tl.arange(0, XBLOCK)

    # Decode row into batch (y1) and channel (y0)
    y0 = rows % 128   # channel index, stride=1 in source
    y1 = rows // 128  # batch index, stride=4032000 in source

    # 2D boundary mask
    row_mask = rows < ynumel   # shape [YBLOCK]
    col_mask = cols < xnumel   # shape [XBLOCK]
    mask2d = row_mask[:, None] & col_mask[None, :]  # shape [YBLOCK, XBLOCK]

    # Coalesced load: fast dim is cols along axis=1 (warp-contiguous)
    # src layout: stride=(4032000, 1, 32000, 128) -> flat index = y0 + 128*col + 4032000*y1
    src_ptrs = y0[:, None] + 128 * cols[None, :] + 4032000 * y1[:, None]
    tile = tl.load(in_ptr0 + src_ptrs, mask=mask2d, other=0.0)

    # Coalesced store: fast dim is cols (stride=1 in destination)
    # dst layout: stride=(4032000, 31500, 250, 1) -> flat index = col + 31500*row
    dst_ptrs = cols[None, :] + 31500 * rows[:, None]
    tl.store(out_ptr0 + dst_ptrs, tile, mask=mask2d)

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
