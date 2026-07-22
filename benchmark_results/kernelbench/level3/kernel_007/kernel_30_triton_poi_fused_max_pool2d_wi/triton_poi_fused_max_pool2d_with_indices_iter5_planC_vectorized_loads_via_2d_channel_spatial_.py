import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_30(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: program_id(0) = spatial index (batch*196 + out_row*14 + out_col)
    #          program_id(1) = channel tile index (XBLOCK-wide contiguous channels)
    # xnumel is passed as snumel=1960 (number of spatial output elements)
    # cnumel=480 is the channel count (hardcoded)
    sid = tl.program_id(0)   # spatial index in [0, 1960)
    cid = tl.program_id(1)   # channel tile index

    # Decode spatial: sid = batch*196 + out_row*14 + out_col
    out_col = sid % 14
    tmp_s   = sid // 14
    out_row = tmp_s % 14
    batch   = tmp_s // 14

    # Channel indices — contiguous (stride=1)
    coff  = cid * XBLOCK + tl.arange(0, XBLOCK)
    cmask = coff < 480

    # Input base pointer offset: coff + in_col*480 + in_row*13440 + batch*376320
    in_col_base = 2 * out_col
    in_row_base = 2 * out_row
    base = coff + 480 * in_col_base + 13440 * in_row_base + 376320 * batch

    # Scalar boundary conditions (same for all channels in tile)
    # Row validity
    row_m1_valid = (in_row_base - 1 >= 0) & (in_row_base - 1 < 28)
    row_0_valid  = (in_row_base     >= 0) & (in_row_base     < 28)
    row_p1_valid = (in_row_base + 1 >= 0) & (in_row_base + 1 < 28)

    # Col validity
    col_m1_valid = (in_col_base - 1 >= 0) & (in_col_base - 1 < 28)
    col_0_valid  = (in_col_base     >= 0) & (in_col_base     < 28)
    col_p1_valid = (in_col_base + 1 >= 0) & (in_col_base + 1 < 28)

    # 9 neighbor loads — each is an XBLOCK-wide contiguous vector load
    # dr=-1, dc=-1
    v00 = tl.load(in_ptr0 + (base - 13440 - 480), mask=cmask & row_m1_valid & col_m1_valid, other=float('-inf'))
    # dr=-1, dc=0
    v01 = tl.load(in_ptr0 + (base - 13440       ), mask=cmask & row_m1_valid & col_0_valid,  other=float('-inf'))
    # dr=-1, dc=+1
    v02 = tl.load(in_ptr0 + (base - 13440 + 480 ), mask=cmask & row_m1_valid & col_p1_valid, other=float('-inf'))
    # dr=0, dc=-1
    v10 = tl.load(in_ptr0 + (base         - 480 ), mask=cmask & row_0_valid  & col_m1_valid, other=float('-inf'))
    # dr=0, dc=0
    v11 = tl.load(in_ptr0 + (base               ), mask=cmask & row_0_valid  & col_0_valid,  other=float('-inf'))
    # dr=0, dc=+1
    v12 = tl.load(in_ptr0 + (base         + 480 ), mask=cmask & row_0_valid  & col_p1_valid, other=float('-inf'))
    # dr=+1, dc=-1
    v20 = tl.load(in_ptr0 + (base + 13440 - 480 ), mask=cmask & row_p1_valid & col_m1_valid, other=float('-inf'))
    # dr=+1, dc=0
    v21 = tl.load(in_ptr0 + (base + 13440       ), mask=cmask & row_p1_valid & col_0_valid,  other=float('-inf'))
    # dr=+1, dc=+1
    v22 = tl.load(in_ptr0 + (base + 13440 + 480 ), mask=cmask & row_p1_valid & col_p1_valid, other=float('-inf'))

    result = tl.maximum(tl.maximum(tl.maximum(v00, v01), tl.maximum(v02, v10)),
                        tl.maximum(tl.maximum(v11, v12), tl.maximum(v20, tl.maximum(v21, v22))))

    # Output: coff + out_col*480 + out_row*6720 + batch*94080
    out_base = coff + 480 * out_col + 6720 * out_row + 94080 * batch
    tl.store(out_ptr0 + out_base, result, mask=cmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 940800
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_30[grid](arg_0, arg_1, 940800)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 480, 28, 28), (376320, 1, 13440, 480), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
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
