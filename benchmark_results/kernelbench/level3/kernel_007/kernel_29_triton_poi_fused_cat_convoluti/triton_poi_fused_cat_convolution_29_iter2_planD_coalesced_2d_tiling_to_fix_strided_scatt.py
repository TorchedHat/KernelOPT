import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=2),
        triton.Config({'XBLOCK': 2},  num_warps=2),
        triton.Config({'XBLOCK': 4},  num_warps=4),
        triton.Config({'XBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=8),
        triton.Config({'XBLOCK': 32}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_29(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Treat XBLOCK as the row-tile size (YBLOCK in plan terminology).
    # The column dimension is always exactly 64 (stride-1 in both input and output).
    # ynumel (rows) = 7840, xnumel_cols (cols) = 64
    ynumel = 7840
    xnumel_cols: tl.constexpr = 64
    pid_y = tl.program_id(0)
    yoffset = pid_y * XBLOCK
    yindex = yoffset + tl.arange(0, XBLOCK)[:, None]   # shape [XBLOCK, 1]
    ymask = yindex < ynumel
    xindex = tl.arange(0, xnumel_cols)[None, :]         # shape [1, 64]
    mask = ymask  # xindex always < 64 = xnumel_cols, no column mask needed
    # Input: flat (7840, 64) with strides (64, 1) -> offset = yindex*64 + xindex
    tmp0 = tl.load(in_ptr0 + (yindex * 64 + xindex), mask, other=0.0)
    # Output: strided (10,64,28,28) with strides (376320,1,13440,480)
    # offset = xindex + 480*yindex  (coalesced: adjacent threads differ by 1 in xindex)
    tl.store(out_ptr0 + (xindex + 480 * yindex), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 501760
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_29[grid](arg_0, arg_1, 501760)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((7840, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 28, 28), (376320, 1, 13440, 480), device='cuda:0', dtype=torch.float32)
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
