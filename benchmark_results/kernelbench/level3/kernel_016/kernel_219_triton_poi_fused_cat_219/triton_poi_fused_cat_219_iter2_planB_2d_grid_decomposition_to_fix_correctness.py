import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 512}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_cat_219(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    # 2D decomposition: inner dim is 32 (channel), outer dim is xnumel=490 (batch-spatial)
    # XBLOCK tiles the outer dimension; inner dim is always 32 (a full warp)
    INNER: tl.constexpr = 32
    x0 = tl.arange(0, INNER)[None, :]             # shape [1, 32], channel indices (inner dim)
    yoffset = tl.program_id(0) * XBLOCK
    y1 = yoffset + tl.arange(0, XBLOCK)[:, None]  # shape [XBLOCK, 1], outer (batch-spatial) indices
    ymask = y1 < xnumel                            # shape [XBLOCK, 1], broadcast to [XBLOCK, 32]
    flat_idx = y1 * 32 + x0                        # shape [XBLOCK, 32], flat input index
    tmp0 = tl.load(in_ptr0 + flat_idx, ymask)
    tl.store(out_ptr0 + (x0 + 1728 * y1), tmp0, ymask)
    tl.store(out_ptr1 + (x0 + 1760 * y1), tmp0, ymask)
    tl.store(out_ptr2 + (x0 + 1792 * y1), tmp0, ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 15680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_219[grid](arg_0, arg_1, arg_2, arg_3, 15680)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 7, 7), (84672, 1, 12096, 1728), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (86240, 1, 12320, 1760), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 7, 7), (87808, 1, 12544, 1792), device='cuda:0', dtype=torch.float32)
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
