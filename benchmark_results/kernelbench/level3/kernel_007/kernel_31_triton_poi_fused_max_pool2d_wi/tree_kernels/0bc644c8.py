import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=2),
        triton.Config({'XBLOCK': 256},  num_warps=2),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 6720) % 14)
    x1 = ((xindex // 480) % 14)
    x6 = xindex
    # Precompute row and col boundary masks once (x2, x1 in [0,13] so _0_ checks always True)
    row_m1 = ((x2 - 1) >= 0) & ((x2 - 1) < 14)  # x2 in [1,13]
    row_p1 = ((x2 + 1) < 14)                      # x2 in [0,12], +1>=0 always True
    col_m1 = ((x1 - 1) >= 0) & ((x1 - 1) < 14)  # x1 in [1,13]
    col_p1 = ((x1 + 1) < 14)                      # x1 in [0,12], +1>=0 always True
    # 9 loads with hoisted masks
    tmp11 = tl.load(in_ptr0 + ((-7200) + x6), mask=row_m1 & col_m1 & xmask, other=float("-inf"))
    tmp17 = tl.load(in_ptr0 + ((-6720) + x6), mask=row_m1 & xmask,           other=float("-inf"))
    tmp24 = tl.load(in_ptr0 + ((-6240) + x6), mask=row_m1 & col_p1 & xmask, other=float("-inf"))
    tmp31 = tl.load(in_ptr0 + ((-480) + x6),  mask=col_m1 & xmask,           other=float("-inf"))
    tmp34 = tl.load(in_ptr0 + (x6),            mask=xmask,                    other=float("-inf"))
    tmp37 = tl.load(in_ptr0 + (480 + x6),      mask=col_p1 & xmask,           other=float("-inf"))
    tmp44 = tl.load(in_ptr0 + (6240 + x6),     mask=row_p1 & col_m1 & xmask, other=float("-inf"))
    tmp47 = tl.load(in_ptr0 + (6720 + x6),     mask=row_p1 & xmask,           other=float("-inf"))
    tmp50 = tl.load(in_ptr0 + (7200 + x6),     mask=row_p1 & col_p1 & xmask, other=float("-inf"))
    tmp51 = tl.maximum(tl.maximum(tl.maximum(tmp11, tmp17),
                       tl.maximum(tmp24, tmp31)),
                       tl.maximum(tl.maximum(tmp34, tmp37),
                       tl.maximum(tmp44, tl.maximum(tmp47, tmp50))))
    tl.store(out_ptr0 + (x6), tmp51, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 940800
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_31[grid](arg_0, arg_1, 940800)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
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
