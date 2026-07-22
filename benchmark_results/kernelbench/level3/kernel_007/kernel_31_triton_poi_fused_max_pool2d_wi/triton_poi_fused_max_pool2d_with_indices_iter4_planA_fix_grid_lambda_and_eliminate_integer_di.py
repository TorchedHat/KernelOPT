import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=2),
        triton.Config({'XBLOCK': 64}, num_warps=2),
        triton.Config({'XBLOCK': 16}, num_warps=1),
        triton.Config({'XBLOCK': 32}, num_warps=1),
        triton.Config({'XBLOCK': 64}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_31(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 3D grid: pid0 = n*C+c in [0,4800), pid1 = h in [0,14), pid2 = w-tile in [0,cdiv(14,XBLOCK))
    pid_nc = tl.program_id(0)   # in [0, N*C) = [0, 4800)
    pid_h  = tl.program_id(1)   # in [0, H) = [0, 14)
    pid_w  = tl.program_id(2)   # in [0, cdiv(W, XBLOCK))

    # Decode n and c from pid_nc (C=480 hardcoded)
    n = pid_nc // 480
    c = pid_nc % 480
    h = pid_h                   # no division needed

    w_offset = pid_w * XBLOCK
    w_range  = w_offset + tl.arange(0, XBLOCK)
    w_mask   = w_range < 14     # W=14

    # Flat base using actual strides (94080, 1, 6720, 480)
    base = n * 94080 + c * 1 + h * 6720 + w_range * 480

    # Boundary masks - cheap scalar/vector comparisons, no division
    row_m1 = h >= 1             # scalar bool
    row_p1 = h <= 12            # scalar bool (h < H-1)
    col_m1 = w_range >= 1       # vector bool
    col_p1 = w_range <= 12      # vector bool (w < W-1)

    tmp11 = tl.load(in_ptr0 + base - 6720 - 480, mask=w_mask & row_m1 & col_m1, other=float('-inf'))
    tmp17 = tl.load(in_ptr0 + base - 6720,        mask=w_mask & row_m1,           other=float('-inf'))
    tmp24 = tl.load(in_ptr0 + base - 6720 + 480,  mask=w_mask & row_m1 & col_p1, other=float('-inf'))
    tmp31 = tl.load(in_ptr0 + base - 480,          mask=w_mask & col_m1,           other=float('-inf'))
    tmp34 = tl.load(in_ptr0 + base,                mask=w_mask,                    other=float('-inf'))
    tmp37 = tl.load(in_ptr0 + base + 480,          mask=w_mask & col_p1,           other=float('-inf'))
    tmp44 = tl.load(in_ptr0 + base + 6720 - 480,  mask=w_mask & row_p1 & col_m1, other=float('-inf'))
    tmp47 = tl.load(in_ptr0 + base + 6720,         mask=w_mask & row_p1,           other=float('-inf'))
    tmp50 = tl.load(in_ptr0 + base + 6720 + 480,  mask=w_mask & row_p1 & col_p1, other=float('-inf'))

    result = tl.maximum(
        tl.maximum(tl.maximum(tmp11, tmp17), tl.maximum(tmp24, tmp31)),
        tl.maximum(tl.maximum(tmp34, tmp37), tl.maximum(tmp44, tl.maximum(tmp47, tmp50)))
    )
    tl.store(out_ptr0 + base, result, mask=w_mask)

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
