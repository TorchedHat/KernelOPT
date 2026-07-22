import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
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

    # Hoist 4 boundary masks ONCE — reused across all 9 loads
    row_m1_ok = x2 >= 1        # equiv to (x2-1)>=0 AND (x2-1)<14 since x2 in [0,13]
    row_p1_ok = x2 < 13        # equiv to (x2+1)>=0 AND (x2+1)<14
    col_m1_ok = x1 >= 1        # equiv to (x1-1)>=0 AND (x1-1)<14
    col_p1_ok = x1 < 13        # equiv to (x1+1)>=0 AND (x1+1)<14
    # row_ok and col_ok are always True for valid xindex — omit entirely

    # 9 loads with simplified masks
    v0 = tl.load(in_ptr0 + ((-7200) + x6), mask=xmask & row_m1_ok & col_m1_ok, other=float("-inf"))
    v1 = tl.load(in_ptr0 + ((-6720) + x6), mask=xmask & row_m1_ok,             other=float("-inf"))
    v2 = tl.load(in_ptr0 + ((-6240) + x6), mask=xmask & row_m1_ok & col_p1_ok, other=float("-inf"))
    v3 = tl.load(in_ptr0 + ((-480)  + x6), mask=xmask & col_m1_ok,             other=float("-inf"))
    v4 = tl.load(in_ptr0 + (x6),           mask=xmask,                          other=float("-inf"))
    v5 = tl.load(in_ptr0 + (480     + x6), mask=xmask & col_p1_ok,             other=float("-inf"))
    v6 = tl.load(in_ptr0 + (6240    + x6), mask=xmask & row_p1_ok & col_m1_ok, other=float("-inf"))
    v7 = tl.load(in_ptr0 + (6720    + x6), mask=xmask & row_p1_ok,             other=float("-inf"))
    v8 = tl.load(in_ptr0 + (7200    + x6), mask=xmask & row_p1_ok & col_p1_ok, other=float("-inf"))

    # Balanced binary tree reduction (depth 3 vs depth 8)
    a = tl.maximum(v0, v1)
    b = tl.maximum(v2, v3)
    c = tl.maximum(v4, v5)
    d = tl.maximum(v6, v7)
    e = tl.maximum(a, b)
    f = tl.maximum(c, d)
    g = tl.maximum(v8, e)
    result = tl.maximum(f, g)

    tl.store(out_ptr0 + (x6), result, mask=xmask)

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
