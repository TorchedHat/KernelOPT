import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 6720) % 14)
    x1 = ((xindex // 480) % 14)
    x6 = xindex

    # Precompute 6 boundary masks once and reuse across all 9 loads
    row_m1_ok = x2 >= 1        # row-1 is valid (x2 >= 1 means x2-1 >= 0)
    row_p1_ok = x2 < 13        # row+1 is valid (x2+1 < 14 means x2 < 13)
    col_m1_ok = x1 >= 1        # col-1 is valid
    col_p1_ok = x1 < 13        # col+1 is valid

    # 9 neighbor loads with eviction_policy='evict_last' to improve L1 reuse
    tmp0 = tl.load(in_ptr0 + (x6 - 7200), mask=xmask & row_m1_ok & col_m1_ok, other=float("-inf"), eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x6 - 6720), mask=xmask & row_m1_ok,             other=float("-inf"), eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x6 - 6240), mask=xmask & row_m1_ok & col_p1_ok, other=float("-inf"), eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x6 - 480),  mask=xmask & col_m1_ok,             other=float("-inf"), eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (x6),        mask=xmask,                          other=float("-inf"), eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x6 + 480),  mask=xmask & col_p1_ok,             other=float("-inf"), eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x6 + 6240), mask=xmask & row_p1_ok & col_m1_ok, other=float("-inf"), eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (x6 + 6720), mask=xmask & row_p1_ok,             other=float("-inf"), eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (x6 + 7200), mask=xmask & row_p1_ok & col_p1_ok, other=float("-inf"), eviction_policy='evict_last')

    # Balanced tree reduction to minimize dependency chain depth
    result = tl.maximum(
        tl.maximum(tl.maximum(tmp0, tmp1), tl.maximum(tmp2, tmp3)),
        tl.maximum(tl.maximum(tmp4, tmp5), tl.maximum(tmp6, tl.maximum(tmp7, tmp8)))
    )
    tl.store(out_ptr0 + (x6), result, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 940800
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_31[grid](arg_0, arg_1, 940800, XBLOCK=128)
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
