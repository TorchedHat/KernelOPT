import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=16),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=16),
        triton.Config({'XBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 5376) % 28)  # row index
    x1 = ((xindex // 192) % 28)   # col index
    x6 = xindex
    row_m1 = x2 >= 1
    row_p1 = x2 <= 26
    col_m1 = x1 >= 1
    col_p1 = x1 <= 26
    m = xmask
    t0 = tl.load(in_ptr0 + (-5568 + x6), mask=m & row_m1 & col_m1, other=float('-inf'))
    t1 = tl.load(in_ptr0 + (-5376 + x6), mask=m & row_m1,          other=float('-inf'))
    t2 = tl.load(in_ptr0 + (-5184 + x6), mask=m & row_m1 & col_p1, other=float('-inf'))
    t3 = tl.load(in_ptr0 + ( -192 + x6), mask=m & col_m1,          other=float('-inf'))
    t4 = tl.load(in_ptr0 + (    0 + x6), mask=m,                   other=float('-inf'))
    t5 = tl.load(in_ptr0 + (  192 + x6), mask=m & col_p1,          other=float('-inf'))
    t6 = tl.load(in_ptr0 + ( 5184 + x6), mask=m & row_p1 & col_m1, other=float('-inf'))
    t7 = tl.load(in_ptr0 + ( 5376 + x6), mask=m & row_p1,          other=float('-inf'))
    t8 = tl.load(in_ptr0 + ( 5568 + x6), mask=m & row_p1 & col_p1, other=float('-inf'))
    result = tl.maximum(t0, tl.maximum(t1, tl.maximum(t2,
             tl.maximum(t3, tl.maximum(t4, tl.maximum(t5,
             tl.maximum(t6, tl.maximum(t7, t8))))))))
    tl.store(out_ptr0 + x6, result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1505280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_9[grid](arg_0, arg_1, 1505280)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
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
