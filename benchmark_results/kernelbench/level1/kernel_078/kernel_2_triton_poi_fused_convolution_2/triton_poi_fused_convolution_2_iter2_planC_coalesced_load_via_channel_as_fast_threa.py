import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # ynumel=24576=256*96, xnumel=3025
    # in_ptr0 strides: (290400,1,5280,96) -> channel(y0) is stride-1
    # 1D flat layout: flat = x2*(256*96) + y1*96 + y0
    # => y0 = flat % 96  (FAST, stride-1 in in_ptr0 -> COALESCED)
    # => y1 = (flat // 96) % 256
    # => x2 = flat // (96*256)
    # YBLOCK=1 in all configs; XBLOCK is the flat block size
    pid = tl.program_id(0)
    flat_index = pid * XBLOCK + tl.arange(0, XBLOCK)
    total = xnumel * ynumel  # 3025 * 24576 = 74317824
    mask = flat_index < total

    y0 = flat_index % 96          # channel [0,95], stride-1 in in_ptr0
    tmp_rem = flat_index // 96
    y1 = tmp_rem % 256            # batch [0,255]
    x2 = tmp_rem // 256           # spatial [0,3024]
    xmask = x2 < xnumel          # guard: 3025 not power-of-2
    full_mask = mask & xmask

    # IDENTICAL load address: y0 + 96*x2 + 290400*y1
    # Consecutive threads differ in y0 (stride-1) -> FULLY COALESCED
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), full_mask)
    # Bias: in_ptr1[y0]
    tmp1 = tl.load(in_ptr1 + y0, full_mask)
    tmp2 = tmp0 + tmp1

    # IDENTICAL store address: x2 + 3025*(y0 + 96*y1)
    y3 = y0 + 96 * y1
    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, full_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
