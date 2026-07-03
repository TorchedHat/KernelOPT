import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D indexing over all N*C*H*W elements in NHWC order
    # Total elements = 256 * 96 * 55 * 55 = 74317824
    # flat = n*(C*HW) + hw*C + c  (NHWC order)
    # c = flat % 96, hw = (flat // 96) % 3025, n = flat // (96*3025)
    C            = 96      # channels, stride-1 in input (NHWC)
    HW           = 3025    # H*W = 55*55
    N_IN_STRIDE  = 290400
    HW_IN_STRIDE = 96
    N_OUT_STRIDE = 290400
    C_OUT_STRIDE = 3025
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = ynumel * xnumel  # 24576 * 3025 = 74317824... wait, 24576*3025 != 74317824
    # Actually: ynumel=24576 (N*C=256*96), xnumel=3025 (H*W)
    # Total = 24576 * 3025 = 74342400... let's compute correctly
    # N=256, C=96, HW=3025 => total = 256*96*3025 = 74342400
    # But plan says 74317824 = 256*96*55*55 = 256*96*3025 = 74342400? 
    # 55*55=3025, 256*96=24576, 24576*3025=74342400
    total = ynumel * xnumel
    mask = offsets < total
    c  = offsets % C
    hw = (offsets // C) % HW
    n  = offsets // (C * HW)
    in_off  = n * N_IN_STRIDE + hw * HW_IN_STRIDE + c
    tmp0 = tl.load(in_ptr0 + in_off, mask=mask)
    tmp1 = tl.load(in_ptr1 + c, mask=mask)
    tmp2 = tmp0 + tmp1
    out_off = n * N_OUT_STRIDE + c * C_OUT_STRIDE + hw
    tl.store(out_ptr0 + out_off, tmp2, mask=mask)

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
