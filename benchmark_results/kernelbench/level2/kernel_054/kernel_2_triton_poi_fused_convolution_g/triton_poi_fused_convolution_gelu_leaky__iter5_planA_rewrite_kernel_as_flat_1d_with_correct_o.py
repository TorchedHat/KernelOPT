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
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D iteration over all numel = xnumel * ynumel = 264257536 elements
    # Use XBLOCK as the flat block size; YBLOCK is fixed to 1
    numel = 64 * 4129024  # 264257536
    pid = tl.program_id(0)
    offset = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offset < numel
    # Decompose flat output index
    x2 = offset // 4129024   # channel index [0,63]
    y3 = offset % 4129024    # spatial flat index [0,4129023]
    y0 = y3 % 64516          # within-channel spatial
    y1 = y3 // 64516         # batch index [0,63]
    # Load inputs
    tmp0 = tl.load(in_ptr0 + (x2 + 64 * y3), mask)
    tmp1 = tl.load(in_ptr1 + x2, mask)
    tmp3 = tl.load(in_ptr2 + x2, mask)
    # bias add + scale mul
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    # leaky relu
    tmp6 = tmp4 > 0.0
    tmp8 = tmp4 * 0.01
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    # gelu
    tmp11 = tmp9 * 0.5
    tmp13 = tmp9 * 0.7071067811865476
    tmp14 = libdevice.erf(tmp13)
    tmp16 = tmp14 + 1.0
    tmp17 = tmp11 * tmp16
    # Store to transposed output layout
    tl.store(out_ptr0 + (y0 + 64516 * x2 + 4129024 * y1), tmp17, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 4129024
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_gelu_leaky_relu_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 4129024, 64)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((64, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 64, 254, 254), (4129024, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
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
