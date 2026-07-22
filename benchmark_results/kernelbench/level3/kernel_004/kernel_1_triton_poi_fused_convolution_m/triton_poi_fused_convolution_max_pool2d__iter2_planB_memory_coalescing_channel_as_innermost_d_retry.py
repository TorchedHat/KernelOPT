import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 64},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 6,  'YBLOCK': 32},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 6,  'YBLOCK': 64},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 6,  'YBLOCK': 32},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 6,  'YBLOCK': 64},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 6,  'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 6,  'YBLOCK': 128}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 3,  'YBLOCK': 64},  num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 6
    ynumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel

    # c = channel index (0..5), stride-1 in output -> warp-innermost for coalescing
    c = xindex
    # flat = N * H_out * W_out index
    flat = yindex
    n = flat // 196
    hw = flat % 196
    h_out = hw // 14
    w_out = hw % 14

    # Input strides: (4704, 784, 28, 1) for (N, C, H_in, W_in)
    # Pool stride=2: h_in = 2*h_out, w_in = 2*w_out
    # base = n*4704 + c*784 + (2*h_out)*28 + (2*w_out)*1
    base = n * 4704 + c * 784 + h_out * 56 + w_out * 2
    mask = ymask & xmask

    # 2x2 pool window offsets: (dh*28 + dw) for dh,dw in {0,1}
    tmp0 = tl.load(in_ptr0 + base,      mask=mask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + base + 1,  mask=mask, eviction_policy='evict_first')
    tmp2 = tl.load(in_ptr0 + base + 28, mask=mask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + base + 29, mask=mask, eviction_policy='evict_first')

    tmp4 = tl.maximum(tmp0, tmp1)
    tmp5 = tl.maximum(tmp4, tmp2)
    tmp6 = tl.maximum(tmp5, tmp3)

    # Output strides: (1176, 1, 84, 6) for (N, C, H_out, W_out)
    # c has stride 1 -> consecutive xindex threads write consecutive addresses (COALESCED)
    out_addr = n * 1176 + c + h_out * 84 + w_out * 6
    tl.store(out_ptr0 + out_addr, tmp6, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 24576
    xnumel = 196
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_1[grid](arg_0, arg_1, 24576, 196)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 6, 28, 28), (4704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 6, 14, 14), (1176, 1, 84, 6), device='cuda:0', dtype=torch.float32)
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
