import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8, 'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 256}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # xnumel = 6  (channels - innermost, warp dimension)
    # ynumel = 802816  (4096 * 14 * 14 - outer dimension)
    xnumel = 6
    ynumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # channels 0..5
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # batch*out_h*out_w
    ymask = yindex < ynumel
    # Decompose yindex into (batch, out_h, out_w)
    out_w  = yindex % 14
    out_hw = yindex // 14
    out_h  = out_hw % 14
    batch  = out_hw // 14
    ch = xindex  # 0..5
    mask = xmask & ymask
    # Input strides: (4704, 784, 28, 1)
    base = 2*out_w + 56*out_h + 784*ch + 4704*batch
    tmp0 = tl.load(in_ptr0 + base,      mask=mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + base + 1,  mask=mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + base + 28, mask=mask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + base + 29, mask=mask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    # Output strides: (1176, 1, 84, 6)
    # Consecutive threads (varying ch) write to consecutive addresses -> COALESCED
    out_addr = ch + 6*out_w + 84*out_h + 1176*batch
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
