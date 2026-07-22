import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 6
    ynumel = 802816
    yoffset = tl.multiple_of(tl.program_id(1) * YBLOCK, YBLOCK)
    yindex = tl.max_contiguous(yoffset + tl.arange(0, YBLOCK), YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.multiple_of(tl.program_id(0) * XBLOCK, XBLOCK)
    xindex = tl.max_contiguous(xoffset + tl.arange(0, XBLOCK), XBLOCK)[None, :]
    xmask = xindex < xnumel
    # c = channel index (0..5), innermost X dimension (stride-1 in output)
    c = xindex
    # Decompose yindex into batch and spatial
    y_hw = yindex % 196
    y_bat = yindex // 196
    y_w = y_hw % 14
    y_h = y_hw // 14
    # Input base address: input strides are (4704, 784, 28, 1)
    # in_ptr0[y_bat, c, 2*y_h, 2*y_w] = y_bat*4704 + c*784 + 2*y_h*28 + 2*y_w*1
    base = y_bat * 4704 + c * 784 + 56 * y_h + 2 * y_w
    tmp0 = tl.load(in_ptr0 + base,      mask=ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (base + 1),  mask=ymask & xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (base + 28), mask=ymask & xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (base + 29), mask=ymask & xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    # Output address: output strides are (1176, 1, 84, 6)
    # out_ptr0[y_bat, c, y_h, y_w] = y_bat*1176 + c*1 + y_h*84 + y_w*6
    out_addr = y_bat * 1176 + c + y_h * 84 + y_w * 6
    tl.store(out_ptr0 + out_addr, tmp6, mask=ymask & xmask)

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
