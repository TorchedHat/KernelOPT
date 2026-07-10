import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=4,  num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_clamp_convolution_div_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flatten the 2D (ynumel=8192, xnumel=65536) space into 1D
    # numel = 8192 * 65536 = 536870912
    # Decompose flat index so channel c = flat % 64 is innermost (stride-1 in NHWC in_ptr0)
    pid = tl.program_id(0)
    flat_range = tl.arange(0, XBLOCK)
    flat_range = tl.max_contiguous(tl.multiple_of(flat_range, XBLOCK), XBLOCK)
    flat = pid * XBLOCK + flat_range
    numel = 536870912
    mask = flat < numel
    # Decompose: innermost = channel c (stride-1 in in_ptr0 NHWC)
    c         = flat % 64
    remainder = flat // 64
    x2        = remainder % 65536
    y1        = remainder // 65536
    # Input NHWC strides (4194304,1,16384,64): address = y1*4194304 + c + x2*64
    tmp0 = tl.load(in_ptr0 + (y1 * 4194304 + c + x2 * 64), mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + c, mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + c, mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tl.maximum(tmp4, 0.0)
    tmp8 = tl.minimum(tmp6, 1.0)
    tmp10 = tmp8 * 2.0
    tmp11 = tl.maximum(tmp10, 0.0)
    tmp12 = tl.minimum(tmp11, 1.0)
    tmp14 = tmp12 * 0.5
    # Output NCHW strides (4194304,65536,256,1): address = y1*4194304 + c*65536 + x2
    tl.store(out_ptr0 + (y1 * 4194304 + c * 65536 + x2), tmp14, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 8192
    xnumel = 65536
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_clamp_convolution_div_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 8192, 65536)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 64, 256, 256), (4194304, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
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
