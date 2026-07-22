import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4, 'YBLOCK': 32},  num_warps=1, num_stages=2),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2, 'YBLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 2, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), mask=xmask & ymask, eviction_policy='evict_last')
    # Load per-row max from out_ptr0 which holds the max values (shape: (128,1,2,2), stride: (4,4,2,1))
    # out_ptr0 is reused as in_ptr1 for the max values: offset = x2 + 4*y1
    tmp_max = tl.load(out_ptr0 + (x2 + 4*y1), mask=xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0 - tmp_max
    tmp2 = tl.exp(tmp1)
    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp2, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 4
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda:0', dtype=torch.float32)
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
