import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_29(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 524288*y1), mask=xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 2048*y0 + 1048576*y1), tmp2, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 2048
    xnumel = 2048
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_29[grid](arg_0, arg_1, arg_2, 2048, 2048)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 256, 16, 128), (524288, 1, 32768, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
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
