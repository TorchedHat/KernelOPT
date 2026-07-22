import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 32,  'XBLOCK': 1},  num_warps=2),
    triton.Config({'YBLOCK': 64,  'XBLOCK': 1},  num_warps=4),
    triton.Config({'YBLOCK': 128, 'XBLOCK': 1},  num_warps=4),
    triton.Config({'YBLOCK': 256, 'XBLOCK': 1},  num_warps=8),
    triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2),
    triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
    triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
    triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 32768
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
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
