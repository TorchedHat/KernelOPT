import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # numel = N * C * H * W = 10 * 3 * 224 * 224 = 1_505_280
    # Output layout: NHWC, strides (150528, 1, 672, 3)
    # Input layout:  NCHW, strides (150528, 50176, 224, 1)
    numel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # flat output index [0, numel)
    xmask = xindex < numel

    # Decompose flat output index into NHWC coordinates
    # out_flat = n*C*H*W + h*W*C + w*C + c  where C=3, H=224, W=224
    c    = xindex % 3                    # channel [0,3)
    tmp  = xindex // 3                   # n*H*W + h*W + w
    w    = tmp % 224                     # width  [0,224)
    tmp2 = tmp // 224                    # n*H + h
    h    = tmp2 % 224                    # height [0,224)
    n    = tmp2 // 224                   # batch  [0,10)

    # Input NCHW address: n*150528 + c*50176 + h*224 + w
    in_idx = n * 150528 + c * 50176 + h * 224 + w
    # Output NHWC address: xindex (flat, stride=1 in output buffer)
    tmp0 = tl.load(in_ptr0 + in_idx, xmask, eviction_policy='evict_first', other=0.0)
    tl.store(out_ptr0 + xindex, tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
