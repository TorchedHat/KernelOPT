import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512,   'YBLOCK': 1}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1024,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 2048,  'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 4096,  'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 8192,  'YBLOCK': 1}, num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 16384, 'YBLOCK': 1}, num_warps=16, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat output index over all 1505280 elements (N=10, C=3, H=224, W=224)
    # xnumel_flat = ynumel * xnumel = 30 * 50176 = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.max_contiguous(
        tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK
    )
    xnumel_flat = 1505280
    xmask = xindex < xnumel_flat
    # Decompose flat output index into NHWC coordinates
    # Output layout: (N=10, H=224, W=224, C=3), strides=(150528, 672, 3, 1)
    # out_flat = n*150528 + hw*3 + c
    c  = xindex % 3           # channel (0,1,2) - fast axis in output
    hw = (xindex // 3) % 50176  # pixel index h*224+w
    n  = xindex // 150528     # batch index
    # Reconstruct NCHW input address: in[n, c, h, w] = n*150528 + c*50176 + hw
    in_idx = n * 150528 + c * 50176 + hw
    tmp0 = tl.load(in_ptr0 + in_idx, xmask)
    # Store to flat output (stride-1, fully coalesced)
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
