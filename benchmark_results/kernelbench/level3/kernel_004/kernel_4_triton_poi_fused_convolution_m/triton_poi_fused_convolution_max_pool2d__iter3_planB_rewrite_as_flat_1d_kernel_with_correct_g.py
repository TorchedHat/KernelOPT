import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel: linearize all 1,638,400 output elements
    # ynumel is passed as 1638400 (total numel), xnumel unused
    numel = ynumel
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    idx = offsets
    # Decompose flat output index into (n, c, oh, ow)
    # Output shape: (4096, 16, 5, 5), strides: (400, 25, 5, 1)
    n    = idx // 400
    rem  = idx % 400
    c    = rem // 25
    rem2 = rem % 25
    oh   = rem2 // 5
    ow   = rem2 % 5
    # Input strides: (1600, 1, 160, 16) for shape (4096, 16, 10, 10)
    # Pool window top-left: ih=oh*2, iw=ow*2
    # base = n*1600 + c*1 + (oh*2)*160 + (ow*2)*16
    base = n * 1600 + c + oh * 320 + ow * 32
    t0 = tl.load(in_ptr0 + base,       mask=mask)
    t1 = tl.load(in_ptr0 + base + 16,  mask=mask)  # iw+1: stride_W=16
    t2 = tl.load(in_ptr0 + base + 160, mask=mask)  # ih+1: stride_H=160
    t3 = tl.load(in_ptr0 + base + 176, mask=mask)  # ih+1,iw+1
    result = tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3))
    # Output offset: strides (400, 25, 5, 1)
    out_off = n * 400 + c * 25 + oh * 5 + ow
    tl.store(out_ptr0 + out_off, result, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 102400
    xnumel = 16
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 102400, 16)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 16, 10, 10), (1600, 1, 160, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
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
