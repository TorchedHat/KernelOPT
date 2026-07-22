import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 1D flat kernel: treat ynumel as total numel (3456), ignore xnumel/YBLOCK
    numel = ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < numel
    w = xindex % 3
    h = (xindex // 3) % 3
    c = (xindex // 9) % 6
    b = xindex // 54
    in_offset  = b * 54 + c * 9 + h * 3 + w
    out_offset = b * 54 + c     + h * 18 + w * 6
    tmp0 = tl.load(in_ptr0 + in_offset, mask=xmask, other=0.0)
    tl.store(out_ptr0 + out_offset, tmp0, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 384
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 6, 3, 3), (54, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 6, 3, 3), (54, 1, 18, 6), device='cuda:0', dtype=torch.float32)
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
