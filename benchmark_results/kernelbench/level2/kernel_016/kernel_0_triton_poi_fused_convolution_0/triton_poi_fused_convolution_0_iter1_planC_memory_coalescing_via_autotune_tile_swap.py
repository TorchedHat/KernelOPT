import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=1),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Treat the problem as 1D over numel = ynumel * xnumel = 30 * 50176 = 1505280
    # but iterate in OUTPUT-contiguous order (C fastest, stride=1 in output)
    # We use only program_id(0) and XBLOCK; YBLOCK=1 so the Y dimension is unused.
    numel = ynumel * xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)
    xmask   = xindex < numel
    # Decompose in OUTPUT-contiguous order: C is fastest (stride=1 in output)
    c     = xindex % 3
    rest  = xindex // 3
    w     = rest % 224
    rest2 = rest // 224
    h     = rest2 % 224
    n     = rest2 // 224
    in_idx  = n * 150528 + c * 50176 + h * 224 + w
    out_idx = n * 150528 + c          + h * 672  + w * 3
    tmp0 = tl.load(in_ptr0 + in_idx, xmask)
    tl.store(out_ptr0 + out_idx, tmp0, xmask)

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
