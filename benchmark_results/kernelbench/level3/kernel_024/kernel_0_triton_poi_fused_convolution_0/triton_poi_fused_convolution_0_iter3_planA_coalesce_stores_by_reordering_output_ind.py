import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4,  num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D iteration over output memory addresses (numel = 1505280)
    # xnumel passed as numel=1505280, ynumel unused (=1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.max_contiguous(
        tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK
    )
    xmask = xindex < xnumel

    # Decode flat output address -> (n, c, h, w)
    # Output layout strides: (150528, 1, 672, 3)
    # Memory order: c innermost (stride-1), w next (stride-3), h (stride-672), n (stride-150528)
    c = xindex % 3
    tmp1 = xindex // 3
    w = tmp1 % 224
    tmp2 = tmp1 // 224
    h = tmp2 % 224
    n = tmp2 // 224

    # Input index: NCHW contiguous, strides (150528, 50176, 224, 1)
    in_idx = n * 150528 + c * 50176 + h * 224 + w
    tmp0 = tl.load(in_ptr0 + in_idx, xmask)

    # Store to flat output address => stride-1, perfectly coalesced
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
