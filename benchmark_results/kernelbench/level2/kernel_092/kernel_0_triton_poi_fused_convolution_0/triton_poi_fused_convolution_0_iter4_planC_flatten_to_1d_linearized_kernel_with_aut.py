import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flatten 2D iteration space into 1D: numel = ynumel * xnumel = 30 * 50176 = 1505280
    # Use only program_id(0) with a flat linear index to avoid YBLOCK/modulus-3 issues
    numel = ynumel * xnumel  # = 1505280
    pid = tl.program_id(0)
    idx = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = idx < numel
    x2 = idx % xnumel          # pixel index [0, 50176)
    y3 = idx // xnumel         # combined batch*channel index [0, 30)
    y0 = y3 % 3                # channel within triplet [0, 3)
    y1 = y3 // 3               # batch index [0, 10)
    tmp0 = tl.load(in_ptr0 + (x2 + xnumel * y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 150528 * y1), tmp0, mask)

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
