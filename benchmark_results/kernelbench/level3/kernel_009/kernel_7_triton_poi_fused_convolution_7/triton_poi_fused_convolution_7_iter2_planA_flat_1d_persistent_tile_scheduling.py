import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flatten the 2D iteration space into 1D: total_numel = 8192 * 9 = 73728
    # XBLOCK is used as the flat block size; YBLOCK=1 (unused in indexing)
    TOTAL_NUMEL = 73728  # ynumel * xnumel = 8192 * 9, compile-time constant
    XNUMEL = 9           # xnumel, compile-time constant
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < TOTAL_NUMEL
    x2 = offsets % XNUMEL          # spatial index [0..8]
    y3 = offsets // XNUMEL         # combined batch+channel [0..8191]
    y0 = y3 % 64                   # channel index [0..63]
    y1 = y3 // 64                  # batch index [0..127]
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_7[grid](arg_0, arg_1, 8192, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
