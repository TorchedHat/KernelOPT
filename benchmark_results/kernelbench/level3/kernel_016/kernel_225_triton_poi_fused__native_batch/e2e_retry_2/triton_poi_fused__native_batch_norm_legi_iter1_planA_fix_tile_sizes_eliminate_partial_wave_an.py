import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 256,  'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 4096, 'XBLOCK': 1}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D tiling: treat all ynumel*xnumel=516096 elements as a single dimension
    # XBLOCK=1 always, YBLOCK is the effective block size
    numel = 57344 * 9  # = 516096
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < numel
    # Decompose flat index into original 2D indices
    x2 = offsets % 9          # kernel-filter index [0..8]
    y3 = offsets // 9         # batch*channel index [0..57343]
    y0 = y3 % 1792            # channel within group
    y1 = y3 // 1792           # batch index
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1792*x2 + 16128*y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 57344
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225[grid](arg_0, arg_1, 57344, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 1792, 3, 3), (16128, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 1792, 3, 3), (16128, 1, 5376, 1792), device='cuda:0', dtype=torch.float32)
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
