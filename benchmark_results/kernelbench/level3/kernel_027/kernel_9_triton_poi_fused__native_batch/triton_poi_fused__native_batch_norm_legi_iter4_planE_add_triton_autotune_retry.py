import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Collapse 2D (ynumel=32768, xnumel=9) into flat 1D numel=294912
    # Use only program_id(0) and XBLOCK for 1D tiling
    numel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    flat_index = xoffset + tl.arange(0, XBLOCK)
    mask = flat_index < numel
    # Reconstruct x2 (0..8) and y3 (0..32767) from flat index
    # flat_index = y3 * 9 + x2  (row-major over y3, x2)
    x2 = flat_index % 9
    y3 = flat_index // 9
    y0 = y3 % 128
    y1 = y3 // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 32768
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9[grid](arg_0, arg_1, 32768, 9)
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
