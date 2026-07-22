import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 1024}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Persistent 1D flat-index kernel using YBLOCK as BLOCK_SIZE
    # total_numel = 8192 * 9 = 73728
    total_numel = 73728
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_start = pid * YBLOCK
    stride = num_programs * YBLOCK
    # Persistent loop: each CTA processes multiple YBLOCK-sized chunks
    while block_start < total_numel:
        offsets = block_start + tl.arange(0, YBLOCK)
        mask = offsets < total_numel
        # Decompose flat index into original (x2, y3) coordinates
        # xnumel = 9 (spatial kernel position)
        x2 = offsets % 9          # 0..8
        y3 = offsets // 9         # 0..8191 (batch*channel)
        y0 = y3 % 64              # output channel index
        y1 = y3 // 64             # batch index
        # Load: input stride (576, 9, 3, 1) -> flat offset = x2 + 9*y3
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask, eviction_policy='evict_last')
        # Store: output stride (576, 1, 192, 64) -> flat offset = y0 + 64*x2 + 576*y1
        tl.store(out_ptr0 + (y0 + 64 * x2 + 576 * y1), tmp0, mask=mask)
        block_start += stride

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_6[grid](arg_0, arg_1, 8192, 9)
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
