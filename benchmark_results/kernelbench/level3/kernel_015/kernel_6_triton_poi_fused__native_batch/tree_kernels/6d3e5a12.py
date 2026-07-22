import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 8, 'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 8, 'XBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 8, 'XBLOCK': 512}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 8, 'XBLOCK': 1024}, num_warps=8, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Persistent 1D kernel: flatten all 27648 elements, each CTA loops over tiles
    # YBLOCK=256, XBLOCK=1 => BLOCK=256 per tile
    BLOCK: tl.constexpr = YBLOCK * XBLOCK
    total: tl.constexpr = 3072 * 9  # 27648
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_start = pid * BLOCK
    while tile_start < total:
        offsets = tile_start + tl.arange(0, BLOCK)
        mask = offsets < total
        flat = offsets
        y3 = flat // 9
        x2 = flat % 9
        y0 = y3 % 96
        y1 = y3 // 96
        tmp0 = tl.load(in_ptr0 + flat, mask=mask)
        tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, mask=mask)
        tile_start += num_programs * BLOCK

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 3072
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_6[grid](arg_0, arg_1, 3072, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
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
