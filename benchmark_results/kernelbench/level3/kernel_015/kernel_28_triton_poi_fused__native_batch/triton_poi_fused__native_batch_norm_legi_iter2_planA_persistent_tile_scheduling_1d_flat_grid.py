import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 128,  'XBLOCK': 2}, num_warps=2, num_stages=3),
    triton.Config({'YBLOCK': 256,  'XBLOCK': 2}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 512,  'XBLOCK': 4}, num_warps=4, num_stages=3),
    triton.Config({'YBLOCK': 1024, 'XBLOCK': 4}, num_warps=8, num_stages=3),
    triton.Config({'YBLOCK': 2048, 'XBLOCK': 8}, num_warps=8, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # YBLOCK = BLOCK (tile size over flattened numel)
    # XBLOCK = TILES_PER_SM (number of tiles each CTA processes)
    numel = 101376  # xnumel * ynumel = 9 * 11264
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    tile_base = pid * YBLOCK
    tile_stride = num_ctas * YBLOCK
    while tile_base < numel:
        offsets = tile_base + tl.arange(0, YBLOCK)
        mask = offsets < numel
        y3 = offsets // 9
        x2 = offsets % 9
        y0 = y3 % 352
        y1 = y3 // 352
        tmp0 = tl.load(in_ptr0 + offsets, mask, eviction_policy='evict_first')
        tl.store(out_ptr0 + (y0 + 352 * x2 + 3168 * y1), tmp0, mask)
        tile_base += tile_stride

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 11264
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28[grid](arg_0, arg_1, 11264, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 352, 3, 3), (3168, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 352, 3, 3), (3168, 1, 1056, 352), device='cuda:0', dtype=torch.float32)
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
