import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 128, 'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 512, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=16),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 1}, num_warps=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel,
                                         YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flatten 2D (ynumel x xnumel) into 1D using YBLOCK as the block size
    # XBLOCK is fixed to 1 in all configs to eliminate X-thread waste
    numel = ynumel * xnumel  # 12288 * 9 = 110592
    BLOCK_SIZE: tl.constexpr = YBLOCK * XBLOCK  # effective 1D block size
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(numel, BLOCK_SIZE)
    tile_id = pid
    while tile_id < num_tiles:
        offset = tile_id * BLOCK_SIZE
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < numel
        x2 = idx % 9
        y3 = idx // 9
        y0 = y3 % 64
        y1 = y3 // 64
        tmp0 = tl.load(in_ptr0 + idx, mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, mask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
