import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # ynumel = NC = 5120 (N*C), xnumel = HW = 50176 (H*W)
    # YBLOCK = CBLOCK (tiles NC dim), XBLOCK = HWBLOCK (tiles HW dim)
    # Repurpose: pid(0) -> HW tiles, pid(1) -> NC tiles
    NC = 5120
    HW = 50176
    C = 512

    pid_hw = tl.program_id(0)   # tiles HW dimension
    pid_nc = tl.program_id(1)   # tiles NC dimension

    # NC indices: [YBLOCK] - these are the channel*batch indices
    nc_idx = pid_nc * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK]
    # HW indices: [XBLOCK] - these are the spatial indices
    hw_idx = pid_hw * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK]

    nc_mask = nc_idx < NC  # [YBLOCK]
    hw_mask = hw_idx < HW  # [XBLOCK]

    n_idx = nc_idx // C    # batch index [YBLOCK]
    c_idx = nc_idx % C     # channel index [YBLOCK]

    # COALESCED LOAD: tile shape [XBLOCK, YBLOCK] = [HWBLOCK, CBLOCK]
    # Source strides: (25690112, 1, 114688, 512) -> in[n,c,h,w] at n*25690112 + c*1 + hw*512
    # Inner dim is C (stride-1) -> coalesced across warp threads (YBLOCK is innermost)
    load_idx = (n_idx[None, :] * 25690112 +
                c_idx[None, :] * 1 +
                hw_idx[:, None] * 512)
    load_mask = hw_mask[:, None] & nc_mask[None, :]  # [XBLOCK, YBLOCK]
    tile = tl.load(in_ptr0 + load_idx, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK]

    # Transpose in registers: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    tile_T = tl.trans(tile)  # [YBLOCK, XBLOCK]

    # COALESCED STORE: tile shape [YBLOCK, XBLOCK] = [CBLOCK, HWBLOCK]
    # Output strides: (25690112, 50176, 224, 1) -> out[n,c,h,w] at n*25690112 + c*50176 + hw*1
    # Inner dim is HW (stride-1) -> coalesced across warp threads (XBLOCK is innermost)
    store_idx = (n_idx[:, None] * 25690112 +
                 c_idx[:, None] * 50176 +
                 hw_idx[None, :] * 1)
    store_mask = nc_mask[:, None] & hw_mask[None, :]  # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_idx, tile_T, mask=store_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 5120
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_cat_convolution_11[grid](arg_0, arg_1, 5120, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 512, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 512, 224, 224), (25690112, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
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
