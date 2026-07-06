import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # Y-tile covers y-dimension (4096 = 32 * 128) - keep ALL as 1D until broadcast
    yoffset = tl.program_id(1) * YBLOCK
    yrange = tl.arange(0, YBLOCK)          # [YBLOCK] 1D
    yindex = yoffset + yrange              # [YBLOCK] 1D
    ymask = yindex < ynumel               # [YBLOCK] 1D
    # X-tile covers x-dimension (31500)
    xoffset = tl.program_id(0) * XBLOCK
    xrange = tl.arange(0, XBLOCK)          # [XBLOCK] 1D
    xindex = xoffset + xrange              # [XBLOCK] 1D
    xmask = xindex < xnumel               # [XBLOCK] 1D
    # Decompose y indices (keep 1D)
    y0 = yindex % 128   # [YBLOCK] 1D - stride-1 dim in source (in_ptr0 stride=1)
    y1 = yindex // 128  # [YBLOCK] 1D - batch dim in source (in_ptr0 stride=4032000)
    # COALESCED LOAD: shape [YBLOCK, XBLOCK]
    # Y is fast dim: y0 varies with stride 1 -> coalesced LDG.128
    # X is slow dim: xindex varies with stride 128 per row
    load_mask = ymask[:, None] & xmask[None, :]                              # [YBLOCK, XBLOCK]
    load_addr = y0[:, None] + 128*xindex[None, :] + 4032000*y1[:, None]     # [YBLOCK, XBLOCK]
    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0, eviction_policy='evict_first')
    # TRANSPOSE in-register: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    tile_t = tl.trans(tile)
    # COALESCED STORE: shape [XBLOCK, YBLOCK]
    # X is fast dim: xindex varies with stride 1 in output -> coalesced STG.128
    store_mask = xmask[:, None] & ymask[None, :]                             # [XBLOCK, YBLOCK]
    store_addr = xindex[:, None] + 31500*yindex[None, :]                     # [XBLOCK, YBLOCK]
    tl.store(out_ptr0 + store_addr, tile_t, mask=store_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
