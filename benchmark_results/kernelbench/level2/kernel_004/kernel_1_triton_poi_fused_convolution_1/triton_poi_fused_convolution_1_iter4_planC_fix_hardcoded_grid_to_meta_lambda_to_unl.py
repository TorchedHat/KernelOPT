import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # Original configs (store-coalesced, load-strided)
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8),
        # NEW: large YBLOCK makes Y the fast-varying warp dim
        # -> loads from in_ptr0 become coalesced (y0 has stride=1)
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
        # num_stages variants for software pipelining
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        # Square/near-square tiles for balanced load+store coalescing
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # 1D arange vectors — broadcast to 2D for address computation
    yoffs = tl.arange(0, YBLOCK)
    xoffs = tl.arange(0, XBLOCK)
    yindex = tl.program_id(1) * YBLOCK + yoffs  # shape [YBLOCK]
    xindex = tl.program_id(0) * XBLOCK + xoffs  # shape [XBLOCK]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    y0 = yindex % 128    # stride-1 in input -> YBLOCK fast-varying = coalesced loads
    y1 = yindex // 128   # stride-4032000 in input
    # 2D load address: y0[:, None] + 128*xindex[None, :] + 4032000*y1[:, None]
    # YBLOCK is fast-varying for loads (y0 stride-1 in input) -> coalesced
    load_addr = y0[:, None] + 128 * xindex[None, :] + 4032000 * y1[:, None]
    load_mask = xmask[None, :] & ymask[:, None]
    tmp0 = tl.load(in_ptr0 + load_addr, mask=load_mask, eviction_policy='evict_last')
    # 2D store address: xindex[None, :] + 31500*yindex[:, None]
    # XBLOCK is fast-varying for stores (stride-1 in output) -> coalesced
    store_addr = xindex[None, :] + 31500 * yindex[:, None]
    store_mask = xmask[None, :] & ymask[:, None]
    tl.store(out_ptr0 + store_addr, tmp0, mask=store_mask, eviction_policy='evict_first')

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
