import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 64},  num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32},  num_warps=4, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # pid(0) -> spatial dimension (xnumel / XBLOCK tiles)
    # pid(1) -> channel dimension (ynumel / YBLOCK tiles)
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK

    # Channel indices (stride-1 in source): make this the LAST axis [None, :] -> innermost for coalesced loads
    ychan = yoffset + tl.arange(0, YBLOCK)   # 1D [YBLOCK]
    ymask = ychan < ynumel
    y0 = ychan % 128        # channel within group [0,128)
    y1 = ychan // 128       # batch index [0,32)

    # Spatial indices (stride-128 in source): make this the FIRST axis [:, None]
    xspa = xoffset + tl.arange(0, XBLOCK)    # 1D [XBLOCK]
    xmask = xspa < xnumel

    # combined_mask for load: shape [XBLOCK, YBLOCK]
    combined_mask_load = xmask[:, None] & ymask[None, :]

    # COALESCED LOAD: tile shape [XBLOCK, YBLOCK], ychan (stride-1) is innermost (last axis)
    # src address: y0 + 128*xspa + 4032000*y1
    src_idx = y0[None, :] + 128 * xspa[:, None] + 4032000 * y1[None, :]  # [XBLOCK, YBLOCK]
    tile = tl.load(in_ptr0 + src_idx, combined_mask_load, eviction_policy='evict_first')

    # Transpose tile to [YBLOCK, XBLOCK] so xspa (stride-1 in output) is innermost for coalesced stores
    tile_T = tl.trans(tile)  # [YBLOCK, XBLOCK]

    # combined_mask for store: shape [YBLOCK, XBLOCK] (transposed)
    combined_mask_store = ymask[:, None] & xmask[None, :]  # [YBLOCK, XBLOCK]

    # COALESCED STORE: xspa (stride-1 in output) is innermost (last axis)
    # Apply multiple_of hint to 1D xspa before broadcasting
    xspa_vec = tl.max_contiguous(tl.multiple_of(xspa, XBLOCK), XBLOCK)  # 1D [XBLOCK]
    # dst address: xspa + 31500*ychan
    dst_idx = xspa_vec[None, :] + 31500 * ychan[:, None]  # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + dst_idx, tile_T, combined_mask_store, eviction_policy='evict_first')

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
