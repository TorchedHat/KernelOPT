import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)   # [XBLOCK] — spatial dim (stride=128 in src)
    yindex = yoffset + tl.arange(0, YBLOCK)   # [YBLOCK] — channel dim (stride=1 in src)
    xmask = xindex < xnumel                    # [XBLOCK] guard xnumel=31500 boundary
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128   # channel within batch-channel block (stride=1 in src)
    y1 = yindex // 128  # batch index
    # COALESCED LOAD: y0 (stride=1 in source) is the fast axis => shape [XBLOCK, YBLOCK]
    # Each warp reads 32 consecutive channel elements — one cache line per warp
    load_offsets = y0[None, :] + 128 * x2[:, None] + 4032000 * y1[None, :]
    tile = tl.load(in_ptr0 + load_offsets, mask=xmask[:, None], other=0.0)  # [XBLOCK, YBLOCK]
    # COALESCED STORE: x2 (stride=1 in dest) is the fast axis => shape [YBLOCK, XBLOCK]
    # tl.trans converts [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    store_offsets = x2[None, :] + 31500 * y3[:, None]   # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_offsets, tl.trans(tile), mask=xmask[None, :])

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
