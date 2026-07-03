import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32}, num_warps=8, num_stages=1),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 128}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(0) * YBLOCK
    xoffset = tl.program_id(1) * XBLOCK

    # LOAD PHASE: swap tile orientation so y (stride-1) is warp-fast -> COALESCED
    # xindex shape: [XBLOCK, 1] (warp-slow), yindex shape: [1, YBLOCK] (warp-fast)
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    xmask = xindex < xnumel
    ymask = yindex < ynumel
    y0 = yindex % 128
    y1 = yindex // 128
    x2 = xindex
    # Coalesced load: [XBLOCK, YBLOCK] tile, y0 is stride-1 and warp-fast
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x2 + 4032000 * y1), mask=xmask & ymask, other=0.0, eviction_policy='evict_first')

    # TRANSPOSE in registers: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK]
    tmp0_T = tl.trans(tmp0)

    # STORE PHASE: original orientation, y is slow [YBLOCK,1], x is fast [1,XBLOCK]
    # xindex_s shape: [1, XBLOCK] (warp-fast, stride-1 in output), yindex_s shape: [YBLOCK, 1]
    yindex_s = yoffset + tl.arange(0, YBLOCK)[:, None]
    xindex_s = xoffset + tl.arange(0, XBLOCK)[None, :]
    ymask_s = yindex_s < ynumel
    xmask_s = xindex_s < xnumel
    y3 = yindex_s
    x2_vec = tl.max_contiguous(tl.multiple_of(xindex_s, [1, XBLOCK]), [1, XBLOCK])
    # Coalesced store: [YBLOCK, XBLOCK] tile, x2 is stride-1 and warp-fast
    tl.store(out_ptr0 + (x2_vec + 31500 * y3), tmp0_T, mask=xmask_s & ymask_s, eviction_policy='evict_first')

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
