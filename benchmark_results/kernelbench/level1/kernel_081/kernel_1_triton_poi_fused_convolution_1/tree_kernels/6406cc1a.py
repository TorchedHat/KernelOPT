import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=2, num_stages=4),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK

    # Phase 1: Coalesced LOAD — Y-threads vary over channel (stride-1 in input)
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    mask = ymask & xmask

    y0 = yindex % 128   # channel (stride-1 in input)
    y1 = yindex // 128  # batch
    x2 = xindex         # spatial

    # Load [YBLOCK, XBLOCK] tile — Y varies over stride-1 channel dim → coalesced
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x2 + 4032000 * y1), mask=mask, other=0.0)  # [YBLOCK, XBLOCK]

    # Phase 2: Transpose in shared memory via tl.trans()
    tmp0_T = tl.trans(tmp0)  # [XBLOCK, YBLOCK]

    # Phase 3: Coalesced STORE — X-threads vary over spatial (stride-1 in output)
    # Apply hints on 1D arange BEFORE broadcasting to enable 128-bit vectorized stores
    # xindex2 is stride-1 in output (contiguous), hint enables 128-bit vectorized stores
    xarange2 = tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK)  # multiples of XBLOCK
    xarange2 = tl.max_contiguous(xarange2, XBLOCK)            # XBLOCK contiguous
    xindex2 = xoffset + xarange2[:, None]  # [XBLOCK, 1]
    yindex2 = yoffset + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK]
    xmask2 = xindex2 < xnumel
    ymask2 = yindex2 < ynumel
    mask2 = xmask2 & ymask2

    tl.store(out_ptr0 + (xindex2 + 31500 * yindex2), tmp0_T, mask=mask2)  # coalesced

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
