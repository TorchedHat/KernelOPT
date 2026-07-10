import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 16},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 256}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    yoffset = tl.program_id(1) * YBLOCK
    yoffs = tl.max_contiguous(tl.multiple_of(yoffset + tl.arange(0, YBLOCK), YBLOCK), YBLOCK)  # 1D [YBLOCK]
    yindex = yoffs             # 1D shape [YBLOCK]
    ymask = yindex < ynumel

    xoffset = tl.program_id(0) * XBLOCK
    xoffs = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)  # 1D [XBLOCK]
    xindex = xoffs             # 1D shape [XBLOCK]
    xmask = xindex < xnumel

    y0 = yindex % 128          # stride-1 in input, varies along YBLOCK (fast) -> COALESCED load
    y1 = yindex // 128

    # 2D combined mask [YBLOCK, XBLOCK]
    mask2d = ymask[:, None] & xmask[None, :]

    # Load: y0[:, None] fast along YBLOCK (stride-1 in input) -> coalesced
    load_addr = y0[:, None] + 128 * xindex[None, :] + 4032000 * y1[:, None]  # [YBLOCK, XBLOCK]
    tmp0 = tl.load(in_ptr0 + load_addr, mask=mask2d, eviction_policy='evict_last')

    # Store: xindex[None, :] fast along XBLOCK (stride-1 in output) -> coalesced
    store_addr = xindex[None, :] + 31500 * yindex[:, None]                   # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_addr, tmp0, mask=mask2d)

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
