import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK':   8}, num_warps=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK':  16}, num_warps=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 16   # y0 range: 0..15 (stride-1 output dim)
    ynumel = 800  # flat (x2, y1): 25*32=800
    # X-tile covers y0 (stride-1 output dim) -> coalesced stores
    xoffset = tl.program_id(0) * XBLOCK
    y0 = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape [1, XBLOCK]
    xmask = y0 < xnumel
    # Y-tile covers flat (x2, y1) index
    yoffset = tl.program_id(1) * YBLOCK
    flat_idx = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = flat_idx < ynumel
    x2 = flat_idx % 25   # kernel spatial index (0..24)
    y1 = flat_idx // 25  # batch*channel outer index (0..31)
    y3 = y1 * 16 + y0    # original yindex = y1*16 + y0, range 0..511
    # Load: input addr = x2 + 25*y3 (same formula as before)
    tmp0 = tl.load(in_ptr0 + (x2 + 25 * y3), xmask & ymask, eviction_policy='evict_last')
    # Store: output addr = y0 + 16*x2 + 400*y1
    # Consecutive x (y0) threads write to consecutive output addresses -> COALESCED
    tl.store(out_ptr0 + (y0 + 16 * x2 + 400 * y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 512
    xnumel = 25
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_14[grid](arg_0, arg_1, 512, 25)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 16, 5, 5), (400, 1, 80, 16), device='cuda:0', dtype=torch.float32)
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
