import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK

    # --- Phase 1: Coalesced LOAD (x is fast dim, stride-1 in input) ---
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1] slow
    xindex_1d = xoffset + tl.arange(0, XBLOCK)         # [XBLOCK] fast
    xindex_1d = tl.max_contiguous(tl.multiple_of(xindex_1d, XBLOCK), XBLOCK)
    xindex = xindex_1d[None, :]                         # [1, XBLOCK]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    # Coalesced load: in_ptr0[x + 50176*y], stride-1 in x
    smem = tl.load(in_ptr0 + (xindex + 50176 * yindex), xmask & ymask,
                   eviction_policy='evict_last', other=0.0)  # shape [YBLOCK, XBLOCK]

    # --- Phase 2: Coalesced STORE (y is fast dim, stride-1 in output y0) ---
    # Transpose the tile so x becomes slow dim, y becomes fast dim
    smem_T = tl.trans(smem)  # shape [XBLOCK, YBLOCK]
    yindex2 = yoffset + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK] NOW fast dim
    xindex2 = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1] NOW slow dim
    ymask2 = yindex2 < ynumel
    xmask2 = xindex2 < xnumel
    y0 = yindex2 % 3    # [1, YBLOCK] stride-1 in output — fast dim
    y1 = yindex2 // 3   # [1, YBLOCK]
    x2 = xindex2        # [XBLOCK, 1]
    # Coalesced store: consecutive threads vary over y (stride-1 in output)
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), smem_T,
             xmask2 & ymask2, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
