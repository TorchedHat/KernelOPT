import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK
    # LOAD PHASE: yindex slow dim [:, None], xindex fast dim [None, :] -> coalesced reads
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # [YBLOCK, 1]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # [1, XBLOCK]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    # Load [YBLOCK, XBLOCK] tile contiguously (stride-1 in x)
    tmp0 = tl.load(in_ptr0 + (xindex + 50176 * yindex), xmask & ymask,
                   eviction_policy='evict_first', other=0.0)
    # STORE PHASE: transpose tile so y is the fast (contiguous) thread dim
    tmp0_T = tl.trans(tmp0)  # [XBLOCK, YBLOCK]
    yindex2 = yoffset + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK] fast dim
    xindex2 = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1] slow dim
    ymask2 = yindex2 < ynumel
    xmask2 = xindex2 < xnumel
    y0_2 = (yindex2 % 3)   # [1, YBLOCK]
    y1_2 = (yindex2 // 3)  # [1, YBLOCK]
    # Store: consecutive threads vary over YBLOCK -> y0_2 varies -> stride-1 stores
    tl.store(out_ptr0 + (y0_2 + 3 * xindex2 + 150528 * y1_2), tmp0_T, xmask2 & ymask2)

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
