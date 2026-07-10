import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Register-tiled NCHW->NHWC transpose for coalesced stores
    # ynumel=30 (N*C=10*3), xnumel=50176 (H*W)
    # Grid axis 0: HW tiles (XBLOCK), axis 1: batch (N=10, YBLOCK=1)
    # CBLOCK=4 (power-of-2 >= C=3) with mask for c<3
    CBLOCK: tl.constexpr = 4   # next power of 2 >= C=3
    N: tl.constexpr = 10
    C: tl.constexpr = 3
    HW: tl.constexpr = 50176

    hw_offset = tl.program_id(0) * XBLOCK
    n_idx     = tl.program_id(1)  # batch index (grid axis 1 = N=10)

    hw_idx = hw_offset + tl.arange(0, XBLOCK)   # [XBLOCK]
    c_idx  = tl.arange(0, CBLOCK)               # [4] = [0,1,2,3]
    hw_mask = hw_idx < HW
    c_mask  = c_idx < C                          # [True, True, True, False]

    # Phase 1: Load NCHW tile coalesced (hw is fast axis, stride-1)
    # load_idx[c, hw] = n*C*HW + c*HW + hw
    load_idx = (n_idx * C * HW
                + c_idx[:, None] * HW
                + hw_idx[None, :])   # [CBLOCK=4, XBLOCK]
    load_mask = c_mask[:, None] & hw_mask[None, :]
    tile = tl.load(in_ptr0 + load_idx, load_mask, other=0.0)
    # tile shape: [4, XBLOCK], loaded with stride-1 in hw -> coalesced

    # Phase 2: Store NHWC with stride-1 in c (fast axis)
    # store_idx[hw, c] = n*HW*C + hw*C + c
    store_idx = (n_idx * HW * C
                 + hw_idx[:, None] * C
                 + c_idx[None, :])   # [XBLOCK, CBLOCK=4]
    store_mask = hw_mask[:, None] & c_mask[None, :]  # [XBLOCK, 4]
    # tl.trans(tile) converts [4, XBLOCK] -> [XBLOCK, 4] for store
    tl.store(out_ptr0 + store_idx, tl.trans(tile), store_mask)
    # store pattern: consecutive c values -> stride-1 COALESCED

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
