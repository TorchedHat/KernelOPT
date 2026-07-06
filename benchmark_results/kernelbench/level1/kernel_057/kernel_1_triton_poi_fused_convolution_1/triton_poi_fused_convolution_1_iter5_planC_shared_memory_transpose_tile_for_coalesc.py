import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Reinterpret: YBLOCK=CBLOCK (C dimension), XBLOCK=HWBLOCK (HW dimension)
    # ynumel=C=128, xnumel=HW=31500
    # 3D grid: axis0=HW tiles, axis1=C tiles, axis2=N (batch)
    hw_tile = tl.program_id(0)  # HW tile index
    c_tile  = tl.program_id(1)  # C tile index
    n_idx   = tl.program_id(2)  # batch index (scalar)
    # YBLOCK walks C dim (stride=1 in NHWC input -> coalesced load)
    c_range  = tl.arange(0, YBLOCK)   # [YBLOCK=CBLOCK]
    # XBLOCK walks HW dim (stride=1 in NCHW output -> coalesced store)
    hw_range = tl.arange(0, XBLOCK)   # [XBLOCK=HWBLOCK]
    c_idx  = c_tile  * YBLOCK + c_range   # [CBLOCK]
    hw_idx = hw_tile * XBLOCK + hw_range  # [HWBLOCK]
    c_mask  = c_idx  < ynumel   # [CBLOCK], ynumel=C=128
    hw_mask = hw_idx < xnumel   # [HWBLOCK], xnumel=HW=31500
    load_mask = c_mask[:, None] & hw_mask[None, :]  # [CBLOCK, HWBLOCK]
    # COALESCED LOAD: consecutive threads differ in c_idx (stride=1 in NHWC input)
    # Input strides: N=4032000, C=1, H=32000, W=128 -> addr = c + 128*hw + 4032000*n
    load_addr = c_idx[:, None] + 128 * hw_idx[None, :] + 4032000 * n_idx
    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)  # [CBLOCK, HWBLOCK]
    # COALESCED STORE: consecutive threads differ in hw_idx (stride=1 in NCHW output)
    # Output strides: N=4032000, C=31500, HW=1 -> addr = hw + 31500*c + 4032000*n
    store_addr = hw_idx[None, :] + 31500 * c_idx[:, None] + 4032000 * n_idx
    tl.store(out_ptr0 + store_addr, tile, mask=load_mask)

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
