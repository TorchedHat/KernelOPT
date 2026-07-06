import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel=4096 (N*C=32*128), xnumel=31500 (H*W=126*250)
    # Input  stride: (4032000, 1, 32000, 128) -- contiguous along C (stride=1)
    # Output stride: (4032000, 31500, 250, 1) -- contiguous along W (stride=1)
    # Shared-memory staged transpose: load with nc as fast axis (coalesced),
    # tl.trans() to swap axes, store with hw as fast axis (coalesced).
    # XBLOCK == YBLOCK (square tile) for tl.trans() to work correctly.
    pid_nc = tl.program_id(0)  # tiles N*C (ynumel=4096)
    pid_hw = tl.program_id(1)  # tiles H*W (xnumel=31500)

    nc_off = pid_nc * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    hw_off = pid_hw * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    nc_mask = nc_off < ynumel
    hw_mask = hw_off < xnumel

    # COALESCED LOAD: nc is the fast (column) axis, C stride=1 in input
    nc_load = nc_off[None, :]   # [1, XBLOCK] - fast axis (stride=1 in input)
    hw_load = hw_off[:, None]   # [YBLOCK, 1] - slow axis
    mask_load = hw_mask[:, None] & nc_mask[None, :]  # [YBLOCK, XBLOCK]
    n_l = nc_load // 128
    c_l = nc_load % 128   # c_l stride=1 in input -> coalesced loads
    h_l = hw_load // 250
    w_l = hw_load % 250
    load_idx = n_l * 4032000 + c_l + h_l * 32000 + w_l * 128
    tmp = tl.load(in_ptr0 + load_idx, mask=mask_load)  # [YBLOCK, XBLOCK]

    # CRITICAL: tl.trans() swaps axes -> [XBLOCK, YBLOCK]
    tmp_t = tl.trans(tmp)

    # COALESCED STORE: hw is the fast (column) axis, stride=1 in output
    nc_store = nc_off[:, None]   # [XBLOCK, 1] - slow axis
    hw_store = hw_off[None, :]   # [1, YBLOCK] - fast axis (stride=1 in output)
    mask_store = nc_mask[:, None] & hw_mask[None, :]  # [XBLOCK, YBLOCK]
    n_s = nc_store // 128
    c_s = nc_store % 128
    store_idx = n_s * 4032000 + c_s * 31500 + hw_store
    tl.store(out_ptr0 + store_idx, tmp_t, mask=mask_store)

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
