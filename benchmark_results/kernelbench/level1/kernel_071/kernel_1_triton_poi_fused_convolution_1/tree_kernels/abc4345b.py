import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 4096 = N*C = 32*128
    # xnumel = 31500 = H*W = 126*250
    # XBLOCK tiles over N*C (fast axis for loads, c stride=1 in input)
    # YBLOCK tiles over H*W (fast axis for stores, hw stride=1 in output)

    pid_nc = tl.program_id(0)   # N*C tile index (XBLOCK)
    pid_hw = tl.program_id(1)   # spatial tile index (YBLOCK)

    nc_start = pid_nc * XBLOCK
    hw_start = pid_hw * YBLOCK

    # nc_idx varies along XBLOCK (fast axis for loads, c stride=1 in input)
    nc_idx = tl.arange(0, XBLOCK)   # [XBLOCK]
    # hw_idx varies along YBLOCK (fast axis for stores, hw stride=1 in output)
    hw_idx = tl.arange(0, YBLOCK)   # [YBLOCK]

    nc = nc_start + nc_idx   # flat N*C index [XBLOCK]
    hw = hw_start + hw_idx   # flat HW index  [YBLOCK]

    n = nc // 128   # batch index [XBLOCK]
    c = nc % 128    # channel index [XBLOCK]

    nc_mask = nc < ynumel   # [XBLOCK]
    hw_mask = hw < xnumel   # [YBLOCK]

    # Load mask shape [YBLOCK, XBLOCK]
    mask_load = hw_mask[:, None] & nc_mask[None, :]

    # COALESCED LOAD: c varies on fast axis [None,:] with stride=1 in input
    # Input stride: (4032000, 1, 32000, 128) -> flat = n*4032000 + hw*128 + c*1
    load_ptrs = (in_ptr0
                 + n[None, :] * 4032000    # [1, XBLOCK]
                 + hw[:, None] * 128       # [YBLOCK, 1] — stride 128, slow axis OK
                 + c[None, :] * 1)         # [1, XBLOCK] — stride 1, fast axis -> coalesced
    tmp = tl.load(load_ptrs, mask=mask_load, other=0.0)  # [YBLOCK, XBLOCK]

    # TRANSPOSE register tile: [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK]
    tmp_t = tl.trans(tmp)  # [XBLOCK, YBLOCK]

    # Store mask shape [XBLOCK, YBLOCK]
    mask_store = nc_mask[:, None] & hw_mask[None, :]

    # Apply contiguity hints to 1D hw BEFORE broadcast
    # hw_start = pid_hw * YBLOCK is always a multiple of YBLOCK
    # arange(YBLOCK) produces YBLOCK consecutive integers
    hw_c = tl.max_contiguous(tl.multiple_of(hw, YBLOCK), YBLOCK)  # [YBLOCK]

    # Build 2D store pointer with hinted hw, then reshape to 1D
    # Output stride: (4032000, 31500, 250, 1) -> flat = n*4032000 + c*31500 + hw*1
    store_ptrs_2d = (out_ptr0
                     + n[:, None] * 4032000   # [XBLOCK, 1]
                     + c[:, None] * 31500     # [XBLOCK, 1]
                     + hw_c[None, :] * 1)     # [1, YBLOCK] — hinted stride-1
    store_ptrs_1d = tl.reshape(store_ptrs_2d, [XBLOCK * YBLOCK])  # [XBLOCK*YBLOCK]
    tmp_1d        = tl.reshape(tmp_t,          [XBLOCK * YBLOCK])
    mask_1d       = tl.reshape(mask_store,     [XBLOCK * YBLOCK])
    tl.store(store_ptrs_1d, tmp_1d, mask=mask_1d)

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
