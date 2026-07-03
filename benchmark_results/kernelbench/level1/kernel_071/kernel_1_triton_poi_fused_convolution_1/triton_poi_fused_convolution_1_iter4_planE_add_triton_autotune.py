import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 8},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024,'YBLOCK': 4},  num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # ynumel = 4096 = N*C = 32*128
    # xnumel = 31500 = H*W = 126*250
    # YBLOCK tiles over N*C (channel dimension, stride=1 in input)
    # XBLOCK tiles over H*W (spatial dimension, stride=1 in output)
    # This makes loads coalesced (c stride=1) and stores coalesced (hw stride=1)

    pid_hw = tl.program_id(0)   # spatial tile index
    pid_nc = tl.program_id(1)   # N*C tile index

    hw_start = pid_hw * XBLOCK
    nc_start = pid_nc * YBLOCK

    # c_idx varies along YBLOCK (fast axis for loads, stride=1 in input)
    nc_idx = tl.arange(0, YBLOCK)   # [YBLOCK]
    # hw_idx varies along XBLOCK (fast axis for stores, stride=1 in output)
    hw_idx = tl.arange(0, XBLOCK)   # [XBLOCK]

    nc = nc_start + nc_idx   # flat N*C index [YBLOCK]
    hw = hw_start + hw_idx   # flat HW index  [XBLOCK]

    n = nc // 128   # batch index [YBLOCK]
    c = nc % 128    # channel index [YBLOCK]

    # Apply contiguity hints to enable 128-bit vectorized loads/stores
    # c is contiguous within each tile (consecutive integers from nc_start % 128)
    c_contig = tl.max_contiguous(tl.multiple_of(c, 1), 1)
    # hw is contiguous within each tile (consecutive integers from hw_start)
    hw_contig = tl.max_contiguous(tl.multiple_of(hw, 1), 1)

    nc_mask = nc < ynumel
    hw_mask = hw < xnumel
    mask2d  = nc_mask[:, None] & hw_mask[None, :]

    # COALESCED LOAD: c varies with stride=1 in input (channels-last layout)
    # Input stride: (4032000, 1, 32000, 128) -> flat = n*4032000 + hw*128 + c
    load_ptrs = (in_ptr0
                 + n[:, None]      * 4032000
                 + hw_contig[None, :] * 128
                 + c_contig[:, None]  * 1)
    tmp = tl.load(load_ptrs, mask=mask2d, other=0.0)

    # COALESCED STORE: hw varies with stride=1 in output (contiguous layout)
    # Output stride: (4032000, 31500, 250, 1) -> flat = n*4032000 + c*31500 + hw
    store_ptrs = (out_ptr0
                  + n[:, None]         * 4032000
                  + c_contig[:, None]  * 31500
                  + hw_contig[None, :] * 1)
    tl.store(store_ptrs, tmp, mask=mask2d)

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
