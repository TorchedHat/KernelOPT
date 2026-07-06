import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Reindex: XBLOCK tiles channel dim (stride=1 in in_ptr0 for coalesced loads)
    # YBLOCK tiles spatial*batch dim
    # This makes loads coalesced (channel is fast dim in NHWC input)
    # Stores to NCHW output: spatial is fast dim, so we accept partial uncoalescing on stores
    # but loads are the primary bottleneck (96.37% L2, 21.6 cycle stalls)

    C = 96
    HW = 3025

    # pid(0) -> channel tiles, pid(1) -> (n, hw) tiles
    c_tile = tl.program_id(0)   # tiles over channel dim
    nhw_tile = tl.program_id(1) # tiles over n*HW

    # Channel indices (fast dim for coalesced NHWC loads)
    c_offs = c_tile * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    c_mask = c_offs < C

    # Spatial*batch indices
    nhw_offs = nhw_tile * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    nhw_mask = nhw_offs < (256 * HW)

    n = nhw_offs // HW
    hw = nhw_offs % HW

    combined_mask = c_mask & nhw_mask

    # NHWC load: stride (290400, 1, 5280, 96) -> index = n*290400 + hw*96 + c
    # hw*96 + c: channel is fast (stride=1), so consecutive threads (varying c) are coalesced
    load_idx = n * 290400 + hw * 96 + c_offs

    tmp0 = tl.load(in_ptr0 + load_idx, combined_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + c_offs, c_mask, other=0.0)
    tmp2 = tmp0 + tmp1

    # NCHW store: stride (290400, 3025, 55, 1) -> index = n*290400 + c*3025 + hw
    # spatial (hw) is fast dim in output, but c varies fastest here -> partially uncoalesced
    # Still better than original where spatial varied fastest for uncoalesced loads
    store_idx = n * 290400 + c_offs * 3025 + hw

    tl.store(out_ptr0 + store_idx, tmp2, combined_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
