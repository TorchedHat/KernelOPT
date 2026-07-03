import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Reinterpret YBLOCK as TILE_HW and XBLOCK as TILE_C for coalesced tiling
    # Input strides: [4032000, 1, 32000, 128] (channels-last-like, C stride=1)
    # Output strides: [4032000, 31500, 250, 1] (NCHW contiguous, HW stride=1)
    C  = 128
    HW = 31500
    pid_hw = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_n  = tl.program_id(2)
    hw_off = pid_hw * YBLOCK + tl.arange(0, YBLOCK)
    c_off  = pid_c  * XBLOCK + tl.arange(0, XBLOCK)
    hw_mask = hw_off < HW
    c_mask  = c_off  < C
    mask    = hw_mask[:, None] & c_mask[None, :]
    n_base  = pid_n * 4032000
    # COALESCED LOAD: innermost dim is c_off (stride=1 in input)
    # Identity: hw_off * 128 = h*32000 + w*128 since hw_off = h*250+w
    load_ptrs  = in_ptr0  + n_base + hw_off[:, None] * 128 + c_off[None, :]
    tile = tl.load(load_ptrs, mask=mask, other=0.0)
    # COALESCED STORE: innermost dim is hw_off (stride=1 in output)
    store_ptrs = out_ptr0 + n_base + c_off[None, :] * 31500 + hw_off[:, None]
    tl.store(store_ptrs, tile, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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
