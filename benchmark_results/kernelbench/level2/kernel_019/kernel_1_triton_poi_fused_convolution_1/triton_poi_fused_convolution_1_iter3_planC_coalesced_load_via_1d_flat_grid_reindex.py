import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Tensor shape: (N=32, C=128, H=126, W=250)
    # Input  strides: (4032000, 1, 32000, 128)  <- C has stride 1 => coalesced load
    # Output strides: (4032000, 31500, 250, 1)  <- W has stride 1 => strided store
    # Grid: pid0 = spatial index in [0, N*H*W=1008000), pid1 = channel tile in [0, 1)
    pid0 = tl.program_id(0)   # spatial index: n*H*W + h*W + w
    pid1 = tl.program_id(1)   # channel tile index

    # Decode spatial index into (n, h, w)
    # W=250, H=126, N=32
    w_idx = pid0 % 250
    tmp   = pid0 // 250
    h_idx = tmp % 126
    n_idx = tmp // 126

    # Channel lane - warp-varying, stride 1 in input
    c_base = pid1 * XBLOCK
    c_lane = tl.arange(0, XBLOCK)
    c_idx  = c_base + c_lane
    cmask  = c_idx < 128

    # Load: input stride (4032000, 1, 32000, 128)
    # c_idx varies with stride 1 => consecutive threads => coalesced
    in_addr = (n_idx * 4032000
               + c_idx * 1
               + h_idx * 32000
               + w_idx * 128)
    tmp0 = tl.load(in_ptr0 + in_addr, mask=cmask)

    # Store: output stride (4032000, 31500, 250, 1)
    # c_idx varies with stride 31500 => strided store (unavoidable given layout)
    out_addr = (n_idx * 4032000
                + c_idx * 31500
                + h_idx * 250
                + w_idx * 1)
    tl.store(out_ptr0 + out_addr, tmp0, mask=cmask)

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
