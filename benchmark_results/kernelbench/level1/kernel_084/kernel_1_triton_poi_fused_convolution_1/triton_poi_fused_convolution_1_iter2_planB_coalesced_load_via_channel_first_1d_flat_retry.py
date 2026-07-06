import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Tensor: (N=32, C=128, H=126, W=250)
    # Input  strides: (4032000, 1, 32000, 128)  <- NHWC, C is stride-1
    # Output strides: (4032000, 31500, 250, 1)  <- contiguous NCHW
    C = 128
    HW = 31500   # H*W = 126*250
    W = 250
    NHW = 1008000  # N*H*W = 32*126*250

    pid_x = tl.program_id(0)   # channel tile (fast axis = stride-1 in input)
    pid_y = tl.program_id(1)   # N*HW spatial tile

    x_off = pid_x * XBLOCK
    x_idx = x_off + tl.arange(0, XBLOCK)   # channel indices, shape [XBLOCK]
    x_mask = x_idx < C                      # always True when XBLOCK=128=C

    y_off = pid_y * YBLOCK
    y_idx = y_off + tl.arange(0, YBLOCK)   # N*HW flat indices, shape [YBLOCK]
    y_mask = y_idx < NHW

    batch   = y_idx // HW          # shape [YBLOCK]
    hw_pos  = y_idx % HW           # shape [YBLOCK]
    hw_row  = hw_pos // W          # shape [YBLOCK]
    hw_col  = hw_pos % W           # shape [YBLOCK]

    # 2D mask: [YBLOCK, XBLOCK]
    mask_2d = y_mask[:, None] & x_mask[None, :]

    # Load: input[batch, hw_row, hw_col, chan] with strides (4032000, 32000, 128, 1)
    # x_idx (chan) is the fast axis -> consecutive threads load consecutive addresses -> COALESCED
    load_addr = (batch[:, None] * 4032000 +
                 hw_row[:, None] * 32000 +
                 hw_col[:, None] * 128 +
                 x_idx[None, :])            # [YBLOCK, XBLOCK]

    tmp = tl.load(in_ptr0 + load_addr, mask=mask_2d)

    # Store: output[batch, chan, hw_row, hw_col] with strides (4032000, 31500, 250, 1)
    store_addr = (batch[:, None] * 4032000 +
                  x_idx[None, :] * 31500 +
                  hw_row[:, None] * 250 +
                  hw_col[:, None] * 1)      # [YBLOCK, XBLOCK]

    tl.store(out_ptr0 + store_addr, tmp, mask=mask_2d)

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
