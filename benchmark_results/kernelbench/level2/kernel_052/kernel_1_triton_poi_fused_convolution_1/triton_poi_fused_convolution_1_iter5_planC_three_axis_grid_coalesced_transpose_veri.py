import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Tensor: (B=32, C=128, H=126, W=250)
    # Input strides:  (4032000, 1, 32000, 128)  -- C is stride-1
    # Output strides: (4032000, 31500, 250, 1)  -- W is stride-1
    # XBLOCK = spatial tile size (32), YBLOCK = channel tile size (128)
    HW = 31500
    pid_x = tl.program_id(0)   # spatial tile
    pid_c = tl.program_id(1)   # channel tile
    pid_b = tl.program_id(2)   # batch index

    # Spatial offsets -- stride-1 in OUTPUT (W dimension)
    x_off = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    x_mask = x_off < HW
    h = x_off // 250   # [XBLOCK]
    w = x_off % 250    # [XBLOCK]

    # Channel offsets -- stride-1 in INPUT (exact fit: YBLOCK=C=128, no mask needed)
    c_off = pid_c * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]

    # Input index: b*4032000 + c*1 + h*32000 + w*128  => shape [XBLOCK, YBLOCK]
    in_idx = (pid_b * 4032000
              + c_off[None, :]        # [1, YBLOCK] stride-1 -> coalesced load
              + h[:, None] * 32000    # [XBLOCK, 1]
              + w[:, None] * 128)     # [XBLOCK, 1]

    # Output index: b*4032000 + c*31500 + h*250 + w*1  => shape [XBLOCK, YBLOCK]
    out_idx = (pid_b * 4032000
               + c_off[None, :] * 31500  # [1, YBLOCK]
               + h[:, None] * 250        # [XBLOCK, 1]
               + w[:, None])             # [XBLOCK, 1] stride-1 -> coalesced store

    mask = x_mask[:, None]  # [XBLOCK, 1] broadcast over YBLOCK
    tmp0 = tl.load(in_ptr0 + in_idx, mask=mask)
    tl.store(out_ptr0 + out_idx, tmp0, mask=mask)

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
