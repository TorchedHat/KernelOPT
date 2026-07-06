import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK = CBLOCK (channel tile), XBLOCK = HWBLOCK (spatial tile)
    # 3D grid: axis0=hw_tile, axis1=c_tile, axis2=n_idx
    hw_tile = tl.program_id(0)   # spatial tile index
    c_tile  = tl.program_id(1)   # channel tile index
    n_idx   = tl.program_id(2)   # batch index
    C  = 96
    HW = 3025
    c_off  = c_tile  * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK] = [CBLOCK]
    hw_off = hw_tile * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK] = [HWBLOCK]
    c_mask  = c_off  < C
    hw_mask = hw_off < HW
    full_mask = c_mask[:, None] & hw_mask[None, :]  # [YBLOCK, XBLOCK]
    # COALESCED LOAD: inner dim is c_off (stride-1 in NHWC input)
    load_idx = c_off[:, None] + 96 * hw_off[None, :] + 290400 * n_idx
    data = tl.load(in_ptr0 + load_idx, mask=full_mask, other=0.0)
    # ADD BIAS broadcast over HW
    bias = tl.load(in_ptr1 + c_off, mask=c_mask, other=0.0)  # [YBLOCK]
    data = data + bias[:, None]
    # COALESCED STORE: inner dim is hw_off (stride-1 in NCHW output)
    store_idx = hw_off[None, :] + 3025 * (c_off[:, None] + 96 * n_idx)
    tl.store(out_ptr0 + store_idx, data, mask=full_mask)

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
