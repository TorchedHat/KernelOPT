import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Corner-turning transpose: YBLOCK=CBLOCK (channels), XBLOCK=SBLOCK (spatial)
    # Loads are coalesced: channels (stride=1 in NHWC) are the fast warp-lane dim
    # Stores are coalesced: spatial (stride=1 in NCHW) are the fast warp-lane dim
    pid_x = tl.program_id(0)  # spatial tile index
    pid_y = tl.program_id(1)  # flat index over N * num_c_tiles
    num_c_tiles = 96 // YBLOCK  # YBLOCK acts as CBLOCK
    n = pid_y // num_c_tiles
    c_tile = pid_y % num_c_tiles
    c_base = c_tile * YBLOCK
    s_base = pid_x * XBLOCK   # XBLOCK acts as SBLOCK
    c_idx = c_base + tl.arange(0, YBLOCK)[None, :]   # [1, CBLOCK] fast dim for load
    s_idx = s_base + tl.arange(0, XBLOCK)[:, None]   # [SBLOCK, 1] slow dim for load
    s_mask = s_idx < 3025
    load_addr = c_idx + 96 * s_idx + 290400 * n      # [SBLOCK, CBLOCK]
    data = tl.load(in_ptr0 + load_addr, s_mask, other=0.0)  # [SBLOCK, CBLOCK]
    bias = tl.load(in_ptr1 + c_idx, None)             # [1, CBLOCK]
    data = data + bias                                 # [SBLOCK, CBLOCK]
    data_T = tl.trans(data)                            # [CBLOCK, SBLOCK] in-register transpose
    c_out = c_base + tl.arange(0, YBLOCK)[:, None]   # [CBLOCK, 1]
    s_out = s_base + tl.arange(0, XBLOCK)[None, :]   # [1, SBLOCK] fast dim for store
    s_out_mask = s_out < 3025
    store_addr = s_out + 3025 * c_out + 290400 * n    # [CBLOCK, SBLOCK]
    tl.store(out_ptr0 + store_addr, data_T, s_out_mask)

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
