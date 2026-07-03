import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK=32 iterates over channels (C=96, stride=1 in input) -> COALESCED LOAD
    # YBLOCK=32 iterates over spatial positions (HW=3025, stride=96 in input)
    # After tl.trans(): XBLOCK over channels (stride=3025 in output), YBLOCK over spatial (stride=1) -> COALESCED STORE
    # spatial_tiles = cdiv(3025, 32) = 95, hardcoded for correct pid decomposition
    spatial_tiles = tl.cdiv(3025, YBLOCK)

    pid_c = tl.program_id(0)   # channel tile index (axis 0)
    pid_ns = tl.program_id(1)  # encodes (batch_idx * spatial_tiles + spatial_tile_idx)
    pid_n = pid_ns // spatial_tiles   # batch index
    pid_s = pid_ns % spatial_tiles    # spatial tile index

    c_off = pid_c * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK] channel offsets
    s_off = pid_s * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK] spatial offsets
    c_mask = c_off < 96    # [XBLOCK]
    s_mask = s_off < 3025  # [YBLOCK]

    # Phase 1: COALESCED LOAD — inner dim=channels (stride=1 in NHWC input)
    # in_ptr0 NHWC strides: N=290400, HW=96 (flat spatial * C), C=1
    # load_idx shape [YBLOCK, XBLOCK]: rows=spatial, cols=channels
    load_idx = pid_n * 290400 + s_off[:, None] * 96 + c_off[None, :]
    load_mask = s_mask[:, None] & c_mask[None, :]
    tile = tl.load(in_ptr0 + load_idx, load_mask, other=0.0)  # [YBLOCK, XBLOCK]

    # Add bias: in_ptr1[c], shape [XBLOCK]
    bias = tl.load(in_ptr1 + c_off, c_mask, other=0.0)  # [XBLOCK]
    tile = tile + bias[None, :]  # broadcast: [YBLOCK, XBLOCK]

    # Phase 2: COALESCED STORE — inner dim=spatial (stride=1 in NCHW output)
    # out_ptr0 NCHW strides: N=290400, C=3025, HW=1
    # store_idx shape [XBLOCK, YBLOCK]: rows=channels, cols=spatial
    store_idx = pid_n * 290400 + c_off[:, None] * 3025 + s_off[None, :]
    store_mask = c_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr0 + store_idx, tl.trans(tile), store_mask)  # tl.trans: [YBLOCK,XBLOCK]->[XBLOCK,YBLOCK]

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
