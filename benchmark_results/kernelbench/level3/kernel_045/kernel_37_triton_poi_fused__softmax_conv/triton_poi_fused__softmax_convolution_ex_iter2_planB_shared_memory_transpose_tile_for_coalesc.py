import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_37(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 32768
    # y1 is a scalar batch index (0..7), one per program_id(1)
    y1 = tl.program_id(1).to(tl.int64)
    # x tile: XBLOCK consecutive x2 values
    xoffset = tl.program_id(0) * XBLOCK
    xindex = (xoffset + tl.arange(0, XBLOCK)).to(tl.int64)  # [XBLOCK], int64
    xmask = xindex < xnumel
    # y0 tile: all 64 channels (YBLOCK=64 covers exactly 0..63)
    y0index = tl.arange(0, YBLOCK).to(tl.int64)  # [YBLOCK], int64, values 0..63
    ymask = y0index < 64  # always True when YBLOCK=64, but needed for correctness

    # Phase 1: COALESCED LOAD along y0 (stride-1 in in_ptr0)
    # load_idx shape: [YBLOCK, XBLOCK]
    # For fixed x2, consecutive y0 values are stride-1 in in_ptr0 -> coalesced!
    load_idx = y0index[:, None] + 64 * xindex[None, :] + 2097152 * y1
    data = tl.load(in_ptr0 + load_idx, mask=ymask[:, None] & xmask[None, :], other=0.0)

    # Bias: in_ptr1[y0], shape [YBLOCK] -> broadcast over x
    bias = tl.load(in_ptr1 + y0index, mask=ymask)  # [YBLOCK]
    data = data + bias[:, None]  # [YBLOCK, XBLOCK]

    # Phase 2: COALESCED STORE along x2 (stride-1 in out_ptr0)
    # For fixed y0, consecutive x2 values are stride-1 in out_ptr0 -> coalesced!
    store_idx = xindex[None, :] + 32768 * y0index[:, None] + 4194304 * y1
    tl.store(out_ptr0 + store_idx, data, mask=ymask[:, None] & xmask[None, :])

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 512
    xnumel = 32768
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_37[grid](arg_0, arg_1, arg_2, 512, 32768, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 64, 64, 512), (2097152, 1, 32768, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 64, 64, 512), (4194304, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
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
