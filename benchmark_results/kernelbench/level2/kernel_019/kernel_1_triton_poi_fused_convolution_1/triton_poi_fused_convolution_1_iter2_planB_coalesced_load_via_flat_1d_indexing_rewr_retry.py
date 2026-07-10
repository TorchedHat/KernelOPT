import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Rewritten for coalesced loads:
    # dim-0 (program_id(0)) tiles over C=128 channels (stride-1 in input)
    # dim-1 (program_id(1)) tiles over outer=n*h*w=1,008,000 elements
    # c_idx varies across warp threads -> stride-1 loads are coalesced
    c_offset = tl.program_id(0) * XBLOCK
    ow_offset = tl.program_id(1) * YBLOCK

    # c_idx: shape [1, XBLOCK] — varies across warp (stride-1 in input)
    c_idx = c_offset + tl.arange(0, XBLOCK)[None, :]
    # ow_idx: shape [YBLOCK, 1] — varies across blocks
    ow_idx = ow_offset + tl.arange(0, YBLOCK)[:, None]

    c_mask = c_idx < 128
    ow_mask = ow_idx < 1008000
    mask = c_mask & ow_mask

    # Decompose ow_idx into (n, h, w)
    # ow_idx = n * 31500 + h * 250 + w
    w_idx = ow_idx % 250
    h_idx = (ow_idx // 250) % 126
    n_idx = ow_idx // 31500

    # Input: channels-last strides (N=4032000, C=1, H=32000, W=128)
    # in[n, c, h, w] = in_ptr0[n*4032000 + c*1 + h*32000 + w*128]
    in_idx = n_idx * 4032000 + c_idx * 1 + h_idx * 32000 + w_idx * 128
    tmp0 = tl.load(in_ptr0 + in_idx, mask=mask)

    # Output: NCHW contiguous strides (N=4032000, C=31500, H=250, W=1)
    # out[n, c, h, w] = out_ptr0[n*4032000 + c*31500 + h*250 + w*1]
    out_idx = n_idx * 4032000 + c_idx * 31500 + h_idx * 250 + w_idx * 1
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
