import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # pid(0) -> x2 (spatial tiles), pid(1) -> y1 (batch index, 32 total)
    x2_offset = tl.program_id(0) * YBLOCK
    x2_range  = x2_offset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    x2_mask   = x2_range < xnumel
    y1        = tl.program_id(1)                             # scalar batch index [0..31]
    # Apply contiguity hints on 1D arange before reshaping to 2D
    y0_1d     = tl.arange(0, XBLOCK)                        # [XBLOCK]
    y0_1d     = tl.max_contiguous(tl.multiple_of(y0_1d, 128), 128)
    y0_range  = y0_1d[None, :]                               # [1, XBLOCK] channel
    # Load from channels-last src: stride = (4032000, 1, 32000, 128)
    load_idx  = y0_range + 128 * x2_range + 4032000 * y1    # [YBLOCK, XBLOCK]
    tmp0      = tl.load(in_ptr0 + load_idx, x2_mask)
    # Store to contiguous dst: stride = (4032000, 31500, 250, 1)
    y3        = y1 * 128 + y0_range                          # [1, XBLOCK]
    store_idx = x2_range + 31500 * y3                        # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_idx, tmp0, x2_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
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
