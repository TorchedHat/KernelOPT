import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Repurpose: XBLOCK=CBLOCK (C tile, innermost), YBLOCK=HWBLOCK (HW tile, outer)
    # 3D grid: program_id(0)=C_tile, program_id(1)=HW_tile, program_id(2)=N
    # Constants for this specific kernel
    C  = 128
    HW = 31500
    N_stride = 4032000

    c_offset  = tl.program_id(0) * XBLOCK   # C tile offset
    hw_offset = tl.program_id(1) * YBLOCK   # HW tile offset
    n         = tl.program_id(2)             # batch index (scalar)

    # C is innermost (cols) for coalesced loads from channels-last input (stride=1 in C)
    c_range  = c_offset  + tl.arange(0, XBLOCK)   # [XBLOCK=CBLOCK]
    hw_range = hw_offset + tl.arange(0, YBLOCK)   # [YBLOCK=HWBLOCK]

    c_mask  = c_range  < C    # [XBLOCK]
    hw_mask = hw_range < HW   # [YBLOCK]

    # 2D tile: [YBLOCK, XBLOCK] — C (XBLOCK) is innermost (cols) for coalesced input loads
    hw2d = hw_range[:, None]   # [YBLOCK, 1]
    c2d  = c_range [None, :]   # [1, XBLOCK]
    mask2d = hw_mask[:, None] & c_mask[None, :]

    # Input: channels-last strides (N=4032000, C=1, H=32000, W=128)
    # For flat HW index hw=h*250+w: h*32000 + w*128 = 128*(h*250+w) = 128*hw
    # load_idx = n*4032000 + c*1 + hw*128  (C stride=1 -> coalesced!)
    load_idx = n * N_stride + c2d + hw2d * C

    # Output: contiguous NCHW strides (N=4032000, C=31500, H=250, W=1)
    # store_idx = n*4032000 + c*31500 + hw*1
    store_idx = n * N_stride + c2d * HW + hw2d

    tmp = tl.load(in_ptr0 + load_idx, mask=mask2d)
    tl.store(out_ptr0 + store_idx, tmp, mask=mask2d)

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
