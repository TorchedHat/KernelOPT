import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # XBLOCK covers channels C (0..95, stride-1 in NHWC input) — coalesced load
    # YBLOCK covers spatial HW positions (0..3024, stride-96 in NHWC input)
    # Grid: (cdiv(3025, YBLOCK), cdiv(96, XBLOCK), 256)

    # Spatial (HW) dimension — program_id(0)
    hw_offset = tl.program_id(0) * YBLOCK
    hw_idx = hw_offset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    hw_mask = hw_idx < 3025

    # Channel dimension — program_id(1)
    c_offset = tl.program_id(1) * XBLOCK
    c_idx = c_offset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK]
    c_mask = c_idx < 96

    # Batch dimension — program_id(2)
    n_idx = tl.program_id(2)

    # Load from NHWC input: addr = n*290400 + hw*96 + c*1
    # Shape: [YBLOCK, XBLOCK] — threads vary in c (stride-1) -> COALESCED
    tmp0 = tl.load(in_ptr0 + (n_idx * 290400 + hw_idx * 96 + c_idx),
                   mask=hw_mask & c_mask, other=0.0)

    # Load bias: shape (96,) indexed by c_idx [1, XBLOCK]
    tmp1 = tl.load(in_ptr1 + c_idx, mask=c_mask, other=0.0)

    # Add bias (broadcast across YBLOCK spatial positions)
    tmp2 = tmp0 + tmp1  # [YBLOCK, XBLOCK]

    # Transpose to [XBLOCK, YBLOCK] for NCHW store
    tmp2_t = tl.trans(tmp2)  # [XBLOCK, YBLOCK]

    # Store to NCHW output: addr = n*290400 + c*3025 + hw*1
    # c_s varies in dim0 [XBLOCK, 1], hw_s varies in dim1 [1, YBLOCK]
    c_s = c_offset + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    hw_s = hw_offset + tl.arange(0, YBLOCK)[None, :]  # [1, YBLOCK]
    tl.store(out_ptr0 + (n_idx * 290400 + c_s * 3025 + hw_s),
             tmp2_t,
             mask=(c_s < 96) & (hw_s < 3025))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
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
