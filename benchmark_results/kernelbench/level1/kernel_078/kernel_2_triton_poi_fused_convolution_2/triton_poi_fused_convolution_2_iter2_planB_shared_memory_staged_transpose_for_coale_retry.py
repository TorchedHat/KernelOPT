import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 3025
    # program_id(0) tiles the x (spatial) dimension: 0..cdiv(3025, XBLOCK)-1
    # program_id(1) tiles the y (channel+batch) dimension: 0..cdiv(24576, YBLOCK)-1
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK

    # xi is the slow dimension (XBLOCK rows), yi is the fast dimension (YBLOCK cols)
    # This makes y (channel, stride-1 in in_ptr0) the fast warp lane dimension for coalesced loads
    # and x (spatial, stride-1 in out_ptr0) the slow dimension for coalesced stores
    xi = tl.arange(0, XBLOCK)[:, None]   # shape [XBLOCK, 1]
    yi = tl.arange(0, YBLOCK)[None, :]   # shape [1, YBLOCK]

    x_abs = xoffset + xi   # spatial index, shape [XBLOCK, 1]
    y_abs = yoffset + yi   # channel+batch index, shape [1, YBLOCK]

    xmask = x_abs < xnumel   # shape [XBLOCK, 1]
    ymask = y_abs < ynumel   # shape [1, YBLOCK]
    load_mask = xmask & ymask  # shape [XBLOCK, YBLOCK]

    # Decompose y_abs into channel (y0_l, stride-1 in in_ptr0) and batch (y1_l)
    y0_l = y_abs % 96      # channel index, shape [1, YBLOCK] — stride-1 in in_ptr0 → coalesced load!
    y1_l = y_abs // 96     # batch index, shape [1, YBLOCK]
    x2_l = x_abs           # spatial index, shape [XBLOCK, 1]

    # Load from NHWC-like input: in_ptr0 has strides (290400, 1, 5280, 96)
    # Address: y0_l + 96*x2_l + 290400*y1_l
    # Adjacent threads (varying yi) differ by 1 in y0_l → coalesced!
    tmp0 = tl.load(in_ptr0 + (y0_l + 96 * x2_l + 290400 * y1_l), load_mask, other=0.0)

    # Load bias: in_ptr1[y0_l], shape [1, YBLOCK] — only depends on channel
    tmp1 = tl.load(in_ptr1 + y0_l, ymask, other=0.0)

    tmp2 = tmp0 + tmp1  # shape [XBLOCK, YBLOCK]

    # Store to NCHW output: out_ptr0 has strides (290400, 3025, 55, 1)
    # Address: x_abs + 3025*y_abs = x2 + 3025*y3 (same mapping as original)
    # This covers all (x,y) pairs in [xoffset..xoffset+XBLOCK) x [yoffset..yoffset+YBLOCK)
    tl.store(out_ptr0 + (x_abs + 3025 * y_abs), tmp2, load_mask)

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
