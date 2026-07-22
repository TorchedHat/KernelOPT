import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_transpose_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: axis 0 encodes (x3, x1) pairs — 128 * 8 = 1024 values
    #          axis 1 encodes x2 — 512 values
    # Each CTA processes one full row of x0 in [0, 96) for a fixed (x3, x1, x2)
    # XBLOCK is unused (kept for signature compatibility); tile size is 128

    pid0 = tl.program_id(0)   # in [0, 1024)
    pid1 = tl.program_id(1)   # in [0, 512)

    x3 = pid0 // 8            # in [0, 128)
    x1 = pid0 % 8             # in [0, 8)
    x2 = pid1                 # in [0, 512)

    x0 = tl.arange(0, 128)    # [0, 1, ..., 127]
    mask = x0 < 96            # only first 96 lanes are valid

    # Load: in_ptr0 shape (1024, 512, 96), strides (49152, 96, 1)
    # First dim index = x3*8 + x1, offset = x0 + 96*x2 + 49152*x1 + 393216*x3
    # x0 is stride-1 => fully coalesced load
    load_offset = x0 + 96 * x2 + 49152 * x1 + 393216 * x3
    tmp0 = tl.load(in_ptr0 + load_offset, mask=mask)

    # Store: out_ptr0 shape (128, 512, 8, 96), strides (393216, 768, 96, 1)
    # Element (x3, x2, x1, x0): offset = x0 + 96*x1 + 768*x2 + 393216*x3
    store_offset = x0 + 96 * x1 + 768 * x2 + 393216 * x3
    tl.store(out_ptr0 + store_offset, tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_8[grid](arg_0, arg_1, 50331648, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
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
