import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)  # tiles along xnumel=31500 (spatial)
    pid_y = tl.program_id(1)  # tiles along ynumel=4096  (batch*channel)
    tx = tl.arange(0, XBLOCK)             # [XBLOCK]
    ty = tl.arange(0, YBLOCK)             # [YBLOCK]
    x2 = (pid_x * XBLOCK + tx)[None, :]  # [1, XBLOCK]
    y3 = (pid_y * YBLOCK + ty)[:, None]  # [YBLOCK, 1]
    y0 = y3 % 128   # channel index within batch
    y1 = y3 // 128  # batch index
    xmask = x2 < xnumel
    ymask = y3 < ynumel
    mask = xmask & ymask
    # Load: input has strides (4032000, 1, 32000, 128) — channels-last layout
    vals = tl.load(in_ptr0 + (y0 + 128 * x2 + 4032000 * y1), mask=mask, other=0.0)
    # Store: output has strides (4032000, 31500, 250, 1) — contiguous NCHW layout
    tl.store(out_ptr0 + (x2 + 31500 * y3), vals, mask=mask)

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
