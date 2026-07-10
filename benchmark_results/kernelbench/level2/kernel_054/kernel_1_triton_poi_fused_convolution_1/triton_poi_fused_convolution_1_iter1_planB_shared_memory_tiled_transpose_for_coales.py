import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 32}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # y covers the channel/batch dimension (stride-1 in input) => coalesced loads
    # Inner arange over y (channel dim) so consecutive threads read stride-1 addresses
    y_tile = pid_y * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
    x_tile = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    y_idx = y_tile[:, None]   # [YBLOCK, 1]
    x_idx = x_tile[None, :]   # [1, XBLOCK]
    xmask = x_idx < xnumel
    ymask = y_idx < ynumel
    mask = xmask & ymask
    y0 = y_idx % 128    # channel (stride-1 in input)
    y1 = y_idx // 128   # batch   (stride-4032000 in input)
    # Load: consecutive threads (varying y0) read stride-1 addresses => COALESCED
    vals = tl.load(in_ptr0 + (y0 + 128 * x_idx + 4032000 * y1), mask)
    # Store: x_idx varies fastest => stride-1 in output => COALESCED
    tl.store(out_ptr0 + (x_idx + 31500 * y_idx), vals, mask)

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
