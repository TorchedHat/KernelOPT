import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_x = tl.program_id(0)  # spatial tile index
    pid_y = tl.program_id(1)  # channel/batch tile index
    y_base = pid_y * YBLOCK
    x_base = pid_x * XBLOCK
    ty = tl.arange(0, YBLOCK)  # fast dim for input coalescing (stride-1 in input)
    tx = tl.arange(0, XBLOCK)  # fast dim for output coalescing (stride-1 in output)
    y_idx = y_base + ty[:, None]   # shape [YBLOCK, 1]
    x_idx = x_base + tx[None, :]   # shape [1, XBLOCK]
    y0 = y_idx % 128   # channel within batch item (stride-1 in input)
    y1 = y_idx // 128  # batch index
    x_mask = x_idx < xnumel
    y_mask = y_idx < ynumel
    mask = y_mask & x_mask
    # COALESCED LOAD: adjacent ty values -> adjacent y0 -> stride-1 in input
    in_ptrs = in_ptr0 + (y0 + 128 * x_idx + 4032000 * y1)
    tile = tl.load(in_ptrs, mask=mask, other=0.0)
    # COALESCED STORE: adjacent tx values -> adjacent x_idx -> stride-1 in output
    out_ptrs = out_ptr0 + (x_idx + 31500 * y_idx)
    tl.store(out_ptrs, tile, mask=mask)

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
