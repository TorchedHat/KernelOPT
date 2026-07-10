import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=3),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(0)
    rows = pid_y * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK]
    cols = pid_x * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK]
    row_mask = rows < ynumel
    col_mask = cols < xnumel
    y0 = rows % 128
    y1 = rows // 128
    load_idx = y0[:, None] + 128 * cols[None, :] + 4032000 * y1[:, None]   # [YBLOCK, XBLOCK]
    load_mask = row_mask[:, None] & col_mask[None, :]
    tile_data = tl.load(in_ptr0 + load_idx, mask=load_mask, other=0.0)
    store_idx = cols[None, :] + 31500 * rows[:, None]   # [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + store_idx, tile_data, mask=load_mask)

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
