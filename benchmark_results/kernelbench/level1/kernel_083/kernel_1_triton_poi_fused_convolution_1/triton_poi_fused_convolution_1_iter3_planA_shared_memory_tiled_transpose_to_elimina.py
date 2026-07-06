import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 3D grid: pid(0)=x tiles, pid(1)=y0 tiles (C=128, stride-1 in input), pid(2)=y1 (batch=32)
    pid_x = tl.program_id(0)   # tile index along xnumel=31500 (stride-128 in input, slow)
    pid_y = tl.program_id(1)   # tile index along y0=C=128 (stride-1 in input, fast)
    pid_b = tl.program_id(2)   # index along y1=batch_outer=32 (scalar per CTA)
    # x_range covers xnumel dimension (stride-128 in input)
    x_range  = pid_x * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK]
    # y0_range covers C=128 dimension (stride-1 in input) — fast axis for coalescing
    y0_range = pid_y * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK], stride-1 in input
    xmask  = x_range  < xnumel
    y0mask = y0_range < 128
    y1 = pid_b
    y0_2d = y0_range[:, None]   # [YBLOCK, 1]
    x_2d  = x_range[None, :]   # [1, XBLOCK]
    mask_2d = y0mask[:, None] & xmask[None, :]
    # COALESCED: y0 (stride-1) is the row axis, consecutive threads read consecutive y0 values
    load_idx  = y0_2d + 128 * x_2d + 4032000 * y1
    tile_data = tl.load(in_ptr0 + load_idx, mask=mask_2d, other=0.0)
    # Store: x (stride-1 in output) is the col axis
    y3_2d     = y0_2d + 128 * y1
    store_idx = x_2d + 31500 * y3_2d
    tl.store(out_ptr0 + store_idx, tile_data, mask=mask_2d)

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
