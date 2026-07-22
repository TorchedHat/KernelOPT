import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1},  num_warps=4),
    triton.Config({'XBLOCK': 4},  num_warps=4),
    triton.Config({'XBLOCK': 8},  num_warps=4),
    triton.Config({'XBLOCK': 16}, num_warps=4),
    triton.Config({'XBLOCK': 32}, num_warps=8),
    triton.Config({'XBLOCK': 64}, num_warps=8),
], key=['xnumel'], reset_to_zero=['out_ptr0'])
@triton.jit
def triton_poi_fused_cat_convolution_26(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Reinterpret: XBLOCK = rows per CTA (YBLOCK in 2D sense)
    # Channel dim is always 128 (full row), tiled by pid_x
    # Row dim (0..7839) is tiled by pid_y with XBLOCK rows per CTA
    CHAN: tl.constexpr = 128
    pid_chan = tl.program_id(0)   # always 0 since cdiv(128,128)=1
    pid_row  = tl.program_id(1)   # row tile index

    chan_off = pid_chan * CHAN + tl.arange(0, CHAN)   # [CHAN]
    row_off  = pid_row  * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]

    chan_idx = chan_off[None, :]   # [1, CHAN]
    row_idx  = row_off[:, None]   # [XBLOCK, 1]

    chan_mask = chan_idx < 128     # [1, CHAN]  -- always True since CHAN=128
    row_mask  = row_idx < 7840    # [XBLOCK, 1]
    mask = chan_mask & row_mask    # [XBLOCK, CHAN]

    in_idx  = row_idx * 128 + chan_idx   # contiguous input (7840,128)
    out_idx = chan_idx + 480 * row_idx   # strided output: stride-1 in chan, stride-480 in row

    tmp0 = tl.load(in_ptr0 + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr0 + out_idx, tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1003520
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_26[grid](arg_0, arg_1, 1003520)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((7840, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 128, 28, 28), (376320, 1, 13440, 480), device='cuda:0', dtype=torch.float32)
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
