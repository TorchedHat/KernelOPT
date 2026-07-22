import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64},  num_warps=4),
    triton.Config({'XBLOCK': 128}, num_warps=4),
    triton.Config({'XBLOCK': 128}, num_warps=8),
    triton.Config({'XBLOCK': 256}, num_warps=4),
    triton.Config({'XBLOCK': 256}, num_warps=8),
], key=['xnumel'])
@triton.jit
def triton_poi_fused_convolution_39(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: axis 0 = channel tiles (XBLOCK wide), axis 1 = batch*spatial tiles (YBLOCK=16 fixed)
    YBLOCK: tl.constexpr = 16
    C: tl.constexpr = 208
    N: tl.constexpr = 1960
    IN_STRIDE_N: tl.constexpr = 208
    OUT_STRIDE_N: tl.constexpr = 512
    pid_x = tl.program_id(0)  # channel tiles
    pid_y = tl.program_id(1)  # batch*spatial tiles
    x0 = pid_x * XBLOCK + tl.arange(0, XBLOCK)   # channel indices [XBLOCK]
    y0 = pid_y * YBLOCK + tl.arange(0, YBLOCK)   # batch*spatial indices [YBLOCK]
    xmask = x0 < C
    ymask = y0 < N
    mask2d = ymask[:, None] & xmask[None, :]
    in_idx  = y0[:, None] * IN_STRIDE_N  + x0[None, :]   # [YBLOCK, XBLOCK]
    out_idx = y0[:, None] * OUT_STRIDE_N + x0[None, :]   # [YBLOCK, XBLOCK]
    tmp0 = tl.load(in_ptr0 + in_idx, mask2d)
    tmp1 = tl.load(in_ptr1 + x0[None, :], xmask[None, :], eviction_policy='evict_last')
    tl.store(out_ptr0 + out_idx, tmp0 + tmp1, mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 407680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_39[grid](arg_0, arg_1, arg_2, 407680)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 208, 14, 14), (40768, 1, 2912, 208), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((208,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 208, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
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
