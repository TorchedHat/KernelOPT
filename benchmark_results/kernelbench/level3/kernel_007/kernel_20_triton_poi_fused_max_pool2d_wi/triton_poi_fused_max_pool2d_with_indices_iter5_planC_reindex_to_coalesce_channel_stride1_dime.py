import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_20(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D grid: pid(0) = spatial position (N*H*W), pid(1) = channel tile
    pid_spatial = tl.program_id(0)   # covers N*H*W = 7840 positions
    pid_c       = tl.program_id(1)   # covers channel tiles

    hw = pid_spatial % (28 * 28)
    n  = pid_spatial // (28 * 28)
    h  = hw // 28
    w  = hw % 28

    c_offset = pid_c * XBLOCK + tl.arange(0, XBLOCK)
    cmask = c_offset < xnumel

    # base: n*200704 + h*7168 + w*256 + c_offset  (c_offset is stride-1)
    base = (n * 200704 + h * 7168 + w * 256).to(tl.int64) + c_offset.to(tl.int64)

    # Scalar boundary checks for 3x3 pooling window (padding=1, stride=1)
    valid_h0 = (h >= 1)       # top row valid (h-1 >= 0)
    valid_h2 = (h <= 26)      # bottom row valid (h+1 < 28)
    valid_w0 = (w >= 1)       # left col valid (w-1 >= 0)
    valid_w2 = (w <= 26)      # right col valid (w+1 < 28)

    # All 9 loads access same channel cache lines (base +- spatial offsets)
    v00 = tl.load(in_ptr0 + base - 7168 - 256, mask=cmask & (valid_h0 & valid_w0), other=float("-inf"))
    v01 = tl.load(in_ptr0 + base - 7168,       mask=cmask & valid_h0,               other=float("-inf"))
    v02 = tl.load(in_ptr0 + base - 7168 + 256, mask=cmask & (valid_h0 & valid_w2), other=float("-inf"))
    v10 = tl.load(in_ptr0 + base        - 256, mask=cmask & valid_w0,               other=float("-inf"))
    v11 = tl.load(in_ptr0 + base,              mask=cmask,                           other=float("-inf"))
    v12 = tl.load(in_ptr0 + base        + 256, mask=cmask & valid_w2,               other=float("-inf"))
    v20 = tl.load(in_ptr0 + base + 7168 - 256, mask=cmask & (valid_h2 & valid_w0), other=float("-inf"))
    v21 = tl.load(in_ptr0 + base + 7168,       mask=cmask & valid_h2,               other=float("-inf"))
    v22 = tl.load(in_ptr0 + base + 7168 + 256, mask=cmask & (valid_h2 & valid_w2), other=float("-inf"))

    result = tl.maximum(
        tl.maximum(tl.maximum(v00, v01), tl.maximum(v02, v10)),
        tl.maximum(tl.maximum(v11, v12), tl.maximum(v20, tl.maximum(v21, v22)))
    )
    tl.store(out_ptr0 + base, result, mask=cmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 2007040
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_20[grid](arg_0, arg_1, 2007040)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
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
