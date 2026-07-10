import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiled decomposition: pid_x covers spatial (1024 elements), pid_y covers batch*channel (8192 elements)
    # YBLOCK is fixed at 8 to enable L2 reuse across batch*channel slices
    YBLOCK: tl.constexpr = 8
    pid_x = tl.program_id(0)   # spatial tile
    pid_y = tl.program_id(1)   # batch*channel tile
    xoffset = pid_x * XBLOCK
    xidx = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK]
    yoffset = pid_y * YBLOCK
    yidx = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    mask = (xidx < 1024) & (yidx < 8192)  # [YBLOCK, XBLOCK]
    x0 = xidx % 32    # output width  [1, XBLOCK]
    x1 = xidx // 32   # output height [1, XBLOCK]
    x2 = yidx         # batch*channel [YBLOCK, 1]
    x3 = x2 * 1024 + xidx   # flat output index [YBLOCK, XBLOCK]
    base = 2*x0 + 128*x1 + 8192*x2  # input base [YBLOCK, XBLOCK]
    tmp0  = tl.load(in_ptr0 + base,        mask=mask, eviction_policy='evict_last')
    tmp1  = tl.load(in_ptr0 + base + 1,    mask=mask, eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr0 + base + 64,   mask=mask, eviction_policy='evict_last')
    tmp5  = tl.load(in_ptr0 + base + 65,   mask=mask, eviction_policy='evict_last')
    tmp7  = tl.load(in_ptr0 + base + 4096, mask=mask, eviction_policy='evict_last')
    tmp9  = tl.load(in_ptr0 + base + 4097, mask=mask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + base + 4160, mask=mask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + base + 4161, mask=mask, eviction_policy='evict_last')
    tmp2  = tl.maximum(tmp0,  tmp1)
    tmp4  = tl.maximum(tmp2,  tmp3)
    tmp6  = tl.maximum(tmp4,  tmp5)
    tmp8  = tl.maximum(tmp6,  tmp7)
    tmp10 = tl.maximum(tmp8,  tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + x3, tmp14, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1[grid](arg_0, arg_1, 8388608)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 32, 32, 64, 64), (4194304, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
