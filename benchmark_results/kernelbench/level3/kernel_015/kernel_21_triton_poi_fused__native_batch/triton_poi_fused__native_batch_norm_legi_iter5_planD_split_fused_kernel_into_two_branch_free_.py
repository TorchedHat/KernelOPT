import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64}, num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def kernel_avgpool_bn_relu_sub(in_out_ptr0, in_ptr0, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128          # channel index (0-127)
    x1 = (xindex // 128) % 28  # spatial H index
    x2 = xindex // 3584        # batch index
    v0 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + x0), mask=xmask, eviction_policy='evict_last')
    v1 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + x0), mask=xmask, eviction_policy='evict_last')
    v2 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + x0), mask=xmask, eviction_policy='evict_last')
    v3 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + x0), mask=xmask, eviction_policy='evict_last')
    pool = (v0 + v1 + v2 + v3) * 0.25
    mean   = tl.load(in_ptr5 + x0, mask=xmask, eviction_policy='evict_last')
    var    = tl.load(in_ptr6 + x0, mask=xmask, eviction_policy='evict_last')
    weight = tl.load(in_ptr7 + x0, mask=xmask, eviction_policy='evict_last')
    bias   = tl.load(in_ptr8 + x0, mask=xmask, eviction_policy='evict_last')
    inv_std = tl.rsqrt(var + 1e-5)
    bn_out = (pool - mean) * inv_std * weight + bias
    relu_out = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), bn_out)
    tl.store(in_out_ptr0 + xindex, relu_out, mask=xmask)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64}, num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    arg_8 = inputs[8]
    arg_9 = inputs[9]
    xnumel = 2007040
    grid = (triton.cdiv(xnumel, 128),)
    kernel_avgpool_bn_relu_sub[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 2007040)
    return arg_9

def get_inputs():
    arg_0 = rand_strided((10, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((31360, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9]

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
