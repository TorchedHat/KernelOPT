import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2764800
    r0_numel = 22
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = (xindex % 900)
    x4 = xindex // 900
    x1 = ((xindex // 900) % 24)
    x5 = xindex

    # Load bias once outside loop (shape [XBLOCK, 1], reused across all 22 iters)
    tmp_bias = tl.load(in_ptr1 + x1, mask=xmask, eviction_policy='evict_last')

    # Fully unrolled 22-element min reduction — no mask, no tl.where, no triton_helpers.min2
    v0  = tl.load(in_ptr0 + (x0 + 900*0  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v1  = tl.load(in_ptr0 + (x0 + 900*1  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v2  = tl.load(in_ptr0 + (x0 + 900*2  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v3  = tl.load(in_ptr0 + (x0 + 900*3  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v4  = tl.load(in_ptr0 + (x0 + 900*4  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v5  = tl.load(in_ptr0 + (x0 + 900*5  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v6  = tl.load(in_ptr0 + (x0 + 900*6  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v7  = tl.load(in_ptr0 + (x0 + 900*7  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v8  = tl.load(in_ptr0 + (x0 + 900*8  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v9  = tl.load(in_ptr0 + (x0 + 900*9  + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v10 = tl.load(in_ptr0 + (x0 + 900*10 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v11 = tl.load(in_ptr0 + (x0 + 900*11 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v12 = tl.load(in_ptr0 + (x0 + 900*12 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v13 = tl.load(in_ptr0 + (x0 + 900*13 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v14 = tl.load(in_ptr0 + (x0 + 900*14 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v15 = tl.load(in_ptr0 + (x0 + 900*15 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v16 = tl.load(in_ptr0 + (x0 + 900*16 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v17 = tl.load(in_ptr0 + (x0 + 900*17 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v18 = tl.load(in_ptr0 + (x0 + 900*18 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v19 = tl.load(in_ptr0 + (x0 + 900*19 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v20 = tl.load(in_ptr0 + (x0 + 900*20 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias
    v21 = tl.load(in_ptr0 + (x0 + 900*21 + 19800*x4), mask=xmask, eviction_policy='evict_first') + tmp_bias

    running_min = tl.minimum(v0, v1)
    running_min = tl.minimum(running_min, v2)
    running_min = tl.minimum(running_min, v3)
    running_min = tl.minimum(running_min, v4)
    running_min = tl.minimum(running_min, v5)
    running_min = tl.minimum(running_min, v6)
    running_min = tl.minimum(running_min, v7)
    running_min = tl.minimum(running_min, v8)
    running_min = tl.minimum(running_min, v9)
    running_min = tl.minimum(running_min, v10)
    running_min = tl.minimum(running_min, v11)
    running_min = tl.minimum(running_min, v12)
    running_min = tl.minimum(running_min, v13)
    running_min = tl.minimum(running_min, v14)
    running_min = tl.minimum(running_min, v15)
    running_min = tl.minimum(running_min, v16)
    running_min = tl.minimum(running_min, v17)
    running_min = tl.minimum(running_min, v18)
    running_min = tl.minimum(running_min, v19)
    running_min = tl.minimum(running_min, v20)
    running_min = tl.minimum(running_min, v21)

    tmp6 = running_min.to(tl.float32)
    tl.store(out_ptr0 + x5, tmp6, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 2764800
    r0_numel = 22
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_min_0[grid](arg_0, arg_1, arg_2, 2764800, 22)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 24, 30, 30), (21600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
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
