import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128}, num_warps=4),
    triton.Config({'XBLOCK': 128}, num_warps=8),
    triton.Config({'XBLOCK': 256}, num_warps=4),
    triton.Config({'XBLOCK': 256}, num_warps=8),
    triton.Config({'XBLOCK': 512}, num_warps=4),
    triton.Config({'XBLOCK': 512}, num_warps=8),
    triton.Config({'XBLOCK': 1024}, num_warps=4),
    triton.Config({'XBLOCK': 1024}, num_warps=8),
], key=['xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool3d_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30505984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 31)
    x1 = ((xindex // 31) % 31)
    x2 = ((xindex // 961) % 31)
    x3 = xindex // 29791
    x4 = (xindex % 29791)
    base = 2*x0 + 126*x1 + 8000*x2 + 252000*x3
    tmp0 = tl.load(in_ptr0 + base, xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + base + 1, xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + base + 63, xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + base + 64, xmask, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr0 + base + 4000, xmask, eviction_policy='evict_first')
    tmp9 = tl.load(in_ptr0 + base + 4001, xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + base + 4063, xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + base + 4064, xmask, eviction_policy='evict_first')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tl.store(out_ptr0 + (x4 + 29792*x3), tmp16, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 30505984
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool3d_convolution_1[grid](arg_0, arg_1, 30505984)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 16, 63, 63, 63), (4032000, 252000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 16, 31, 31, 31), (476672, 29792, 961, 31, 1), device='cuda:0', dtype=torch.float32)
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
