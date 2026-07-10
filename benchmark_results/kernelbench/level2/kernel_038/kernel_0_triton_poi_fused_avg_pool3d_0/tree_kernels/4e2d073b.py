import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = xindex // 1024
    x3 = xindex
    # base: stride-2 in x0, but pairs (base, base+1) are contiguous.
    # tl.max_contiguous(expr, 16): 16 consecutive threads have contiguous base values
    # tl.multiple_of(expr, 2): base is always even-aligned (2*x0 is always even)
    base = tl.multiple_of(tl.max_contiguous(2*x0 + 128*x1 + 8192*x2, 16), 2)
    tmp0  = tl.load(in_ptr0 + base,        mask=xmask, eviction_policy='evict_first')
    tmp1  = tl.load(in_ptr0 + base + 1,    mask=xmask, eviction_policy='evict_first')
    tmp3  = tl.load(in_ptr0 + base + 64,   mask=xmask, eviction_policy='evict_first')
    tmp5  = tl.load(in_ptr0 + base + 65,   mask=xmask, eviction_policy='evict_first')
    tmp7  = tl.load(in_ptr0 + base + 4096, mask=xmask, eviction_policy='evict_first')
    tmp9  = tl.load(in_ptr0 + base + 4097, mask=xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + base + 4160, mask=xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + base + 4161, mask=xmask, eviction_policy='evict_first')
    tmp2  = tmp0 + tmp1
    tmp4  = tmp2 + tmp3
    tmp6  = tmp4 + tmp5
    tmp8  = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + (x3), tmp16, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 16777216
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_avg_pool3d_0[grid](arg_0, arg_1, 16777216, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 32, 32, 64, 64), (4194304, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 32, 16, 32, 32), (524288, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
