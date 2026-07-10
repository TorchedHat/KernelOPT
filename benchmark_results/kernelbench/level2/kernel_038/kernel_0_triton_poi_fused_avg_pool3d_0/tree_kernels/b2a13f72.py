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
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)           # output W in [0,32)
    x1 = ((xindex // 32) % 32)   # output H in [0,32)
    x2 = xindex // 1024           # batch/ch/D
    x3 = xindex
    # Compute even/odd input W addresses separately with contiguity hints
    # w_even = 2*x0 is always even (stride-1 pair), w_odd = 2*x0+1 is always odd
    w_even = tl.multiple_of(2 * x0, 2)       # always even, hint alignment
    w_odd  = tl.multiple_of(2 * x0 + 1, 1)  # always odd
    row_base = 128*x1 + 8192*x2
    # 8 loads: w_even and w_odd are explicit contiguous pairs, compiler can vectorize
    tmp0  = tl.load(in_ptr0 + w_even + row_base,        mask=xmask)  # D=0, H=2*x1,   W=2*x0
    tmp1  = tl.load(in_ptr0 + w_odd  + row_base,        mask=xmask)  # D=0, H=2*x1,   W=2*x0+1
    tmp3  = tl.load(in_ptr0 + w_even + row_base + 64,   mask=xmask)  # D=0, H=2*x1+1, W=2*x0
    tmp5  = tl.load(in_ptr0 + w_odd  + row_base + 64,   mask=xmask)  # D=0, H=2*x1+1, W=2*x0+1
    tmp7  = tl.load(in_ptr0 + w_even + row_base + 4096, mask=xmask)  # D=1, H=2*x1,   W=2*x0
    tmp9  = tl.load(in_ptr0 + w_odd  + row_base + 4096, mask=xmask)  # D=1, H=2*x1,   W=2*x0+1
    tmp11 = tl.load(in_ptr0 + w_even + row_base + 4160, mask=xmask)  # D=1, H=2*x1+1, W=2*x0
    tmp13 = tl.load(in_ptr0 + w_odd  + row_base + 4160, mask=xmask)  # D=1, H=2*x1+1, W=2*x0+1
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
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
