import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_51(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Output-linear order: output strides (100352, 1, 7168, 512), C is fastest (stride=1)
    c    = xindex % 64          # channel index (stride=1 in output)
    hw_n = xindex // 64         # combined spatial+batch
    h_w  = hw_n % 196           # h*w position (14*14=196)
    n    = hw_n // 196          # batch index
    h    = h_w // 14            # height index
    w    = h_w % 14             # width index
    # Input strides (12544, 1, 896, 64): in[n,c,h,w] = n*12544 + c*1 + h*896 + w*64
    x_in = n * 12544 + c + h * 896 + w * 64
    tmp0 = tl.load(in_ptr0 + x_in, xmask)
    tmp1 = tl.load(in_ptr1 + c, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + xindex, tmp2, xmask)  # coalesced stride-1 store

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 125440
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_51[grid](arg_0, arg_1, arg_2, 125440)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 64, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
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
