import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = (xindex < xnumel) & ymask
    # Decompose output indices
    x_ow = (xindex % 128)   # output col  [0..127]
    x_oh = xindex // 128    # output row  [0..15]
    y_c  = (yindex % 128)   # channel     [0..127]
    y_b  = yindex // 128    # batch       [0..7]
    # Base input address: input strides (2097152, 8192, 256, 1)
    # ih=2*x_oh, iw=2*x_ow  =>  base = y_b*2097152 + y_c*8192 + 512*x_oh + 2*x_ow
    base = 2097152*y_b + 8192*y_c + 512*x_oh + 2*x_ow
    # Load 2x2 pool window — pairs {base,base+1} and {base+256,base+257} are contiguous
    tmp0 = tl.load(in_ptr0 + base,       xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + base + 1,   xmask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + base + 256, xmask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + base + 257, xmask, eviction_policy='evict_first')
    tmp6 = tl.maximum(tl.maximum(tl.maximum(tmp0, tmp1), tmp3), tmp5)
    # Output strides: (262144, 1, 16384, 128) => out[b,c,oh,ow]
    out_idx = 262144*y_b + y_c + 16384*x_oh + 128*x_ow
    tl.store(out_ptr0 + out_idx, tmp6, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1024
    xnumel = 2048
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_exp_max_pool2d_with_indices_sub_10[grid](arg_0, arg_1, 1024, 2048)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 128, 32, 256), (2097152, 8192, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 128, 16, 128), (262144, 1, 16384, 128), device='cuda:0', dtype=torch.float32)
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
