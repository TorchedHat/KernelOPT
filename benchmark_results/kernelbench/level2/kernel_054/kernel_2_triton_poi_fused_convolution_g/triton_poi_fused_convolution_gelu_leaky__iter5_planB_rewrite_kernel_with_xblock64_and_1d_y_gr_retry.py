import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 64}, num_warps=4),
        triton.Config({'YBLOCK': 128}, num_warps=4),
        triton.Config({'YBLOCK': 256}, num_warps=8),
        triton.Config({'YBLOCK': 512}, num_warps=8),
        triton.Config({'YBLOCK': 1024}, num_warps=8),
    ],
    key=['ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK=64 covers all 64 channels in one block -- no xmask needed
    # program_id(0) drives Y only
    yoffset = tl.program_id(0) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < ynumel
    xindex = tl.arange(0, XBLOCK)[None, :]             # shape [1, 64]
    # Decompose yindex: y1=batch [0,64), y0=spatial [0,64516)
    y0 = yindex % 64516
    y1 = yindex // 64516
    # Load: in_ptr0 strides (4129024,1,16256,64) -> flat index = channel + 64*yindex
    tmp0 = tl.load(in_ptr0 + (xindex + 64 * yindex), ymask, eviction_policy='evict_last')
    # Load bias per channel: in_ptr1 shape [64] stride (1,)
    tmp1 = tl.load(in_ptr1 + xindex, eviction_policy='evict_last')
    # Load scale per channel: in_ptr2 shape [64,1,1] stride (1,1,1)
    tmp3 = tl.load(in_ptr2 + xindex, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1, 1], 0.0, tl.float32)
    tmp6 = tmp4 > tmp5
    tmp7 = tl.full([1, 1], 0.01, tl.float32)
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tl.full([1, 1], 0.5, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp13 = tmp9 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = tl.full([1, 1], 1.0, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    # Store: out_ptr0 strides (4129024,64516,254,1) -> flat index = y0 + 64516*channel + 4129024*batch
    tl.store(out_ptr0 + (y0 + 64516 * xindex + 4129024 * y1), tmp17, ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 4129024
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_gelu_leaky_relu_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 4129024, 64)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((64, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 64, 254, 254), (4129024, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
