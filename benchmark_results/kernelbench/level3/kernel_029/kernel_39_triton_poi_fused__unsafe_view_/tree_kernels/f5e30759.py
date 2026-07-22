import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=1),
        triton.Config({'XBLOCK': 64},  num_warps=1),
        triton.Config({'XBLOCK': 64},  num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=2),
        triton.Config({'XBLOCK': 128}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1693440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x3 = xindex
    x_div32 = xindex // 32
    x1 = x_div32 % 588
    x2 = xindex // 18816
    x1_mod49 = x1 % 49
    x1_div49 = x1 // 49
    x1_mod49_div7 = x1_mod49 // 7
    x1_mod49_mod7 = x1_mod49 % 7
    x2_mod3 = x2 % 3
    x2_div3_mod3 = (x2 // 3) % 3
    tmp0 = (-4) + 7*x2_div3_mod3 + x1_mod49_div7
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-4) + 7*x2_mod3 + x1_mod49_mod7
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-23040) + x0 + 32*x1_div49 + 384*x1_mod49_mod7 + 2688*x2_mod3 + 5376*x1_mod49_div7 + 37632*x2_div3_mod3 + 75264*(x2 // 9)), mask=tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-60) + 7*x2_mod3 + 14*x1_mod49_div7 + 98*x2_div3_mod3 + 196*(x2 // 9) + x1_mod49_mod7), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + ((-60) + 7*x2_mod3 + 14*x1_mod49_div7 + 98*x2_div3_mod3 + 196*(x2 // 9) + x1_mod49_mod7), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.math.fma(tmp14, 0.002604166666666667, 1e-05)
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 32*x1_div49), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr4 + (x0 + 32*x1_div49), mask=tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.math.fma(tmp20, tmp21, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 1693440
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_clone_constant_pad_nd_convolution_native_layer_norm_permute_transpose_view_39[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1693440)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 196, 1), (196, 1, 1960), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 196, 1), (196, 1, 1984), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((90, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
