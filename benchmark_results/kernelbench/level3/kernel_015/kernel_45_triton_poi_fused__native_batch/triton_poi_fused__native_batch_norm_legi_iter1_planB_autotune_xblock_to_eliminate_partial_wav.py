import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 564480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 288)
    x1 = ((xindex // 288) % 14)
    x2 = xindex // 4032
    x3 = xindex // 288
    x4 = xindex
    tmp21 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 4.0, tl.float32)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp4, tmp15, tmp19)
    tmp22 = tmp20 - tmp21
    tmp24 = tl.full([1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.sqrt_rn(tmp25)
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = (tmp27 / tmp26)
    tmp29 = tl.full([1], 1.0, tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp22 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full([1], 0, tl.int32)
    tmp37 = tl.maximum(tmp36, tmp35)
    tl.store(in_out_ptr0 + (x4), tmp37, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 564480
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_45[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 564480)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((10, 288, 14, 14), (56448, 1, 4032, 288), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((7840, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
