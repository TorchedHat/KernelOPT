import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8, 'YBLOCK': 8}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 8, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 8}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 8}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 501760
    xnumel = 96
    xpid = tl.program_id(0)
    ypid = tl.program_id(1)
    xoffset = xpid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    yoffset = ypid * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    x2 = xindex
    y0 = (yindex % 50176)
    y1 = yindex // 50176
    y3 = yindex
    tmp17 = tl.load(in_ptr3 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 50176*(x2) + 1605632*y1), mask=tmp4 & xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (32*y3 + ((-32) + x2)), mask=tmp9 & xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 96, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), mask=tmp11 & xmask & ymask, eviction_policy='evict_first', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt_rn(tmp21)
    tmp23 = tl.full([1, 1], 1, tl.int32)
    tmp24 = (tmp23 / tmp22)
    tmp25 = tl.full([1, 1], 1.0, tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1, 1], 0, tl.int32)
    tmp33 = tl.maximum(tmp32, tmp31)
    tl.store(in_out_ptr0 + (x2 + 96*y3), tmp33, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    ynumel = 501760
    xnumel = 96
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 501760, 96)
    return arg_7

def get_inputs():
    arg_0 = rand_strided((10, 96, 224, 224), (4816896, 1, 21504, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 224, 224), (1605632, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7]

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
