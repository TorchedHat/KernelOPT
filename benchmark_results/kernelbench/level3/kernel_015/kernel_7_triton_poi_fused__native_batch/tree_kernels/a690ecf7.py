import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp_mean   = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')  # mean
    tmp_var    = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')  # var
    tmp_weight = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')  # weight
    tmp_bias   = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')  # bias
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), mask=tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), mask=tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    # Pre-compute per-channel BN scale and fused bias (algebraically equivalent)
    tmp_scale      = tmp_weight / tl.sqrt(tmp_var + 1e-5)   # weight/sqrt(var+eps)
    tmp_fused_bias = tmp_bias - tmp_mean * tmp_scale         # bias - mean*scale
    # Two-FMA BN: out = x * scale + fused_bias
    tmp_normed = tmp16 * tmp_scale + tmp_fused_bias
    tmp_relu   = tl.maximum(tl.full([1], 0, tl.float32), tmp_normed)
    tl.store(in_out_ptr0 + (x2), tmp_relu, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    xnumel = 4014080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 4014080)
    return arg_7

def get_inputs():
    arg_0 = rand_strided((10, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
