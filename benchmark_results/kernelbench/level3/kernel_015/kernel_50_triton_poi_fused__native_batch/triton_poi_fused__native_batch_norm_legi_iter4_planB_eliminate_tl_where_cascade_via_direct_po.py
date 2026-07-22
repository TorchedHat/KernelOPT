import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 878080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 448)
    x1 = ((xindex // 448) % 14)
    x2 = xindex // 6272
    x3 = xindex // 448
    x4 = xindex
    tmp51 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr10 + (x0), mask=xmask, eviction_policy='evict_last')
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
    tmp13 = tmp11 * 0.25
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    # in_ptr1 is now the consolidated (1960, 192) tensor passed from caller
    # Access pattern: 192*x3 + (x0 - 256) for channels 256-447
    tmp_direct = tl.load(in_ptr1 + (192*x3 + (x0 - 256)), mask=(~tmp4) & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp4, tmp15, tmp_direct)
    tmp_diff = tmp50 - tmp51
    tmp_rsqrt = tl.math.rsqrt(tmp53 + 1e-5)
    tmp_norm = tmp_diff * tmp_rsqrt
    tmp_scaled = tmp_norm * tmp62 + tmp64
    tmp67 = tl.maximum(0.0, tmp_scaled)
    tl.store(in_out_ptr0 + (x4), tmp67, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    arg_8 = inputs[8]
    arg_9 = inputs[9]
    arg_10 = inputs[10]
    arg_11 = inputs[11]
    xnumel = 878080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, 878080, XBLOCK=128)
    return arg_11

def get_inputs():
    arg_0 = rand_strided((10, 448, 14, 14), (87808, 1, 6272, 448), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((7840, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    arg_11 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11]

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
