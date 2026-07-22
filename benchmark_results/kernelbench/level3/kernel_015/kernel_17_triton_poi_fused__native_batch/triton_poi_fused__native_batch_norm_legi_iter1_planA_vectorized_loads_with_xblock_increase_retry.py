import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xindex = tl.max_contiguous(tl.multiple_of(xindex, 1024), 1024)
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 28)
    x2 = xindex // 3584
    x4 = xindex
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 14336*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 14336*x2), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + 256*x1 + 14336*x2), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + 256*x1 + 14336*x2), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 4.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.full([1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt_rn(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = (tmp15 / tmp14)
    tmp17 = tl.full([1], 1.0, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = tl.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x4), tmp25, None)
    tl.store(out_ptr1 + (x0 + 384*x3), tmp8, None)
    tl.store(out_ptr2 + (x0 + 416*x3), tmp8, None)
    tl.store(out_ptr3 + (x0 + 448*x3), tmp8, None)
    tl.store(out_ptr4 + (x0 + 480*x3), tmp8, None)
    tl.store(out_ptr5 + (x0 + 512*x3), tmp8, None)

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
    xnumel = 1003520
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 1003520, XBLOCK=128)
    return arg_10

def get_inputs():
    arg_0 = rand_strided((31360, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 128, 28, 28), (301056, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((10, 128, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((10, 128, 28, 28), (351232, 1, 12544, 448), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((10, 128, 28, 28), (376320, 1, 13440, 480), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 128, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10]

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
