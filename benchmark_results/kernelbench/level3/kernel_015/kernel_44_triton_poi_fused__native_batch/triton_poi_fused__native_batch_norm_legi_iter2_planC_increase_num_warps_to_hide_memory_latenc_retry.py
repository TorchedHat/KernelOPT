import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 14)
    x2 = xindex // 3584
    x4 = xindex
    x3 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 14336*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 14336*x2), xmask)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + 512*x1 + 14336*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + 512*x1 + 14336*x2), xmask)
    tmp9 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp25, xmask)
    tl.store(out_ptr1 + (x0 + 512*x3), tmp8, xmask)
    tl.store(out_ptr2 + (x0 + 544*x3), tmp8, xmask)
    tl.store(out_ptr3 + (x0 + 576*x3), tmp8, xmask)
    tl.store(out_ptr4 + (x0 + 608*x3), tmp8, xmask)
    tl.store(out_ptr5 + (x0 + 640*x3), tmp8, xmask)
    tl.store(out_ptr6 + (x0 + 672*x3), tmp8, xmask)
    tl.store(out_ptr7 + (x0 + 704*x3), tmp8, xmask)
    tl.store(out_ptr8 + (x0 + 736*x3), tmp8, xmask)
    tl.store(out_ptr9 + (x0 + 768*x3), tmp8, xmask)
    tl.store(out_ptr10 + (x0 + 800*x3), tmp8, xmask)
    tl.store(out_ptr11 + (x0 + 832*x3), tmp8, xmask)
    tl.store(out_ptr12 + (x0 + 864*x3), tmp8, xmask)
    tl.store(out_ptr13 + (x0 + 896*x3), tmp8, xmask)
    tl.store(out_ptr14 + (x0 + 928*x3), tmp8, xmask)
    tl.store(out_ptr15 + (x0 + 960*x3), tmp8, xmask)
    tl.store(out_ptr16 + (x0 + 992*x3), tmp8, xmask)
    tl.store(out_ptr17 + (x0 + 1024*x3), tmp8, xmask)

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
    arg_12 = inputs[12]
    arg_13 = inputs[13]
    arg_14 = inputs[14]
    arg_15 = inputs[15]
    arg_16 = inputs[16]
    arg_17 = inputs[17]
    arg_18 = inputs[18]
    arg_19 = inputs[19]
    arg_20 = inputs[20]
    arg_21 = inputs[21]
    arg_22 = inputs[22]
    xnumel = 501760
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_44[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, 501760)
    return arg_22

def get_inputs():
    arg_0 = rand_strided((7840, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 256, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((10, 256, 14, 14), (106624, 1, 7616, 544), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((10, 256, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((10, 256, 14, 14), (119168, 1, 8512, 608), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 256, 14, 14), (125440, 1, 8960, 640), device='cuda:0', dtype=torch.float32)
    arg_11 = rand_strided((10, 256, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    arg_12 = rand_strided((10, 256, 14, 14), (137984, 1, 9856, 704), device='cuda:0', dtype=torch.float32)
    arg_13 = rand_strided((10, 256, 14, 14), (144256, 1, 10304, 736), device='cuda:0', dtype=torch.float32)
    arg_14 = rand_strided((10, 256, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    arg_15 = rand_strided((10, 256, 14, 14), (156800, 1, 11200, 800), device='cuda:0', dtype=torch.float32)
    arg_16 = rand_strided((10, 256, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    arg_17 = rand_strided((10, 256, 14, 14), (169344, 1, 12096, 864), device='cuda:0', dtype=torch.float32)
    arg_18 = rand_strided((10, 256, 14, 14), (175616, 1, 12544, 896), device='cuda:0', dtype=torch.float32)
    arg_19 = rand_strided((10, 256, 14, 14), (181888, 1, 12992, 928), device='cuda:0', dtype=torch.float32)
    arg_20 = rand_strided((10, 256, 14, 14), (188160, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    arg_21 = rand_strided((10, 256, 14, 14), (194432, 1, 13888, 992), device='cuda:0', dtype=torch.float32)
    arg_22 = rand_strided((10, 256, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22]

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
