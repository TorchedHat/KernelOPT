import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_175(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, xnumel, XBLOCK : tl.constexpr):
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 896)
    x1 = ((xindex // 896) % 7)
    x2 = xindex // 6272
    x4 = xindex
    x3 = xindex // 896
    tmp0 = tl.load(in_ptr0 + (x0 + 1792*x1 + 25088*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (896 + x0 + 1792*x1 + 25088*x2), xmask)
    tmp3 = tl.load(in_ptr0 + (12544 + x0 + 1792*x1 + 25088*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (13440 + x0 + 1792*x1 + 25088*x2), xmask)
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
    tl.store(out_ptr1 + (x0 + 1152*x3), tmp8, xmask)
    tl.store(out_ptr2 + (x0 + 1184*x3), tmp8, xmask)
    tl.store(out_ptr3 + (x0 + 1216*x3), tmp8, xmask)
    tl.store(out_ptr4 + (x0 + 1248*x3), tmp8, xmask)
    tl.store(out_ptr5 + (x0 + 1280*x3), tmp8, xmask)
    tl.store(out_ptr6 + (x0 + 1312*x3), tmp8, xmask)
    tl.store(out_ptr7 + (x0 + 1344*x3), tmp8, xmask)
    tl.store(out_ptr8 + (x0 + 1376*x3), tmp8, xmask)
    tl.store(out_ptr9 + (x0 + 1408*x3), tmp8, xmask)
    tl.store(out_ptr10 + (x0 + 1440*x3), tmp8, xmask)
    tl.store(out_ptr11 + (x0 + 1472*x3), tmp8, xmask)
    tl.store(out_ptr12 + (x0 + 1504*x3), tmp8, xmask)
    tl.store(out_ptr13 + (x0 + 1536*x3), tmp8, xmask)
    tl.store(out_ptr14 + (x0 + 1568*x3), tmp8, xmask)
    tl.store(out_ptr15 + (x0 + 1600*x3), tmp8, xmask)
    tl.store(out_ptr16 + (x0 + 1632*x3), tmp8, xmask)
    tl.store(out_ptr17 + (x0 + 1664*x3), tmp8, xmask)
    tl.store(out_ptr18 + (x0 + 1696*x3), tmp8, xmask)
    tl.store(out_ptr19 + (x0 + 1728*x3), tmp8, xmask)
    tl.store(out_ptr20 + (x0 + 1760*x3), tmp8, xmask)
    tl.store(out_ptr21 + (x0 + 1792*x3), tmp8, xmask)
    tl.store(out_ptr22 + (x0 + 1824*x3), tmp8, xmask)
    tl.store(out_ptr23 + (x0 + 1856*x3), tmp8, xmask)
    tl.store(out_ptr24 + (x0 + 1888*x3), tmp8, xmask)
    tl.store(out_ptr25 + (x0 + 1920*x3), tmp8, xmask)



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
    arg_23 = inputs[23]
    arg_24 = inputs[24]
    arg_25 = inputs[25]
    arg_26 = inputs[26]
    arg_27 = inputs[27]
    arg_28 = inputs[28]
    arg_29 = inputs[29]
    arg_30 = inputs[30]
    xnumel = 439040
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_175[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30, 439040, XBLOCK=128)
    return arg_30

def get_inputs():
    arg_0 = rand_strided((1960, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((896,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((896,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((896,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((896,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 896, 7, 7), (43904, 1, 6272, 896), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 896, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((10, 896, 7, 7), (58016, 1, 8288, 1184), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((10, 896, 7, 7), (59584, 1, 8512, 1216), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((10, 896, 7, 7), (61152, 1, 8736, 1248), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 896, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    arg_11 = rand_strided((10, 896, 7, 7), (64288, 1, 9184, 1312), device='cuda:0', dtype=torch.float32)
    arg_12 = rand_strided((10, 896, 7, 7), (65856, 1, 9408, 1344), device='cuda:0', dtype=torch.float32)
    arg_13 = rand_strided((10, 896, 7, 7), (67424, 1, 9632, 1376), device='cuda:0', dtype=torch.float32)
    arg_14 = rand_strided((10, 896, 7, 7), (68992, 1, 9856, 1408), device='cuda:0', dtype=torch.float32)
    arg_15 = rand_strided((10, 896, 7, 7), (70560, 1, 10080, 1440), device='cuda:0', dtype=torch.float32)
    arg_16 = rand_strided((10, 896, 7, 7), (72128, 1, 10304, 1472), device='cuda:0', dtype=torch.float32)
    arg_17 = rand_strided((10, 896, 7, 7), (73696, 1, 10528, 1504), device='cuda:0', dtype=torch.float32)
    arg_18 = rand_strided((10, 896, 7, 7), (75264, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    arg_19 = rand_strided((10, 896, 7, 7), (76832, 1, 10976, 1568), device='cuda:0', dtype=torch.float32)
    arg_20 = rand_strided((10, 896, 7, 7), (78400, 1, 11200, 1600), device='cuda:0', dtype=torch.float32)
    arg_21 = rand_strided((10, 896, 7, 7), (79968, 1, 11424, 1632), device='cuda:0', dtype=torch.float32)
    arg_22 = rand_strided((10, 896, 7, 7), (81536, 1, 11648, 1664), device='cuda:0', dtype=torch.float32)
    arg_23 = rand_strided((10, 896, 7, 7), (83104, 1, 11872, 1696), device='cuda:0', dtype=torch.float32)
    arg_24 = rand_strided((10, 896, 7, 7), (84672, 1, 12096, 1728), device='cuda:0', dtype=torch.float32)
    arg_25 = rand_strided((10, 896, 7, 7), (86240, 1, 12320, 1760), device='cuda:0', dtype=torch.float32)
    arg_26 = rand_strided((10, 896, 7, 7), (87808, 1, 12544, 1792), device='cuda:0', dtype=torch.float32)
    arg_27 = rand_strided((10, 896, 7, 7), (89376, 1, 12768, 1824), device='cuda:0', dtype=torch.float32)
    arg_28 = rand_strided((10, 896, 7, 7), (90944, 1, 12992, 1856), device='cuda:0', dtype=torch.float32)
    arg_29 = rand_strided((10, 896, 7, 7), (92512, 1, 13216, 1888), device='cuda:0', dtype=torch.float32)
    arg_30 = rand_strided((10, 896, 7, 7), (94080, 1, 13440, 1920), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30]

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
