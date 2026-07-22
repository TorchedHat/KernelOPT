import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr18 + (x0 + 1056*x3), tmp8, xmask)
    tl.store(out_ptr19 + (x0 + 1088*x3), tmp8, xmask)
    tl.store(out_ptr20 + (x0 + 1120*x3), tmp8, xmask)
    tl.store(out_ptr21 + (x0 + 1152*x3), tmp8, xmask)
    tl.store(out_ptr22 + (x0 + 1184*x3), tmp8, xmask)
    tl.store(out_ptr23 + (x0 + 1216*x3), tmp8, xmask)
    tl.store(out_ptr24 + (x0 + 1248*x3), tmp8, xmask)
    tl.store(out_ptr25 + (x0 + 1280*x3), tmp8, xmask)
    tl.store(out_ptr26 + (x0 + 1312*x3), tmp8, xmask)
    tl.store(out_ptr27 + (x0 + 1344*x3), tmp8, xmask)
    tl.store(out_ptr28 + (x0 + 1376*x3), tmp8, xmask)
    tl.store(out_ptr29 + (x0 + 1408*x3), tmp8, xmask)
    tl.store(out_ptr30 + (x0 + 1440*x3), tmp8, xmask)
    tl.store(out_ptr31 + (x0 + 1472*x3), tmp8, xmask)
    tl.store(out_ptr32 + (x0 + 1504*x3), tmp8, xmask)
    tl.store(out_ptr33 + (x0 + 1536*x3), tmp8, xmask)
    tl.store(out_ptr34 + (x0 + 1568*x3), tmp8, xmask)
    tl.store(out_ptr35 + (x0 + 1600*x3), tmp8, xmask)
    tl.store(out_ptr36 + (x0 + 1632*x3), tmp8, xmask)
    tl.store(out_ptr37 + (x0 + 1664*x3), tmp8, xmask)
    tl.store(out_ptr38 + (x0 + 1696*x3), tmp8, xmask)
    tl.store(out_ptr39 + (x0 + 1728*x3), tmp8, xmask)
    tl.store(out_ptr40 + (x0 + 1760*x3), tmp8, xmask)
    tl.store(out_ptr41 + (x0 + 1792*x3), tmp8, xmask)



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
    arg_31 = inputs[31]
    arg_32 = inputs[32]
    arg_33 = inputs[33]
    arg_34 = inputs[34]
    arg_35 = inputs[35]
    arg_36 = inputs[36]
    arg_37 = inputs[37]
    arg_38 = inputs[38]
    arg_39 = inputs[39]
    arg_40 = inputs[40]
    arg_41 = inputs[41]
    arg_42 = inputs[42]
    arg_43 = inputs[43]
    arg_44 = inputs[44]
    arg_45 = inputs[45]
    arg_46 = inputs[46]
    xnumel = 501760
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_44[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30, arg_31, arg_32, arg_33, arg_34, arg_35, arg_36, arg_37, arg_38, arg_39, arg_40, arg_41, arg_42, arg_43, arg_44, arg_45, arg_46, 501760, XBLOCK=128)
    return arg_46

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
    arg_23 = rand_strided((10, 256, 14, 14), (206976, 1, 14784, 1056), device='cuda:0', dtype=torch.float32)
    arg_24 = rand_strided((10, 256, 14, 14), (213248, 1, 15232, 1088), device='cuda:0', dtype=torch.float32)
    arg_25 = rand_strided((10, 256, 14, 14), (219520, 1, 15680, 1120), device='cuda:0', dtype=torch.float32)
    arg_26 = rand_strided((10, 256, 14, 14), (225792, 1, 16128, 1152), device='cuda:0', dtype=torch.float32)
    arg_27 = rand_strided((10, 256, 14, 14), (232064, 1, 16576, 1184), device='cuda:0', dtype=torch.float32)
    arg_28 = rand_strided((10, 256, 14, 14), (238336, 1, 17024, 1216), device='cuda:0', dtype=torch.float32)
    arg_29 = rand_strided((10, 256, 14, 14), (244608, 1, 17472, 1248), device='cuda:0', dtype=torch.float32)
    arg_30 = rand_strided((10, 256, 14, 14), (250880, 1, 17920, 1280), device='cuda:0', dtype=torch.float32)
    arg_31 = rand_strided((10, 256, 14, 14), (257152, 1, 18368, 1312), device='cuda:0', dtype=torch.float32)
    arg_32 = rand_strided((10, 256, 14, 14), (263424, 1, 18816, 1344), device='cuda:0', dtype=torch.float32)
    arg_33 = rand_strided((10, 256, 14, 14), (269696, 1, 19264, 1376), device='cuda:0', dtype=torch.float32)
    arg_34 = rand_strided((10, 256, 14, 14), (275968, 1, 19712, 1408), device='cuda:0', dtype=torch.float32)
    arg_35 = rand_strided((10, 256, 14, 14), (282240, 1, 20160, 1440), device='cuda:0', dtype=torch.float32)
    arg_36 = rand_strided((10, 256, 14, 14), (288512, 1, 20608, 1472), device='cuda:0', dtype=torch.float32)
    arg_37 = rand_strided((10, 256, 14, 14), (294784, 1, 21056, 1504), device='cuda:0', dtype=torch.float32)
    arg_38 = rand_strided((10, 256, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    arg_39 = rand_strided((10, 256, 14, 14), (307328, 1, 21952, 1568), device='cuda:0', dtype=torch.float32)
    arg_40 = rand_strided((10, 256, 14, 14), (313600, 1, 22400, 1600), device='cuda:0', dtype=torch.float32)
    arg_41 = rand_strided((10, 256, 14, 14), (319872, 1, 22848, 1632), device='cuda:0', dtype=torch.float32)
    arg_42 = rand_strided((10, 256, 14, 14), (326144, 1, 23296, 1664), device='cuda:0', dtype=torch.float32)
    arg_43 = rand_strided((10, 256, 14, 14), (332416, 1, 23744, 1696), device='cuda:0', dtype=torch.float32)
    arg_44 = rand_strided((10, 256, 14, 14), (338688, 1, 24192, 1728), device='cuda:0', dtype=torch.float32)
    arg_45 = rand_strided((10, 256, 14, 14), (344960, 1, 24640, 1760), device='cuda:0', dtype=torch.float32)
    arg_46 = rand_strided((10, 256, 14, 14), (351232, 1, 25088, 1792), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, arg_27, arg_28, arg_29, arg_30, arg_31, arg_32, arg_33, arg_34, arg_35, arg_36, arg_37, arg_38, arg_39, arg_40, arg_41, arg_42, arg_43, arg_44, arg_45, arg_46]

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
