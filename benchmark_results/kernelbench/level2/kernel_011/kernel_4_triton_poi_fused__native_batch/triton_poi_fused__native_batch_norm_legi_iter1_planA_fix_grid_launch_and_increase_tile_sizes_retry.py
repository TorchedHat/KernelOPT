import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=16, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147968
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 17)
    y4 = yindex // 17
    y2 = yindex // 289
    y5 = (yindex % 289)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4352 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4480 + x3 + 256*y0 + 8704*y4), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (8*y2 + (x3 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (8*y2 + (x3 // 16)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tmp6 - tmp7
    tmp10 = tl.full([1, 1], 4624.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y5 + 289*x3 + 36992*y2), tmp19, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    ynumel = 147968
    xnumel = 128
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 147968, 128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((512, 128, 34, 34), (147968, 1, 4352, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((512, 128, 17, 17), (36992, 289, 17, 1), device='cuda:0', dtype=torch.float32)
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
