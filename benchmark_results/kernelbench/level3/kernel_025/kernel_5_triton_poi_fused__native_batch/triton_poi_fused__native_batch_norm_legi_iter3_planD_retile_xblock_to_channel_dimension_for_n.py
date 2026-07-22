import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 501760
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x0 = xindex
    y1 = (yindex % 50176)
    y2 = yindex // 50176
    y3 = yindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 64*y1 + 3211264*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp10 = tl.math.rsqrt(tmp5)
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp23 = tl.math.rsqrt(tmp20)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = tl.maximum(tmp30, tmp29)
    tl.store(out_ptr0 + (y1 + 50176*x0 + 3211264*y2), tmp31, xmask & ymask)

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
    ynumel = 640
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176)
    return arg_10

def get_inputs():
    arg_0 = rand_strided((10, 64, 224, 224), (3211264, 1, 14336, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((501760, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((10, 64, 224, 224), (3211264, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
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
