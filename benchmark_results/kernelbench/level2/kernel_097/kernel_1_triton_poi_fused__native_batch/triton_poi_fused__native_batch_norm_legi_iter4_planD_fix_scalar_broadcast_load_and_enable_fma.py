import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
], key=['xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + tl.zeros([XBLOCK], dtype=tl.int32), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp10 = libdevice.rsqrt(tmp7)
    tmp13 = tmp4 * tmp10
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp20 = tmp17 + tmp18
    tmp22 = tl.sigmoid(tmp20)
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x2), tmp23, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_add_addmm_div_mul_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
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
