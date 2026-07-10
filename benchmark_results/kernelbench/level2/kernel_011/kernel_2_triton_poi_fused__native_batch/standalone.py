import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    xoffset = pid * XBLOCK
    stride = num_programs * XBLOCK
    while xoffset < xnumel:
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
    x0 = (xindex % 128)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp_eps = tl.full([1], 1e-05, tl.float32)
        tmp_var_eps = tmp5 + tmp_eps
        tmp_inv_std = libdevice.rsqrt(tmp_var_eps)
        tmp_scale = tmp14 * tmp_inv_std
        tmp_offset = tl.math.fma(-tmp3, tmp_scale, tmp16)
        tmp17 = tl.math.fma(tmp2, tmp_scale, tmp_offset)
        tmp18 = libdevice.tanh(tmp17)
        tl.store(in_out_ptr0 + (x2), tmp18, xmask)
        xoffset += stride


def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 75759616
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_tanh_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 75759616, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((512, 128, 34, 34), (147968, 1, 4352, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
