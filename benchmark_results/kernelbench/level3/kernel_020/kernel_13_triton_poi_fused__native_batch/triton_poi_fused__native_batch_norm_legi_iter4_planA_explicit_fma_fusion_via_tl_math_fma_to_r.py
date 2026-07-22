import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    # FMA 1: fuse (tmp3 + 1e-5) into rsqrt arg to avoid separate FADD
    inv_std = tl.math.rsqrt(tl.math.fma(tmp3, 1.0, 1e-5))
    tmp11 = tmp2 * inv_std
    # FMA 2: fuse (tmp11 * tmp12 + tmp14) to avoid separate FMUL + FADD
    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
    tmp17 = tl.maximum(tmp15, 0.0)
    tmp19 = tl.minimum(tmp17, 6.0)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1128960
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
