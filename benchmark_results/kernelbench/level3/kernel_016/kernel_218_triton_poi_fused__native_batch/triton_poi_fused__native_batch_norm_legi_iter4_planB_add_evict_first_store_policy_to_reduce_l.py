import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_218(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 831040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1696)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp_inv_std = tl.math.rsqrt(tmp3 + tl.full([1], 1e-05, tl.float32))
    tmp_xhat = (tmp0 - tmp1) * tmp_inv_std
    tmp_out = tl.math.fma(tmp12, tmp_xhat, tmp14)
    tmp_relu = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), tmp_out)
    tl.store(out_ptr0 + (x2), tmp_relu, xmask, cache_modifier='.cs')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 831040
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_218[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 831040, XBLOCK=128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((10, 1696, 7, 7), (83104, 1, 11872, 1696), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1696,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1696,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1696,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1696,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 1696, 7, 7), (83104, 1, 11872, 1696), device='cuda:0', dtype=torch.float32)
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
