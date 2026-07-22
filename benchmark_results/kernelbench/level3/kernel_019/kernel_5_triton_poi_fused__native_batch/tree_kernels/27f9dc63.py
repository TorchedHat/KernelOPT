import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    # Preload 4 x 64-float channel arrays once per block (1 KB total)
    # tl.arange(0,64) is a compile-time constant range -> vectorized 256-byte load
    c_idx = tl.arange(0, 64)
    xmask_c = c_idx < 64
    smem_mean   = tl.load(in_ptr0 + c_idx, mask=xmask_c)  # [64] float32, loaded once per block
    smem_var    = tl.load(in_ptr1 + c_idx, mask=xmask_c)
    smem_weight = tl.load(in_ptr2 + c_idx, mask=xmask_c)
    smem_bias   = tl.load(in_ptr3 + c_idx, mask=xmask_c)
    tl.debug_barrier()  # memory fence before per-thread indexed reads

    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64).to(tl.int32)

    tmp0  = tl.load(in_out_ptr0 + (x2), mask=xmask)
    tmp1  = tl.gather(smem_mean,   x0, 0)
    tmp3  = tl.gather(smem_var,    x0, 0)
    tmp12 = tl.gather(smem_weight, x0, 0)
    tmp14 = tl.gather(smem_bias,   x0, 0)

    tmp2  = tmp0 - tmp1
    tmp4  = tl.full([1], 1e-05, tl.float32)
    tmp5  = tmp3 + tmp4
    tmp10 = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2007040
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
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
