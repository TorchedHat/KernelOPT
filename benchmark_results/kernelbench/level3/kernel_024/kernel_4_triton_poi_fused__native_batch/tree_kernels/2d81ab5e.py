import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    # GROUP_SIZE=64 divides 62720 evenly (62720 = 980 * 64), no clamping needed
    GROUP_SIZE: tl.constexpr = 64
    NUM_GROUPS: tl.constexpr = 980  # = 62720 // 64
    pid_group = pid // GROUP_SIZE
    pid_in_group = pid % GROUP_SIZE
    swizzled_pid = pid_in_group * NUM_GROUPS + pid_group
    xoffset = swizzled_pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    # Replace int-div-sqrt chain with rsqrt for FMA fusion
    tmp5 = tmp3 + 1e-05
    inv_std = tl.math.rsqrt(tmp5)  # single MUFU.RSQ; enables FFMA below
    tmp11 = tmp2 * inv_std
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    # Use float zero for ReLU to avoid int->float cast
    tmp17 = tl.maximum(tl.zeros([1], tl.float32), tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 8028160
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 8028160, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
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
