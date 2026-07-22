import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    # Flat 1D layout - no 2D tiling, no SP_BLOCK/CBLOCK decomposition
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]   # [XBLOCK]
    # xnumel=3010560 is exactly divisible by XBLOCK=128 - no mask needed
    x0 = xindex % 24   # channel index per element
    # Load channel params with evict_last - keeps 96 bytes hot in L1
    mean_val   = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
    var_val    = tl.load(in_ptr1 + x0, eviction_policy='evict_last')
    weight_val = tl.load(in_ptr2 + x0, eviction_policy='evict_last')
    bias_val   = tl.load(in_ptr3 + x0, eviction_policy='evict_last')
    tmp0 = tl.load(in_out_ptr0 + xindex)   # main data load
    inv_std = libdevice.rsqrt(var_val + 1e-5)
    normed  = (tmp0 - mean_val) * inv_std * weight_val + bias_val
    result  = tl.maximum(tl.full([XBLOCK], 0.0, tl.float32), normed)
    tl.store(in_out_ptr0 + xindex, result)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_relu_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 24, 112, 112), (301056, 1, 2688, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
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
