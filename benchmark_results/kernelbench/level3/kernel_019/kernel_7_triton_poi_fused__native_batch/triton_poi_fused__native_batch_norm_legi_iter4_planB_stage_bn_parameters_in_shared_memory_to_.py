import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    mean   = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    var    = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    weight = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    bias   = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    x_val  = tl.load(in_out_ptr0 + (x2), xmask)
    # FMA-fused BN inference: out = (x - mean) * rsqrt(var + eps) * weight + bias
    inv_std = tl.math.rsqrt(var + 1e-05)                    # single MUFU.RSQ instruction
    x_norm  = tl.math.fma(x_val, inv_std, -mean * inv_std)  # FFMA: x*inv_std - mean*inv_std
    out     = tl.math.fma(x_norm, weight, bias)              # FFMA: x_norm*weight + bias
    result  = tl.maximum(0.0, out)                           # ReLU
    tl.store(in_out_ptr0 + (x2), result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 4014080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4014080, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
