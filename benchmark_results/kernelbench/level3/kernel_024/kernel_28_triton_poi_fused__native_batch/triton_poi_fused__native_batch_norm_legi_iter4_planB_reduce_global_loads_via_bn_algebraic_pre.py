import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1728)
    # Issue all parameter loads first — their latency overlaps with each other
    tmp_mean   = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')  # mean
    tmp_var    = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')  # var
    tmp_weight = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')  # weight
    tmp_bias   = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')  # bias
    # Compute fused scale and bias BEFORE loading main data
    tmp_eps    = tl.full([1], 1e-05, tl.float32)
    tmp_scale  = tmp_weight * libdevice.rsqrt(tmp_var + tmp_eps)  # scale = w/sqrt(v+e)
    tmp_fbias  = tl.fma(-tmp_mean, tmp_scale, tmp_bias)           # fbias = b - mean*scale
    # Now load main data — its latency is partially hidden by param arithmetic above
    tmp_input  = tl.load(in_out_ptr0 + (x2), xmask)              # streaming, no evict_last
    # Apply linear transform + ReLU
    tmp_out    = tl.fma(tmp_input, tmp_scale, tmp_fbias)
    tmp_relu   = tl.maximum(tl.zeros_like(tmp_out), tmp_out)
    tl.store(in_out_ptr0 + (x2), tmp_relu, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 3456
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_28[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3456, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((2, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1728,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1728,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1728,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1728,), (1,), device='cuda:0', dtype=torch.float32)
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
