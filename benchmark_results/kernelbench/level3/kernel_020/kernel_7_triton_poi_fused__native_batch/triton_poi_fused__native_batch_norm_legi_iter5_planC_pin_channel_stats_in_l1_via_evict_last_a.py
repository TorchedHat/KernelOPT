import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 96
    mean   = tl.load(in_ptr0 + x0, xmask, other=0.0, eviction_policy='evict_last')
    var    = tl.load(in_ptr1 + x0, xmask, other=0.0, eviction_policy='evict_last')
    weight = tl.load(in_ptr2 + x0, xmask, other=1.0, eviction_policy='evict_last')
    bias   = tl.load(in_ptr3 + x0, xmask, other=0.0, eviction_policy='evict_last')
    inv_std = tl.math.rsqrt(var + 1e-5)
    scale   = weight * inv_std
    offset  = bias - mean * scale
    x_val  = tl.load(in_out_ptr0 + xindex, xmask, other=0.0, eviction_policy='evict_first')
    result = x_val * scale + offset
    result = tl.maximum(result, 0.0)
    result = tl.minimum(result, 6.0)
    tl.store(in_out_ptr0 + xindex, result, xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
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
