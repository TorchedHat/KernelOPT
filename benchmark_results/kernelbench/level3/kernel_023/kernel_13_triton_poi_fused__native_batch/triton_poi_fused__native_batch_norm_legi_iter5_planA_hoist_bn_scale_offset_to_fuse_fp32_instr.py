import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    # Load BN parameters first (broadcast/evict_last, scalar per channel)
    mean = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    var = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    weight = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    bias = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    # Precompute scale and offset (hoisted, not per-element)
    eps = tl.full([1], 1e-05, tl.float32)
    scale = weight * libdevice.rsqrt(var + eps)
    offset = bias - mean * scale
    # Load input x and compute BN result via 2 FMAs: x*scale + offset
    x = tl.load(in_out_ptr0 + (x2), xmask)
    result = x * scale + offset
    # HardTanh clamp [0.0, 6.0]
    result = tl.maximum(result, tl.full([1], 0.0, tl.float32))
    result = tl.minimum(result, tl.full([1], 6.0, tl.float32))
    tl.store(in_out_ptr0 + (x2), result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1128960
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960)
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
