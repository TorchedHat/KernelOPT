import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    # Load channel-indexed constants first (evict_last, hit L1)
    tmp_mean   = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')  # mean
    tmp_var    = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')  # var
    tmp_weight = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')  # weight
    tmp_beta   = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')  # beta
    # Precompute per-channel scale and shift (FMA-friendly)
    scale = libdevice.rsqrt(tmp_var + 1e-5) * tmp_weight   # rsqrt*weight
    shift = tmp_beta - tmp_mean * scale                     # beta - mean*scale
    # Load per-element data and bias
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')  # x
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')        # bias
    # 2-FMA chain: (x + bias) * scale + shift
    tmp_out = (tmp0 + tmp1) * scale + shift
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tl.maximum(tmp18, tmp_out)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 12845056
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 12845056)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((8, 128, 112, 112), (1605632, 1, 14336, 128), device='cuda:0', dtype=torch.float32)
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
