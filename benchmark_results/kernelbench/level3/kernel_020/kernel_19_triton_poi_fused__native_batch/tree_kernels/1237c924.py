import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=8),
        triton.Config({'XBLOCK': 2048}, num_warps=16),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0  = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
    tmp1  = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3  = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    # Precompute per-channel constants (hoisted by compiler; depend only on x0)
    tmp_eps          = tl.full([1], 1e-05, tl.float32)
    tmp_inv_std      = tl.math.rsqrt(tmp3 + tmp_eps)    # 1/sqrt(var+eps)
    tmp_scale        = tmp12 * tmp_inv_std               # weight/sqrt(var+eps)
    tmp_shifted_bias = tmp14 - tmp1 * tmp_scale          # bias - mean*scale  [neg-FMA]
    # Hot path: single FMA per element — PTX emits FFMA
    tmp_out = tmp0 * tmp_scale + tmp_shifted_bias
    # hardtanh [0, 6]
    tmp_zero = tl.full([1], 0.0, tl.float32)
    tmp_six  = tl.full([1], 6.0, tl.float32)
    tmp_out  = tl.maximum(tmp_out, tmp_zero)
    tmp_out  = tl.minimum(tmp_out, tmp_six)
    tl.store(in_out_ptr0 + (x2), tmp_out, xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1881600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((240,), (1,), device='cuda:0', dtype=torch.float32)
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
