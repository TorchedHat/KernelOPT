import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=16),
        triton.Config({'XBLOCK': 2048}, num_warps=16),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    # === CTA startup: preload all 96 channel params once into registers ===
    # tl.arange must be power of 2, so use 128 with a mask for the 96-element load
    c_idx  = tl.arange(0, 128)
    c_mask = c_idx < 96
    mean_all   = tl.load(in_ptr0 + c_idx, mask=c_mask, other=0.0)  # [128] floats, valid for [0..95]
    var_all    = tl.load(in_ptr1 + c_idx, mask=c_mask, other=0.0)  # [128] floats
    weight_all = tl.load(in_ptr2 + c_idx, mask=c_mask, other=0.0)  # [128] floats
    bias_all   = tl.load(in_ptr3 + c_idx, mask=c_mask, other=0.0)  # [128] floats
    invstd_all = tl.math.rsqrt(var_all + 1e-5)                      # [128] floats, rsqrt done ONCE per CTA

    # === Hot path: per-element computation ===
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96).to(tl.int32)   # channel index per element, shape [XBLOCK]

    tmp0 = tl.load(in_out_ptr0 + (x2), mask=xmask)
    # Gather from register-resident channel arrays (no L1TEX stall)
    mean_x   = tl.gather(mean_all,   x0, 0)   # shape [XBLOCK]
    invstd_x = tl.gather(invstd_all, x0, 0)
    weight_x = tl.gather(weight_all, x0, 0)
    bias_x   = tl.gather(bias_all,   x0, 0)

    tmp2 = tmp0 - mean_x
    tmp11 = tmp2 * invstd_x
    tmp13 = tmp11 * weight_x
    tmp15 = tmp13 + bias_x
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, mask=xmask)

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
