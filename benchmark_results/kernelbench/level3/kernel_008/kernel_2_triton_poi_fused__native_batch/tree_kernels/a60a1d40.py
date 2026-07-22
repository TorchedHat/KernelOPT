import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    # Block-level load of all 24 channel scalars ONCE per block
    # tl.arange must be power of 2, so use 32 with mask for the 24 valid elements
    chan_idx = tl.arange(0, 32)
    chan_mask = chan_idx < 24
    all_mean   = tl.load(in_ptr0 + chan_idx, mask=chan_mask, other=0.0)  # [32] with 24 valid
    all_var    = tl.load(in_ptr1 + chan_idx, mask=chan_mask, other=0.0)
    all_weight = tl.load(in_ptr2 + chan_idx, mask=chan_mask, other=0.0)
    all_bias   = tl.load(in_ptr3 + chan_idx, mask=chan_mask, other=0.0)
    # Per-element computation using register-resident channel scalars
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)  # channel index per element [0..23]
    tmp0  = tl.load(in_out_ptr0 + x2, None)
    # Use tl.gather to index register arrays by per-element channel index (no L1TEX stall)
    tmp1  = tl.gather(all_mean,   x0, 0)   # mean[channel]
    tmp3  = tl.gather(all_var,    x0, 0)   # var[channel]
    tmp12 = tl.gather(all_weight, x0, 0)   # weight[channel]
    tmp14 = tl.gather(all_bias,   x0, 0)   # bias[channel]
    tmp_eps   = tl.full([1], 1e-05, tl.float32)
    tmp_var_e = tmp3 + tmp_eps
    tmp_scale = tmp12 * libdevice.rsqrt(tmp_var_e)
    tmp_sb    = tl.math.fma(-tmp1, tmp_scale, tmp14)
    tmp_out   = tl.math.fma(tmp0, tmp_scale, tmp_sb)
    tmp17     = tl.maximum(tl.full([1], 0.0, tl.float32), tmp_out)
    tl.store(in_out_ptr0 + x2, tmp17, None)

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
