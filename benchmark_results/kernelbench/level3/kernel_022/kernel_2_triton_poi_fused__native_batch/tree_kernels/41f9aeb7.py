import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    # --- Preamble: precompute 24 channel scalars once per CTA ---
    c_idx    = tl.arange(0, 32)                           # [32] (power-of-2 >= 24)
    c_mask   = c_idx < 24
    mean_c   = tl.load(in_ptr0 + c_idx, mask=c_mask, other=0.0)   # [32] mean per channel
    var_c    = tl.load(in_ptr1 + c_idx, mask=c_mask, other=0.0)   # [32] var per channel
    weight_c = tl.load(in_ptr2 + c_idx, mask=c_mask, other=0.0)   # [32] BN weight (gamma)
    bias_c   = tl.load(in_ptr3 + c_idx, mask=c_mask, other=0.0)   # [32] BN bias (beta)
    scale_c  = libdevice.rsqrt(var_c + 1e-5) * weight_c           # [32] rsqrt fires 32 times only!
    # --- Per-element body (no rsqrt, no evict_last scalar loads) ---
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]          # [XBLOCK]
    xmask   = xindex < xnumel
    x2      = xindex
    x0      = (xindex % 24)                              # channel index [XBLOCK]
    tmp0    = tl.load(in_out_ptr0 + x2, xmask)           # input element
    mean_val  = tl.gather(mean_c,  x0, 0)                # [XBLOCK] register lookup
    scale_val = tl.gather(scale_c, x0, 0)                # [XBLOCK] register lookup
    bias_val  = tl.gather(bias_c,  x0, 0)                # [XBLOCK] register lookup
    normed  = (tmp0 - mean_val) * scale_val + bias_val
    result  = tl.maximum(tl.full([1], 0.0, tl.float32), normed)
    tl.store(in_out_ptr0 + x2, result, xmask)

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
