import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr):
    # Standard single-tile-per-CTA: no loop, no NUM_SMS
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex % 8192
    x2 = xindex
    # No mask needed: xnumel=8388608 is exactly divisible by XBLOCK=128
    bias_val = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
    x_val    = tl.load(in_out_ptr0 + x2)
    weight   = tl.load(in_ptr1 + x0, eviction_policy='evict_last')
    mean     = tl.load(in_ptr2 + x0, eviction_policy='evict_last')
    var      = tl.load(in_ptr3 + x0, eviction_policy='evict_last')
    gamma    = tl.load(in_ptr4 + x0, eviction_policy='evict_last')
    beta     = tl.load(in_ptr5 + x0, eviction_policy='evict_last')
    # FMA-friendly BN: group terms to expose a*b+c patterns
    inv_std     = tl.math.rsqrt(var + 1e-5)
    scale       = weight * gamma * inv_std   # FMUL chain
    mean_scaled = mean * scale               # FMUL
    offset      = beta - mean_scaled         # FSUB
    bias_scaled = bias_val * scale           # FMUL
    combined    = bias_scaled + offset       # FADD
    out         = x_val * scale + combined   # FFMA - single fused instruction
    tl.store(in_out_ptr0 + x2, out)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8388608, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
