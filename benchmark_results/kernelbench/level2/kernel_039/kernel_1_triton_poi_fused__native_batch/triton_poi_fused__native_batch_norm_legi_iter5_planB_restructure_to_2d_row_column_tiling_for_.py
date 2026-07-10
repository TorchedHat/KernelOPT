import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: XBLOCK=8192 (full column dimension), XROWS=4 rows per CTA
    # Grid = (256,) = 1024 rows / 4 rows per CTA
    XROWS: tl.constexpr = 4
    row_tile  = tl.program_id(0)
    row_start = row_tile * XROWS
    col_idx   = tl.arange(0, XBLOCK)  # [8192] full broadcast dimension
    # Load 5 broadcast vectors ONCE per CTA (no modulo needed)
    bias_val = tl.load(in_ptr0 + col_idx, eviction_policy='evict_last')
    weight   = tl.load(in_ptr1 + col_idx, eviction_policy='evict_last')
    mean     = tl.load(in_ptr2 + col_idx, eviction_policy='evict_last')
    var      = tl.load(in_ptr3 + col_idx, eviction_policy='evict_last')
    gamma    = tl.load(in_ptr4 + col_idx, eviction_policy='evict_last')
    beta     = tl.load(in_ptr5 + col_idx, eviction_policy='evict_last')
    # Precompute scale/offset once, reused across all XROWS rows
    inv_std     = tl.math.rsqrt(var + 1e-5)
    scale       = weight * gamma * inv_std
    mean_scaled = mean * scale
    offset      = beta - mean_scaled
    bias_scaled = bias_val * scale
    combined    = bias_scaled + offset
    # Loop over XROWS rows, reusing broadcast data in registers
    for r in tl.static_range(XROWS):
        row   = row_start + r
        x_val = tl.load(in_out_ptr0 + row * XBLOCK + col_idx, eviction_policy='evict_first')
        out   = x_val * scale + combined  # FFMA
        tl.store(in_out_ptr0 + row * XBLOCK + col_idx, out, eviction_policy='evict_first')

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
