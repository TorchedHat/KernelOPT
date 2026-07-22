import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr = 128):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex

    # Load all 24 BN params as compact [32] vectors (power-of-2 arange) - 4 cache lines total
    # Only 24 elements are valid; mask the extra 8 with other=0.0
    chan_idx  = tl.arange(0, 32)         # shape [32], power-of-2
    chan_mask = chan_idx < 24            # only 24 channels are valid
    smem_mean   = tl.load(in_ptr0 + chan_idx, mask=chan_mask, other=0.0)
    smem_var    = tl.load(in_ptr1 + chan_idx, mask=chan_mask, other=0.0)
    smem_weight = tl.load(in_ptr2 + chan_idx, mask=chan_mask, other=0.0)
    smem_bias   = tl.load(in_ptr3 + chan_idx, mask=chan_mask, other=0.0)

    # Per-lane channel index (same logic as before)
    cta_chan_base = xoffset % 24
    x0 = (cta_chan_base + tl.arange(0, XBLOCK)) % 24  # shape [XBLOCK], values in [0,23]

    # Gather from preloaded [32] vectors into per-lane [XBLOCK] values
    tmp0   = tl.load(in_out_ptr0 + x2, xmask)
    mean   = tl.gather(smem_mean,   x0, 0)
    var    = tl.gather(smem_var,    x0, 0)
    weight = tl.gather(smem_weight, x0, 0)
    bias   = tl.gather(smem_bias,   x0, 0)

    eps     = tl.full([1], 1e-05, tl.float32)
    inv_std = tl.math.rsqrt(var + eps)
    scale   = weight * inv_std
    shift   = bias - mean * scale
    # Explicit FMA: fuses multiply and add into single FFMA PTX instruction
    out     = tl.math.fma(tmp0, scale, shift)
    out     = tl.maximum(tl.zeros([XBLOCK], dtype=tl.float32), out)
    tl.store(in_out_ptr0 + x2, out, xmask)

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
